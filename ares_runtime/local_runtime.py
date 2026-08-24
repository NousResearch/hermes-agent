"""Operator-owned lifecycle for a locally installed Ares runtime.

This is deliberately separate from the candidate-custody activation path.  It
owns one local, source-tracked runtime for an operator who wants a stable
agent while their checkout remains a development worktree.  It never imports
from that worktree after ``ares setup`` has completed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_SCHEMA = 2
_DEFAULT_UPSTREAM_REMOTE = "https://github.com/NousResearch/hermes-agent.git"
_DEFAULT_UPSTREAM_BRANCH = "main"


class AresLocalRuntimeError(RuntimeError):
    """Raised when the explicit local-runtime contract is not satisfied."""


def _desktop_launch_arguments(
    executable: Path,
    *,
    platform: str,
    environment: Mapping[str, str],
) -> list[str]:
    """Prefer XWayland only when Electron would otherwise hide its window."""

    arguments = [str(executable)]
    is_wayland = environment.get("XDG_SESSION_TYPE", "").strip().lower() == "wayland" or bool(
        environment.get("WAYLAND_DISPLAY", "").strip()
    )
    has_xwayland = bool(environment.get("DISPLAY", "").strip())
    if platform == "linux" and is_wayland and has_xwayland:
        arguments.append("--ozone-platform=x11")
    return arguments


@dataclass(frozen=True)
class AresLocalPaths:
    """All state owned by the local Ares runtime controller."""

    state_root: Path
    data_root: Path
    agent_home: Path
    launcher_path: Path
    unit_path: Path

    @property
    def config_path(self) -> Path:
        return self.state_root / "config.json"

    @property
    def lock_path(self) -> Path:
        return self.state_root / "control.lock"

    @property
    def releases_dir(self) -> Path:
        return self.data_root / "releases"

    @property
    def staging_dir(self) -> Path:
        return self.data_root / "staging"

    @property
    def current_link(self) -> Path:
        return self.data_root / "current"

    @property
    def previous_link(self) -> Path:
        return self.data_root / "previous"


def _default_paths() -> AresLocalPaths:
    home = Path.home()
    agent_home = Path(
        os.environ.get("ARES_HOME", str(home / ".ares"))
    ).expanduser()
    launcher_dir = Path(
        os.environ.get("ARES_BIN_DIR", str(home / ".local" / "bin"))
    ).expanduser()
    return AresLocalPaths(
        state_root=agent_home / "runtime-state",
        data_root=agent_home / "runtime",
        agent_home=agent_home,
        launcher_path=launcher_dir / "ares",
        unit_path=home / ".config" / "systemd" / "user" / "ares-gateway.service",
    )


class AresLocalRuntime:
    """Build, select, and launch the one stable local Ares runtime.

    ``current`` is the only active-runtime pointer.  ``previous`` is solely a
    rollback target.  Repository metadata is held separately in ``config`` so
    there is no duplicate active-runtime truth and no development-worktree
    fallback after setup.
    """

    def __init__(self, paths: AresLocalPaths | None = None) -> None:
        self.paths = paths or _default_paths()

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.paths.state_root.mkdir(parents=True, exist_ok=True)
        with self.paths.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _ensure_layout(self) -> None:
        self.paths.state_root.mkdir(parents=True, exist_ok=True)
        self.paths.releases_dir.mkdir(parents=True, exist_ok=True)
        self.paths.staging_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _run(
        args: Sequence[str | Path],
        *,
        cwd: Path | None = None,
        capture: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [str(arg) for arg in args]
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            check=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise AresLocalRuntimeError(
                f"command failed ({completed.returncode}): {' '.join(command)}"
                + (f"\n{detail}" if detail else "")
            )
        return completed

    @staticmethod
    def _git_output(source: Path, *args: str) -> str:
        return AresLocalRuntime._run(
            ["git", "-C", source, *args], capture=True
        ).stdout.strip()

    @staticmethod
    def _require_revision(value: object) -> str:
        revision = str(value)
        if not _REVISION_RE.fullmatch(revision):
            raise AresLocalRuntimeError("invalid Ares release revision")
        return revision

    def _release_dir(self, revision: str) -> Path:
        return self.paths.releases_dir / self._require_revision(revision)

    def _release_source(self, revision: str) -> Path:
        source = self._release_dir(revision) / "source"
        if not source.is_dir():
            raise AresLocalRuntimeError(f"release {revision} is not installed")
        return source

    @staticmethod
    def _python_for(source: Path) -> Path:
        return source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _release_from_link(self, link: Path, label: str) -> tuple[str, Path] | None:
        if not link.is_symlink():
            return None
        source = link.resolve(strict=True)
        try:
            relative = source.relative_to(self.paths.releases_dir.resolve())
        except ValueError as exc:
            raise AresLocalRuntimeError(f"{label} pointer escapes the Ares release directory") from exc
        if len(relative.parts) != 2 or relative.parts[1] != "source":
            raise AresLocalRuntimeError(f"{label} pointer has an invalid release layout")
        revision = self._require_revision(relative.parts[0])
        if source != self._release_source(revision).resolve():
            raise AresLocalRuntimeError(f"{label} pointer does not match its release")
        return revision, source

    def active_release(self) -> tuple[str, Path]:
        value = self._release_from_link(self.paths.current_link, "current")
        if value is None:
            raise AresLocalRuntimeError("Ares is not set up; run `ares setup --source <checkout>`")
        return value

    def previous_release(self) -> tuple[str, Path] | None:
        return self._release_from_link(self.paths.previous_link, "previous")

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _atomic_link(path: Path, target: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        try:
            os.symlink(str(target), temporary)
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()

    def _read_config(self) -> dict[str, object]:
        try:
            raw = json.loads(self.paths.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AresLocalRuntimeError("Ares source configuration is missing; run `ares setup`") from exc
        except json.JSONDecodeError as exc:
            raise AresLocalRuntimeError("Ares source configuration is invalid") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") not in {1, _CONFIG_SCHEMA}:
            raise AresLocalRuntimeError("Ares source configuration has an unsupported schema")
        remote = raw.get("remote")
        branch = raw.get("branch")
        if not isinstance(remote, str) or not remote.strip() or "\n" in remote:
            raise AresLocalRuntimeError("Ares source configuration has an invalid remote")
        if not isinstance(branch, str) or not branch.strip() or "\n" in branch:
            raise AresLocalRuntimeError("Ares source configuration has an invalid branch")
        upstream_remote = raw.get("upstream_remote", _DEFAULT_UPSTREAM_REMOTE)
        upstream_branch = raw.get("upstream_branch", _DEFAULT_UPSTREAM_BRANCH)
        if not isinstance(upstream_remote, str) or not upstream_remote.strip() or "\n" in upstream_remote:
            raise AresLocalRuntimeError("Ares source configuration has an invalid upstream remote")
        if not isinstance(upstream_branch, str) or not upstream_branch.strip() or "\n" in upstream_branch:
            raise AresLocalRuntimeError("Ares source configuration has an invalid upstream branch")
        raw["upstream_remote"] = upstream_remote
        raw["upstream_branch"] = upstream_branch
        return raw

    def _write_config(
        self,
        *,
        remote: str,
        branch: str,
        upstream_remote: str = _DEFAULT_UPSTREAM_REMOTE,
        upstream_branch: str = _DEFAULT_UPSTREAM_BRANCH,
    ) -> None:
        self._atomic_json(
            self.paths.config_path,
            {
                "schema_version": _CONFIG_SCHEMA,
                "remote": remote,
                "branch": branch,
                "upstream_remote": upstream_remote,
                "upstream_branch": upstream_branch,
            },
        )

    def _activate(self, revision: str) -> None:
        target = self._release_source(revision).resolve()
        current = self._release_from_link(self.paths.current_link, "current")
        if current is not None and current[1] == target:
            return
        if current is not None:
            self._atomic_link(self.paths.previous_link, current[1])
        self._atomic_link(self.paths.current_link, target)

    def _build_environment(self, source: Path) -> dict[str, str]:
        """Return a clean build environment scoped to this Ares installation."""

        environment = os.environ.copy()
        for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
            environment.pop(name, None)
        # Candidate builds must be profile-isolated and must resolve the Node
        # version that the Ares runtime owns, never an ambient system Node.
        environment["HERMES_HOME"] = str(self.paths.agent_home)
        environment["UV_PROJECT_ENVIRONMENT"] = str(source / ".venv")
        node_dirs = [self.paths.agent_home / "node" / "bin", self.paths.agent_home / "node"]
        existing_path = environment.get("PATH", "")
        environment["PATH"] = os.pathsep.join(
            [str(directory) for directory in node_dirs if directory.is_dir()]
            + ([existing_path] if existing_path else [])
        )
        return environment

    def _agent_environment(self) -> dict[str, str]:
        """Return the process environment for the isolated Ares agent home."""

        environment = os.environ.copy()
        environment["HERMES_HOME"] = str(self.paths.agent_home)
        environment["ARES_MANAGED_RUNTIME"] = "1"
        return environment

    def _managed_npm(self) -> str | None:
        """Resolve npm from Ares's private managed Node installation."""

        from hermes_constants import (
            bootstrap_hermes_managed_node,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(self.paths.agent_home)
        try:
            return bootstrap_hermes_managed_node()
        finally:
            reset_hermes_home_override(token)

    def _seed_agent_home(self, source_home: Path) -> bool:
        """Create an independent Ares home from the useful Hermes settings once.

        This is an explicit migration, not a runtime fallback.  Secrets and
        settings are copied so the already working provider configuration is
        available immediately; later changes are independent in each home.
        """

        source_home = source_home.expanduser().resolve()
        marker = self.paths.agent_home / "ares-migration.json"
        existing_marker: dict[str, object] = {}
        if marker.is_file():
            try:
                raw_marker = json.loads(marker.read_text(encoding="utf-8"))
                if isinstance(raw_marker, dict):
                    existing_marker = raw_marker
            except json.JSONDecodeError:
                raise AresLocalRuntimeError("Ares migration record is invalid") from None
        if self.paths.agent_home.exists():
            managed_entries = {"runtime", "runtime-state"}
            if not existing_marker and any(
                entry.name not in managed_entries for entry in self.paths.agent_home.iterdir()
            ):
                return False
        else:
            self.paths.agent_home.mkdir(parents=True, exist_ok=False)
        copied: list[str] = []
        for name in ("config.yaml", ".env", "auth.json", "active_profile"):
            candidate = source_home / name
            destination = self.paths.agent_home / name
            if candidate.is_file() and not destination.exists():
                shutil.copy2(candidate, destination)
                copied.append(name)
        for name in ("profiles", "skills", "plugins"):
            candidate = source_home / name
            if candidate.is_dir() and not (self.paths.agent_home / name).exists():
                shutil.copytree(candidate, self.paths.agent_home / name, symlinks=True)
                copied.append(name)
        previous_copied = existing_marker.get("copied", [])
        if not isinstance(previous_copied, list) or not all(
            isinstance(value, str) for value in previous_copied
        ):
            previous_copied = []
        self._atomic_json(
            self.paths.agent_home / "ares-migration.json",
            {
                "schema_version": 1,
                "source_home": str(source_home),
                "copied": sorted(set(previous_copied) | set(copied)),
                "migrated_at": int(time.time()),
            },
        )
        return bool(copied)

    def _provision_context_governor_key(self, source: Path) -> None:
        """Initialize the Ares-owned key only for the configured strict engine."""

        python = self._python_for(source)
        if not python.is_file():
            raise AresLocalRuntimeError("stable Ares Python is missing during Context Governor setup")
        program = """
from pathlib import Path
import shutil
import yaml
from hermes_constants import get_hermes_home
from plugins.context_engine._context_governor.key_state import ContextGovernorKeyError, ContextGovernorKeyState

config_path = Path(get_hermes_home()) / 'config.yaml'
config = yaml.safe_load(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
if (config or {}).get('context', {}).get('engine') == 'ri-context-governor':
    binary = shutil.which('context-governor')
    if not binary:
        raise RuntimeError('context-governor binary is unavailable')
    state = ContextGovernorKeyState(get_hermes_home(), binary)
    try:
        binding = state.active_binding()
    except ContextGovernorKeyError as exc:
        if exc.code != 'MissingGovernedKey':
            raise
        binding = state.initialize_first_install()
    binding.close()
"""
        self._run(
            [python, "-c", program],
            cwd=source,
            env=self._agent_environment(),
        )

    def _build_runtime(self, source: Path, *, desktop: bool) -> None:
        from hermes_cli.managed_uv import ensure_uv

        uv = ensure_uv()
        if not uv:
            raise AresLocalRuntimeError("`uv` is required to build the stable Ares runtime")
        environment = self._build_environment(source)
        self._run(
            # Current Hermes intentionally rejects wheel/non-editable installs.
            # Ares retains immutable source beside the venv, so its supported
            # editable install remains release-safe and reproducible.
            [str(uv), "sync", "--locked", "--extra", "all", "--no-dev"],
            cwd=source,
            env=environment,
        )
        python = self._python_for(source)
        if not python.is_file():
            raise AresLocalRuntimeError("Ares runtime build did not create its Python interpreter")
        self._run(
            [python, "-c", "import ares_runtime.local_runtime; import hermes_cli.main"],
            cwd=source,
            env=self._build_environment(source),
        )
        if desktop:
            npm = self._managed_npm()
            if npm is None:
                raise AresLocalRuntimeError(
                    "Ares could not provision its managed npm for the Desktop build"
                )
            desktop_environment = self._build_environment(source)
            # The Desktop build requires workspace dev dependencies even when
            # the Python runtime deliberately excludes development extras.
            self._run([npm, "ci", "--include=dev"], cwd=source, env=desktop_environment)
            self._run([npm, "run", "pack"], cwd=source / "apps" / "desktop", env=desktop_environment)
            if self._desktop_binary(source) is None:
                raise AresLocalRuntimeError("Ares Desktop build completed without an executable")

    @staticmethod
    def _desktop_binary(source: Path) -> Path | None:
        if sys.platform == "darwin":
            candidate = source / "apps" / "desktop" / "release" / "mac" / "Ares.app" / "Contents" / "MacOS" / "Ares"
        elif os.name == "nt":
            candidate = source / "apps" / "desktop" / "release" / "win-unpacked" / "Ares.exe"
        else:
            candidate = source / "apps" / "desktop" / "release" / "linux-unpacked" / "Ares"
        return candidate if candidate.is_file() else None

    def _materialize(self, source_spec: str, revision: str, *, desktop: bool) -> None:
        self._ensure_layout()
        final_dir = self._release_dir(revision)
        if final_dir.exists():
            source = self._release_source(revision)
            self._build_runtime(source, desktop=desktop and self._desktop_binary(source) is None)
            return
        staging = self.paths.staging_dir / f"{revision}.{uuid.uuid4().hex}"
        source = staging / "source"
        try:
            self._run(["git", "clone", "--no-local", source_spec, source])
            self._run(["git", "-C", source, "checkout", "--detach", revision])
            self._build_runtime(source, desktop=desktop)
            self._atomic_json(
                staging / "release.json",
                {
                    "revision": revision,
                    "source": source_spec,
                    "installed_at": int(time.time()),
                },
            )
            os.replace(staging, final_dir)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    @staticmethod
    def _remote_revision(remote: str, branch: str) -> str:
        """Resolve one remote branch without trusting an ambient checkout."""

        output = AresLocalRuntime._run(
            ["git", "ls-remote", remote, f"refs/heads/{branch}"], capture=True
        ).stdout.strip()
        fields = output.split()
        if not fields:
            raise AresLocalRuntimeError(
                f"remote {remote!r} does not expose branch {branch!r}"
            )
        return AresLocalRuntime._require_revision(fields[0])

    def _release_metadata(self, revision: str) -> dict[str, object]:
        """Read the small release descriptor used only for update short-circuiting."""

        path = self._release_dir(revision) / "release.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _materialize_upstream_candidate(
        self,
        *,
        downstream_remote: str,
        downstream_revision: str,
        upstream_remote: str,
        upstream_branch: str,
        upstream_revision: str,
        desktop: bool,
    ) -> str:
        """Build one isolated Ares candidate from upstream plus the downstream delta.

        The staging checkout never touches the operator's Ares worktree.  It
        starts at the immutable upstream revision, applies the exact delta from
        the common ancestor to the configured Ares revision, and is built
        before its release directory becomes visible.  A merge or build failure
        therefore leaves ``current`` unchanged.
        """

        self._ensure_layout()
        downstream_revision = self._require_revision(downstream_revision)
        upstream_revision = self._require_revision(upstream_revision)
        staging = self.paths.staging_dir / f"candidate.{uuid.uuid4().hex}"
        source = staging / "source"
        patch_path = staging / "ares.patch"
        try:
            self._run(["git", "clone", "--no-local", downstream_remote, source])
            self._run(["git", "-C", source, "checkout", "--detach", downstream_revision])
            self._run(["git", "-C", source, "remote", "add", "ares-upstream", upstream_remote])
            self._run(
                ["git", "-C", source, "fetch", "--no-tags", "ares-upstream", upstream_branch]
            )
            fetched_upstream = self._require_revision(
                self._git_output(source, "rev-parse", "FETCH_HEAD")
            )
            if fetched_upstream != upstream_revision:
                raise AresLocalRuntimeError(
                    "upstream changed while preparing the Ares release candidate"
                )
            merge_base = self._require_revision(
                self._git_output(source, "merge-base", downstream_revision, upstream_revision)
            )
            downstream_patch = self._run(
                [
                    "git",
                    "-C",
                    source,
                    "diff",
                    "--binary",
                    "--full-index",
                    f"{merge_base}..{downstream_revision}",
                ],
                capture=True,
            ).stdout
            patch_path.write_text(downstream_patch, encoding="utf-8")
            self._run(["git", "-C", source, "checkout", "--detach", upstream_revision])
            if downstream_patch:
                self._run(["git", "-C", source, "apply", "--index", "--3way", patch_path])
            cached_diff = subprocess.run(
                ["git", "-C", str(source), "diff", "--cached", "--quiet"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if cached_diff.returncode not in {0, 1}:
                detail = (cached_diff.stderr or cached_diff.stdout).strip()
                raise AresLocalRuntimeError(
                    "could not inspect the staged Ares patch"
                    + (f": {detail}" if detail else "")
                )
            has_downstream_delta = cached_diff.returncode == 1
            if has_downstream_delta:
                tree = self._git_output(source, "write-tree")
                candidate_environment = os.environ.copy()
                candidate_environment.update(
                    {
                        "GIT_AUTHOR_NAME": "Ares Runtime",
                        "GIT_AUTHOR_EMAIL": "ares-runtime@localhost",
                        "GIT_COMMITTER_NAME": "Ares Runtime",
                        "GIT_COMMITTER_EMAIL": "ares-runtime@localhost",
                    }
                )
                candidate_revision = self._require_revision(
                    self._run(
                        [
                            "git",
                            "-C",
                            source,
                            "commit-tree",
                            tree,
                            "-p",
                            upstream_revision,
                            "-m",
                            f"Ares release candidate from Hermes {upstream_revision}",
                        ],
                        capture=True,
                        env=candidate_environment,
                    ).stdout.strip()
                )
                self._run(["git", "-C", source, "reset", "--hard", candidate_revision])
            else:
                candidate_revision = upstream_revision
            final_dir = self._release_dir(candidate_revision)
            if final_dir.exists():
                existing = self._release_metadata(candidate_revision)
                if (
                    existing.get("upstream_revision") != upstream_revision
                    or existing.get("downstream_revision") != downstream_revision
                ):
                    raise AresLocalRuntimeError(
                        f"release candidate revision collision: {candidate_revision}"
                    )
                self._build_runtime(self._release_source(candidate_revision), desktop=desktop)
                return candidate_revision
            self._build_runtime(source, desktop=desktop)
            self._atomic_json(
                staging / "release.json",
                {
                    "revision": candidate_revision,
                    "source": downstream_remote,
                    "downstream_revision": downstream_revision,
                    "upstream_remote": upstream_remote,
                    "upstream_branch": upstream_branch,
                    "upstream_revision": upstream_revision,
                    "installed_at": int(time.time()),
                },
            )
            os.replace(staging, final_dir)
            return candidate_revision
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _install_launcher(self) -> None:
        self.paths.launcher_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"export ARES_HOME={str(self.paths.agent_home)!r}\n"
            f"export ARES_BIN_DIR={str(self.paths.launcher_path.parent)!r}\n"
            f"runtime_root={str(self.paths.current_link)!r}\n"
            "python=\"$runtime_root/.venv/bin/python\"\n"
            "if [[ ! -x \"$python\" ]]; then\n"
            "  printf '%s\\n' 'Ares runtime is not installed; run ares setup from the Ares checkout.' >&2\n"
            "  exit 1\n"
            "fi\n"
            "cd \"$runtime_root\"\n"
            "exec \"$python\" -m ares_runtime.local_runtime \"$@\"\n"
        )
        temporary = self.paths.launcher_path.with_name(f".{self.paths.launcher_path.name}.{uuid.uuid4().hex}")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o755)
        os.replace(temporary, self.paths.launcher_path)

    def _install_gateway_unit(self) -> None:
        self.paths.unit_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "[Unit]\n"
            "Description=Ares stable gateway\n"
            "After=network-online.target\n"
            "Wants=network-online.target\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"Environment=HERMES_HOME={self.paths.agent_home}\n"
            f"ExecStart={self.paths.launcher_path} gateway foreground\n"
            "Restart=on-failure\n"
            "RestartSec=3\n\n"
            "TimeoutStopSec=210\n\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )
        temporary = self.paths.unit_path.with_name(f".{self.paths.unit_path.name}.{uuid.uuid4().hex}")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, self.paths.unit_path)

    @staticmethod
    def _systemd_environment() -> dict[str, str]:
        """Resolve the normal user bus when Ares starts outside a login shell."""

        environment = os.environ.copy()
        runtime_dir = environment.get("XDG_RUNTIME_DIR")
        if not runtime_dir:
            candidate = Path("/run/user") / str(os.getuid())
            if candidate.is_dir():
                runtime_dir = str(candidate)
                environment["XDG_RUNTIME_DIR"] = runtime_dir
        if runtime_dir and not environment.get("DBUS_SESSION_BUS_ADDRESS"):
            bus = Path(runtime_dir) / "bus"
            if bus.exists():
                environment["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"
        return environment

    def _systemctl(self, *args: str, required: bool = True) -> bool:
        if shutil.which("systemctl") is None:
            if required:
                raise AresLocalRuntimeError("systemd user services are unavailable on this host")
            return False
        completed = subprocess.run(
            ["systemctl", "--user", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._systemd_environment(),
            check=False,
        )
        if completed.returncode and required:
            detail = (completed.stderr or completed.stdout).strip()
            raise AresLocalRuntimeError(
                f"systemctl --user {' '.join(args)} failed"
                + (f": {detail}" if detail else "")
            )
        return completed.returncode == 0

    def _handoff_gateway(self, *, legacy_active: bool) -> None:
        self._systemctl("daemon-reload")
        if legacy_active:
            self._systemctl("disable", "--now", "hermes-gateway.service")
        try:
            self._systemctl("enable", "ares-gateway.service")
            self._systemctl("restart", "ares-gateway.service")
            time.sleep(1)
            if not self._systemctl("is-active", "--quiet", "ares-gateway.service", required=False):
                raise AresLocalRuntimeError("Ares gateway did not remain active after startup")
        except Exception:
            self._systemctl("disable", "--now", "ares-gateway.service", required=False)
            if legacy_active:
                self._systemctl("enable", "--now", "hermes-gateway.service", required=False)
            raise

    def setup(
        self,
        source: Path,
        *,
        desktop: bool,
        gateway: bool,
        seed_from: Path,
        upstream_remote: str = _DEFAULT_UPSTREAM_REMOTE,
        upstream_branch: str = _DEFAULT_UPSTREAM_BRANCH,
    ) -> tuple[str, bool]:
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise AresLocalRuntimeError(f"Ares source checkout does not exist: {source}")
        if self._git_output(source, "rev-parse", "--is-inside-work-tree") != "true":
            raise AresLocalRuntimeError(f"not a Git checkout: {source}")
        revision = self._require_revision(self._git_output(source, "rev-parse", "HEAD"))
        try:
            remote = self._git_output(source, "remote", "get-url", "origin")
        except AresLocalRuntimeError:
            remote = str(source)
        try:
            branch = self._git_output(source, "symbolic-ref", "--quiet", "--short", "HEAD")
        except AresLocalRuntimeError:
            branch = "main"
        with self.locked():
            old_active = self._release_from_link(self.paths.current_link, "current")
            legacy_active = self._systemctl("is-active", "--quiet", "hermes-gateway.service", required=False)
            self._materialize(str(source), revision, desktop=desktop)
            seeded = self._seed_agent_home(seed_from)
            self._provision_context_governor_key(self._release_source(revision))
            self._activate(revision)
            self._write_config(
                remote=remote,
                branch=branch,
                upstream_remote=upstream_remote,
                upstream_branch=upstream_branch,
            )
            self._install_launcher()
            if gateway:
                self._install_gateway_unit()
                try:
                    self._handoff_gateway(legacy_active=legacy_active)
                except Exception:
                    if old_active is not None:
                        self._atomic_link(self.paths.current_link, old_active[1])
                        # The launcher resolves through `current`; regenerate it
                        # before reviving the prior gateway, otherwise a failed
                        # candidate can strand rollback on its new wrapper.
                        self._install_launcher()
                        self._systemctl("enable", "ares-gateway.service", required=False)
                        self._systemctl("restart", "ares-gateway.service", required=False)
                    else:
                        self.paths.current_link.unlink(missing_ok=True)
                    raise
        return revision, seeded

    def update(self, *, desktop: bool) -> tuple[str, bool]:
        with self.locked():
            config = self._read_config()
            remote = str(config["remote"])
            branch = str(config["branch"])
            upstream_remote = str(config["upstream_remote"])
            upstream_branch = str(config["upstream_branch"])
            downstream_revision = self._remote_revision(remote, branch)
            upstream_revision = self._remote_revision(upstream_remote, upstream_branch)
            current = self._release_from_link(self.paths.current_link, "current")
            if current is not None:
                metadata = self._release_metadata(current[0])
                if (
                    metadata.get("downstream_revision") == downstream_revision
                    and metadata.get("upstream_revision") == upstream_revision
                    and metadata.get("upstream_remote") == upstream_remote
                    and metadata.get("upstream_branch") == upstream_branch
                ):
                    return current[0], False
            old_active = current
            revision = self._materialize_upstream_candidate(
                downstream_remote=remote,
                downstream_revision=downstream_revision,
                upstream_remote=upstream_remote,
                upstream_branch=upstream_branch,
                upstream_revision=upstream_revision,
                desktop=desktop,
            )
            self._activate(revision)
            if self.paths.unit_path.exists():
                try:
                    self._install_gateway_unit()
                    self._systemctl("daemon-reload")
                    self._systemctl("restart", "ares-gateway.service")
                    time.sleep(1)
                    if not self._systemctl("is-active", "--quiet", "ares-gateway.service", required=False):
                        raise AresLocalRuntimeError("Ares gateway did not remain active after update")
                except Exception:
                    if old_active is not None:
                        self._atomic_link(self.paths.current_link, old_active[1])
                        self._systemctl("restart", "ares-gateway.service", required=False)
                    raise
            return revision, True

    def rollback(self) -> str:
        with self.locked():
            current = self.active_release()
            previous = self.previous_release()
            if previous is None:
                raise AresLocalRuntimeError("no previous Ares runtime is available for rollback")
            self._atomic_link(self.paths.current_link, previous[1])
            self._atomic_link(self.paths.previous_link, current[1])
            if self.paths.unit_path.exists():
                self._systemctl("restart", "ares-gateway.service")
                time.sleep(1)
                if not self._systemctl("is-active", "--quiet", "ares-gateway.service", required=False):
                    self._atomic_link(self.paths.current_link, current[1])
                    self._atomic_link(self.paths.previous_link, previous[1])
                    self._systemctl("restart", "ares-gateway.service", required=False)
                    raise AresLocalRuntimeError("Ares gateway did not remain active after rollback")
            return previous[0]

    @staticmethod
    def _source_cleanliness(source: Path) -> tuple[bool, str]:
        """Return whether the selected release source has a clean Git tree."""
        probe = subprocess.run(
            ["git", "-C", str(source), "status", "--short"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if probe.returncode:
            return False, "could not inspect selected release Git state"
        changed = [line for line in probe.stdout.splitlines() if line.strip()]
        if not changed:
            return True, "clean"
        return False, f"dirty ({len(changed)} path(s))"

    def doctor(self) -> list[tuple[str, bool, str]]:
        checks: list[tuple[str, bool, str]] = []
        try:
            revision, source = self.active_release()
            checks.append(("active runtime", True, revision))
        except AresLocalRuntimeError as exc:
            checks.append(("active runtime", False, str(exc)))
            return checks
        python = self._python_for(source)
        checks.append(("stable Python", python.is_file(), str(python)))
        clean, cleanliness = self._source_cleanliness(source)
        checks.append(("selected release tree", clean, cleanliness))
        if python.is_file():
            probe = subprocess.run(
                [python, "-c", "import ares_runtime.local_runtime; import hermes_cli.main"],
                cwd=source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            checks.append(
                ("Ares runtime imports", probe.returncode == 0, (probe.stderr or "ok").strip())
            )
            try:
                from hermes_cli.sqlite_runtime import probe_sqlite_runtime

                sqlite = probe_sqlite_runtime(python)
                if sqlite is None:
                    checks.append(("SQLite runtime", False, "could not probe selected interpreter"))
                elif sqlite.wal_reset_vulnerable:
                    checks.append(
                        (
                            "SQLite runtime",
                            False,
                            f"vulnerable SQLite {sqlite.sqlite_version_string}",
                        )
                    )
                else:
                    checks.append(("SQLite runtime", True, sqlite.sqlite_version_string))
            except Exception as exc:
                checks.append(("SQLite runtime", False, f"probe failed: {type(exc).__name__}"))
        context_probe = """
from pathlib import Path
import json
import yaml
from hermes_constants import get_hermes_home

config_path = Path(get_hermes_home()) / 'config.yaml'
config = yaml.safe_load(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
engine = (config or {}).get('context', {}).get('engine', 'compressor')
if engine == 'ri-context-governor':
    from plugins.context_engine._context_governor import ContextGovernorEngine
    capabilities = ContextGovernorEngine().probe_activation()
    print(json.dumps({'engine': capabilities['engine'], 'strict_probe': 'passed'}))
else:
    print(json.dumps({'engine': engine, 'strict_probe': 'not configured'}))
"""
        try:
            probe = self._run(
                [python, "-c", context_probe],
                cwd=source,
                capture=True,
                env=self._agent_environment(),
            )
            checks.append(("Context Governor strict probe", True, probe.stdout.strip()))
        except AresLocalRuntimeError as exc:
            checks.append(("Context Governor strict probe", False, str(exc)))
        mcp_probe = """
import json
from hermes_cli.config import load_config_readonly
from tools.mcp_tool import probe_mcp_server_tools

config = load_config_readonly() or {}
servers = config.get('mcp_servers') or {}
enabled = sorted(
    name for name, value in servers.items()
    if isinstance(value, dict) and value.get('enabled', True) is not False
)
probed = probe_mcp_server_tools()
missing = [name for name in enabled if name not in probed]
print(json.dumps({'enabled': enabled, 'probed': sorted(probed), 'missing': missing}))
"""
        try:
            probe = subprocess.run(
                [python, "-c", mcp_probe],
                cwd=source,
                env=self._agent_environment(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=150,
                check=False,
            )
            if probe.returncode:
                checks.append(("MCP readiness", False, "bounded probe failed"))
            else:
                result = json.loads(probe.stdout)
                missing = result.get("missing") or []
                checks.append(
                    (
                        "MCP readiness",
                        not missing,
                        "all enabled servers responded" if not missing else f"unavailable: {', '.join(missing)}",
                    )
                )
        except subprocess.TimeoutExpired:
            checks.append(("MCP readiness", False, "bounded probe timed out"))
        except Exception as exc:
            checks.append(("MCP readiness", False, f"probe failed: {type(exc).__name__}"))
        gateway_active = self._systemctl("is-active", "--quiet", "ares-gateway.service", required=False)
        checks.append(("Ares gateway", gateway_active, "active" if gateway_active else "inactive"))
        return checks

    def status(self) -> list[str]:
        current = self._release_from_link(self.paths.current_link, "current")
        previous = self.previous_release()
        try:
            config = self._read_config()
        except AresLocalRuntimeError as exc:
            return [f"Ares status: {exc}"]
        return [
            f"active: {current[0] if current else 'none'}",
            f"previous: {previous[0] if previous else 'none'}",
            f"remote: {config['remote']}",
            f"branch: {config['branch']}",
            f"gateway: {'active' if self._systemctl('is-active', '--quiet', 'ares-gateway.service', required=False) else 'inactive'}",
        ]

    def _exec_hermes(self, arguments: Sequence[str]) -> None:
        _, source = self.active_release()
        python = self._python_for(source)
        if not python.is_file():
            raise AresLocalRuntimeError("stable Ares Python is missing; run `ares update`")
        os.chdir(source)
        os.execve(
            str(python),
            [str(python), "-m", "hermes_cli.main", *arguments],
            self._agent_environment(),
        )

    def tui(self, arguments: Sequence[str]) -> None:
        self._exec_hermes(["--tui", *arguments])

    def chat(self, arguments: Sequence[str]) -> None:
        self._exec_hermes(arguments)

    def gateway(self, action: str) -> None:
        if action == "foreground":
            self._exec_hermes(["gateway"])
        if action == "start":
            self._systemctl("enable", "--now", "ares-gateway.service")
        elif action == "stop":
            self._systemctl("disable", "--now", "ares-gateway.service")
        elif action == "restart":
            self._systemctl("restart", "ares-gateway.service")
        elif action == "status":
            active = self._systemctl("is-active", "--quiet", "ares-gateway.service", required=False)
            print("Ares gateway is " + ("active" if active else "inactive"))
            if not active:
                raise AresLocalRuntimeError("Ares gateway is inactive")
        else:
            raise AresLocalRuntimeError(f"unsupported gateway action: {action}")

    def auth(self, args, passthrough: Sequence[str]) -> None:
        """Delegate to hermes auth with Ares home environment."""
        # Build the hermes auth command line from args
        hermes_args = ["auth"]
        
        # Handle the case where user runs: ares auth spotify status
        # This gets parsed as auth_action='spotify', provider='status'
        # But hermes expects: hermes auth spotify status
        if args.auth_action == "spotify":
            if args.provider in ("login", "status", "logout"):
                # User ran: ares auth spotify <action>
                hermes_args.append("spotify")
                hermes_args.append(args.provider)
            elif args.provider is None and args.spotify_action:
                # User ran: ares auth spotify --spotify-action <action>
                hermes_args.append("spotify")
                hermes_args.append(args.spotify_action)
            else:
                # User ran: ares auth <action> spotify (e.g., ares auth add spotify)
                # This is not the normal pattern for spotify but handle it
                hermes_args.append(args.auth_action)
                if args.provider:
                    hermes_args.append(args.provider)
        elif args.provider == "spotify" and args.auth_action:
            # User ran: ares auth <action> spotify (e.g., ares auth add spotify)
            hermes_args.append(args.auth_action)
            hermes_args.append(args.provider)
        elif args.provider == "spotify" and not args.auth_action:
            # User ran: ares auth spotify (with --spotify-action flag)
            hermes_args.append("spotify")
            hermes_args.append(args.spotify_action or "login")
        else:
            # Normal provider + action pattern
            if args.auth_action:
                hermes_args.append(args.auth_action)
            if args.provider:
                hermes_args.append(args.provider)
        
        if args.auth_type:
            hermes_args.extend(["--type", args.auth_type])
        if args.label:
            hermes_args.extend(["--label", args.label])
        if args.api_key:
            hermes_args.extend(["--api-key", args.api_key])
        if args.portal_url:
            hermes_args.extend(["--portal-url", args.portal_url])
        if args.inference_url:
            hermes_args.extend(["--inference-url", args.inference_url])
        if args.client_id:
            hermes_args.extend(["--client-id", args.client_id])
        if args.scope:
            hermes_args.extend(["--scope", args.scope])
        if args.no_browser:
            hermes_args.append("--no-browser")
        if args.timeout:
            hermes_args.extend(["--timeout", str(args.timeout)])
        if args.insecure:
            hermes_args.append("--insecure")
        if args.ca_bundle:
            hermes_args.extend(["--ca-bundle", args.ca_bundle])
        if args.target:
            hermes_args.append(args.target)
        if args.redirect_uri:
            hermes_args.extend(["--redirect-uri", args.redirect_uri])
        # Add any passthrough args
        hermes_args.extend(passthrough)
        self._exec_hermes(hermes_args)

    def desktop(self, *, rebuild: bool) -> None:
        revision, source = self.active_release()
        if rebuild:
            with self.locked():
                self._build_runtime(source, desktop=True)
        executable = self._desktop_binary(source)
        if executable is None:
            raise AresLocalRuntimeError(
                "Ares Desktop is not built; run `ares update` or `ares desktop --rebuild`"
            )
        environment = self._agent_environment()
        environment["HERMES_DESKTOP_HERMES_ROOT"] = str(source)
        environment["HERMES_DESKTOP_PYTHON"] = str(self._python_for(source))
        environment["HERMES_DESKTOP_APP_NAME"] = "Ares"
        subprocess.Popen(
            _desktop_launch_arguments(executable, platform=sys.platform, environment=environment),
            cwd=source,
            env=environment,
            start_new_session=True,
        )
        print(f"Ares Desktop started from stable release {revision}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ares",
        description="Manage the stable local Ares runtime independently from its development checkout.",
    )
    subparsers = parser.add_subparsers(dest="command")
    setup = subparsers.add_parser("setup", help="Build and select a stable runtime from a Git checkout")
    setup.add_argument("--source", type=Path, default=Path.cwd(), help="Ares checkout to install (default: current directory)")
    setup.add_argument(
        "--seed-from",
        type=Path,
        default=Path.home() / ".hermes",
        help="copy settings and credentials from this Hermes home only when ~/.ares does not yet exist",
    )
    setup.add_argument("--no-desktop", action="store_true", help="Do not build or install Desktop")
    setup.add_argument("--no-gateway", action="store_true", help="Do not install or start the Ares gateway service")
    setup.add_argument(
        "--upstream-remote",
        default=_DEFAULT_UPSTREAM_REMOTE,
        help="Hermes upstream Git remote used to construct future release candidates",
    )
    setup.add_argument(
        "--upstream-branch",
        default=_DEFAULT_UPSTREAM_BRANCH,
        help="Hermes upstream branch used to construct future release candidates",
    )
    update = subparsers.add_parser("update", help="Build and atomically select the configured remote branch")
    update.add_argument("--no-desktop", action="store_true", help="Do not build Desktop for this release")
    subparsers.add_parser("rollback", help="Return to the previous stable runtime")
    subparsers.add_parser("doctor", help="Check the selected runtime and gateway")
    subparsers.add_parser("status", help="Show the selected runtime, remote, and gateway")
    desktop = subparsers.add_parser("desktop", help="Launch the selected Ares Desktop application")
    desktop.add_argument("--rebuild", action="store_true", help="Build Desktop in the selected stable runtime first")
    subparsers.add_parser("tui", help="Launch the selected TUI")
    subparsers.add_parser("chat", help="Launch the selected Ares CLI")
    gateway = subparsers.add_parser("gateway", help="Manage the selected Ares gateway service")
    gateway.add_argument("action", choices=("start", "stop", "restart", "status", "foreground"))
    # Auth subcommand - delegates to hermes auth with Ares home
    auth = subparsers.add_parser("auth", help="Manage pooled provider credentials in Ares home")
    auth.add_argument("auth_action", nargs="?", default="", help="Auth action (add, list, remove, reset, status, logout, spotify)")
    auth.add_argument("provider", nargs="?", help="Provider id (e.g., openai-codex, openrouter)")
    auth.add_argument("--type", dest="auth_type", choices=["oauth", "api-key", "api_key"], help="Credential type to add")
    auth.add_argument("--label", help="Optional display label")
    auth.add_argument("--api-key", help="API key value (otherwise prompted securely)")
    auth.add_argument("--portal-url", help="Nous portal base URL")
    auth.add_argument("--inference-url", help="Nous inference base URL")
    auth.add_argument("--client-id", help="OAuth client id")
    auth.add_argument("--scope", help="OAuth scope override")
    auth.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser for OAuth login")
    auth.add_argument("--timeout", type=float, help="OAuth/network timeout in seconds")
    auth.add_argument("--insecure", action="store_true", help="Disable TLS verification for OAuth login")
    auth.add_argument("--ca-bundle", help="Custom CA bundle for OAuth login")
    auth.add_argument("--target", help="Credential index, entry id, or exact label (for remove)")
    auth.add_argument("--spotify-action", choices=["login", "status", "logout"], default="login", help="Spotify auth action")
    auth.add_argument("--redirect-uri", help="Allow-listed localhost redirect URI for your Spotify app")
    parser.add_argument("--version", action="store_true", help="Print the selected stable runtime revision")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.command in {"chat", "tui"}:
        args.arguments = passthrough
    elif passthrough:
        parser.error("unrecognized arguments: " + " ".join(passthrough))
    runtime = AresLocalRuntime()
    try:
        if args.version:
            revision, _ = runtime.active_release()
            print(f"Ares {revision}")
        elif args.command == "setup":
            revision, seeded = runtime.setup(
                args.source,
                desktop=not args.no_desktop,
                gateway=not args.no_gateway,
                seed_from=args.seed_from,
                upstream_remote=args.upstream_remote,
                upstream_branch=args.upstream_branch,
            )
            print(f"Ares stable runtime selected: {revision}")
            if seeded:
                print(f"Ares home seeded once from: {args.seed_from}")
        elif args.command == "update":
            revision, changed = runtime.update(desktop=not args.no_desktop)
            print(("Updated" if changed else "Already current") + f" Ares runtime: {revision}")
        elif args.command == "rollback":
            print(f"Rolled back Ares runtime to: {runtime.rollback()}")
        elif args.command == "doctor":
            checks = runtime.doctor()
            for label, passed, detail in checks:
                print(f"{'PASS' if passed else 'FAIL'} {label}: {detail}")
            if not all(passed for _, passed, _ in checks):
                raise AresLocalRuntimeError("Ares doctor found failed checks")
        elif args.command == "status":
            print("\n".join(runtime.status()))
        elif args.command == "desktop":
            runtime.desktop(rebuild=args.rebuild)
        elif args.command == "tui":
            runtime.tui(args.arguments)
        elif args.command == "chat":
            runtime.chat(args.arguments)
        elif args.command == "gateway":
            runtime.gateway(args.action)
        elif args.command == "auth":
            runtime.auth(args, passthrough)
        else:
            runtime.tui(())
    except AresLocalRuntimeError as exc:
        print(f"ares: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
