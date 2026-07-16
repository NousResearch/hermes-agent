"""``hermes dev sync`` — the single provision verb for source checkouts.

Replaces the scattered provision steps that used to live in ``install.sh``
(venv + python deps), ``_update_node_dependencies`` (node deps),
``_build_web_ui`` (web build), and ``_desktop_build_needed`` /
``_write_desktop_build_stamp`` (desktop build + stamp).

The logic is intentionally kept in this module (not in the argparse
subcommand file) so it is testable without argparse — callers pass a
``tree_root`` path and get a :class:`SyncReport` back.

See ``docs/plans/updater-rework/04-phase3-ejected-dev.md`` task 3.2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DevSyncError(RuntimeError):
    """A mandatory checkout provisioning step failed."""


# ---------------------------------------------------------------------------
# Tree-kind detection (Python-side mirror of the Rust launcher's TreeKind)
# ---------------------------------------------------------------------------

def detect_tree_kind(tree_root: Path) -> str:
    """Return ``"slot"`` or ``"checkout"`` for *tree_root*.

    A **slot** (managed bundle) has ``manifest.json`` at its root.
    A **checkout** (source/ejected) has ``pyproject.toml`` + ``.git``.

    This is the Python-side counterpart of the Rust launcher's
    ``resolve_tree_root`` (phase 1, task 1.1).  The launcher detects this
    at exec time; we detect it here for the ``hermes dev`` subcommand.
    """
    if (tree_root / "manifest.json").is_file():
        return "slot"
    return "checkout"


# ---------------------------------------------------------------------------
# ArtifactStamp — content-hash gating, generalized from _desktop_build_needed
# ---------------------------------------------------------------------------

class ArtifactStamp:
    """Content-hash stamp that gates whether an artifact needs rebuilding.

    Generalizes the desktop build-stamp pattern (``_desktop_build_needed``
    / ``_write_desktop_build_stamp`` in ``hermes_cli/main.py``) into a
    reusable class.  The hashing logic is ported verbatim from
    ``_compute_desktop_content_hash`` — do not reinvent it.

    Usage::

        stamp = ArtifactStamp(
            stamp_file=hermes_home / "tui-build-stamp.json",
            source_globs=["ui-tui/src", "ui-tui/package.json"],
            project_root=tree_root,
            dist_sentinel=tree_root / "ui-tui" / "dist" / "entry.js",
        )
        if stamp.needs_build():
            ...  # run the build
            stamp.write_stamp()
    """

    def __init__(
        self,
        *,
        stamp_file: Path,
        source_globs: list[str],
        project_root: Path,
        dist_sentinel: Optional[Path] = None,
    ) -> None:
        self.stamp_file = stamp_file
        self.source_globs = source_globs
        self.project_root = project_root
        self.dist_sentinel = dist_sentinel

    # -- public API --

    def needs_build(self) -> bool:
        """Return True when the artifact is stale or missing."""
        # Missing dist → definitely needs build
        if self.dist_sentinel is not None and not self.dist_sentinel.exists():
            return True

        # Missing stamp → needs build
        if not self.stamp_file.is_file():
            return True

        try:
            stamp_data = json.loads(self.stamp_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True

        saved_hash = stamp_data.get("contentHash")
        if not saved_hash:
            return True

        current_hash = self.compute_content_hash()
        return current_hash != saved_hash

    def write_stamp(self) -> None:
        """Write the stamp file after a successful build."""
        try:
            self.stamp_file.parent.mkdir(parents=True, exist_ok=True)
            content_hash = self.compute_content_hash()
            stamp_data = {
                "contentHash": content_hash,
                "builtAt": datetime.now(timezone.utc).isoformat(),
            }
            self.stamp_file.write_text(
                json.dumps(stamp_data, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            # Never let stamp-writing block or fail a build
            logger.debug("Failed to write build stamp: %s", exc)

    # -- hashing (ported from _compute_desktop_content_hash) --

    def compute_content_hash(self) -> str:
        """Return a SHA-256 hex digest of all source files matching the globs.

        Walks each source glob relative to ``project_root``, honoring
        ``.gitignore`` via *pathspec* (same library the desktop stamp uses)
        so ``node_modules/``, ``dist/``, ``*.pyc`` etc. are skipped
        automatically without a hardcoded skip-list.
        """
        h = hashlib.sha256()

        def _hash_file(path: Path) -> None:
            rel = str(path.relative_to(self.project_root))
            h.update(rel.encode())
            h.update(b"\0")
            try:
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
            except (OSError, IOError):
                pass
            h.update(b"\0")

        from pathspec import PathSpec

        gitignore = self.project_root / ".gitignore"
        lines: list[str] = []
        if gitignore.is_file():
            lines = gitignore.read_text(encoding="utf-8").splitlines()
        spec = PathSpec.from_lines("gitignore", lines)

        for glob_pat in self.source_globs:
            base = self.project_root / glob_pat
            if not base.exists():
                continue
            if base.is_file():
                rel = str(base.relative_to(self.project_root))
                if not spec.match_file(rel):
                    _hash_file(base)
                continue
            # Directory walk — prune ignored dirs in-place
            for dirpath, dirnames, filenames in os.walk(base, topdown=True):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not spec.match_file(
                        str((Path(dirpath) / d).relative_to(self.project_root))
                    )
                ]
                for fn in sorted(filenames):
                    fp = Path(dirpath) / fn
                    rel = str(fp.relative_to(self.project_root))
                    if not spec.match_file(rel):
                        _hash_file(fp)

        return h.hexdigest()


# ---------------------------------------------------------------------------
# Sync report
# ---------------------------------------------------------------------------

@dataclass
class SyncReport:
    """Summary of what ``dev sync`` built or skipped."""

    venv_created: bool = False
    venv_synced: bool = False
    launcher_installed: bool = False
    node_deps_installed: bool = False
    tui_built: bool = False
    web_built: bool = False
    desktop_built: bool = False
    feature_ledger_applied: bool = False
    launcher_ok_deleted: bool = False
    skipped: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        """Return human-readable summary lines for printing."""
        lines: list[str] = []
        if self.venv_created:
            lines.append("  ✓ venv created")
        elif self.venv_synced:
            lines.append("  ✓ venv synced")
        else:
            lines.append("  · venv already up to date")

        if self.launcher_installed:
            lines.append("  ✓ launcher installed")
        if self.node_deps_installed:
            lines.append("  ✓ node deps installed")
        if self.tui_built:
            lines.append("  ✓ TUI dist built")
        if self.web_built:
            lines.append("  ✓ web dist built")
        if self.desktop_built:
            lines.append("  ✓ desktop packed")
        if self.feature_ledger_applied:
            lines.append("  ✓ feature ledger applied")
        for item in self.skipped:
            lines.append(f"  · {item}")
        return lines


# ---------------------------------------------------------------------------
# Dev sync — the single provision verb
# ---------------------------------------------------------------------------

def _stamp_dir(tree_root: Path) -> Path:
    """Return the directory for build stamps (inside the tree's .hermes-dev/).

    Stamps live inside the checkout (not ``$HERMES_HOME``) so that
    multiple worktrees each have independent stamp state.
    """
    return tree_root / ".hermes-dev"


def _tui_stamp(tree_root: Path) -> ArtifactStamp:
    """ArtifactStamp for the TUI dist build."""
    tui_dir = tree_root / "ui-tui"
    return ArtifactStamp(
        stamp_file=_stamp_dir(tree_root) / "tui-build-stamp.json",
        source_globs=[
            "ui-tui/src",
            "ui-tui/packages",
            "ui-tui/package.json",
            "ui-tui/package-lock.json",
            "ui-tui/tsconfig.json",
            "package.json",
            "package-lock.json",
        ],
        project_root=tree_root,
        dist_sentinel=tui_dir / "dist" / "entry.js",
    )


def _web_stamp(tree_root: Path) -> ArtifactStamp:
    """ArtifactStamp for the web dist build."""
    web_dir = tree_root / "web"
    return ArtifactStamp(
        stamp_file=_stamp_dir(tree_root) / "web-build-stamp.json",
        source_globs=[
            "web/src",
            "web/public",
            "web/package.json",
            "web/vite.config.ts",
            "web/vite.config.js",
            "package.json",
            "package-lock.json",
        ],
        project_root=tree_root,
        dist_sentinel=tree_root / "hermes_cli" / "web_dist" / ".vite" / "manifest.json",
    )


def _desktop_stamp(tree_root: Path) -> ArtifactStamp:
    """ArtifactStamp for the desktop build."""
    return ArtifactStamp(
        stamp_file=_stamp_dir(tree_root) / "desktop-build-stamp.json",
        source_globs=[
            "apps/desktop",
            "package.json",
            "package-lock.json",
        ],
        project_root=tree_root,
        dist_sentinel=tree_root / "apps" / "desktop" / "dist" / "index.html",
    )


def _has_desktop_build(tree_root: Path) -> bool:
    """Return True if a previous desktop build exists (same heuristic as cmd_update)."""
    desktop_dir = tree_root / "apps" / "desktop"
    if (desktop_dir / "dist" / "index.html").exists():
        return True
    # Check for packaged executables
    release_dir = desktop_dir / "release"
    if release_dir.is_dir():
        for p in release_dir.rglob("*"):
            if p.is_file() and p.stat().st_size > 100_000:
                return True
    return False


def run(
    tree_root: Path,
    *,
    watch: bool = False,
    only: Optional[list[str]] = None,
    desktop: bool = False,
    runner: Optional["SubprocessRunner"] = None,
) -> SyncReport:
    """Provision a source checkout in one verb.

    Steps (each gated where applicable):
    1. venv: ``uv venv .venv`` if missing; ``uv sync --extra all --locked``
       (fall back to ``uv pip install -e .[all]`` when lockfile is stale)
    2. install the release launcher into ``.hermes-launcher/`` (best-effort)
    3. node deps: root + ``ui-tui`` + ``web`` workspaces
    4. builds, each gated by an ArtifactStamp: tui dist, web dist,
       and desktop pack ONLY if ``--desktop`` or a previous desktop build exists
    5. apply the feature ledger if present (phase 5; soft-import)
    6. delete the venv's ``.launcher-ok`` stamp whenever step 1 touched the venv
    7. print a summary table

    Args:
        tree_root: Root of the source checkout.
        watch: If True, supervise the selected frontend dev processes after sync.
        only: If given, only run these steps (``"venv"``, ``"node"``,
              ``"tui"``, ``"web"``, ``"desktop"``).
        desktop: Force desktop build even if no previous build exists.
        runner: Optional :class:`SubprocessRunner` for testability.

    Returns:
        :class:`SyncReport` summarizing what was built/skipped.
    """
    if runner is None:
        runner = SubprocessRunner()

    report = SyncReport()
    steps = only or ["venv", "launcher", "node", "tui", "web", "desktop", "ledger"]

    # --- Step 1: venv ---
    if "venv" in steps:
        report = _sync_venv(tree_root, report, runner)

    # --- Step 2: launcher ---
    if "launcher" in steps:
        report = _install_launcher(tree_root, report, runner)

    # --- Step 3: node deps ---
    if "node" in steps:
        report = _sync_node_deps(tree_root, report, runner)

    # --- Step 4: builds ---
    if "tui" in steps:
        report = _build_tui(tree_root, report, runner)
    if "web" in steps:
        report = _build_web(tree_root, report, runner)
    if "desktop" in steps and (desktop or _has_desktop_build(tree_root)):
        report = _build_desktop(tree_root, report, runner)

    # --- Step 5: feature ledger ---
    if "ledger" in steps:
        report = _apply_feature_ledger(tree_root, report)

    # --- Step 6: delete .launcher-ok ---
    if report.venv_created or report.venv_synced:
        launcher_ok = tree_root / ".venv" / ".launcher-ok"
        if launcher_ok.exists():
            try:
                launcher_ok.unlink()
                report.launcher_ok_deleted = True
            except OSError:
                pass

    # --- Step 7: summary ---
    for line in report.summary_lines():
        print(line)

    if watch:
        _watch(tree_root, steps, desktop=desktop, runner=runner)

    return report


def _sync_venv(
    tree_root: Path, report: SyncReport, runner: "SubprocessRunner"
) -> SyncReport:
    """Step 1: create/sync the venv via managed uv."""
    venv_dir = tree_root / ".venv"

    # Ensure uv is available
    try:
        from hermes_cli.managed_uv import ensure_uv

        uv_bin = ensure_uv()
        uv_bin = str(uv_bin) if uv_bin else None
    except Exception:
        uv_bin = None

    if not uv_bin:
        import shutil

        uv_bin = shutil.which("uv")
        if not uv_bin:
            raise DevSyncError("venv: managed uv is unavailable")

    if not venv_dir.is_dir():
        # Create the venv
        result = runner.run([uv_bin, "venv", str(venv_dir)], cwd=tree_root)
        _require_success("venv creation", result)
        report.venv_created = True

    # Sync deps: uv sync --extra all --locked, fall back to uv pip install -e .[all]
    lockfile = tree_root / "uv.lock"
    if lockfile.is_file():
        result = runner.run(
            [uv_bin, "sync", "--extra", "all", "--locked"], cwd=tree_root
        )
        if result.returncode == 0:
            report.venv_synced = True
        else:
            # Lockfile is stale — fall back
            result = runner.run(
                [uv_bin, "pip", "install", "-e", ".[all]"],
                cwd=tree_root,
                env={"VIRTUAL_ENV": str(venv_dir)},
            )
            _require_success("venv dependency sync", result)
            report.venv_synced = True
    else:
        # No lockfile — use pip install directly
        result = runner.run(
            [uv_bin, "pip", "install", "-e", ".[all]"],
            cwd=tree_root,
            env={"VIRTUAL_ENV": str(venv_dir)},
        )
        _require_success("venv dependency sync", result)
        report.venv_synced = True

    return report


def _install_launcher(
    tree_root: Path, report: SyncReport, runner: "SubprocessRunner"
) -> SyncReport:
    """Step 2: install the release launcher into .hermes-launcher/."""
    launcher_dir = tree_root / ".hermes-launcher"
    try:
        from hermes_cli.subcommands.adopt import _platform_suffix, DEFAULT_RELEASE_BASE
        import stat
        import urllib.request

        suffix = _platform_suffix()
        base = DEFAULT_RELEASE_BASE
        url = f"{base.rstrip('/')}/hermes-{suffix}"

        launcher_dir.mkdir(parents=True, exist_ok=True)
        dest = launcher_dir / ("hermes.exe" if sys.platform == "win32" else "hermes")

        with urllib.request.urlopen(url) as resp:  # noqa: S310
            data = resp.read()
        dest.write_bytes(data)
        dest.chmod(dest.stat().st_mode | stat.S_IRWXU)
        report.launcher_installed = True
    except Exception as exc:
        raise DevSyncError(f"launcher install failed: {exc}") from exc

    return report


def _sync_node_deps(
    tree_root: Path, report: SyncReport, runner: "SubprocessRunner"
) -> SyncReport:
    """Step 3: install node deps — root + ui-tui + web workspaces.

    Mirrors ``_update_node_dependencies`` argv exactly: root install with
    ``--workspaces=false`` first, then ``--workspace ui-tui --workspace web``.
    """
    try:
        from hermes_constants import find_node_executable, with_hermes_node_path
    except ImportError:
        raise DevSyncError("node deps: hermes_constants is not importable")

    npm = find_node_executable("npm")
    if not npm:
        raise DevSyncError("node deps: npm not found")

    if not (tree_root / "package.json").exists():
        raise DevSyncError("node deps: package.json is missing")

    extra_args = ["--no-fund", "--no-audit", "--progress=false"]
    env = with_hermes_node_path()

    # Step 1: root install (no workspace recursion)
    root_args = [*extra_args, "--workspaces=false"]
    result = runner.run_npm(npm, tree_root, extra_args=tuple(root_args), env=env)
    if result.returncode != 0:
        _require_success("node root install", result)

    # Step 2: install only the workspaces (ui-tui, web)
    ws_args = [*extra_args, "--workspace", "ui-tui", "--workspace", "web"]
    result = runner.run_npm(npm, tree_root, extra_args=tuple(ws_args), env=env)
    if result.returncode == 0:
        report.node_deps_installed = True
    else:
        _require_success("node workspace install", result)

    return report


def _build_tui(
    tree_root: Path, report: SyncReport, runner: "SubprocessRunner"
) -> SyncReport:
    """Step 4a: build the TUI dist, gated by ArtifactStamp."""
    tui_dir = tree_root / "ui-tui"
    if not (tui_dir / "package.json").exists():
        raise DevSyncError("tui build: ui-tui/package.json is missing")

    stamp = _tui_stamp(tree_root)
    if not stamp.needs_build():
        report.skipped.append("tui dist up to date")
        return report

    try:
        from hermes_constants import find_node_executable, with_hermes_node_path
    except ImportError:
        raise DevSyncError("tui build: hermes_constants is not importable")

    npm = find_node_executable("npm")
    if not npm:
        raise DevSyncError("tui build: npm not found")

    env = with_hermes_node_path()
    # esbuild the TUI
    result = runner.run([npm, "run", "build"], cwd=tui_dir, env=env)
    if result.returncode == 0:
        stamp.write_stamp()
        report.tui_built = True
    else:
        _require_success("tui build", result)

    return report


def _build_web(
    tree_root: Path, report: SyncReport, runner: "SubprocessRunner"
) -> SyncReport:
    """Step 4b: build the web dist, gated by ArtifactStamp."""
    web_dir = tree_root / "web"
    if not (web_dir / "package.json").exists():
        raise DevSyncError("web build: web/package.json is missing")

    stamp = _web_stamp(tree_root)
    if not stamp.needs_build():
        report.skipped.append("web dist up to date")
        return report

    try:
        from hermes_constants import find_node_executable, with_hermes_node_path
    except ImportError:
        raise DevSyncError("web build: hermes_constants is not importable")

    npm = find_node_executable("npm")
    if not npm:
        raise DevSyncError("web build: npm not found")

    env = with_hermes_node_path()
    result = runner.run([npm, "run", "build"], cwd=web_dir, env=env)
    if result.returncode == 0:
        stamp.write_stamp()
        report.web_built = True
    else:
        _require_success("web build", result)

    return report


def _build_desktop(
    tree_root: Path, report: SyncReport, runner: "SubprocessRunner"
) -> SyncReport:
    """Step 4c: build the desktop app, gated by ArtifactStamp."""
    desktop_dir = tree_root / "apps" / "desktop"
    if not (desktop_dir / "package.json").exists():
        raise DevSyncError("desktop build: apps/desktop/package.json is missing")

    stamp = _desktop_stamp(tree_root)
    if not stamp.needs_build():
        report.skipped.append("desktop up to date")
        return report

    try:
        from hermes_constants import find_node_executable, with_hermes_node_path
    except ImportError:
        raise DevSyncError("desktop build: hermes_constants is not importable")

    npm = find_node_executable("npm")
    if not npm:
        raise DevSyncError("desktop build: npm not found")

    env = with_hermes_node_path()
    # Source mode: "build"; packaged mode: "pack"
    source_mode = not _has_desktop_build(tree_root) or not _desktop_packaged_exists(desktop_dir)
    build_script = "build" if source_mode else "pack"
    result = runner.run([npm, "run", build_script], cwd=desktop_dir, env=env)
    if result.returncode == 0:
        stamp.write_stamp()
        report.desktop_built = True
    else:
        _require_success("desktop build", result)

    return report


def _desktop_packaged_exists(desktop_dir: Path) -> bool:
    """Return True if a packaged desktop executable exists."""
    release_dir = desktop_dir / "release"
    if not release_dir.is_dir():
        return False
    if sys.platform == "darwin":
        candidates = list(release_dir.glob("mac*/Hermes.app/Contents/MacOS/Hermes"))
    elif sys.platform == "win32":
        candidates = [
            release_dir / "win-unpacked" / "Hermes.exe",
            release_dir / "win-ia32-unpacked" / "Hermes.exe",
            release_dir / "win-arm64-unpacked" / "Hermes.exe",
        ]
    else:
        candidates = [
            release_dir / "linux-unpacked" / "hermes",
            release_dir / "linux-unpacked" / "Hermes",
            release_dir / "linux-arm64-unpacked" / "hermes",
            release_dir / "linux-arm64-unpacked" / "Hermes",
        ]
    return any(p.exists() for p in candidates)


def _apply_feature_ledger(tree_root: Path, report: SyncReport) -> SyncReport:
    """Step 5: apply the feature ledger (phase 5 task 5.2).

    Calls ``apply_ledger`` from ``tools.lazy_deps`` with the tree's venv
    python. Failures are warnings (never fail the sync for a feature install).
    """
    try:
        from tools.lazy_deps import apply_ledger
    except ImportError:
        report.skipped.append("feature ledger skipped (module absent)")
        return report

    venv_python = tree_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = tree_root / "venv" / "bin" / "python"
    if not venv_python.exists():
        report.skipped.append("feature ledger skipped (no venv python)")
        return report

    try:
        results = apply_ledger(str(venv_python))
        failed = [name for name, status in results.items() if status.startswith("failed")]
        if failed:
            report.skipped.append(f"feature ledger: {len(failed)} failed: {', '.join(failed)}")
        report.feature_ledger_applied = True
    except Exception as exc:
        report.skipped.append(f"feature ledger error: {exc}")
        report.feature_ledger_applied = False

    return report


# ---------------------------------------------------------------------------
# SubprocessRunner — injectable for testing
# ---------------------------------------------------------------------------

class SubprocessRunner:
    """Thin wrapper around subprocess for testability.

    In production, this just calls ``subprocess.run``.  In tests, a mock
    or fake can be passed to :func:`run` to assert the right argv shapes
    without actually running anything.
    """

    def run(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        run_env = {**os.environ, **(env or {})}
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=run_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def run_npm(
        self,
        npm: str,
        cwd: Path,
        *,
        extra_args: tuple[str, ...] = (),
        capture_output: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a deterministic npm install (ci → install fallback).

        Mirrors ``_run_npm_install_deterministic``: prefers ``npm ci``
        when a lockfile exists, falls back to ``npm install``.
        """
        run_env = {**os.environ, **(env or {}), "CI": "1"}

        lockfile = cwd / "package-lock.json"
        if lockfile.exists():
            ci_cmd = [npm, "ci", *extra_args]
            ci_result = subprocess.run(
                ci_cmd,
                cwd=str(cwd),
                env=run_env,
                capture_output=capture_output,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if ci_result.returncode == 0:
                return ci_result

        install_cmd = [npm, "install", *extra_args]
        return subprocess.run(
            install_cmd,
            cwd=str(cwd),
            env=run_env,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def watch(self, commands: list[tuple[list[str], Path]]) -> int:
        """Run dev processes together; stop peers when the first exits."""
        processes = [subprocess.Popen(command, cwd=str(cwd)) for command, cwd in commands]
        try:
            while True:
                for process in processes:
                    code = process.poll()
                    if code is not None:
                        return code
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def _require_success(step: str, result: subprocess.CompletedProcess) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "unknown error").strip()
    raise DevSyncError(f"{step} failed: {detail}")


def _watch(
    tree_root: Path,
    steps: list[str],
    *,
    desktop: bool,
    runner: SubprocessRunner,
) -> None:
    try:
        from hermes_constants import find_node_executable
    except ImportError as exc:
        raise DevSyncError("watch: hermes_constants is not importable") from exc
    npm = find_node_executable("npm")
    if not npm:
        raise DevSyncError("watch: npm not found")
    commands: list[tuple[list[str], Path]] = []
    if "tui" in steps:
        commands.append(([npm, "run", "dev"], tree_root / "ui-tui"))
    if "web" in steps:
        commands.append(([npm, "run", "dev"], tree_root / "web"))
    if "desktop" in steps and desktop:
        commands.append(([npm, "run", "dev"], tree_root / "apps" / "desktop"))
    if not commands:
        raise DevSyncError("watch: select tui, web, or --desktop")
    code = runner.watch(commands)
    if code != 0:
        raise DevSyncError(f"watch process exited with code {code}")
