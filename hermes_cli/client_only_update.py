"""Client-only Desktop update for runtime-free remote installs.

A remote-mode Desktop client can be an intentional checkout with no local
venv — the agent runtime lives on the connected host. The POSIX hand-off
used to treat a missing ``venv/bin/hermes`` as a broken local install and
abort before any code moved.

This module is the smallest supported path for that surface:

* classify runtime-free remote mode vs a broken local install vs a full
  install that must keep using ``hermes update``
* advance the git checkout and rebuild the Desktop app
* roll the tree back when git, dependency, or build work fails
* never fleet-restart or otherwise touch a remote gateway
* never rewrite saved Desktop connections

Stdlib + ``git`` / ``npm`` only so a machine without a Hermes venv can still
run it via ``python3 -m hermes_cli.client_only_update``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

UpdateKind = Literal["full_install", "client_only", "broken_local"]
REMOTE_CONNECTION_KINDS = frozenset({"remote", "ssh", "cloud"})
REMOTE_CONNECTION_MODES = frozenset({"remote", "ssh", "cloud"})

RunCommand = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class UpdateSurface:
    """Facts used to choose a local update path. No hostnames or tokens."""

    has_venv_hermes: bool
    has_venv_python: bool
    remote_mode: bool
    has_bootstrap_marker: bool = False


@dataclass(frozen=True)
class ClientOnlyUpdateResult:
    ok: bool
    exit_code: int
    message: str
    kind: UpdateKind
    installed_commit: str = ""
    rolled_back: bool = False
    rebuilt_desktop: bool = False
    fleet_restarted: bool = False
    connections_rewritten: bool = False


def classify_update_kind(surface: UpdateSurface) -> UpdateKind:
    """Choose the local update path from install + connection facts.

    A runnable venv pair is always a full install — remote mode must not
    strip dependency/fleet work from a machine that actually has a runtime.
    Missing both venv files *and* a remote/ssh/cloud surface is the
    intentional runtime-free client. Anything else (partial venv, local
    mode, bootstrap-only wreckage) is a broken local install.
    """
    if surface.has_venv_hermes and surface.has_venv_python:
        return "full_install"
    if (
        surface.remote_mode
        and not surface.has_venv_hermes
        and not surface.has_venv_python
    ):
        return "client_only"
    return "broken_local"


def remote_mode_from_connection_docs(
    connection: Mapping[str, object] | None,
    connections: Mapping[str, object] | None,
) -> bool:
    """True when saved Desktop connection state points at a remote backend.

    ``connection.json`` ``mode`` and the v2 registry primary/last-used row
    are both consulted. A legacy ``mode=local`` file does not win over a
    registry primary that is ssh/remote/cloud.
    """
    if _mapping_mode_is_remote(connection):
        return True
    if connections is None:
        return False

    launch_mode = str(connections.get("launchMode") or "primary")
    target_id = connections.get("primary")
    if launch_mode == "last-used":
        target_id = connections.get("lastUsed") or target_id

    rows = connections.get("connections")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if target_id and row.get("id") != target_id:
            continue
        kind = str(row.get("kind") or "").strip().lower()
        if kind in REMOTE_CONNECTION_KINDS:
            return True
        if target_id:
            return False
    return False


def inspect_install_root(install_root: Path, *, windows: bool = False) -> UpdateSurface:
    """Read venv/bootstrap signals from a checkout. Does not read connection files."""
    bins = [install_root / name / ("Scripts" if windows else "bin") for name in (".venv", "venv")]
    hermes_name = "hermes.exe" if windows else "hermes"
    python_names = ("python.exe", "python") if windows else ("python3", "python")
    has_hermes = any(_is_executable(scripts / hermes_name) for scripts in bins)
    has_python = any(_is_executable(scripts / name) for scripts in bins for name in python_names)
    marker = (install_root / ".hermes-bootstrap-complete").is_file()
    return UpdateSurface(
        has_venv_hermes=has_hermes,
        has_venv_python=has_python,
        remote_mode=False,
        has_bootstrap_marker=marker,
    )


def load_json_object(path: Path) -> dict[str, object] | None:
    """Best-effort object JSON loader. Missing/invalid files are None."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def run_client_only_update(
    install_root: Path,
    *,
    branch: str = "main",
    hermes_home: Path | None = None,
    remote_mode: bool = True,
    force_client_only: bool = False,
    run: RunCommand | None = None,
    build_command: Sequence[str] | None = None,
    skip_desktop_build: bool = False,
    relaunch_target: Path | None = None,
) -> ClientOnlyUpdateResult:
    """Advance the checkout and rebuild Desktop without a local runtime.

    ``force_client_only`` is the Desktop hand-off signal: the app already
    classified this as runtime-free remote mode. Without it, a missing venv
    in local mode stays a broken-local refusal.
    """
    install_root = install_root.resolve()
    hermes_home = (hermes_home or install_root.parent).resolve()
    runner = run or _run
    windows = os.name == "nt"
    surface = inspect_install_root(install_root, windows=windows)
    if force_client_only or remote_mode:
        surface = UpdateSurface(
            has_venv_hermes=surface.has_venv_hermes,
            has_venv_python=surface.has_venv_python,
            remote_mode=True,
            has_bootstrap_marker=surface.has_bootstrap_marker,
        )
    kind = classify_update_kind(surface)

    if kind == "full_install":
        return ClientOnlyUpdateResult(
            ok=False,
            exit_code=64,
            message=(
                "This checkout has a local Hermes runtime. "
                "Use `hermes update` (full install), not the client-only path."
            ),
            kind=kind,
        )
    if kind == "broken_local":
        return ClientOnlyUpdateResult(
            ok=False,
            exit_code=3,
            message=(
                f"Update aborted: {install_root / 'venv' / ('Scripts' if windows else 'bin') / 'hermes'} "
                "is missing. The install needs repair (run the Hermes installer or hermes doctor)."
            ),
            kind=kind,
        )

    if not (install_root / ".git").exists():
        return ClientOnlyUpdateResult(
            ok=False,
            exit_code=1,
            message=f"Update aborted: {install_root} is not a git checkout.",
            kind=kind,
        )

    pre_sha = _git_output(runner, install_root, ["rev-parse", "HEAD"])
    if not pre_sha:
        return ClientOnlyUpdateResult(
            ok=False,
            exit_code=1,
            message="Update aborted: could not read HEAD.",
            kind=kind,
        )

    # Never use rollback to clean up changes that belong to the user.
    status = runner(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=install_root, env=_git_env(),
    )
    if status.returncode != 0 or status.stdout.strip():
        return ClientOnlyUpdateResult(
            ok=False, exit_code=1, kind=kind, installed_commit=pre_sha,
            message="Update refused: could not verify a clean checkout or local changes exist. Preserve or commit them before retrying.",
        )

    try:
        _git_check(runner, install_root, ["fetch", "origin", branch])
        _git_check(
            runner,
            install_root,
            ["merge", "--ff-only", f"origin/{branch}"],
        )
    except ClientOnlyGitError as exc:
        rolled = _rollback_to(runner, install_root, pre_sha)
        return ClientOnlyUpdateResult(
            ok=False,
            exit_code=1,
            message=f"Update failed during git: {exc}",
            kind=kind,
            installed_commit=pre_sha if rolled else "",
            rolled_back=rolled,
        )

    post_sha = _git_output(runner, install_root, ["rev-parse", "HEAD"]) or pre_sha
    rebuilt = False
    if not skip_desktop_build:
        try:
            rebuilt = _rebuild_desktop(
                install_root,
                runner=runner,
                build_command=build_command,
                hermes_home=hermes_home,
                expected_commit=post_sha,
                relaunch_target=relaunch_target,
            )
        except (ClientOnlyBuildError, OSError) as exc:
            rolled = _rollback_to(runner, install_root, pre_sha)
            return ClientOnlyUpdateResult(
                ok=False,
                exit_code=6,
                message=f"Desktop rebuild failed: {exc}",
                kind=kind,
                installed_commit=pre_sha if rolled else post_sha,
                rolled_back=rolled,
                rebuilt_desktop=False,
            )

    _write_client_receipt(
        hermes_home,
        branch=branch,
        pre_sha=pre_sha,
        post_sha=post_sha,
        rebuilt_desktop=rebuilt,
    )
    return ClientOnlyUpdateResult(
        ok=True,
        exit_code=0,
        message="Client update complete.",
        kind=kind,
        installed_commit=post_sha,
        rolled_back=False,
        rebuilt_desktop=rebuilt,
    )


class ClientOnlyGitError(RuntimeError):
    pass


class ClientOnlyBuildError(RuntimeError):
    pass


def _mapping_mode_is_remote(doc: Mapping[str, object] | None) -> bool:
    if not doc:
        return False
    return str(doc.get("mode") or "").strip().lower() in REMOTE_CONNECTION_MODES


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _run(args: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    merged = os.environ.copy() if env is None else dict(env)
    merged.setdefault("GIT_TERMINAL_PROMPT", "0")
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        env=merged,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_env() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def _git_check(run: RunCommand, cwd: Path, args: Sequence[str]) -> None:
    result = run(["git", *args], cwd=cwd, env=_git_env())
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip() or f"git {' '.join(args)} failed"
        raise ClientOnlyGitError(err)


def _git_output(run: RunCommand, cwd: Path, args: Sequence[str]) -> str:
    result = run(["git", *args], cwd=cwd, env=_git_env())
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _rollback_to(run: RunCommand, cwd: Path, sha: str) -> bool:
    try:
        _git_check(run, cwd, ["reset", "--keep", sha])
        return True
    except ClientOnlyGitError:
        return False


def _desktop_layout(output: Path) -> tuple[Path, Path, Path]:
    if sys.platform == "darwin":
        arch = "arm64" if platform.machine() == "arm64" else "x64"
        bundle = output / ("mac-arm64" if arch == "arm64" else "mac") / "Hermes.app"
        return bundle, bundle / "Contents/MacOS/Hermes", bundle / "Contents/Resources"
    if sys.platform.startswith("linux"):
        name = "linux-arm64-unpacked" if platform.machine() in ("arm64", "aarch64") else "linux-unpacked"
        bundle = output / name
        return bundle, bundle / "Hermes", bundle / "resources"
    raise ClientOnlyBuildError("Client-only packaging is supported by the POSIX handoff only.")


def _require_desktop_closed(executables: Sequence[Path], runner: RunCommand, cwd: Path) -> None:
    # Match the full executable path, never an argv substring or bot process.
    paths = {str(executable) for executable in executables}
    if sys.platform.startswith("linux"):
        # Linux `ps comm` reports a basename, unlike macOS. /proc exposes
        # the actual executable without confusing a similarly named process.
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                target = os.readlink(entry / "exe").removesuffix(" (deleted)")
            except (FileNotFoundError, PermissionError):
                continue
            if target in paths:
                raise ClientOnlyBuildError("Desktop is running or reopened during the update. Close it fully and retry.")
        return
    result = runner(["ps", "-axo", "pid=,comm="], cwd=cwd)
    if result.returncode:
        raise ClientOnlyBuildError("Cannot verify that Desktop is closed; retry after closing it.")
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[1] in paths:
            raise ClientOnlyBuildError("Desktop is running or reopened during the update. Close it fully and retry.")


def _verify_desktop_bundle(output: Path, commit: str) -> Path:
    bundle, executable, resources = _desktop_layout(output)
    stamp = load_json_object(resources / "install-stamp.json")
    if not stamp or stamp.get("commit") != commit:
        raise ClientOnlyBuildError("Packaged Desktop commit does not match the updated checkout.")
    if not _is_executable(executable) or not (resources / "app.asar.unpacked/dist/index.html").is_file():
        raise ClientOnlyBuildError("Packaged Desktop executable or application files are missing.")
    return bundle


def _sign_and_verify_macos_bundle(desktop_dir: Path, staging_dir: Path) -> None:
    """Sign and strictly verify a staged macOS app before it can replace the live app.

    The desktop module owns the signing policy and implementation. Import it only
    for this macOS gate so the runtime-free updater remains usable with Python's
    standard library and does not load backend or configuration state up front.
    The shared fixup may use a configured local identity, identifier-pinned
    ad-hoc signing, or its legacy ad-hoc fallback. Ad-hoc signing can require
    TCC permissions to be granted again, so a fallback is reported as a warning
    by the shared implementation rather than treated as a publisher signature.
    """
    try:
        from hermes_cli import main_desktop
    except Exception as exc:
        raise ClientOnlyBuildError(f"macOS signing support is unavailable: {exc}") from exc

    if not main_desktop._desktop_macos_relaunchable_fixup(
        desktop_dir,
        publisher_signing_configured=False,
        release_dir=staging_dir,
    ):
        raise ClientOnlyBuildError(
            "Staged macOS Desktop bundle could not be signed; the previous app was kept."
        )

    bundle, _executable, _resources = _desktop_layout(staging_dir)
    codesign = shutil.which("codesign")
    if not codesign:
        raise ClientOnlyBuildError(
            "macOS codesign is unavailable; the previous Desktop app was kept."
        )
    verification = main_desktop._codesign_verify(
        codesign, bundle, check=False, text=True
    )
    if verification.returncode != 0:
        detail = (verification.stderr or verification.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ClientOnlyBuildError(
            "Staged macOS Desktop bundle failed strict code-signature verification"
            f"{suffix}; the previous app was kept."
        )


def _rebuild_desktop(
    install_root: Path,
    *,
    runner: RunCommand,
    build_command: Sequence[str] | None,
    hermes_home: Path,
    expected_commit: str,
    relaunch_target: Path | None = None,
) -> bool:
    desktop_dir = install_root / "apps" / "desktop"
    if not (desktop_dir / "package.json").is_file():
        raise ClientOnlyBuildError("Desktop package.json is missing; no app was built.")

    env = os.environ.copy()
    managed_node = hermes_home / "node" / "bin"
    if managed_node.is_dir():
        env["PATH"] = f"{managed_node}{os.pathsep}{env.get('PATH', '')}"
    # Resolve npm only after adding the managed runtime used by GUI installs.
    npm = shutil.which("npm", path=env.get("PATH"))
    if not npm and not build_command:
        raise ClientOnlyBuildError("No Node/npm runtime on PATH")
    env["CI"] = "1"
    env["CSC_IDENTITY_AUTO_DISCOVERY"] = "false"
    for key in (
        "ELECTRON_RUN_AS_NODE", "CSC_LINK", "CSC_KEY_PASSWORD", "APPLE_SIGNING_IDENTITY",
        "APPLE_NOTARY_PROFILE", "APPLE_API_KEY", "APPLE_API_KEY_ID", "APPLE_API_ISSUER",
    ):
        env.pop(key, None)

    release = desktop_dir / "release"
    canonical, executable, _ = _desktop_layout(release)
    executables = [executable]
    if relaunch_target:
        executables.append(relaunch_target / "Contents/MacOS/Hermes" if sys.platform == "darwin" else relaunch_target)
    _require_desktop_closed(executables, runner, install_root)
    # before-pack deletes its output directory. It must never target the
    # installed bundle, even after the original Desktop PID has exited.
    backup_root = hermes_home / "backups" / "desktop-client-updates"
    backup_root.mkdir(parents=True, exist_ok=True)
    operation = Path(tempfile.mkdtemp(prefix="update-", dir=backup_root))
    output = operation / "release"
    if build_command:
        commands = [(list(build_command), desktop_dir)]
    else:
        platform_flag = "--mac" if sys.platform == "darwin" else "--linux"
        arch = "arm64" if platform.machine() in ("arm64", "aarch64") else "x64"
        commands = [
            ([npm, "ci", "--include=dev"], install_root),
            ([npm, "run", "build", "--workspace", "apps/desktop"], install_root),
            ([npm, "run", "builder", "--workspace", "apps/desktop", "--",
              platform_flag, f"--{arch}", "--dir", "--publish", "never",
              f"-c.directories.output={output}"], install_root),
        ]
    for command, cwd in commands:
        result = runner(command, cwd=cwd, env=env)
        if result.returncode != 0:
            tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-15:])
            raise ClientOnlyBuildError(tail or f"{' '.join(command)} failed")

    candidate = _verify_desktop_bundle(output, expected_commit)
    if sys.platform == "darwin":
        _sign_and_verify_macos_bundle(desktop_dir, output)
    _require_desktop_closed(executables, runner, install_root)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    previous = operation / "previous-app"
    had_previous = canonical.exists()
    if had_previous:
        canonical.rename(previous)
    try:
        candidate.rename(canonical)
    except OSError:
        if had_previous:
            previous.rename(canonical)
        raise
    return True


def _write_client_receipt(
    hermes_home: Path,
    *,
    branch: str,
    pre_sha: str,
    post_sha: str,
    rebuilt_desktop: bool,
) -> None:
    receipt_dir = hermes_home / "logs" / "update_receipts"
    try:
        receipt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "kind": "client_only",
            "outcome": "success",
            "branch": branch,
            "pre_update": {"sha": pre_sha},
            "post_update": {"sha": post_sha},
            "steps": [
                {"name": "git_ff_only", "ok": True},
                {"name": "desktop_rebuild", "ok": rebuilt_desktop},
                {"name": "fleet_restart", "ok": True, "skipped": True, "reason": "client_only"},
            ],
        }
        (receipt_dir / "latest.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime-free remote Desktop client update")
    parser.add_argument("--install-root", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--hermes-home")
    parser.add_argument("--relaunch-target", help="Desktop app or executable selected by the native handoff")
    parser.add_argument(
        "--client-only",
        action="store_true",
        help="Caller already classified this as a runtime-free remote client",
    )
    parser.add_argument("--connection-file", help="Optional connection.json path for classification")
    parser.add_argument("--connections-file", help="Optional connections.json path for classification")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    install_root = Path(args.install_root)
    connection = load_json_object(Path(args.connection_file)) if args.connection_file else None
    connections = load_json_object(Path(args.connections_file)) if args.connections_file else None
    remote_mode = args.client_only or remote_mode_from_connection_docs(connection, connections)
    result = run_client_only_update(
        install_root,
        branch=args.branch,
        hermes_home=Path(args.hermes_home) if args.hermes_home else None,
        remote_mode=remote_mode,
        force_client_only=args.client_only,
        relaunch_target=Path(args.relaunch_target) if args.relaunch_target else None,
    )
    if result.installed_commit:
        print(f"INSTALLED_COMMIT={result.installed_commit}")
    print(result.message)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
