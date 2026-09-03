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
    scripts = install_root / "venv" / ("Scripts" if windows else "bin")
    hermes_name = "hermes.exe" if windows else "hermes"
    python_names = ("python.exe", "python") if windows else ("python3", "python")
    has_hermes = _is_executable(scripts / hermes_name)
    has_python = any(_is_executable(scripts / name) for name in python_names)
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
            )
        except ClientOnlyBuildError as exc:
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
    merged = os.environ.copy()
    if env:
        merged.update(env)
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
        _git_check(run, cwd, ["reset", "--hard", sha])
        return True
    except ClientOnlyGitError:
        return False


def _rebuild_desktop(
    install_root: Path,
    *,
    runner: RunCommand,
    build_command: Sequence[str] | None,
) -> bool:
    desktop_dir = install_root / "apps" / "desktop"
    if not (desktop_dir / "package.json").is_file():
        return False

    command = list(build_command or _default_desktop_build_command())
    if not command:
        raise ClientOnlyBuildError("no Node/npm runtime on PATH")

    env = os.environ.copy()
    managed_node = Path(os.environ.get("HERMES_HOME") or install_root.parent) / "node" / "bin"
    if managed_node.is_dir():
        env["PATH"] = f"{managed_node}{os.pathsep}{env.get('PATH', '')}"

    result = runner(command, cwd=desktop_dir, env=env)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()
        raise ClientOnlyBuildError(tail or f"{' '.join(command)} failed")
    return True


def _default_desktop_build_command() -> list[str]:
    override = os.environ.get("HERMES_CLIENT_ONLY_BUILD_CMD")
    if override:
        return override.split()
    npm = shutil.which("npm")
    if npm:
        return [npm, "run", "build"]
    return []


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
    parser.add_argument(
        "--client-only",
        action="store_true",
        help="Caller already classified this as a runtime-free remote client",
    )
    parser.add_argument("--connection-file", help="Optional connection.json path for classification")
    parser.add_argument("--connections-file", help="Optional connections.json path for classification")
    parser.add_argument("--skip-desktop-build", action="store_true")
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
        skip_desktop_build=args.skip_desktop_build,
    )
    if result.installed_commit:
        print(f"INSTALLED_COMMIT={result.installed_commit}")
    print(result.message)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
