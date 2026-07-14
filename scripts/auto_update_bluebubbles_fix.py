#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, NamedTuple


OFFICIAL_URL = "https://github.com/NousResearch/hermes-agent.git"
FORK_URL = "https://github.com/1960697431/hermes-agent.git"
OFFICIAL_SLUG = "nousresearch/hermes-agent"
FORK_SLUG = "1960697431/hermes-agent"
DEFAULT_BRANCH = "fix/bluebubbles-canonical-chat-id"
AUTO_BRANCH = "auto/bluebubbles-update"
DEFAULT_REPO = Path.home() / ".hermes" / "hermes-agent"
DEFAULT_STATE_DIR = Path.home() / ".hermes" / "auto-update" / "bluebubbles-fix"
DEFAULT_LOG_FILE = Path.home() / ".hermes" / "logs" / "bluebubbles-auto-update.log"
DEPENDENCY_FILES = {
    "pyproject.toml",
    "uv.lock",
    "setup-hermes.sh",
    "constraints-termux.txt",
}


class CommandError(RuntimeError):
    def __init__(self, cmd: list[str], cwd: Path, returncode: int):
        super().__init__(
            f"command failed with exit code {returncode}: {shlex.join(cmd)}"
        )
        self.cmd = cmd
        self.cwd = cwd
        self.returncode = returncode


class RemoteLayout(NamedTuple):
    upstream: str
    fork: str


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another BlueBubbles auto-update run is active") from exc
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, *args) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def normalize_remote_url(url: str) -> str:
    return url.lower().removesuffix(".git").replace(":", "/")


def remote_matches(url: str, slug: str) -> bool:
    return slug.lower().removesuffix(".git") in normalize_remote_url(url)


def parse_remote_rows(output: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], parts[2].strip("()")))
    return rows


def resolve_remote_layout(rows: Iterable[tuple[str, str, str]]) -> RemoteLayout:
    upstream = None
    fork = None
    for name, url, direction in rows:
        if direction != "fetch":
            continue
        if upstream is None and remote_matches(url, OFFICIAL_SLUG):
            upstream = name
        if fork is None and remote_matches(url, FORK_SLUG):
            fork = name
    if upstream is None:
        raise ValueError("official NousResearch/hermes-agent remote is missing")
    if fork is None:
        raise ValueError("1960697431/hermes-agent fork remote is missing")
    return RemoteLayout(upstream=upstream, fork=fork)


def tracked_status_paths(status_lines: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for line in status_lines:
        if not line or line.startswith("??") or line.startswith("!!"):
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            paths.append(path)
    return paths


def needs_dependency_sync(changed_paths: Iterable[str]) -> bool:
    return any(path in DEPENDENCY_FILES for path in changed_paths)


def build_bluebubbles_validation_commands(python: str) -> list[list[str]]:
    return [
        [
            python,
            "-m",
            "py_compile",
            "gateway/platforms/bluebubbles.py",
        ],
        ["scripts/run_tests.sh", "tests/gateway/test_bluebubbles.py"],
    ]


def run_command(
    cmd: list[str | Path],
    cwd: Path,
    logger: Logger,
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [str(part) for part in cmd]
    logger(f"$ ({cwd}) {shlex.join(rendered)}")
    completed = subprocess.run(
        rendered,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in completed.stdout.splitlines():
        logger(f"  {line}")
    if check and completed.returncode != 0:
        raise CommandError(rendered, cwd, completed.returncode)
    return completed


def git_stdout(repo: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def ref_exists(repo: Path, ref: str) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.returncode == 0


def find_uv() -> str | None:
    for candidate in (
        shutil.which("uv"),
        str(Path.home() / ".local" / "bin" / "uv"),
        str(Path.home() / ".cargo" / "bin" / "uv"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def python_can_import(python: Path, module: str) -> bool:
    completed = subprocess.run(
        [str(python), "-c", f"import {module}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.returncode == 0


def ensure_remote_layout(repo: Path, logger: Logger) -> RemoteLayout:
    rows = parse_remote_rows(git_stdout(repo, ["remote", "-v"]))
    names = {name for name, _, _ in rows}

    if not any(remote_matches(url, OFFICIAL_SLUG) for _, url, d in rows if d == "fetch"):
        name = "upstream" if "upstream" not in names else "official"
        run_command(["git", "remote", "add", name, OFFICIAL_URL], repo, logger)

    rows = parse_remote_rows(git_stdout(repo, ["remote", "-v"]))
    names = {name for name, _, _ in rows}
    if not any(remote_matches(url, FORK_SLUG) for _, url, d in rows if d == "fetch"):
        name = "cherry" if "cherry" not in names else "fork"
        run_command(["git", "remote", "add", name, FORK_URL], repo, logger)

    rows = parse_remote_rows(git_stdout(repo, ["remote", "-v"]))
    return resolve_remote_layout(rows)


def remove_staging_worktree(repo: Path, staging: Path, logger: Logger) -> None:
    if staging.exists():
        run_command(
            ["git", "worktree", "remove", "--force", staging],
            repo,
            logger,
            check=False,
        )
    if staging.exists():
        shutil.rmtree(staging)
    run_command(["git", "worktree", "prune"], repo, logger, check=False)
    run_command(["git", "branch", "-D", AUTO_BRANCH], repo, logger, check=False)


def ensure_staging_venv_link(repo: Path, staging: Path, logger: Logger) -> None:
    source = repo / "venv"
    target = staging / "venv"
    if target.exists() or target.is_symlink():
        return
    if source.exists():
        target.symlink_to(source, target_is_directory=True)
        logger(f"linked staging venv to {source}")


def sync_dependencies(project_dir: Path, repo: Path, logger: Logger) -> None:
    uv = find_uv()
    env = os.environ.copy()
    env["UV_NO_CONFIG"] = "1"
    env["UV_PROJECT_ENVIRONMENT"] = str(repo / "venv")
    if uv is not None:
        run_command(
            [uv, "sync", "--extra", "all", "--extra", "dev", "--locked"],
            project_dir,
            logger,
            env=env,
        )
        return

    python = repo / "venv" / "bin" / "python"
    run_command([python, "-m", "pip", "install", "-e", ".[all,dev]"], project_dir, logger)


def validate_bluebubbles(staging: Path, logger: Logger) -> None:
    python = staging / "venv" / "bin" / "python"
    for cmd in build_bluebubbles_validation_commands(str(python)):
        run_command(cmd, staging, logger)


def prepare_staging(
    repo: Path,
    staging: Path,
    source_ref: str,
    upstream_ref: str,
    logger: Logger,
) -> str:
    remove_staging_worktree(repo, staging, logger)
    run_command(["git", "worktree", "add", "--detach", staging, source_ref], repo, logger)
    ensure_staging_venv_link(repo, staging, logger)
    run_command(["git", "checkout", "-B", AUTO_BRANCH], staging, logger)
    try:
        run_command(["git", "rebase", upstream_ref], staging, logger)
    except CommandError:
        run_command(["git", "rebase", "--abort"], staging, logger, check=False)
        raise
    return git_stdout(staging, ["rev-parse", "HEAD"])


def deploy_to_runtime_repo(
    repo: Path,
    branch: str,
    fork_remote: str,
    old_rev: str,
    new_ref: str,
    logger: Logger,
    *,
    sync_runtime_dependencies: bool,
    restart_gateway: bool,
) -> None:
    status_lines = git_stdout(repo, ["status", "--porcelain=v1"]).splitlines()
    tracked_paths = tracked_status_paths(status_lines)
    if tracked_paths:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_command(
            ["git", "stash", "push", "-m", f"auto-update tracked backup {stamp}"],
            repo,
            logger,
        )

    run_command(["git", "fetch", fork_remote, branch], repo, logger)
    if ref_exists(repo, branch):
        run_command(["git", "switch", branch], repo, logger)
    else:
        run_command(["git", "switch", "-c", branch], repo, logger)
    run_command(["git", "reset", "--hard", new_ref], repo, logger)

    if sync_runtime_dependencies:
        sync_dependencies(repo, repo, logger)

    python = repo / "venv" / "bin" / "python"
    run_command(
        [python, "-m", "py_compile", "gateway/platforms/bluebubbles.py"],
        repo,
        logger,
    )
    run_command([repo / "venv" / "bin" / "hermes", "--version"], repo, logger)

    if restart_gateway:
        try:
            run_command([repo / "venv" / "bin" / "hermes", "gateway", "restart"], repo, logger)
        except CommandError:
            logger("gateway restart failed; rolling back runtime checkout")
            run_command(["git", "reset", "--hard", old_rev], repo, logger, check=False)
            run_command(
                [repo / "venv" / "bin" / "hermes", "gateway", "restart"],
                repo,
                logger,
                check=False,
            )
            raise
        run_command([repo / "venv" / "bin" / "hermes", "gateway", "status"], repo, logger)


def changed_paths_between(repo: Path, old_rev: str, new_rev: str) -> list[str]:
    output = git_stdout(repo, ["diff", "--name-only", old_rev, new_rev])
    return [line for line in output.splitlines() if line]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep the local Hermes BlueBubbles fix branch rebased and deployed."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-restart-gateway", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    repo = args.repo.expanduser().resolve()
    state_dir = args.state_dir.expanduser().resolve()
    staging = state_dir / "staging"
    logger = Logger(args.log_file.expanduser())
    lock_path = state_dir / "update.lock"

    try:
        with FileLock(lock_path):
            logger("starting Hermes BlueBubbles auto-update")
            logger(f"runtime repo: {repo}")
            layout = ensure_remote_layout(repo, logger)
            logger(f"upstream remote: {layout.upstream}; fork remote: {layout.fork}")

            run_command(["git", "fetch", "--prune", layout.upstream], repo, logger)
            run_command(["git", "fetch", "--prune", layout.fork], repo, logger)

            upstream_ref = f"{layout.upstream}/main"
            fork_ref = f"{layout.fork}/{args.branch}"
            source_ref = fork_ref if ref_exists(repo, fork_ref) else args.branch
            if not ref_exists(repo, source_ref):
                raise SystemExit(f"source branch not found: {source_ref}")

            old_rev = git_stdout(repo, ["rev-parse", "HEAD"])
            new_rev = prepare_staging(repo, staging, source_ref, upstream_ref, logger)
            changed_paths = changed_paths_between(repo, old_rev, new_rev)
            pytest_missing = not python_can_import(repo / "venv" / "bin" / "python", "pytest")
            if needs_dependency_sync(changed_paths) or pytest_missing:
                sync_dependencies(repo, repo, logger)
            validate_bluebubbles(staging, logger)

            if args.dry_run:
                logger(f"dry run complete; validated candidate {new_rev}")
                return 0

            run_command(
                ["git", "push", "--force-with-lease", layout.fork, f"HEAD:{args.branch}"],
                staging,
                logger,
            )
            deploy_to_runtime_repo(
                repo,
                args.branch,
                layout.fork,
                old_rev,
                f"{layout.fork}/{args.branch}",
                logger,
                sync_runtime_dependencies=needs_dependency_sync(changed_paths),
                restart_gateway=not args.no_restart_gateway,
            )
            logger(f"completed Hermes BlueBubbles auto-update at {new_rev}")
            return 0
    except Exception as exc:
        logger(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
