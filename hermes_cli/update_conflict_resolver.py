"""Agent-assisted conflict resolution for ``hermes update`` maintainer syncs.

This module is intentionally narrow: it only handles the durable fork fleet
path where ``patched-main`` is maintained by merging ``upstream/main`` and
pushing Kamell's fork.  The live checkout is never used as the resolution
surface; conflicts are replayed in a temporary worktree and only fast-forwarded
back after the resolver produced a clean merge commit.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import TextIO


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class UpdateConflictResolverConfig:
    enabled: bool = False
    timeout_seconds: int = 1800
    model: str = ""
    provider: str = ""
    reasoning_effort: str = "high"
    push: bool = True
    max_turns: int = 1000


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    raw = str(value).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def _as_int(value: object, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def load_update_conflict_resolver_config(config: dict | None = None) -> UpdateConflictResolverConfig:
    """Read update conflict resolver settings from config/env.

    Environment variables are intended as one-shot overrides for maintainers:
    ``HERMES_UPDATE_AUTO_RESOLVE``, ``HERMES_UPDATE_AUTO_RESOLVE_TIMEOUT``,
    ``HERMES_UPDATE_AUTO_RESOLVE_MODEL``, ``HERMES_UPDATE_AUTO_RESOLVE_PROVIDER``,
    ``HERMES_UPDATE_AUTO_RESOLVE_REASONING_EFFORT``,
    ``HERMES_UPDATE_AUTO_RESOLVE_PUSH``, and
    ``HERMES_UPDATE_AUTO_RESOLVE_MAX_TURNS``.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}

    update_cfg = (config or {}).get("update", {})
    if not isinstance(update_cfg, dict):
        update_cfg = {}

    enabled_raw = os.getenv("HERMES_UPDATE_AUTO_RESOLVE")
    enabled = _as_bool(
        enabled_raw if enabled_raw is not None else update_cfg.get("auto_resolve_conflicts"),
        default=False,
    )

    timeout_raw = os.getenv("HERMES_UPDATE_AUTO_RESOLVE_TIMEOUT")
    timeout_seconds = _as_int(
        timeout_raw if timeout_raw is not None else update_cfg.get("auto_resolve_timeout"),
        default=1800,
        minimum=60,
    )

    max_turns_raw = os.getenv("HERMES_UPDATE_AUTO_RESOLVE_MAX_TURNS")
    max_turns = _as_int(
        max_turns_raw if max_turns_raw is not None else update_cfg.get("auto_resolve_max_turns"),
        default=1000,
        minimum=1,
    )

    push_raw = os.getenv("HERMES_UPDATE_AUTO_RESOLVE_PUSH")
    push = _as_bool(
        push_raw if push_raw is not None else update_cfg.get("auto_resolve_push", True),
        default=True,
    )

    return UpdateConflictResolverConfig(
        enabled=enabled,
        timeout_seconds=timeout_seconds,
        model=str(
            os.getenv("HERMES_UPDATE_AUTO_RESOLVE_MODEL")
            or update_cfg.get("auto_resolve_model")
            or ""
        ).strip(),
        provider=str(
            os.getenv("HERMES_UPDATE_AUTO_RESOLVE_PROVIDER")
            or update_cfg.get("auto_resolve_provider")
            or ""
        ).strip(),
        reasoning_effort=str(
            os.getenv("HERMES_UPDATE_AUTO_RESOLVE_REASONING_EFFORT")
            or update_cfg.get("auto_resolve_reasoning_effort")
            or "high"
        ).strip(),
        push=push,
        max_turns=max_turns,
    )


def _emit(stream: TextIO, message: str = "") -> None:
    print(message, file=stream)


def _run_git(
    git_cmd: list[str],
    cwd: Path,
    args: list[str],
    *,
    check: bool = False,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        git_cmd + args,
        cwd=cwd,
        capture_output=capture_output,
        text=True,
        check=check,
    )


def _git_stdout(git_cmd: list[str], cwd: Path, args: list[str]) -> str:
    result = _run_git(git_cmd, cwd, args, check=True)
    return (result.stdout or "").strip()


def _unmerged_files(git_cmd: list[str], cwd: Path) -> list[str]:
    result = _run_git(git_cmd, cwd, ["diff", "--name-only", "--diff-filter=U"])
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _worktree_root() -> Path:
    try:
        from hermes_constants import get_hermes_home

        root = get_hermes_home() / "tmp" / "update-conflict-resolver"
    except Exception:
        root = Path.home() / ".hermes" / "tmp" / "update-conflict-resolver"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_branch_name() -> str:
    return f"hermes-update-resolve-{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}"


def _abort_live_merge(git_cmd: list[str], cwd: Path, stream: TextIO) -> bool:
    abort = _run_git(git_cmd, cwd, ["merge", "--abort"])
    if abort.returncode == 0:
        _emit(stream, "  ✓ Aborted conflicted live merge before using temp worktree")
        return True

    reset = _run_git(git_cmd, cwd, ["reset", "--merge"])
    if reset.returncode == 0:
        _emit(stream, "  ✓ Reset conflicted live merge before using temp worktree")
        return True

    _emit(stream, "  ✗ Could not abort/reset the conflicted live merge.")
    err = (abort.stderr or reset.stderr or "").strip()
    if err:
        _emit(stream, f"    {err.splitlines()[0]}")
    return False


def _build_resolver_prompt(
    *,
    worktree: Path,
    branch_name: str,
    conflict_files: list[str],
    merge_stderr: str,
    push_enabled: bool,
) -> str:
    conflicts = "\n".join(f"- {path}" for path in conflict_files) or "- unknown"
    merge_error = merge_stderr.strip() or "No stderr captured."
    push_note = (
        "The parent updater will push HEAD:patched-main after verification. Do not push."
        if push_enabled
        else "Push is disabled for this run. Do not push."
    )
    return textwrap.dedent(
        f"""
        You are a dedicated Hermes update conflict resolver.

        Task: resolve a failed maintainer merge of upstream/main into Kamell's patched-main fork branch.

        Repository worktree: {worktree}
        Temporary branch: {branch_name}
        Failed operation: git merge --no-edit upstream/main
        Conflicted files:
        {conflicts}

        Merge stderr:
        {merge_error}

        Hard rules:
        - Work only inside {worktree}.
        - Do not run `hermes update` recursively.
        - Preserve Kamell's durable patched-main fleet behavior.
        - Prefer upstream code unless it conflicts with explicit patched-main behavior.
        - Never delete fork-specific behavior unless upstream clearly supersedes it.
        - Resolve all conflict markers and unmerged paths.
        - Run focused checks for every touched area. At minimum, if updater files are touched, run:
          python -m py_compile hermes_cli/main.py hermes_cli/update_conflict_resolver.py
          python -m pytest tests/hermes_cli/test_update_autostash.py tests/hermes_cli/test_cmd_update.py -q -o 'addopts='
        - Stage resolved files and create the merge commit. Use `git commit --no-edit` if MERGE_HEAD exists.
        - {push_note}

        Useful verification before finishing:
        - git diff --name-only --diff-filter=U must be empty.
        - git status --short should be clean after committing.
        - git merge-base --is-ancestor upstream/main HEAD should pass.

        If safe resolution is not possible, stop and say exactly why in your final response.
        """
    ).strip()


def _resolver_command(prompt: str, cfg: UpdateConflictResolverConfig) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "chat",
        "-q",
        prompt,
        "-t",
        "terminal,file,skills,session_search",
        "-s",
        "hermes-agent",
        "--max-turns",
        str(cfg.max_turns),
    ]
    if cfg.provider:
        cmd.extend(["--provider", cfg.provider])
    if cfg.model:
        cmd.extend(["--model", cfg.model])
    return cmd


def _resolver_env(
    cfg: UpdateConflictResolverConfig,
    *,
    worktree: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_UPDATE_RESOLVER"] = "1"
    if worktree is not None:
        env["HERMES_UPDATE_RESOLVER_WORKTREE"] = str(worktree)
    if cfg.reasoning_effort:
        env["HERMES_REASONING_EFFORT"] = cfg.reasoning_effort
    env.setdefault("HERMES_MAX_ITERATIONS", str(cfg.max_turns))
    return env


def _commit_if_needed(git_cmd: list[str], worktree: Path, stream: TextIO) -> bool:
    unmerged = _unmerged_files(git_cmd, worktree)
    if unmerged:
        _emit(stream, "  ✗ Resolver left unmerged files:")
        for path in unmerged:
            _emit(stream, f"    {path}")
        return False

    status = _git_stdout(git_cmd, worktree, ["status", "--porcelain"])
    if not status:
        return True

    _run_git(git_cmd, worktree, ["add", "-A"], check=True)
    merge_head = _run_git(git_cmd, worktree, ["rev-parse", "-q", "--verify", "MERGE_HEAD"])
    if merge_head.returncode == 0:
        commit = _run_git(git_cmd, worktree, ["commit", "--no-edit"])
    else:
        commit = _run_git(
            git_cmd,
            worktree,
            ["commit", "-m", "merge: sync patched-main with upstream main"],
        )
    if commit.returncode != 0:
        _emit(stream, "  ✗ Could not create merge commit after resolver finished.")
        if commit.stderr.strip():
            _emit(stream, f"    {commit.stderr.strip().splitlines()[0]}")
        return False
    _emit(stream, "  ✓ Created merge commit from resolver changes")
    return True


def _cleanup_worktree(git_cmd: list[str], cwd: Path, worktree: Path, branch_name: str) -> None:
    subprocess.run(
        git_cmd + ["worktree", "remove", "--force", str(worktree)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        git_cmd + ["branch", "-D", branch_name],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)


def run_patched_main_conflict_resolver(
    git_cmd: list[str],
    cwd: Path,
    *,
    merge_stderr: str = "",
    config: dict | None = None,
    stream: TextIO | None = None,
) -> bool:
    """Resolve a patched-main/upstream conflict in a temp worktree.

    Returns True only when the temp worktree produced a clean commit, the commit
    was pushed if configured, and the live checkout was fast-forwarded to
    ``origin/patched-main``.  On failure, leaves the temp worktree in place for
    manual inspection and returns False.
    """
    out = stream or sys.stdout
    cfg = load_update_conflict_resolver_config(config)
    if not cfg.enabled:
        return False

    _emit(out, "  → Auto-resolving patched-main merge conflict in a temp worktree...")
    if not _abort_live_merge(git_cmd, cwd, out):
        return False

    branch_name = _make_branch_name()
    worktree = Path(tempfile.mkdtemp(prefix=f"{branch_name}-", dir=str(_worktree_root())))
    try:
        add = _run_git(
            git_cmd,
            cwd,
            ["worktree", "add", "-b", branch_name, str(worktree), "patched-main"],
        )
        if add.returncode != 0:
            _emit(out, "  ✗ Could not create resolver worktree.")
            if add.stderr.strip():
                _emit(out, f"    {add.stderr.strip().splitlines()[0]}")
            shutil.rmtree(worktree, ignore_errors=True)
            return False

        merge = _run_git(git_cmd, worktree, ["merge", "--no-edit", "upstream/main"])
        conflict_files = _unmerged_files(git_cmd, worktree)
        if merge.returncode != 0 and not conflict_files:
            _emit(out, "  ✗ Temp worktree merge failed without conflict files.")
            if merge.stderr.strip():
                _emit(out, f"    {merge.stderr.strip().splitlines()[0]}")
            _emit(out, f"    Resolver worktree left at: {worktree}")
            return False

        if conflict_files:
            _emit(out, f"  → Spawn resolver agent ({cfg.timeout_seconds // 60}m timeout)")
            prompt = _build_resolver_prompt(
                worktree=worktree,
                branch_name=branch_name,
                conflict_files=conflict_files,
                merge_stderr=merge.stderr or merge_stderr,
                push_enabled=cfg.push,
            )
            try:
                # Run the resolver agent from the stable live checkout, not from
                # the conflicted temp worktree.  The temp worktree can contain
                # conflict markers in hermes_cli/main.py; using it as cwd makes
                # ``python -m hermes_cli.main`` import broken code before the
                # resolver ever gets a chance to fix anything.
                agent = subprocess.run(
                    _resolver_command(prompt, cfg),
                    cwd=cwd,
                    env=_resolver_env(cfg, worktree=worktree),
                    timeout=cfg.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                _emit(out, f"  ✗ Resolver agent timed out after {cfg.timeout_seconds}s.")
                _emit(out, f"    Resolver worktree left at: {worktree}")
                return False
            if agent.returncode != 0:
                _emit(out, f"  ✗ Resolver agent exited with code {agent.returncode}.")
                _emit(out, f"    Resolver worktree left at: {worktree}")
                return False
        else:
            _emit(out, "  ✓ Temp worktree merged upstream/main without conflicts")

        if not _commit_if_needed(git_cmd, worktree, out):
            _emit(out, f"    Resolver worktree left at: {worktree}")
            return False

        ancestor = _run_git(git_cmd, worktree, ["merge-base", "--is-ancestor", "upstream/main", "HEAD"])
        if ancestor.returncode != 0:
            _emit(out, "  ✗ Resolver result does not contain upstream/main.")
            _emit(out, f"    Resolver worktree left at: {worktree}")
            return False

        if cfg.push:
            push = _run_git(git_cmd, worktree, ["push", "origin", "HEAD:patched-main"])
            if push.returncode != 0:
                _emit(out, "  ✗ Could not push resolver result to origin/patched-main.")
                if push.stderr.strip():
                    _emit(out, f"    {push.stderr.strip().splitlines()[0]}")
                _emit(out, f"    Resolver worktree left at: {worktree}")
                return False
            _emit(out, "  ✓ Pushed resolver result to origin/patched-main")

        if cfg.push:
            _run_git(git_cmd, cwd, ["fetch", "origin", "--quiet"], check=True)
            ff_target = "origin/patched-main"
        else:
            ff_target = branch_name
        ff = _run_git(git_cmd, cwd, ["merge", "--ff-only", ff_target])
        if ff.returncode != 0:
            _emit(out, "  ✗ Could not fast-forward live patched-main to resolver result.")
            if ff.stderr.strip():
                _emit(out, f"    {ff.stderr.strip().splitlines()[0]}")
            _emit(out, f"    Resolver worktree left at: {worktree}")
            return False

        _emit(out, "  ✓ Live patched-main fast-forwarded to resolver result")
        _cleanup_worktree(git_cmd, cwd, worktree, branch_name)
        return True
    except subprocess.CalledProcessError as exc:
        _emit(out, f"  ✗ Resolver verification command failed: {' '.join(map(str, exc.cmd))}")
        _emit(out, f"    Resolver worktree left at: {worktree}")
        return False
