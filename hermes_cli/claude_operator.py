"""Claude Code worker operator — launch and manage Claude Code workers in tmux.

Hermes treats Claude Code (and later Codex / Gemini) as a worker fleet. This
module is the smallest useful primitive for the **Dispatcher** role from
``operator-plan.md``: spawn a Claude Code worker in a predictably-named tmux
session, capture its log, and answer ``list``/``status``/``attach``/``stop``.

Safety contract
---------------

* Default permission mode is ``auto``. The worker auto-accepts edits and
  most low-risk tool calls but cannot bypass the safety hooks Bryan has
  installed (pre-push-guardian, worktree-guard, careful, supabase-guard).
* ``bypassPermissions`` / ``--dangerously-skip-permissions`` is **refused**.
  Hermes never spawns workers in that mode; the refusal lives in
  :func:`validate_permission_mode` and is exercised by tests.
  ``dontAsk`` is also not allowed — Hermes does not blanket-suppress prompts.
* Production-destructive surfaces (Railway, GitHub PRs, prod DB) are not
  touched by this module. Workers operate inside a workdir they were given;
  Hermes' Watcher / Verifier roles enforce the rest.

The module exposes both a pure-logic API (used by tests + future callers)
and a thin tmux-driven side-effect layer (``spawn``, ``list_sessions``,
``status``, ``stop``).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from hermes_constants import get_hermes_home

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Tmux session namespace. Every worker session lives under this prefix so
#: ``list_sessions`` can filter cleanly and ``hermes/_supervisor`` is reserved.
TMUX_PREFIX = "hermes"

#: Max length for any single slug segment. Keeps tmux session names short
#: enough for human readability in ``tmux ls``.
MAX_SLUG_LEN = 24

#: Permission modes that are accepted. Mirrors Claude Code's CLI flag.
#: ``auto`` is Bryan's required default — auto-accepts edits + low-risk tool
#: calls without bypassing safety hooks. ``acceptEdits`` is still accepted for
#: callers that want narrower auto-acceptance; ``default`` falls back to
#: per-call prompts (used for risky DDL where Hermes forwards prompts to Bryan).
ALLOWED_PERMISSION_MODES = frozenset({"auto", "acceptEdits", "default", "plan"})

#: Permission modes the operator hard-refuses. Bypass mode defeats the safety
#: hooks Bryan already has installed. ``dontAsk`` is also refused — Hermes
#: doesn't blanket-suppress prompts; risky calls must surface to Bryan.
REFUSED_PERMISSION_MODES = frozenset(
    {"bypassPermissions", "bypass", "dangerouslySkipPermissions", "dontAsk"}
)


class OperatorError(RuntimeError):
    """Raised for any operator-level safety or configuration violation."""


# ---------------------------------------------------------------------------
# Pure logic — slug + naming + paths + command construction
# ---------------------------------------------------------------------------


def slugify(value: str, max_len: int = MAX_SLUG_LEN) -> str:
    """Lower-case kebab-case slug, ASCII-only, bounded by ``max_len``.

    Used for the intent / worker / project segments of a tmux session
    name. Empty results raise :class:`OperatorError` — every segment must
    be non-empty so ``list_sessions`` filtering stays unambiguous.
    """
    if value is None:
        raise OperatorError("slug input must not be None")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    cleaned = cleaned[:max_len].strip("-")
    if not cleaned:
        raise OperatorError(f"slug is empty after normalization: {value!r}")
    return cleaned


def session_name(project: str, worker: str, intent: str) -> str:
    """Compose a deterministic tmux session name.

    Shape: ``hermes/<project>/<worker>/<intent>``. Matches the conventions
    in ``operator-plan.md`` section 4 so Watcher / Narrator / Archivist can
    identify the project + worker class + intent from the session name
    without consulting an out-of-band registry.
    """
    return "/".join(
        (TMUX_PREFIX, slugify(project), slugify(worker), slugify(intent))
    )


def operator_home() -> Path:
    """Root directory for operator-managed artifacts (logs, prompt files)."""
    return get_hermes_home() / "claude-operator"


def _session_safe(name: str) -> str:
    """Filesystem-safe rendering of a tmux session name (slashes → dashes)."""
    return name.replace("/", "-")


def log_path(name: str) -> Path:
    """Path to the worker's append-only log file."""
    return operator_home() / "logs" / f"{_session_safe(name)}.log"


def prompt_path(name: str) -> Path:
    """Path to the worker's pinned initial prompt (for replay / inspection)."""
    return operator_home() / "prompts" / f"{_session_safe(name)}.md"


def validate_permission_mode(mode: str) -> str:
    """Return ``mode`` if it's allowed; raise :class:`OperatorError` otherwise.

    This is the single chokepoint for the "no bypass" safety contract.
    Both the CLI surface and any future programmatic caller route through
    here so the refusal cannot be silently skipped.
    """
    if mode in REFUSED_PERMISSION_MODES:
        raise OperatorError(
            f"permission mode {mode!r} is refused: Hermes never spawns workers "
            "in bypass / dontAsk modes. Use 'auto' (default), 'acceptEdits', "
            "or 'default' (per-call prompts)."
        )
    if mode not in ALLOWED_PERMISSION_MODES:
        raise OperatorError(
            f"permission mode {mode!r} is unknown. Allowed: "
            f"{sorted(ALLOWED_PERMISSION_MODES)}"
        )
    return mode


def build_claude_command(
    *,
    workdir: Path,
    permission_mode: str = "auto",
    prompt: Optional[str] = None,
    agent: Optional[str] = None,
    binary: str = "claude",
    extra_args: Optional[Iterable[str]] = None,
) -> list[str]:
    """Construct the argv for invoking Claude Code.

    The constructed list is what the tmux launcher will exec. Pure: no
    shell quoting concerns, no environment side effects. Tests assert
    that ``auto`` lands in the argv by default and that bypass never does.
    """
    validate_permission_mode(permission_mode)
    argv: list[str] = [binary, "--permission-mode", permission_mode]
    if workdir:
        argv.extend(["--add-dir", str(workdir)])
    if agent:
        argv.extend(["--agent", agent])
    if extra_args:
        argv.extend(extra_args)
    if prompt:
        argv.append(prompt)
    return argv


def tmux_launch_command(
    *,
    session: str,
    workdir: Path,
    claude_argv: list[str],
    log_file: Path,
    tmux_binary: str = "tmux",
) -> list[str]:
    """Compose the tmux argv that creates the detached worker session.

    Uses ``tmux new-session -d`` so the spawn is non-blocking. The worker's
    stdout/stderr is piped through ``tee -a`` to ``log_file`` for replay.
    The inner command is a single shell string (tmux requires that for
    ``new-session``), built with ``shlex.quote`` to keep prompts containing
    quotes from breaking the launch.
    """
    inner = " ".join(shlex.quote(tok) for tok in claude_argv)
    shell_cmd = (
        f"cd {shlex.quote(str(workdir))} && "
        f"{inner} 2>&1 | tee -a {shlex.quote(str(log_file))}"
    )
    return [
        tmux_binary,
        "new-session",
        "-d",
        "-s",
        session,
        "-c",
        str(workdir),
        shell_cmd,
    ]


def attach_command(session: str, tmux_binary: str = "tmux") -> str:
    """Shell snippet a human can run to watch a worker live."""
    return f"{tmux_binary} attach -t {shlex.quote(session)}"


# ---------------------------------------------------------------------------
# Side-effect layer — actually talk to tmux
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerSession:
    """One live tmux session managed by the operator."""

    name: str
    log_file: Path
    created_at: float


def _ensure_tmux(tmux_binary: str = "tmux") -> None:
    if not shutil.which(tmux_binary):
        raise OperatorError(
            f"{tmux_binary!r} not found on PATH. Install tmux (e.g. `brew install tmux`)."
        )


def _ensure_claude(binary: str = "claude") -> None:
    if not shutil.which(binary):
        raise OperatorError(
            f"{binary!r} not found on PATH. Install Claude Code "
            "(https://docs.claude.com/claude-code) and ensure it's in PATH."
        )


def _prepare_paths(name: str) -> tuple[Path, Path]:
    log = log_path(name)
    prompt = prompt_path(name)
    log.parent.mkdir(parents=True, exist_ok=True)
    prompt.parent.mkdir(parents=True, exist_ok=True)
    return log, prompt


def spawn(
    *,
    project: str,
    worker: str,
    intent: str,
    workdir: Path,
    prompt: str,
    permission_mode: str = "auto",
    agent: Optional[str] = None,
    binary: str = "claude",
    tmux_binary: str = "tmux",
    extra_args: Optional[Iterable[str]] = None,
) -> WorkerSession:
    """Spawn a Claude Code worker in a detached tmux session.

    Honors the safety contract: refuses bypass permission modes via
    :func:`validate_permission_mode`. Writes the initial prompt to a
    pinned file under ``~/.hermes/claude-operator/prompts/`` so the
    operator (and the Archivist role) can replay or audit it later.
    """
    validate_permission_mode(permission_mode)

    workdir = Path(workdir).expanduser().resolve()
    if not workdir.is_dir():
        raise OperatorError(f"workdir does not exist or is not a directory: {workdir}")

    _ensure_tmux(tmux_binary)
    _ensure_claude(binary)

    name = session_name(project, worker, intent)
    if _session_exists(name, tmux_binary=tmux_binary):
        raise OperatorError(
            f"tmux session {name!r} already exists. Stop it first or pick a "
            "different intent slug."
        )

    log_file, prompt_file = _prepare_paths(name)
    prompt_file.write_text(prompt, encoding="utf-8")

    claude_argv = build_claude_command(
        workdir=workdir,
        permission_mode=permission_mode,
        prompt=prompt,
        agent=agent,
        binary=binary,
        extra_args=extra_args,
    )
    tmux_argv = tmux_launch_command(
        session=name,
        workdir=workdir,
        claude_argv=claude_argv,
        log_file=log_file,
        tmux_binary=tmux_binary,
    )
    subprocess.run(tmux_argv, check=True)

    return WorkerSession(name=name, log_file=log_file, created_at=time.time())


def _session_exists(name: str, tmux_binary: str = "tmux") -> bool:
    result = subprocess.run(
        [tmux_binary, "has-session", "-t", name],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def list_sessions(tmux_binary: str = "tmux") -> list[WorkerSession]:
    """Return every active ``hermes/...`` tmux session."""
    if not shutil.which(tmux_binary):
        return []
    result = subprocess.run(
        [tmux_binary, "list-sessions", "-F", "#{session_name}\t#{session_created}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    sessions: list[WorkerSession] = []
    for line in result.stdout.splitlines():
        if not line.startswith(f"{TMUX_PREFIX}/"):
            continue
        try:
            name, created = line.split("\t", 1)
            created_at = float(created)
        except ValueError:
            continue
        sessions.append(
            WorkerSession(
                name=name,
                log_file=log_path(name),
                created_at=created_at,
            )
        )
    return sessions


def status(name: str, tail_lines: int = 30, tmux_binary: str = "tmux") -> str:
    """Return a short human-readable status report for ``name``."""
    alive = _session_exists(name, tmux_binary=tmux_binary)
    log_file = log_path(name)
    parts: list[str] = []
    parts.append(f"session: {name}")
    parts.append(f"alive:   {'yes' if alive else 'no'}")
    parts.append(f"log:     {log_file}")
    if log_file.exists():
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-tail_lines:]
            parts.append("")
            parts.append(f"--- last {len(tail)} log lines ---")
            parts.extend(tail)
        except OSError as exc:
            parts.append(f"(log read failed: {exc})")
    else:
        parts.append("(no log file yet)")
    return "\n".join(parts)


def stop(name: str, tmux_binary: str = "tmux") -> bool:
    """Kill a tmux session. Returns ``True`` if a session was killed."""
    if not _session_exists(name, tmux_binary=tmux_binary):
        return False
    subprocess.run([tmux_binary, "kill-session", "-t", name], check=True)
    return True


# ---------------------------------------------------------------------------
# argparse surface — wired into the top-level CLI by main.py
# ---------------------------------------------------------------------------


def build_parser(parent_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``claude-operator`` subcommand tree."""
    parser = parent_subparsers.add_parser(
        "claude-operator",
        help="Spawn and manage Claude Code workers in tmux",
        description=(
            "Hermes-managed Claude Code worker fleet. Spawns workers in tmux "
            "with --permission-mode auto (never bypass / dontAsk). Names "
            "sessions predictably (hermes/<project>/<worker>/<intent>) so "
            "list/status/attach all work without an external registry. See "
            "operator-plan.md for the full role split (Dispatcher / Watcher "
            "/ Verifier / Archivist / Narrator)."
        ),
    )
    sub = parser.add_subparsers(dest="operator_action")

    p_spawn = sub.add_parser("spawn", help="Launch a Claude Code worker in tmux")
    p_spawn.add_argument("--project", required=True, help="Project slug (e.g. 'lyra')")
    p_spawn.add_argument("--worker", default="claude", help="Worker class (default: claude)")
    p_spawn.add_argument("--intent", required=True, help="Short intent slug (e.g. 'fix-stale-agm')")
    p_spawn.add_argument("--workdir", required=True, help="Directory the worker operates in (typically a worktree)")
    p_spawn.add_argument(
        "--permission-mode",
        default="auto",
        help=(
            "Claude Code permission mode (default: auto). "
            "Allowed: auto, acceptEdits, default, plan. "
            "Bypass / dontAsk modes are refused."
        ),
    )
    p_spawn.add_argument("--agent", default=None, help="Optional Claude Code agent preset name")
    p_spawn.add_argument(
        "--prompt",
        default=None,
        help="Initial prompt text. Mutually exclusive with --prompt-file.",
    )
    p_spawn.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a file containing the initial prompt.",
    )
    p_spawn.add_argument(
        "--binary",
        default="claude",
        help="Claude Code binary name on PATH (default: claude)",
    )

    p_list = sub.add_parser("list", aliases=["ls"], help="List active worker sessions")
    p_list.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")

    p_status = sub.add_parser("status", help="Show status + log tail for one worker")
    p_status.add_argument("session", help="Full session name (e.g. hermes/lyra/claude/fix-stale-agm)")
    p_status.add_argument("--tail", type=int, default=30, help="Log tail line count (default: 30)")

    p_attach = sub.add_parser("attach", help="Print the tmux attach command for a worker")
    p_attach.add_argument("session", help="Full session name")

    p_stop = sub.add_parser("stop", help="Kill a worker's tmux session")
    p_stop.add_argument("session", help="Full session name")

    return parser


def operator_command(args: argparse.Namespace) -> int:
    """Dispatch ``hermes claude-operator …`` invocations."""
    action = getattr(args, "operator_action", None)
    if action == "spawn":
        return _cmd_spawn(args)
    if action in ("list", "ls"):
        return _cmd_list(args)
    if action == "status":
        return _cmd_status(args)
    if action == "attach":
        return _cmd_attach(args)
    if action == "stop":
        return _cmd_stop(args)
    print(
        "Usage: hermes claude-operator {spawn|list|status|attach|stop} ...",
        file=sys.stderr,
    )
    return 2


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt and args.prompt_file:
        raise OperatorError("--prompt and --prompt-file are mutually exclusive")
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    raise OperatorError("one of --prompt or --prompt-file is required")


def _cmd_spawn(args: argparse.Namespace) -> int:
    try:
        prompt = _resolve_prompt(args)
        session = spawn(
            project=args.project,
            worker=args.worker,
            intent=args.intent,
            workdir=Path(args.workdir),
            prompt=prompt,
            permission_mode=args.permission_mode,
            agent=args.agent,
            binary=args.binary,
        )
    except OperatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"tmux launch failed: {exc}", file=sys.stderr)
        return 1
    print(f"spawned: {session.name}")
    print(f"log:     {session.log_file}")
    print(f"attach:  {attach_command(session.name)}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    sessions = list_sessions()
    if getattr(args, "json", False):
        import json as _json

        payload = [
            {
                "name": s.name,
                "log_file": str(s.log_file),
                "created_at": s.created_at,
            }
            for s in sessions
        ]
        print(_json.dumps(payload, indent=2))
        return 0
    if not sessions:
        print("(no active hermes workers)")
        return 0
    for s in sessions:
        age_s = int(time.time() - s.created_at)
        print(f"{s.name}  age={age_s}s  log={s.log_file}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    print(status(args.session, tail_lines=args.tail))
    return 0


def _cmd_attach(args: argparse.Namespace) -> int:
    print(attach_command(args.session))
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    killed = stop(args.session)
    if killed:
        print(f"killed: {args.session}")
        return 0
    print(f"no active session: {args.session}", file=sys.stderr)
    return 1
