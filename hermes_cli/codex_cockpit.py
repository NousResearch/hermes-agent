"""Codex cockpit helpers.

This module keeps the Hermes/Codex integration split cleanly:

* Hermes is the cockpit: status, launch orchestration, context review.
* Codex is the driver: coding work runs in Codex app-server or codex exec.

The functions here are intentionally gateway/CLI-neutral so slash commands,
tests, and future UI surfaces can share the same parsing and rendering.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_READOUT = {
    "include_git_status": True,
    "include_recent_tool_events": True,
    "include_checks": True,
    "max_events": 25,
}

DEFAULT_CONTEXT_HELPER = {
    "enabled": False,
    "harvest_launches": True,
    "review_model": "",
    "max_output_chars": 8000,
    "auto_promote_memory": True,
    "auto_promote_skills": True,
    "min_confidence": 0.75,
    "notify_on_stage": True,
    "propose_memory": True,
    "propose_skills": True,
    "auto_promote": False,
}


@dataclass(frozen=True)
class CodexCockpitConfig:
    enabled: bool
    driver: str
    default_model: str
    default_worktree_root: str
    branch_prefix: str
    repo_allowlist: tuple[str, ...]
    readout: dict[str, Any]
    context_helper: dict[str, Any]


@dataclass(frozen=True)
class CodexCommand:
    action: str
    tokens: tuple[str, ...]
    raw_args: str


@dataclass(frozen=True)
class LaunchPlan:
    repo_root: str
    worktree_path: str
    branch: str
    command: str
    task_id: str
    prompt_preview: str


def _home_code_root() -> str:
    return str(Path.home() / "Code")


def load_cockpit_config(config: Mapping[str, Any] | None) -> CodexCockpitConfig:
    """Return normalized cockpit config with conservative defaults."""
    raw = {}
    if isinstance(config, Mapping):
        maybe = config.get("codex_cockpit", {})
        if isinstance(maybe, Mapping):
            raw = dict(maybe)

    readout = dict(DEFAULT_READOUT)
    if isinstance(raw.get("readout"), Mapping):
        readout.update(raw["readout"])
    context_helper = dict(DEFAULT_CONTEXT_HELPER)
    if isinstance(raw.get("context_helper"), Mapping):
        context_helper.update(raw["context_helper"])

    allowlist_raw = raw.get("repo_allowlist")
    if isinstance(allowlist_raw, str):
        allowlist = [allowlist_raw]
    elif isinstance(allowlist_raw, Iterable):
        allowlist = [str(item) for item in allowlist_raw if str(item).strip()]
    else:
        allowlist = [_home_code_root()]

    normalized_allowlist = tuple(
        str(Path(os.path.expanduser(path)).resolve())
        for path in allowlist
        if str(path).strip()
    )

    return CodexCockpitConfig(
        enabled=_as_bool(raw.get("enabled", True)),
        driver=str(raw.get("driver") or "codex_app_server"),
        default_model=str(raw.get("default_model") or "gpt-5.5"),
        default_worktree_root=str(
            raw.get("default_worktree_root") or "/private/tmp/hermes-codex"
        ),
        branch_prefix=str(raw.get("branch_prefix") or "codex/"),
        repo_allowlist=normalized_allowlist,
        readout=readout,
        context_helper=context_helper,
    )


def parse_codex_command(raw_args: str) -> CodexCommand:
    """Parse `/codex ...` into an action and shlex tokens."""
    raw_args = raw_args or ""
    try:
        tokens = tuple(shlex.split(raw_args))
    except ValueError as exc:
        return CodexCommand("error", (str(exc),), raw_args)
    if not tokens:
        return CodexCommand("status", (), raw_args)
    action = tokens[0].lower()
    if action in {"?", "help"}:
        return CodexCommand("help", tokens[1:], raw_args)
    return CodexCommand(action, tokens[1:], raw_args)


def render_help() -> str:
    return "\n".join(
        [
            "**Codex Cockpit**",
            "`/codex status` - show runtime, auth, active agent, git, and pending context",
            "`/codex last` - show the latest assistant reply for this session",
            "`/codex checks` - summarize recent validation-looking background processes",
            "`/codex context` - show pending memory/skill promotions",
            "`/codex learn status|pending|apply|discard` - review Codex learning proposals",
            "`/codex launch <repo> <prompt>` - start Codex in a clean worktree",
        ]
    )


def render_status(
    config: Mapping[str, Any] | None,
    *,
    active_agent: Any = None,
    process_sessions: Optional[list[dict[str, Any]]] = None,
    transcript: Optional[list[dict[str, Any]]] = None,
    cwd: Optional[str] = None,
    session_key: Optional[str] = None,
) -> str:
    """Render a compact cockpit readout."""
    cockpit = load_cockpit_config(config)
    runtime = _current_runtime(config)
    codex_ok, codex_version = _codex_binary_status()
    auth_line = _codex_auth_line()

    lines = [
        "**Codex Cockpit**",
        f"- Enabled: {_yes_no(cockpit.enabled)}",
        f"- Driver: `{cockpit.driver}`",
        f"- Hermes runtime: `{runtime}`",
        f"- Codex CLI: {_format_binary(codex_ok, codex_version)}",
        f"- Codex auth: {auth_line}",
        f"- Default model: `{cockpit.default_model}`",
        f"- Worktree root: `{cockpit.default_worktree_root}`",
    ]
    if session_key:
        lines.append(f"- Session key: `{session_key}`")

    lines.extend(_active_agent_lines(active_agent))

    if cockpit.readout.get("include_recent_tool_events", True):
        tool_names = _recent_tool_names(
            transcript or [],
            _positive_int(cockpit.readout.get("max_events"), 25),
        )
        if tool_names:
            rendered_tools = ", ".join(f"`{name}`" for name in tool_names)
            lines.append(f"- Recent tools: {rendered_tools}")
        else:
            lines.append("- Recent tools: none")

    if cockpit.readout.get("include_git_status", True):
        lines.extend(_git_lines(cwd))

    if cockpit.readout.get("include_checks", True):
        check_count = len(_validation_processes(process_sessions or []))
        lines.append(f"- Recent checks: {check_count}")

    mem_count, skill_count = _pending_context_counts()
    lines.append(f"- Pending context: {mem_count} memory, {skill_count} skill")
    learning_count = _pending_learning_count()
    if learning_count:
        lines.append(f"- Pending Codex learning: {learning_count}")

    last = latest_assistant_text(transcript or [])
    if last:
        lines.append(f"- Last reply: {_truncate_one_line(last, 140)}")
    return "\n".join(lines)


def render_last(
    transcript: Optional[list[dict[str, Any]]],
    *,
    active_agent: Any = None,
) -> str:
    text = latest_assistant_text(transcript or [])
    thread_id = _agent_attr(active_agent, "_last_codex_thread_id")
    turn_id = _agent_attr(active_agent, "_last_codex_turn_id")
    parts = ["**Codex Last**"]
    if thread_id:
        parts.append(f"- Thread: `{thread_id}`")
    if turn_id:
        parts.append(f"- Turn: `{turn_id}`")
    if text:
        parts.append("")
        parts.append(text[-2500:])
    else:
        parts.append("- No assistant reply recorded for this session yet.")
    return "\n".join(parts)


def render_checks(process_sessions: Optional[list[dict[str, Any]]]) -> str:
    checks = _validation_processes(process_sessions or [])[:8]
    if not checks:
        return "**Codex Checks**\nNo recent validation-looking background processes found."
    lines = ["**Codex Checks**"]
    for proc in checks:
        status = str(proc.get("status") or "unknown")
        exit_code = proc.get("exit_code")
        suffix = f" exit={exit_code}" if exit_code is not None else ""
        command = _truncate_one_line(str(proc.get("command") or ""), 120)
        preview = _truncate_one_line(str(proc.get("output_preview") or ""), 160)
        lines.append(f"- `{proc.get('session_id', 'process')}` {status}{suffix}: `{command}`")
        if preview:
            lines.append(f"  {preview}")
    return "\n".join(lines)


def render_context() -> str:
    memory_pending, skill_pending = _pending_context_records()
    lines = ["**Codex Context**"]
    if not memory_pending and not skill_pending:
        lines.append("No pending memory or skill promotions.")
        return "\n".join(lines)

    if memory_pending:
        lines.append(f"Pending memory: {len(memory_pending)}")
        for record in memory_pending[:5]:
            lines.append(
                f"- `{record.get('id', '?')}` {record.get('origin', 'foreground')}: "
                f"{_truncate_one_line(str(record.get('summary') or ''), 140)}"
            )
    if skill_pending:
        lines.append(f"Pending skills: {len(skill_pending)}")
        for record in skill_pending[:5]:
            lines.append(
                f"- `{record.get('id', '?')}` {record.get('origin', 'foreground')}: "
                f"{_truncate_one_line(str(record.get('summary') or ''), 140)}"
            )
    lines.append("Promote or discard staged items with the existing `/memory` and skill review flows.")
    return "\n".join(lines)


def render_learn(tokens: tuple[str, ...], config: Mapping[str, Any] | None) -> str:
    """Render or apply `/codex learn ...` commands."""
    from hermes_cli import codex_learning

    action = (tokens[0].lower() if tokens else "status")
    if action == "status":
        return codex_learning.render_learn_status(config)
    if action == "pending":
        return codex_learning.render_learn_pending()
    if action == "apply":
        target = tokens[1] if len(tokens) > 1 else ""
        return codex_learning.apply_learning(target, config)
    if action in {"discard", "reject", "drop"}:
        target = tokens[1] if len(tokens) > 1 else ""
        return codex_learning.discard_learning(target)
    return (
        "Unknown `/codex learn` action. Use: "
        "`/codex learn status|pending|apply <id|all>|discard <id|all>`."
    )


def prepare_launch(
    raw_args: str,
    config: Mapping[str, Any] | None,
    *,
    cwd: Optional[str] = None,
    now: Optional[float] = None,
) -> tuple[Optional[LaunchPlan], Optional[str]]:
    """Validate `/codex launch` args and produce the background command."""
    cockpit = load_cockpit_config(config)
    if not cockpit.enabled:
        return None, "Codex cockpit is disabled in `codex_cockpit.enabled`."

    parsed = parse_codex_command(raw_args)
    tokens = parsed.tokens if parsed.action == "launch" else tuple()
    if len(tokens) < 2:
        return None, "Usage: `/codex launch <repo> <prompt>`"

    repo_arg = tokens[0]
    prompt = " ".join(tokens[1:]).strip()
    if not prompt:
        return None, "Usage: `/codex launch <repo> <prompt>`"

    repo_path = Path(os.path.expanduser(repo_arg))
    if not repo_path.is_absolute():
        repo_path = Path(cwd or os.getcwd()) / repo_path
    repo_path = repo_path.resolve()

    repo_root, error = resolve_git_root(str(repo_path))
    if error:
        return None, error
    assert repo_root is not None

    if not _path_allowed(repo_root, cockpit.repo_allowlist):
        allowed = ", ".join(f"`{p}`" for p in cockpit.repo_allowlist) or "(none)"
        return None, f"Repository `{repo_root}` is outside `codex_cockpit.repo_allowlist`: {allowed}"

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now or time.time()))
    slug = _slug(prompt)
    branch_prefix = _safe_branch_prefix(cockpit.branch_prefix)
    branch = f"{branch_prefix}{slug}-{timestamp}"
    root = Path(os.path.expanduser(cockpit.default_worktree_root)).resolve()
    worktree_path = root / f"{Path(repo_root).name}-{slug}-{timestamp}"
    task_id = f"codex_{slug}_{timestamp}"

    command = " && ".join(
        [
            f"mkdir -p {shlex.quote(str(root))}",
            f"git -C {shlex.quote(repo_root)} fetch origin",
            (
                f"git -C {shlex.quote(repo_root)} worktree add "
                f"-b {shlex.quote(branch)} {shlex.quote(str(worktree_path))} origin/main"
            ),
            (
                "codex exec "
                f"-C {shlex.quote(str(worktree_path))} "
                f"--model {shlex.quote(cockpit.default_model)} "
                "--sandbox workspace-write "
                f"{shlex.quote(prompt)}"
            ),
        ]
    )
    return (
        LaunchPlan(
            repo_root=repo_root,
            worktree_path=str(worktree_path),
            branch=branch,
            command=command,
            task_id=task_id,
            prompt_preview=_truncate_one_line(prompt, 120),
        ),
        None,
    )


def resolve_git_root(path: str) -> tuple[Optional[str], Optional[str]]:
    try:
        proc = subprocess.run(
            ["git", "-C", path, "rev-parse", "--show-toplevel"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return None, f"Could not inspect `{path}` as a git repo: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "not a git repository").strip()
        return None, f"`{path}` is not a git repository: {detail}"
    return str(Path(proc.stdout.strip()).resolve()), None


def latest_assistant_text(transcript: list[dict[str, Any]]) -> str:
    for msg in reversed(transcript):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _current_runtime(config: Mapping[str, Any] | None) -> str:
    try:
        from hermes_cli.codex_runtime_switch import get_current_runtime

        return get_current_runtime(dict(config or {}))
    except Exception:
        return "unknown"


def _codex_binary_status() -> tuple[bool, Optional[str]]:
    try:
        from hermes_cli.codex_runtime_switch import check_codex_binary_ok

        return check_codex_binary_ok()
    except Exception as exc:
        return False, str(exc)


def _codex_auth_line() -> str:
    try:
        from hermes_cli.auth import get_codex_auth_status

        status = get_codex_auth_status()
    except Exception as exc:
        return f"unknown ({exc})"
    if status.get("logged_in"):
        source = status.get("source") or status.get("auth_mode") or "logged in"
        return f"logged in ({source})"
    error = status.get("error")
    return f"not logged in ({error})" if error else "not logged in"


def _active_agent_lines(active_agent: Any) -> list[str]:
    if active_agent is None:
        return ["- Active agent: no"]
    lines = ["- Active agent: yes"]
    for label, attr in (
        ("Model", "model"),
        ("Provider", "provider"),
        ("API mode", "api_mode"),
        ("CWD", "session_cwd"),
        ("Codex thread", "_last_codex_thread_id"),
        ("Codex turn", "_last_codex_turn_id"),
        ("Codex error", "_last_codex_error"),
    ):
        value = _agent_attr(active_agent, attr)
        if value:
            lines.append(f"  {label}: `{value}`")
    return lines


def _agent_attr(active_agent: Any, attr: str) -> str:
    if active_agent is None:
        return ""
    value = getattr(active_agent, attr, "")
    if isinstance(value, (str, int, float)):
        return str(value)
    return ""


def _git_lines(cwd: Optional[str]) -> list[str]:
    if not cwd:
        return ["- Git: no cwd"]
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, "status", "-sb"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return [f"- Git: unavailable ({exc})"]
    if proc.returncode != 0:
        return ["- Git: not a git workspace"]
    first = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
    return [f"- Git: `{first}`" if first else "- Git: clean/unknown"]


def _validation_processes(process_sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    needles = (
        "pytest",
        "npm test",
        "npm run test",
        "npm run build",
        "git diff --check",
        "tsc",
        "ruff",
        "mypy",
    )
    filtered = []
    for proc in process_sessions:
        command = str(proc.get("command") or "").lower()
        if any(needle in command for needle in needles):
            filtered.append(proc)
    return sorted(filtered, key=lambda item: str(item.get("started_at") or ""), reverse=True)


def _recent_tool_names(transcript: list[dict[str, Any]], limit: int) -> list[str]:
    names: list[str] = []
    for msg in reversed(transcript):
        tool_name = msg.get("tool_name")
        if isinstance(tool_name, str) and tool_name.strip():
            names.append(tool_name.strip())
        calls = msg.get("tool_calls")
        if isinstance(calls, list):
            for call in reversed(calls):
                if not isinstance(call, Mapping):
                    continue
                fn = call.get("function")
                if isinstance(fn, Mapping):
                    name = fn.get("name")
                    if isinstance(name, str) and name.strip():
                        names.append(name.strip())
                if len(names) >= limit:
                    return names
        if len(names) >= limit:
            return names
    return names


def _pending_context_counts() -> tuple[int, int]:
    memory, skills = _pending_context_records()
    return len(memory), len(skills)


def _pending_context_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from tools import write_approval

        return (
            write_approval.list_pending("memory"),
            write_approval.list_pending("skills"),
        )
    except Exception:
        return [], []


def _pending_learning_count() -> int:
    try:
        from hermes_cli import codex_learning

        return codex_learning.count_pending_proposals()
    except Exception:
        return 0


def _path_allowed(path: str, allowlist: tuple[str, ...]) -> bool:
    candidate = Path(path).resolve()
    for root in allowlist:
        try:
            candidate.relative_to(Path(root).resolve())
            return True
        except ValueError:
            continue
    return False


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug or "task")[:48].strip("-") or "task"


def _safe_branch_prefix(prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "-", prefix.strip())
    if not cleaned:
        cleaned = "codex/"
    return cleaned if cleaned.endswith("/") else f"{cleaned}/"


def _truncate_one_line(text: str, limit: int) -> str:
    one_line = " ".join(str(text).split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: max(0, limit - 3)].rstrip() + "..."


def _format_binary(ok: bool, version: Optional[str]) -> str:
    if ok:
        return f"OK `{version or 'installed'}`"
    return f"missing ({version or 'install codex'})"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}
    return bool(value)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
