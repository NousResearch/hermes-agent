"""Agent-harness accuracy shared metrics: file-edit matching, loop guards, recovery after a tool
error, terminal command outcomes and bad model replies.

These tune the agent loop itself: which fuzzy-match strategies earn their keep, how often the loop
guards fire, whether a model recovers after a failed tool call, which kinds of commands fail, and
which models return empty / refused / truncated replies. Producers pass RAW values; every builder
maps them onto closed enums. Nothing here reads command text, file paths, tool arguments or reply
text beyond the structural checks Hermes already makes (think-block stripping for "visible text").

Hermes-owned agent loops (the background self-improvement review and the curator fork) are not user
work and never count. Recording is a no-op unless shared metrics are enabled for the owning profile,
and never raises into the caller. Stdlib-only at import time: tool modules import this on their hot
path.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_BACKGROUND_ORIGIN = "background_review"


def _enabled() -> bool:
    try:
        from .relay_shared_metrics import enabled

        return enabled()
    except Exception:
        return False


def _record(mark_attr: str, data: dict[str, str]) -> None:
    try:
        from . import shared_metrics_contract as contract
        from .relay_shared_metrics import record_process_mark

        record_process_mark(getattr(contract, mark_attr), data)
    except Exception:
        logger.debug("Shared-metrics %s not recorded", mark_attr, exc_info=True)


def _background_tool_call() -> bool:
    try:
        from tools.skill_provenance import is_background_review

        return is_background_review()
    except Exception:
        return False


def _user_agent(agent: Any) -> bool:
    """The review/curator forks tag themselves with the background write origin."""
    return agent is not None and getattr(agent, "_memory_write_origin", None) != _BACKGROUND_ORIGIN


def _route(agent: Any) -> dict[str, str]:
    from .shared_metrics_model import model_route

    return model_route(getattr(agent, "provider", None), getattr(agent, "model", None))


# ---- file edits ------------------------------------------------------------------------------

@dataclass
class _EditProbe:
    """What the fuzzy matcher saw during one edit tool call (several hunks for V4A)."""

    strategies: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)


_EDIT_PROBE: contextvars.ContextVar[_EditProbe | None] = contextvars.ContextVar(
    "shared_metrics_edit_probe", default=None,
)


def note_edit_match(strategy: str | None, miss: str | None = None) -> None:
    """Called by the fuzzy matcher: the strategy that landed a replacement, or why none did
    (``no_match`` / ``ambiguous``). A no-op outside a metered edit tool call."""
    probe = _EDIT_PROBE.get()
    if probe is None:
        return
    if strategy:
        probe.strategies.append(strategy)
    elif miss:
        probe.misses.append(miss)


def _strategy_rank() -> dict[str, int]:
    from tools.fuzzy_match import STRATEGIES

    return {name: index for index, (name, _fn) in enumerate(STRATEGIES)}


# The patch tool's ``mode`` argument: ``patch`` is the V4A multi-hunk format.
_EDIT_MODES = {"replace": "replace", "patch": "v4a", "whole_file": "whole_file"}


def file_edit_fields(*, tool: str, mode: Any, result: Any, probe: _EditProbe) -> dict[str, str]:
    from .shared_metrics_contract import FILE_EDIT_MODES, FILE_EDIT_STRATEGIES

    data = _parsed(result)
    # patch results carry ``success``; write_file reports bytes written, or an ``error``.
    if data.get("success") is True or (data and "success" not in data and not data.get("error")):
        outcome = "already_applied" if data.get("no_change") is True else "applied"
    elif "ambiguous" in probe.misses:
        outcome = "ambiguous"
    elif "no_match" in probe.misses:
        outcome = "no_match"
    else:
        outcome = "failed"
    strategy = "none"
    if outcome == "applied" and probe.strategies:
        # A V4A patch lands several hunks: report the loosest strategy it needed.
        rank = _strategy_rank()
        strategy = max(probe.strategies, key=lambda name: rank.get(name, -1))
    mode_value = _EDIT_MODES.get(str(mode or "").strip().lower(), "replace")
    return {
        "match_strategy": strategy if strategy in FILE_EDIT_STRATEGIES else "none",
        "mode": mode_value if mode_value in FILE_EDIT_MODES else "replace",
        "outcome": outcome,
        "tool": tool,
    }


def record_file_edit(tool: str, mode: str, call: Callable[[], str]) -> str:
    """Run one ``patch`` / ``write_file`` tool call and count how its edit landed; returns the
    call's result unchanged. The matcher reports into a context-local probe, so strategies from
    unrelated callers (skill patches, V4A validation of another call) never mix in."""
    if not _enabled() or _background_tool_call():
        return call()
    probe = _EditProbe()
    token = _EDIT_PROBE.set(probe)
    try:
        result = call()
    finally:
        _EDIT_PROBE.reset(token)
    try:
        _record("FILE_EDIT_MARK", file_edit_fields(tool=tool, mode=mode, result=result, probe=probe))
    except Exception:
        logger.debug("Shared-metrics file edit not recorded", exc_info=True)
    return result


def _parsed(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    try:
        data = json.loads(result) if isinstance(result, (str, bytes)) else None
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---- terminal command outcomes ---------------------------------------------------------------

def _kinds(kind: str, *names: str) -> dict[str, str]:
    return dict.fromkeys(names, kind)


_COMMAND_KINDS: dict[str, str] = {
    **_kinds("git", "git", "gh", "hub", "git-lfs", "tig"),
    **_kinds(
        "package_manager", "pip", "pip3", "uv", "uvx", "poetry", "pipx", "conda", "mamba", "micromamba",
        "npm", "pnpm", "yarn", "apt", "apt-get", "brew", "dnf", "yum", "pacman", "apk", "gem", "bundle",
        "composer", "nix", "snap", "choco", "winget", "port",
    ),
    **_kinds(
        "build", "make", "cmake", "ninja", "cargo", "go", "gradle", "gradlew", "mvn", "bazel", "meson",
        "gcc", "g++", "cc", "clang", "clang++", "rustc", "javac", "tsc", "dotnet", "msbuild", "scons",
        "swift", "xcodebuild", "zig",
    ),
    **_kinds(
        "test_runner", "pytest", "py.test", "tox", "nox", "jest", "vitest", "mocha", "rspec", "phpunit",
        "ctest", "playwright",
    ),
    **_kinds("python", "python", "python3", "ipython", "pypy", "pypy3"),
    **_kinds("node", "node", "npx", "deno", "bun", "ts-node", "tsx", "nodejs"),
    **_kinds(
        "shell_builtin", "cd", "echo", "export", "source", ".", "set", "unset", "pwd", "printf", "test",
        "[", "[[", "true", "false", "exit", "alias", "read", "eval", "type", "hash", "ulimit", "umask",
        "wait", "pushd", "popd", "shopt", "trap", "declare", "local", "if", "for", "while", "case", "{",
    ),
    **_kinds("shell", "bash", "sh", "zsh", "dash", "fish", "ksh", "pwsh", "powershell", "cmd"),
    **_kinds(
        "file_ops", "ls", "cat", "head", "tail", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "find",
        "grep", "egrep", "rg", "sed", "awk", "chmod", "chown", "ln", "du", "df", "tree", "wc", "sort",
        "uniq", "diff", "tar", "zip", "unzip", "gzip", "gunzip", "fd", "stat", "file", "less", "more",
        "realpath", "readlink", "basename", "dirname", "cut", "tr", "xargs", "tee", "jq", "patch",
        "md5sum", "sha256sum", "truncate", "which",
    ),
    **_kinds(
        "network", "curl", "wget", "ssh", "scp", "rsync", "ping", "nc", "dig", "nslookup", "http", "telnet",
        "sftp", "ftp", "traceroute",
    ),
    **_kinds(
        "container", "docker", "podman", "kubectl", "docker-compose", "helm", "nerdctl", "minikube", "kind",
        "k9s", "apptainer", "singularity",
    ),
}
# Prefixes that only change how the real program runs; the program after them is the kind.
_COMMAND_WRAPPERS = frozenset({"sudo", "env", "time", "nohup", "exec", "command", "builtin", "nice", "stdbuf"})
_TOKEN_RE = re.compile(r"\s*([^\s;|&()<>]+)")
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_PYTHON_RE = re.compile(r"python\d+(\.\d+)?")
# Only the command's opening is inspected; a huge heredoc never gets scanned.
_COMMAND_HEAD_CHARS = 256
_MAX_PREFIX_TOKENS = 4


def command_kind(command: Any) -> str:
    """A closed kind from the command's first program word (after env assignments and wrappers
    like ``sudo``). The word itself is never exported."""
    text = command[:_COMMAND_HEAD_CHARS] if isinstance(command, str) else ""
    position = 0
    for _ in range(_MAX_PREFIX_TOKENS + 1):
        match = _TOKEN_RE.match(text, position)
        if match is None:
            return "other"
        position = match.end()
        word = match.group(1).strip("'\"")
        if _ASSIGNMENT_RE.match(word) or word.lower() in _COMMAND_WRAPPERS or word.startswith("-"):
            continue  # env assignments, wrappers and their flags (``sudo -E``)
        name = word.rsplit("/", 1)[-1].lower()
        if name in _COMMAND_KINDS:
            return _COMMAND_KINDS[name]
        return "python" if _PYTHON_RE.fullmatch(name) else "other"
    return "other"


# Exit statuses a shell reports for a foreground child killed by SIGKILL / SIGTERM (OOM killer,
# an external kill); a negative status is Python's own "terminated by signal N".
_KILLED_STATUSES = frozenset({137, 143})


def terminal_outcome(result: Any) -> str | None:
    """How one foreground command ended, from the backend's raw result dict. Hermes' own deadline
    and interrupt set flags on the result, so a command's own ``exit 124`` reads ``nonzero``."""
    if not isinstance(result, dict):
        return None
    if result.get("hermes_timed_out") is True:
        return "timeout"
    if result.get("hermes_interrupted") is True:
        return "killed"
    code = result.get("returncode")
    if not isinstance(code, int) or isinstance(code, bool):
        return None
    if code == 0:
        return "ok"
    return "killed" if code < 0 or code in _KILLED_STATUSES else "nonzero"


def terminal_outcome_fields(*, command: Any, backend: Any, outcome: str) -> dict[str, str]:
    from .shared_metrics_contract import TERMINAL_BACKENDS

    backend_value = str(backend or "").strip().lower()
    return {
        "backend": backend_value if backend_value in TERMINAL_BACKENDS else "other",
        "command_kind": command_kind(command),
        "outcome": outcome,
    }


def record_terminal_outcome(command: Any, backend: Any, result: Any = None, *, outcome: str | None = None) -> None:
    """Count one foreground terminal command that reached an exit status (or Hermes' deadline).
    Commands handed to the background, refused by a guard, or whose backend failed never ran to
    an exit status and are not counted here (``hermes.execution_backend.count`` covers them)."""
    try:
        if not _enabled() or _background_tool_call():
            return
        from .shared_metrics_loop import _UNMETERED

        if _UNMETERED.get():
            return
        resolved = outcome or terminal_outcome(result)
        if resolved is None:
            return
        _record("TERMINAL_OUTCOME_MARK", terminal_outcome_fields(command=command, backend=backend, outcome=resolved))
    except Exception:
        logger.debug("Shared-metrics terminal outcome not recorded", exc_info=True)


# ---- per-turn agent state (loop guards, recovery) -----------------------------------------------

@dataclass
class _TurnState:
    """Per-turn harness bookkeeping; ``agent._harness_metrics_turn`` is reset at turn start."""

    round_outcomes: list[tuple[str, bool]] = field(default_factory=list)
    pending_errors: list[str] = field(default_factory=list)
    guard_signals: set[tuple[str, str]] = field(default_factory=set)


def _turn_state(agent: Any) -> _TurnState:
    state = getattr(agent, "_harness_metrics_turn", None)
    if not isinstance(state, _TurnState):
        state = _TurnState()
        agent._harness_metrics_turn = state
    return state


def _active(agent: Any) -> bool:
    return _user_agent(agent) and _enabled()


# ---- loop guards -----------------------------------------------------------------------------

_GUARD_DETECTORS = {
    **dict.fromkeys(("repeated_exact_failure_block", "repeated_exact_failure_warning"), "exact_failure"),
    **dict.fromkeys(("idempotent_no_progress_block", "idempotent_no_progress_warning"), "idempotent_no_progress"),
    **dict.fromkeys(("same_tool_failure_halt", "same_tool_failure_warning"), "same_tool_failure"),
    "identical_call_streak_halt": "identical_call_streak", "identical_call_streak": "identical_call_streak",
    "identical_cycle_halt": "identical_cycle", "identical_cycle": "identical_cycle",
    "loop_web_search_cap": "web_search_cap", "loop_subagent_cap": "subagent_cap",
}


def _record_guard(agent: Any, signal: str, detector: str) -> None:
    state = _turn_state(agent)
    if (signal, detector) in state.guard_signals:
        return  # a detector that keeps firing is one stuck turn, not N
    state.guard_signals.add((signal, detector))
    _record("LOOP_GUARD_MARK", {**_route(agent), "detector": detector, "signal": signal})


def record_guardrail_decision(agent: Any, action: Any, code: Any) -> None:
    """A tool-loop guardrail verdict: ``warn`` (the call ran, the model was told it is repeating)
    is ``repeated_tool_call``; ``block``/``halt`` (the harness stopped it) is ``loop_detected``.
    Once per turn per detector and signal."""
    try:
        detector = _GUARD_DETECTORS.get(str(code or ""))
        signal = {"warn": "repeated_tool_call", "block": "loop_detected", "halt": "loop_detected"}.get(str(action or ""))
        if detector is None or signal is None or not _active(agent):
            return
        _record_guard(agent, signal, detector)
    except Exception:
        logger.debug("Shared-metrics loop guard not recorded", exc_info=True)


def record_guardrail_warnings(agent: Any, decision: Any, stall_kind: str | None) -> None:
    """The warnings one tool result carried: a guardrail ``warn`` verdict and/or an identical-call
    stall notice (``stall_kind`` names its detector)."""
    if getattr(decision, "action", None) == "warn":
        record_guardrail_decision(agent, "warn", getattr(decision, "code", None))
    if stall_kind:
        record_guardrail_decision(agent, "warn", stall_kind)


_BUDGET_EXIT_PREFIXES = ("budget_exhausted", "max_iterations_reached(")


# ---- recovery after a tool error -------------------------------------------------------------

def _tool_name(name: Any) -> str:
    from .shared_metrics_contract import tool_metric_name

    return tool_metric_name({"tool_name": name})


def observe_tool_outcome(agent: Any, tool_name: Any, is_error: bool) -> None:
    """Every committed tool result of the current round, in the model's emission order."""
    try:
        if _active(agent):
            _turn_state(agent).round_outcomes.append((str(tool_name or ""), bool(is_error)))
    except Exception:
        logger.debug("Shared-metrics tool outcome not observed", exc_info=True)


def _record_recovery(agent: Any, route: dict[str, str], tool: str, next_tool: str, next_outcome: str) -> None:
    _record("TOOL_RECOVERY_MARK", {
        **route, "next_outcome": next_outcome, "next_tool": next_tool, "tool": _tool_name(tool),
    })


def finish_tool_round(agent: Any) -> None:
    """Resolve the previous round's failed calls against this round's first calls, then carry
    this round's failures forward. The next call to the same tool is the retry; with none, the
    model switched tools and its first call is what it tried instead."""
    try:
        if not _active(agent):
            return
        state = _turn_state(agent)
        outcomes, state.round_outcomes = state.round_outcomes, []
        if not outcomes:
            return
        if state.pending_errors:
            route = _route(agent)
            for tool in state.pending_errors:
                same = next((failed for name, failed in outcomes if name == tool), None)
                failed = outcomes[0][1] if same is None else same
                _record_recovery(
                    agent, route, tool, "different" if same is None else "same", "error" if failed else "success",
                )
        state.pending_errors = [name for name, failed in outcomes if failed]
    except Exception:
        logger.debug("Shared-metrics tool recovery not recorded", exc_info=True)


def finish_turn(agent: Any, turn_exit_reason: Any, final_response: Any, *, interrupted: Any, failed: Any) -> None:
    """Turn end: the iteration cap, and failed calls the model never followed with another
    call — ``no_tool_call`` when it answered in text, ``gave_up`` when the turn ended without a
    reply from it (halted, budget spent, interrupted, errored)."""
    try:
        if not _active(agent):
            return
        state = _turn_state(agent)
        reason = str(turn_exit_reason or "")
        if reason.startswith(_BUDGET_EXIT_PREFIXES):
            _record_guard(agent, "iteration_cap", "iteration_budget")
        pending, state.pending_errors = state.pending_errors, []
        if not pending:
            return
        answered = (
            reason.startswith("text_response(") and bool(final_response) and not interrupted and not failed
        )
        route = _route(agent)
        for tool in pending:
            _record_recovery(agent, route, tool, "none", "no_tool_call" if answered else "gave_up")
    except Exception:
        logger.debug("Shared-metrics turn end not recorded", exc_info=True)


# ---- model reply issues ------------------------------------------------------------------------

_FINISH_ISSUES = {"content_filter": "refusal", "length": "truncated_length", "incomplete": "truncated_length"}


def record_reply_finish(agent: Any, response: Any, finish_reason: Any) -> None:
    """A primary response whose structured finish reason is a refusal or an output-length stop
    (transports already promote Anthropic ``stop_reason=refusal`` and an OpenAI ``message.refusal``
    to ``content_filter``). Those mostly leave the loop before normal intake, so they are counted
    where the reason is derived; the latch keeps intake from counting the same response again."""
    try:
        issue = _FINISH_ISSUES.get(str(finish_reason or ""))
        if issue is None or not _active(agent):
            return
        agent._harness_metrics_reply = response
        _record("MODEL_REPLY_ISSUE_MARK", {**_route(agent), "issue": issue})
    except Exception:
        logger.debug("Shared-metrics reply finish not recorded", exc_info=True)


def reply_content_issue(agent: Any, assistant_message: Any) -> str:
    if getattr(assistant_message, "tool_calls", None):
        return "none"
    content = getattr(assistant_message, "content", None)
    if isinstance(content, str) and content.strip() and agent._has_content_after_think_block(content):
        return "none"
    return "reasoning_only" if agent._extract_reasoning(assistant_message) else "empty"


def record_reply_content(agent: Any, response: Any, assistant_message: Any) -> None:
    """Every other primary response: ``empty`` (no visible text, no tool call, no reasoning),
    ``reasoning_only`` (reasoning but nothing to show or run), else ``none`` — one row per
    response, so issue rates have their own denominator."""
    try:
        if not _active(agent):
            return
        if getattr(agent, "_harness_metrics_reply", None) is response:
            agent._harness_metrics_reply = None  # already counted at the finish reason
            return
        _record("MODEL_REPLY_ISSUE_MARK", {**_route(agent), "issue": reply_content_issue(agent, assistant_message)})
    except Exception:
        logger.debug("Shared-metrics reply content not recorded", exc_info=True)
