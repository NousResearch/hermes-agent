"""Host-observed completion evidence read from a transcript.

"Done" is a claim; what actually ran is a fact. This module turns OpenAI-shaped message dicts
(the live turn's ``messages`` or ``SessionDB.get_messages`` rows) into the facts: which tools ran
with which key argument and exit code, which file edits landed, and whether a passing check
(a test / build / lint / CI command that exited 0) ran after the last edit.

Two readers grade "done" on it: the ``/goal`` judge (``hermes_cli/goals_evidence.py``) and the
``todo_list`` result, which marks an item completed after an unchecked edit as unverified
(``annotate_todo_result``). Both are read-only and append to a tool result or the judge's own
prompt, never the system prompt.

A terminal result that carries the verification ledger's own verdict (``verification_evidence``,
present when ``agent.verify_on_stop`` is on) wins over the command-shape heuristic below. Edits
made through the shell (``sed -i``, ``cat >``) are invisible here, so a transcript whose only
edits went through the terminal reads as "no edits" and nothing is annotated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from agent.tool_dispatch_helpers import _extract_landed_file_mutation_paths
from agent.tool_result_classification import FILE_MUTATING_TOOL_NAMES, file_mutation_result_landed

# Commands whose exit code says whether the work holds up: test runners, builds, type checkers,
# linters and CI status. A command matching none of these is ordinary activity, not a check.
_CHECK_COMMAND_RE = re.compile(
    r"""(?x)(?:^|[\s;&|(/])(?:
        pytest|py\.test|tox|nox|unittest|run_tests(?:\.sh)?|scripts/check\b|
        (?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|build|lint|check|typecheck|ci)\b|
        vitest|jest|tsc|mypy|pyright|ruff\s+check|eslint|
        cargo\s+(?:test|build|check|clippy|nextest)\b|
        go\s+(?:test|build|vet)\b|
        make\s+(?:test|check|build|lint|ci)\b|ctest|
        mvn\s+(?:\S+\s+)*(?:test|verify|package)\b|gradlew?\s+(?:test|build|check)\b|
        dotnet\s+(?:test|build)\b|swift\s+(?:test|build)\b|
        gh\s+(?:pr\s+checks|run\s+(?:watch|view))\b|hermes\s+verify\b
    )"""
)
# ``a | b`` reports b's status and ``a || true`` reports 0 whatever a did, so the exit code of
# such a command does not belong to the check in it.
_STATUS_HIDING_RE = re.compile(r"\|\|?")
_ARG_KEYS = ("path", "command", "query", "url", "name", "pattern", "task_id")
_DETAIL_CHARS = 80


@dataclass(frozen=True)
class ToolEvent:
    """One tool call joined with its result."""

    name: str
    detail: str  # ``key=value`` for the first informative argument, or ""
    exit_code: Optional[int]
    is_edit: bool  # a file write/patch whose result says it landed
    is_check: bool  # a test/build/lint/CI command
    check_passed: bool  # is_check, and its own status was observed as 0
    paths: tuple[str, ...] = ()  # files a landed edit wrote
    edit_failed: bool = False  # a file write/patch whose result came back without landing


def _parse_json(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _call_parts(call: Any) -> tuple[str, dict, str]:
    """``(name, args, call_id)`` of a transcript tool_call (OpenAI or flattened shape)."""
    if not isinstance(call, dict):
        return "?", {}, ""
    fn: dict = call["function"] if isinstance(call.get("function"), dict) else call
    args = _parse_json(fn.get("arguments") or {})
    return str(fn.get("name") or "?"), args if isinstance(args, dict) else {}, str(call.get("id") or call.get("call_id") or "")


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return ""


def _check_status(command: str, result: Any) -> tuple[bool, bool]:
    """``(is_check, passed)`` for a terminal call. The ledger's verdict, when the result carries
    one, decides; otherwise the command shape plus an exit code only it could have produced."""
    data = result if isinstance(result, dict) else {}
    ledger = data.get("verification_evidence")
    if isinstance(ledger, dict) and ledger.get("status"):
        return True, ledger.get("status") == "passed"
    if not _CHECK_COMMAND_RE.search(command):
        return False, False
    return True, data.get("exit_code") == 0 and not _STATUS_HIDING_RE.search(command)


def _event(call: Any, results: dict[str, Any]) -> ToolEvent:
    name, args, call_id = _call_parts(call)
    detail = ""
    for key in _ARG_KEYS:
        value = args.get(key)
        if value not in (None, "", [], {}):
            text = " ".join(str(value).split())
            detail = f"{key}={text[:_DETAIL_CHARS - 1] + '…' if len(text) > _DETAIL_CHARS else text}"
            break
    raw_result = results.get(call_id)
    data = _parse_json(_result_text(raw_result)) if raw_result is not None else None
    exit_code = data.get("exit_code") if isinstance(data, dict) else None
    is_check, passed = (
        _check_status(str(args.get("command") or ""), data) if name == "terminal" else (False, False)
    )
    landed = name in FILE_MUTATING_TOOL_NAMES and file_mutation_result_landed(name, _result_text(raw_result))
    return ToolEvent(
        name=name, detail=detail, exit_code=exit_code if isinstance(exit_code, int) else None,
        is_edit=landed, is_check=is_check, check_passed=passed,
        paths=tuple(_extract_landed_file_mutation_paths(name, args, _result_text(raw_result))) if landed else (),
        edit_failed=name in FILE_MUTATING_TOOL_NAMES and raw_result is not None and not landed,
    )


def tool_events(messages: Iterable[Any]) -> list[ToolEvent]:
    """Every tool call in ``messages``, oldest first, joined with its result by call id."""
    rows = [m for m in messages or () if isinstance(m, dict)]
    results = {str(m.get("tool_call_id")): m.get("content") for m in rows if m.get("role") == "tool" and m.get("tool_call_id")}
    events: list[ToolEvent] = []
    for row in rows:
        if row.get("role") != "assistant" or not row.get("tool_calls"):
            continue
        calls = _parse_json(row.get("tool_calls"))
        events.extend(_event(call, results) for call in calls or () if isinstance(calls, list))
    return events


def current_turn(messages: Iterable[Any]) -> list[Any]:
    """The rows after the last user message (the turn in flight or just finished)."""
    rows = list(messages or ())
    for index in range(len(rows) - 1, -1, -1):
        if isinstance(rows[index], dict) and rows[index].get("role") == "user":
            return rows[index + 1:]
    return rows


def _label(event: ToolEvent) -> str:
    return f"{event.name} {event.detail}".strip()


def check_after_last_edit(events: list[ToolEvent]) -> dict[str, Any]:
    """Whether the last landed edit is covered by a passing check run after it.

    ``state``: ``no_edits`` (nothing to verify), ``verified``, ``failed`` (the newest check after
    the edit exited non-zero), ``hidden`` (a check ran but a pipe or ``||`` hid its status) or
    ``unchecked`` (no check after the edit). ``reason`` is one plain sentence for the model.
    """
    last_edit = max((i for i, e in enumerate(events) if e.is_edit), default=None)
    if last_edit is None:
        return {"state": "no_edits", "reason": ""}
    edit = _label(events[last_edit])
    checks = [e for e in events[last_edit + 1:] if e.is_check]
    if not checks:
        return {"state": "unchecked", "reason": f"no test/build/CI command ran after the last edit ({edit})"}
    newest = checks[-1]
    if newest.check_passed:
        return {"state": "verified", "reason": f"{_label(newest)} passed after the last edit"}
    if newest.exit_code not in (None, 0):
        return {"state": "failed", "reason": f"the last check after the edit failed ({_label(newest)} → exit {newest.exit_code})"}
    return {"state": "hidden", "reason": f"the check after the last edit ({_label(newest)}) ran behind a pipe or `||`, so its own exit status is unknown"}


def annotate_todo_result(result: Any, before: dict[str, str], messages: Optional[list]) -> Any:
    """Mark items this write moved to ``completed`` as unverified when a landed edit has no
    passing check after it. Adds ``unverified: {ids, reason}`` to the todo JSON; never blocks
    the write and leaves any other result (errors, reads) untouched."""
    data = _parse_json(result)
    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return result
    newly = [str(t.get("id")) for t in data["todos"]
             if isinstance(t, dict) and t.get("status") == "completed" and before.get(str(t.get("id"))) != "completed"]
    if not newly:
        return result
    verdict = check_after_last_edit(tool_events(messages or ()))
    if verdict["state"] in ("no_edits", "verified"):
        return result
    data["unverified"] = {
        "ids": newly,
        "reason": f"Marked completed, but {verdict['reason']}. Run the check, or say why it can't be run.",
    }
    return json.dumps(data, ensure_ascii=False)
