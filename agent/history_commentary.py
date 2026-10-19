"""Display-only Codex commentary projection shared by REST and gateway history.

Provider items and the stored reasoning string remain untouched for model replay.
Only the projected fields may be used for public transcript text.
"""

import json
from contextlib import contextmanager
from typing import Any

from agent.redact import redact_sensitive_text
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from utils import is_truthy_value


@contextmanager
def _owning_home(home):
    token = set_hermes_home_override(str(home)) if home is not None else None
    try:
        yield
    finally:
        if token is not None:
            reset_hermes_home_override(token)


def visible_commentary(text: str, *, strip_thinking=None) -> str:
    """Use the same think stripping and secret redaction as live delivery."""
    if strip_thinking is None:
        from agent.agent_runtime_helpers import strip_think_blocks

        strip_thinking = lambda value: strip_think_blocks(None, value)

    visible = strip_thinking(text).strip()
    return redact_sensitive_text(visible) if visible else visible


def _phase_message_items(message: dict, phases: frozenset[str | None]) -> list[str]:
    items = message.get("codex_message_items")
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except (TypeError, ValueError):
            return []
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if (
            not isinstance(item, dict)
            or item.get("type") != "message"
            or item.get("role") != "assistant"
        ):
            continue
        phase, content = item.get("phase"), item.get("content")
        if phase is not None and not isinstance(phase, str):
            continue
        normalized_phase = phase.strip().lower() if isinstance(phase, str) else None
        if normalized_phase not in phases or not isinstance(content, list):
            continue
        text = "".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
            and part["text"].strip()
        ).strip()
        if text:
            result.append(text)
    return result


def _commentary_items(message: dict) -> list[str]:
    return _phase_message_items(message, frozenset({"commentary"}))


def _final_items(message: dict) -> list[str]:
    # Unphased assistant messages are normalized as final content too.
    return _phase_message_items(message, frozenset({"final", "final_answer", None}))


def _without_flattened_commentary(reasoning: str, commentary: list[str]) -> str:
    """Omit exact whole-line public segments from a display copy, not source data.

    Current normalization separates parts with two newlines; older rows may
    have one. If a private segment happens to be identical, origin is ambiguous:
    omit both occurrences rather than leaking the public text or choosing the
    wrong one. Surrounding reasoning is retained in its original line style.
    """
    remaining = reasoning
    for text in sorted(set(commentary), key=len, reverse=True):
        search_from = 0
        while (at := remaining.find(text, search_from)) >= 0:
            end = at + len(text)
            if (at == 0 or remaining[at - 1] == "\n") and (
                end == len(remaining) or remaining[end] == "\n"
            ):
                before, after = remaining[:at], remaining[end:]
                prefix, suffix = before.rstrip("\n"), after.lstrip("\n")
                separator = (
                    "\n\n"
                    if before.endswith("\n\n") or after.startswith("\n\n")
                    else "\n"
                )
                remaining = prefix + (separator if prefix and suffix else "") + suffix
                search_from = 0
            else:
                search_from = end
    return remaining


def _project_one(message: dict, *, enabled: bool) -> dict:
    if message.get("role") != "assistant" or message.get("display_kind") == "hidden":
        return message
    raw = _commentary_items(message)
    if not raw:
        return message
    projected = dict(message)
    # A disabled profile must not recover raw sidecar text on the frontend.
    projected["display_commentary"] = []
    if enabled:
        projected["display_commentary"] = [
            part for text in raw if (part := visible_commentary(text))
        ]
    reasoning = (
        message.get("reasoning")
        or message.get("reasoning_content")
        or message.get("reasoning_details")
        or ""
    )
    if isinstance(reasoning, str):
        projected["display_reasoning"] = _without_flattened_commentary(reasoning, raw)

    # A nonempty canonical answer can also be filled by stream recovery after
    # the provider sidecar was captured. Without a matching final item there
    # is no way to distinguish "commentary + final" from a real final that
    # merely starts with the same words. Never delete canonical answer bytes.
    # Sanitize the display copy when it contains raw commentary and avoid
    # duplicating a leading commentary item in a separate Desktop bubble.
    content = message.get(
        "display_content", message.get("content", message.get("text"))
    )
    if isinstance(content, str) and any(text in content for text in raw):
        finals = _final_items(message)
        if not (finals and content.strip() == "\n".join(finals)):
            projected["display_content"] = visible_commentary(content)
            if any(content.startswith(text) for text in raw):
                projected["display_commentary"] = []
    return projected


def project_history_commentary(messages: list[dict], *, home: Any = None) -> list[dict]:
    """Project a batch inside the owning profile's config and redaction scope."""
    if not any(
        isinstance(message, dict) and message.get("codex_message_items")
        for message in messages
    ):
        return messages
    with _owning_home(home):
        from hermes_cli.config import load_config

        try:
            display = load_config().get("display") or {}
            enabled = is_truthy_value(
                display.get("show_commentary"), default=True
            ) and is_truthy_value(
                display.get("interim_assistant_messages"), default=True
            )
        except Exception:
            enabled = False  # unreadable policy must not publish raw provider items
        return [
            _project_one(message, enabled=enabled)
            if isinstance(message, dict)
            else message
            for message in messages
        ]
