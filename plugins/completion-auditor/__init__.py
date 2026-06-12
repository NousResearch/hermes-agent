"""completion-auditor — audit-only evidence-alignment observer.

The first implementation slice is intentionally small:

* plugin loading is opt-in through ``plugins.enabled``;
* ``post_tool_call`` records metadata-only evidence per ``session_id``/``turn_id``;
* ``post_llm_call`` writes one ``hermes-completion-audit-v1`` JSONL record;
* no hook returns replacement text, so user-visible final responses are not
  mutated by this plugin.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import load_settings
from .evidence import clear as _clear_evidence
from .evidence import pop_turn, record_tool_call, size as _ledger_size
from .report import build_record, write_record

logger = logging.getLogger(__name__)


def _on_post_tool_call(**kwargs: Any) -> None:
    settings = load_settings()
    if not settings.audit_enabled:
        return None
    ok = record_tool_call(
        include_result_excerpt=settings.include_tool_result_excerpt,
        max_result_excerpt_chars=settings.max_result_excerpt_chars,
        redact_secrets=settings.redact_secrets,
        **kwargs,
    )
    if not ok:
        logger.debug("completion-auditor: uncorrelatable post_tool_call payload")
    return None


def _on_post_llm_call(
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
    assistant_response: str | None = None,
    **_: Any,
) -> None:
    settings = load_settings()
    if not settings.audit_enabled:
        return None
    evidence = pop_turn(session_id, turn_id)
    record = build_record(
        settings=settings,
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        assistant_response=assistant_response,
        evidence=evidence,
    )
    try:
        write_record(settings, record)
    except Exception as exc:  # pragma: no cover - fail-open runtime guard
        logger.warning("completion-auditor: failed to write audit record: %s", exc)
    return None


def register(ctx: Any) -> None:
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)


# Test helpers; intentionally private so runtime users stick to plugin hooks.
def _reset_for_tests() -> None:
    _clear_evidence()


def _ledger_size_for_tests() -> int:
    return _ledger_size()
