"""ACP form elicitation (``elicitation/create``) as the transport for Hermes' ``clarify`` tool."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable

logger = logging.getLogger(__name__)

ELICITATION_METHOD = "elicitation/create"
_ANSWER_FIELD = "answer"


def client_supports_form_elicitation(initialize_params: Any) -> bool:
    """ACP names an elicitation mode as supported only by its presence (``"form": {}``), under
    v1 ``clientCapabilities`` or v2 ``capabilities``. Read from the raw initialize params:
    the pinned SDK's ``ClientCapabilities`` model drops the field."""
    if not isinstance(initialize_params, dict):
        return False
    for key in ("clientCapabilities", "capabilities"):
        capabilities = initialize_params.get(key)
        elicitation = capabilities.get("elicitation") if isinstance(capabilities, dict) else None
        if isinstance(elicitation, dict) and elicitation.get("form") is not None:
            return True
    return False


def _answer_schema(choices: list[str] | None, multi_select: bool) -> dict[str, Any]:
    if choices and multi_select:
        return {"type": "array", "items": {"type": "string", "enum": list(choices)}}
    if choices:
        return {"type": "string", "enum": list(choices)}
    return {"type": "string"}


def build_clarify_elicitation(session_id: str, question: str, choices: list[str] | None, multi_select: bool,
                              questions: list[dict] | None = None) -> dict[str, Any]:
    """``elicitation/create`` params for one clarify call: a single ``answer`` field, or one
    required field per batch entry keyed by its ``qid``."""
    if questions:
        properties = {
            entry["qid"]: {**_answer_schema(entry["choices"], entry["multi_select"]), "title": entry["question"]}
            for entry in questions
        }
        message = question or "\n".join(entry["question"] for entry in questions)
    else:
        properties = {_ANSWER_FIELD: _answer_schema(choices, multi_select)}
        message = question
    return {
        "sessionId": session_id, "mode": "form", "message": message,
        "requestedSchema": {"type": "object", "properties": properties, "required": list(properties)},
    }


def _send_request(conn: Any) -> Callable[[str, dict], Any]:
    """The SDK connection's generic JSON-RPC request. ``agent-client-protocol`` 0.9 has no typed
    elicitation method, and ``ext_method`` prefixes ``_`` onto the method name."""
    return conn._conn.send_request


def _resolve_timeout(timeout: float | None) -> float | None:
    """``None`` → ``agent.clarify_timeout``, the knob the gateway/CLI/TUI read; ``<= 0`` waits
    without a deadline, as on the other surfaces."""
    if timeout is None:
        from tools.clarify_gateway import get_clarify_timeout

        timeout = get_clarify_timeout()
    return float(timeout) if timeout > 0 else None


def make_clarify_callback(conn: Any, loop: asyncio.AbstractEventLoop, session_id: str,
                          timeout: float | None = None) -> Callable[..., Any]:
    """Return a batch-capable ``clarify_callback`` that asks over ``elicitation/create`` from the
    agent's worker thread and blocks for the answer.

    Accept returns the answer (the batch form ``{"answers": {qid: value}}``); decline and cancel
    are a skip (``""``, or no answers for a batch); no answer before the timeout is the timeout
    sentinel (``None`` for a batch). A failed request raises, which ``clarify_tool`` reports."""
    from agent.async_utils import safe_schedule_threadsafe
    from tools.clarify_tool import TIMEOUT_RESPONSE

    def _callback(question: str, choices: list[str] | None, multi_select: bool = False,
                  questions: list[dict] | None = None) -> Any:
        params = build_clarify_elicitation(session_id, question, choices, multi_select, questions)
        future = safe_schedule_threadsafe(
            _send_request(conn)(ELICITATION_METHOD, params), loop, logger=logger,
            log_message="Clarify elicitation: failed to schedule on loop",
        )
        if future is None:
            raise RuntimeError("could not schedule the ACP elicitation request")
        try:
            response = future.result(timeout=_resolve_timeout(timeout))
        except FutureTimeout:
            future.cancel()
            logger.warning("Clarify elicitation timed out")
            return None if questions else TIMEOUT_RESPONSE
        accepted = isinstance(response, dict) and response.get("action") == "accept"
        content = response.get("content") if accepted else None
        content = content if isinstance(content, dict) else {}
        if questions:
            return {"answers": {entry["qid"]: content[entry["qid"]] for entry in questions if entry["qid"] in content}}
        return content.get(_ANSWER_FIELD, "")

    return _callback
