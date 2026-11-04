"""Clarify prompt delivery for the API server.

A ``/api/sessions/{id}/chat/stream`` run streams its own SSE queue and never subscribes to the
gateway's live event stream, so a clarify prompt published only there parks the turn in
``tools.clarify_gateway`` until the clarify timeout with the question never reaching the client.
``notify_clarify_prompt`` hands the prompt to the queue of the run that asked it; the client answers
with ``POST /v1/runs/{run_id}/clarify``.

Kept in a sibling module because ``gateway/platforms/api_server.py`` is already over its line cap:
the facade binds this function onto ``APIServerAdapter`` at the end of the module.
"""

from __future__ import annotations

from typing import Any


def notify_clarify_prompt(adapter: Any, *, session_id: str, entry: Any, generation: int,
                          session_key: str = "") -> None:
    """Forward a clarify prompt to the SSE queue of the session-stream run that asked it.

    One session can have several runs in ``_run_statuses`` (every turn adds one) and dict order is
    oldest-first, so matching the FIRST hit would enqueue the prompt into a finished turn's dead
    queue and the live client would never see it. Finished runs' queues are pruned here and the
    NEWEST live run for this session gets the prompt.
    """
    from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES

    wanted = {str(session_id or ""), str(session_key or "")}
    best = None
    for rid, status in list(adapter._run_statuses.items()):
        if rid not in adapter._run_event_queues:
            continue
        if str(status.get("status") or "") in TERMINAL_STATUSES:
            adapter._run_event_queues.pop(rid, None)
            continue
        sid = str(status.get("session_id") or "")
        if sid and (sid in wanted or str(session_key or "").endswith(sid)):
            key = (float(status.get("created_at") or 0.0), float(status.get("updated_at") or 0.0))
            if best is None or key > best[0]:
                best = (key, rid)
    run_id = best[1] if best else ""
    events = adapter._run_event_queues.get(run_id) if run_id else None
    if events is None:
        return
    events.enqueue("clarify.request", {
        "kind": "clarify",
        "request_id": str(getattr(entry, "clarify_id", "") or ""),
        "execution_generation": generation,
        "question": str(getattr(entry, "question", "") or ""),
        "choices": list(getattr(entry, "choices", None) or []),
        "multi_select": bool(getattr(entry, "multi_select", False)),
    })
