"""Plugin observers for the prompts a gateway session owner opens for a person to answer.

``on_session_prompt`` fires once when the owner opens a prompt for a session's viewers, carrying the id
that answers it; ``on_session_prompt_settled`` fires once when that prompt stops being open. The prompts:
an ``approval.request`` / ``clarify.request`` event (``PendingControls``; answered by ``approval.respond`` /
``clarify.respond`` with ``prompt_id``) and a server-to-client request frame whose method is in
``PERSON_METHODS`` (``session_server_requests._ask``; answered by a response frame or ``clarify.lock`` with
the frame's ``srq-`` id). Window requests a client app answers on its own never fire.

Observers only: return values are ignored and a failing plugin never affects the prompt. Calls run in
order on one worker thread, so plugin code never runs under ``event_stream.lock`` or
``session_server_requests._lock``, on the agent thread or on the event loop; each runs in the owning
profile's scope (``session_authorities.owner_scope``). ``params`` is a deep copy with every string
force-redacted (as ``tools/human_input_hooks.py`` redacts its ``prompt``) inside that owner scope, so
the owning profile's registered vault values are scrubbed too, and never holds an answer.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import queue
import sys
import threading
from copy import deepcopy

logger = logging.getLogger(__name__)

OPENED = "on_session_prompt"
SETTLED = "on_session_prompt_settled"

# The request-frame methods a PERSON answers. Explicit on purpose: a new window request (theme,
# tour, preview, reads) must never start notifying anyone by default.
PERSON_METHODS = frozenset({"sudo", "secret", "clarify", "vault.unlock_prompt", "vault.code", "vault.save_login"})

_calls: queue.SimpleQueue = queue.SimpleQueue()
_worker_lock = threading.Lock()
_worker: threading.Thread | None = None


class _Opened:
    """One prompt that got ``on_session_prompt``: what its settled call repeats, and whose scope."""
    __slots__ = ("authority", "execution_generation", "kind", "prompt_id", "session_id")

    def __init__(self, authority, session_id, kind, prompt_id, execution_generation):
        self.authority = authority
        self.session_id = session_id
        self.kind = kind
        self.prompt_id = prompt_id
        self.execution_generation = execution_generation

    def payload(self):
        return {"session_id": self.session_id, "kind": self.kind, "prompt_id": self.prompt_id,
                "execution_generation": self.execution_generation}


class PromptTable(dict):
    """``PendingControls.pending``: a prompt that leaves the table is settled for observers, whichever
    path removes it (``snapshot``, a ``remote_responders`` answer, a managed worker's ``prompt_settled``)."""

    def __init__(self):
        super().__init__()
        self.observed: dict[str, _Opened] = {}

    def __delitem__(self, prompt_id):
        super().__delitem__(prompt_id)
        _settle(self.observed.pop(prompt_id, None), "settled")

    def pop(self, prompt_id, *default):
        held = prompt_id in self
        value = super().pop(prompt_id, *default)
        if held:
            _settle(self.observed.pop(prompt_id, None), "settled")
        return value

    # A copy is a plain-dict snapshot (as ``dict.copy`` already returns): removing from it never
    # settles anything and never touches this table's ``observed``.
    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return deepcopy(dict(self), memo)


def control_opened(authority, session_id, controls, prompt_id) -> None:
    """The authority published *prompt_id*'s ``approval.request`` / ``clarify.request`` (it is in
    ``controls.pending``, where ``respond`` finds it)."""
    table = controls.pending
    held = table.get(prompt_id)
    if held is None or prompt_id in table.observed:
        return
    prompt = held[1]
    opened = _Opened(authority, session_id, prompt["kind"], prompt_id, prompt.get("execution_generation"))
    table.observed[prompt_id] = opened
    _enqueue(OPENED, authority, {**opened.payload(), "params": deepcopy(prompt)})


def request_opened(events, session_id, request_id, method, params, replay) -> _Opened | None:
    """``_ask`` wrote request *request_id* (already in ``_open``): fire for a person's prompt and return
    the handle :func:`request_settled` takes, else None."""
    if not replay or method not in PERSON_METHODS:
        return None
    authority = _frame_owner(events, session_id)
    opened = _Opened(authority, session_id, method, request_id, None)
    _enqueue(OPENED, authority, {**opened.payload(), "params": deepcopy(params)})
    return opened


def request_settled(opened: _Opened | None, reason: str) -> None:
    """``_ask`` published the request's ``request.cancel`` with *reason*."""
    _settle(opened, reason)


def _settle(opened: _Opened | None, reason: str) -> None:
    if opened is not None:
        _enqueue(SETTLED, opened.authority, {**opened.payload(), "reason": reason})


def _frame_owner(events, session_id):
    """The served authority whose live *session_id* is the stream the frame went out on; None when this
    process runs no gateway API (no runner to ask), where the asker's own scope applies."""
    run_api = sys.modules.get("gateway.run_api")
    runner = getattr(run_api, "_ACTIVE_RUNNER", None)
    if runner is None:
        return None
    from gateway.session_authorities import all_authorities
    from gateway.session_server_requests import owned_by
    return next((a for a in all_authorities(runner) if owned_by(a, session_id, events)), None)


def _enqueue(hook_name, authority, payload) -> None:
    global _worker
    _calls.put((contextvars.copy_context(), hook_name, authority, payload))
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_drain, daemon=True, name="session-prompt-hooks")
            _worker.start()


def _drain() -> None:
    while True:
        context, hook_name, authority, payload = _calls.get()
        context.run(_dispatch, hook_name, authority, payload)


def _dispatch(hook_name, authority, payload) -> None:
    from hermes_cli.config_backend import ConfigBackendUnavailable
    try:
        scope = contextlib.nullcontext()
        if authority is not None:
            from gateway.session_authorities import owner_scope
            scope = owner_scope(authority)
        with scope:
            if "params" in payload:  # in the owner's scope: its registered vault values are scrubbed too
                payload = {**payload, "params": _redacted(payload["params"])}
            from hermes_cli.lifecycle import invoke_hook
            invoke_hook(hook_name, **payload)
    except (Exception, ConfigBackendUnavailable):  # health: allow BLE001 -- observers never break a prompt
        logger.debug("%s dispatch failed", hook_name, exc_info=True)


def _redacted(value):
    if isinstance(value, str):
        try:
            from agent.redact import redact_approval_display
            return redact_approval_display(value, force=True)
        except Exception:  # health: allow BLE001 -- never fall back to the raw text
            logger.debug("session prompt hook redaction failed", exc_info=True)
            return ""
    if isinstance(value, dict):
        return {key: _redacted(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(item) for item in value]
    return value
