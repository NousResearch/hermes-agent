"""``/api/sessions/{session_id}/asks``: the prompts a session's turn waits on, as API objects.

An ask is a dangerous-command approval (``approval``), a clarify question (``question``) or a masked
secret/sudo request (``secret``), recorded by the session's live event stream (gateway/session_asks.py).
Answers go through the session authority exactly as ``POST /v1/runs/{id}/approval`` and ``/clarify``
do (``api_server_authority_runs.respond_prompt``): same prompt id, same execution generation, same
native waiter, so there is no second approval queue. A secret value is never accepted here (it only
travels in an attached viewer's response frame); a secret ask can only be dismissed.

Bearer API key auth, scoped to the request's profile (``/p/<profile>/`` selects it): a session of
another profile is not found. The key reaches every session of its profile (Desktop, TUI, messaging
too), as the other ``/api/sessions`` routes do. Room grants name runs, not sessions, and are refused.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway.platforms.api_server_authority_runs import respond_prompt, store_error_response
from gateway.session_asks import STATUSES, approval_outcome, ledger_of, question_outcome, retired_of
from gateway.session_contract import SessionActor, SessionRef
from hermes_state_runtime import RuntimeStoreError

CAPABILITY_ENDPOINTS = (
    ("session_asks", ("GET", "/api/sessions/{session_id}/asks")),
    ("session_ask", ("GET", "/api/sessions/{session_id}/asks/{ask_id}")),
    ("session_ask_answer", ("POST", "/api/sessions/{session_id}/asks/{ask_id}/answer")),
    ("session_ask_dismiss", ("POST", "/api/sessions/{session_id}/asks/{ask_id}/dismiss")))
# Long enough for the agent thread to record a settle the answer just caused.
_SETTLE_WAIT_S, _SETTLE_POLL_S = 2.0, 0.05
_READ_ACTOR = frozenset({"session:read", "session:operator"})


def _http_routes(self) -> list[tuple[str, str, Any]]:
    base = "/api/sessions/{session_id}/asks"
    return [("GET", base, _bound(self, _handle_list)), ("GET", base + "/{ask_id}", _bound(self, _handle_get)),
            ("POST", base + "/{ask_id}/answer", _bound(self, _handle_answer)),
            ("POST", base + "/{ask_id}/dismiss", _bound(self, _handle_dismiss))]


def _bound(adapter, handler):
    async def route(request):
        return await handler(adapter, request)
    route.__name__ = handler.__name__
    return route


def _error(message: str, status: int, code: str, **extra) -> web.Response:
    from gateway.platforms.api_server import _openai_error
    body = _openai_error(message, code=code)
    body["error"].update(extra)
    return web.json_response(body, status=status)


async def _session_ledger(adapter, request) -> tuple[Optional[tuple], Optional[web.Response]]:
    """``((authority, session_id, ledger | None), None)`` for the request's own profile, else an error.
    A stored session with no live conversation reads the asks it had when it was retired, if any."""
    if adapter._room_grant_token(request):
        return None, _error("Room grants cannot read session asks; use the run routes.", 403,
                            "room_grant_not_supported")
    if (auth_err := adapter._check_auth(request)) is not None:
        return None, auth_err
    session_id = request.match_info["session_id"]
    from gateway.session_authorities import require_active_authority
    try:
        authority = require_active_authority(adapter.gateway_runner, lambda: adapter._ensure_session_db())
    except RuntimeStoreError as exc:
        return None, _store_error(exc)
    live = authority.sessions.get(session_id)
    if live is None:
        _session, err = await adapter._get_existing_session_or_404(session_id)
        return (None, err) if err else ((authority, session_id, retired_of(authority).get(session_id)), None)
    try:  # the authority's own read check: deleted, history-mount and profile rules
        authority.authorize(SessionActor("api", authority.profile_id, _READ_ACTOR, "api-asks"),
                            SessionRef(authority.profile_id, session_id), "session:read")
    except RuntimeStoreError as exc:
        return None, _store_error(exc)
    ledger = ledger_of(live.event_stream)
    if (earlier := retired_of(authority).take(session_id)) is not None:
        ledger.absorb(earlier)  # a restored conversation keeps the asks it had before retirement
    return (authority, session_id, ledger), None


def _store_error(exc: RuntimeStoreError) -> web.Response:
    if exc.reason in ("not_found", "permission_denied", "profile_mismatch"):
        return _error("Session not found", 404, "session_not_found")
    from gateway.platforms.api_server import _openai_error
    return store_error_response(_openai_error, exc)


def _statuses(raw: Optional[str]) -> Optional[tuple[str, ...]]:
    if raw is None or raw == "":
        return ("pending",)
    if raw == "all":
        return STATUSES
    picked = tuple(s.strip() for s in raw.split(","))
    return picked if all(s in STATUSES for s in picked) else None


async def _handle_list(adapter, request) -> web.Response:
    """GET /api/sessions/{session_id}/asks?status=pending|all|<s>,<s> — newest first."""
    scope, err = await _session_ledger(adapter, request)
    if err:
        return err
    statuses = _statuses(request.query.get("status"))
    if statuses is None:
        return _error("status must be 'all' or a comma list of: " + ", ".join(STATUSES), 400, "invalid_status")
    _authority, session_id, ledger = scope
    asks = ledger.list(statuses) if ledger is not None else []
    return web.json_response({"object": "list", "session_id": session_id, "data": asks})


async def _owned_ask(adapter, request):
    scope, err = await _session_ledger(adapter, request)
    if err:
        return None, None, err
    ledger, ask_id = scope[2], request.match_info["ask_id"]
    ask = ledger.get(ask_id) if ledger is not None else None
    if ask is None:
        return None, None, _error(f"Ask not found: {ask_id}", 404, "ask_not_found")
    return scope, ask, None


async def _handle_get(adapter, request) -> web.Response:
    """GET /api/sessions/{session_id}/asks/{ask_id}."""
    _scope, ask, err = await _owned_ask(adapter, request)
    return err or web.json_response(ask.as_dict())


def _approval_response(ask, body) -> tuple:
    from tools.approval import canonical_approval_choice
    choice = canonical_approval_choice(body.get("choice"))
    if set(body) != {"choice"} or choice not in ask.payload.get("choices", ()):
        return None, None, "choice must be one of: " + ", ".join(ask.payload.get("choices", ()))
    return {"choice": choice}, {"choice": choice}, None


_QUESTION_SHAPE = 'answer with {"text": "..."} or {"choice": <one of the question\'s choices>}'


def _one_answer(body, choices, multi) -> tuple:
    """``{"text": str}`` (free text, or a choice label) or ``{"choice": label | [labels]}`` -> the raw
    answer the clarify tool reads (a multi-select is the JSON array of labels it decodes) and the
    answer the ask records; ``(None, None)`` when malformed."""
    if not isinstance(body, dict):
        return None, None
    if set(body) == {"text"} and isinstance(body["text"], str) and body["text"].strip():
        return body["text"], {"text": body["text"]}
    picked = body.get("choice") if set(body) == {"choice"} else None
    picks = picked if isinstance(picked, list) and multi else [picked]
    if picks and all(isinstance(p, str) and p in choices for p in picks):
        return (json.dumps(picks, ensure_ascii=False) if multi else picks[0]), {"choice": picked}
    return None, None


def _question_response(ask, body) -> tuple:
    if "questions" in ask.payload:
        return _batch_response(ask, body)
    raw, recorded = _one_answer(body, ask.payload.get("choices") or [], bool(ask.payload.get("multi_select")))
    return (None, None, _QUESTION_SHAPE) if raw is None else ({"answer": raw}, recorded, None)


def _batch_response(ask, body) -> tuple:
    """A local viewer's question panel (Desktop, TUI): ``{"answers": {qid: answer | null}}`` naming
    every question (``null`` skips one), or one question's ``text`` / ``choice`` when it holds one."""
    questions = {q["qid"]: q for q in ask.payload["questions"]}
    given = body.get("answers") if set(body) == {"answers"} else (
        {next(iter(questions)): body} if len(questions) == 1 else None)
    if not isinstance(given, dict) or set(given) != set(questions):
        return None, None, 'answer with {"answers": {<qid>: {"text"|"choice": ...} | null}} for ' + ", ".join(questions)
    answers = {}
    for qid, one in given.items():
        q = questions[qid]
        raw = None if one is None else _one_answer(one, q.get("choices") or [], bool(q.get("multi_select")))[0]
        if one is not None and raw is None:
            return None, None, f"{qid}: {_QUESTION_SHAPE}"
        answers[qid] = raw
    return {"answers": answers}, {"answers": answers}, None


_ANSWER_SHAPES = {"approval": _approval_response, "question": _question_response}
# The native answer a dismiss sends: a deny, or the clarify skip (an empty answer).
_DISMISS_RESPONSES = {"approval": {"choice": "deny"}, "question": {"answer": ""}}
_RESPOND_KIND = {"approval": "approval", "question": "clarify"}


def _expected_settle(ask, response: Optional[dict]) -> tuple:
    """The ``(status, answer)`` the ask's native waiter reports once *response* reaches it."""
    if ask.kind == "approval":
        return approval_outcome("resolved", {"choice": response["choice"]})[:2]
    if response is not None and "answers" in response:
        return "answered", {"answers": response["answers"]}
    if response is None or not response["answer"].strip():
        return "dismissed", None  # a skip, a cancel-all or a secret settled with no value
    return question_outcome(response["answer"])[:2]


async def _deliver(authority, session_id, ask, response: Optional[dict]) -> dict:
    """Send *response* to the ask's native waiter: the authority's prompt controls, or for a viewer
    request (a secret, a question panel) its response frame on the session's own stream."""
    if ask.execution_generation is not None:
        return await respond_prompt(authority, session_id, ask.execution_generation, ask.id,
                                    _RESPOND_KIND[ask.kind], response, operator=True)
    from gateway import session_server_requests as requests
    live = authority.sessions.get(session_id)
    result = {"value": ""} if ask.kind == "secret" else (response if response and "answers" in response else {})
    settled = live is not None and requests.resolve(
        {"id": ask.id, "result": result},
        session_ok=lambda sid, events: sid == session_id and events is live.event_stream)
    return {"status": "resolved" if settled else "already_resolved"}


async def _settle(scope, ask, status: str, recorded: Optional[dict], response: Optional[dict]) -> web.Response:
    """Send *response* to the ask's native waiter, recording the close as *status* by ``api``. One API
    answer at a time: a concurrent one is told the ask is closed."""
    authority, session_id, ledger = scope
    if not ledger.claim(ask.id, (status, recorded), _expected_settle(ask, response)):
        return _closed(ledger, ask.id)
    try:
        outcome = await _deliver(authority, session_id, ask, response)
    except RuntimeStoreError as exc:
        ledger.release(ask.id)
        if exc.reason in ("approval_not_pending", "stale_generation"):
            return _closed(ledger, ask.id)
        return _store_error(exc)
    if outcome.get("status") != "resolved":
        ledger.release(ask.id)
        return _closed(ledger, ask.id)
    deadline = asyncio.get_running_loop().time() + _SETTLE_WAIT_S
    while (current := ledger.get(ask.id)).status == "pending" and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(_SETTLE_POLL_S)
    return web.json_response(current.as_dict())


def _closed(ledger, ask_id) -> web.Response:
    ask = ledger.get(ask_id)
    return _error(f"Ask is closed: {ask_id}", 409, "ask_closed", ask=ask.as_dict() if ask else None)


async def _handle_answer(adapter, request) -> web.Response:
    """POST /api/sessions/{session_id}/asks/{ask_id}/answer — the body is the answer itself."""
    scope, ask, err = await _owned_ask(adapter, request)
    if err:
        return err
    body, err = await adapter._read_json_body(request)
    if err:
        return err
    if ask.kind == "secret":
        return _error("A secret is entered only in an attached client's masked prompt; dismiss it instead.",
                      400, "answer_with_secret")
    if ask.status != "pending":
        return _closed(scope[2], ask.id)
    response, recorded, problem = _ANSWER_SHAPES[ask.kind](ask, body)
    if problem:
        return _error(problem, 400, "invalid_answer")
    return await _settle(scope, ask, "answered", recorded, response)


async def _handle_dismiss(adapter, request) -> web.Response:
    """POST /api/sessions/{session_id}/asks/{ask_id}/dismiss — close with no answer (an approval is denied)."""
    scope, ask, err = await _owned_ask(adapter, request)
    if err:
        return err
    if ask.status != "pending":
        return _closed(scope[2], ask.id)
    return await _settle(scope, ask, "dismissed", None, _DISMISS_RESPONSES.get(ask.kind))
