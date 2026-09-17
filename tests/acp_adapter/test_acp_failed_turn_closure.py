"""A failed turn must not leave its accepted user row as the durable conversation tail.

#108033 closed this for the gateway (``gateway/run_turn.py::_hmwa_close_failed_turn``,
keyed on ``SessionDB.latest_conversation_role``), but only for turns that come back
through the gateway. ``acp_adapter/server.py::_finish_turn`` never inspects the terminal
result, and the paths that produce these envelopes — the content-policy refusal return,
``_Trunc.end_turn``, the overflow builders, the codex runtime — persist and return
without reaching ``finalize_turn``. Standalone ACP therefore used to persist the user row
and stop; the next prompt appended a second user row, ``repair_message_sequence`` merged
the pair into one user instruction, and the provider was asked to act on the failed
request again.

Closed at the one seam every envelope passes through:
``agent/conversation_loop.py::close_durable_failed_turn``.

These tests drive the real ``HermesACPAgent.prompt()`` path with a real ``AIAgent``, a
real ``SessionDB`` and a loopback fixture provider. Nothing here is mocked at the
transcript layer: every assertion reads rows back out of SQLite, or reads the request
body the provider actually received.

Context overflow is deliberately NOT closed here — see the two overflow tests at the end.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

_FIXTURE_MODEL = "fixture-model"


# ── loopback fixture provider ──────────────────────────────────────────────────


class _FixtureProvider:
    """Scriptable OpenAI-compatible endpoint.

    ``script`` is consumed one entry per request; ``sticky`` (when set) answers every
    request and is what drives the retry-exhaustion classes. An entry is a dict:

    ``{"kind": "finish", "finish_reason": ..., "content": ...}`` — HTTP 200 completion
    ``{"kind": "http", "status": ..., "message": ...}``          — provider error
    ``{"kind": "empty_stream"}``                                  — 200 with no usable SSE
    ``{"kind": "reasoning_only", "reasoning": ...}``              — hidden reasoning, no text
    """

    def __init__(self) -> None:
        self.script: list[dict[str, Any]] = []
        self.sticky: dict[str, Any] | None = None
        self.requests: list[dict[str, Any]] = []
        self.received = threading.Event()   # set when a request arrives
        self.release = threading.Event()    # a {"kind": "hold"} spec waits on this
        provider = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: D102 - silence stderr spam
                pass

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.requests.append(body)
                if provider.sticky is not None:
                    spec = provider.sticky
                elif provider.script:
                    spec = provider.script.pop(0)
                else:
                    spec = {"kind": "finish", "finish_reason": "stop", "content": "fixture reply"}
                provider.received.set()
                if spec["kind"] == "hold":
                    provider.release.wait(timeout=20)
                    spec = {"kind": "finish", "finish_reason": "stop", "content": ""}
                status, mime, payload = self._render(body, spec)
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _render(self, body, spec):
                kind = spec["kind"]
                usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
                if kind == "http":
                    return spec["status"], "application/json", json.dumps({"error": {
                        "message": spec["message"], "type": "invalid_request_error",
                        "code": spec.get("code", "fixture_rejected"),
                    }}).encode()
                if kind == "empty_stream":
                    return 200, "text/event-stream", b"data: [DONE]\n\n"
                delta = (
                    {"role": "assistant", "reasoning_content": spec["reasoning"]}
                    if kind == "reasoning_only"
                    else {"role": "assistant", "content": spec.get("content", "")}
                )
                finish_reason = spec.get("finish_reason", "stop")
                if body.get("stream"):
                    chunks = [
                        {"id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 1,
                         "model": _FIXTURE_MODEL,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                        {"id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 1,
                         "model": _FIXTURE_MODEL,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                         "usage": usage},
                    ]
                    return 200, "text/event-stream", (
                        "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
                    ).encode()
                message = {"role": "assistant"}
                message.update({k: v for k, v in delta.items() if k != "role"})
                return 200, "application/json", json.dumps({
                    "id": "chatcmpl-fixture", "object": "chat.completion", "created": 1,
                    "model": _FIXTURE_MODEL,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": usage,
                }).encode()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def user_rows_of_last_request(self) -> list[Any]:
        return [m["content"] for m in self.requests[-1]["messages"] if m.get("role") == "user"]

    def shutdown(self) -> None:
        self._server.shutdown()


class _RecordingConn:
    """Minimal ACP client: records every ``session_update`` the server emits."""

    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any) -> None:
        self.updates.append(update)

    async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no tool approval should be requested in these fixtures")

    def agent_texts(self) -> list[str]:
        out = []
        for u in self.updates:
            text = getattr(u, "text", None) or getattr(getattr(u, "content", None), "text", None)
            if text and type(u).__name__.lower().startswith("agent"):
                out.append(text)
        return out

    def all_texts(self) -> list[str]:
        out = []
        for u in self.updates:
            text = getattr(u, "text", None) or getattr(getattr(u, "content", None), "text", None)
            if isinstance(text, str):
                out.append(text)
        return out


# ── harness ───────────────────────────────────────────────────────────────────


class _Harness:
    def __init__(self, server, manager, db, db_path, provider, conn):
        self.server, self.manager, self.db = server, manager, db
        self.db_path, self.provider, self.conn = db_path, provider, conn

    def new_session(self, cwd: str) -> str:
        return self.manager.create_session(cwd=cwd).session_id

    def prompt(self, session_id: str, text: str):
        from acp_adapter.server import TextContentBlock

        return asyncio.run(self.server.prompt(
            prompt=[TextContentBlock(type="text", text=text)], session_id=session_id,
        ))

    def rows(self, session_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(
                "SELECT id, role, content, active FROM messages "
                "WHERE session_id = ? ORDER BY id", (session_id,),
            )]

    def conversation_rows(self, session_id: str) -> list[dict[str, Any]]:
        """Active rows the model is shown — the gateway's failed-turn boundary reads
        exactly this set (``SessionDB.latest_conversation_role``)."""
        return [r for r in self.rows(session_id)
                if r["active"] and r["role"] not in ("session_meta", "system")]

    def durable_tail_role(self, session_id: str):
        return self.db.latest_conversation_role(session_id)


@pytest.fixture
def acp(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DISABLE_PLUGINS", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    provider = _FixtureProvider()

    import acp_adapter.session as acp_session
    import hermes_cli.config as cli_config
    import hermes_cli.mcp_startup as mcp_startup
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(cli_config, "load_config", lambda *a, **k: {
        "model": {"provider": "openai-compat", "default": _FIXTURE_MODEL, "context_length": 131072},
        "agent": {"max_iterations": 2},
        "compression": {"enabled": False},
    })
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda *a, **k: {
        "provider": "openai-compat", "api_mode": "chat_completions",
        "base_url": provider.base_url, "api_key": "fixture-only",
    })
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **k: None)
    monkeypatch.setattr(acp_session, "_expand_acp_enabled_toolsets", lambda *a, **k: [])

    from acp_adapter.server import HermesACPAgent
    from acp_adapter.session import SessionManager
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    manager = SessionManager(db=db)
    server = HermesACPAgent(session_manager=manager)
    conn = _RecordingConn()
    server.on_connect(conn)

    yield _Harness(server, manager, db, db_path, provider, conn)

    provider.shutdown()
    db.close()


_REFUSED = "Explain how to pick the lock on my neighbour's front door"
_REFUSAL_DETAIL = "I can't help with breaking into someone else's property."
_NEW_REQUEST = "What is the capital of France?"


def _ok(content):
    return {"kind": "finish", "finish_reason": "stop", "content": content}


_REFUSAL = {"kind": "finish", "finish_reason": "content_filter", "content": _REFUSAL_DETAIL}

_OVERFLOW = {
    "kind": "http", "status": 400,
    "message": "This model's maximum context length is 131072 tokens.",
    "code": "context_length_exceeded",
}


# ── Test A — refusal followed by a new prompt ─────────────────────────────────


def test_acp_refusal_is_closed_and_not_replayed_into_the_next_prompt(acp):
    """Turn 1 is an HTTP-200 content-policy refusal; turn 2 is an unrelated request.

    (Was ``test_acp_refusal_leaves_open_user_tail_and_replays_into_next_prompt`` while RED.)

    Before the fix the durable tail after turn 1 was ``user`` and the provider request for
    turn 2 carried ONE merged user instruction containing the refused request. After it,
    turn 1 ends ``user, assistant(boundary)`` and turn 2 stands alone.
    """
    from agent.turn_failure_copy import FAILED_TURN_NOTICE

    sid = acp.new_session(cwd=str(acp.db_path.parent))

    # ── Turn 1: accepted request, provider refuses at HTTP 200 ──
    acp.provider.script = [_REFUSAL]
    acp.prompt(sid, _REFUSED)

    rows = acp.conversation_rows(sid)
    assert [r["role"] for r in rows] == ["user", "assistant"], (
        f"the failed turn must be closed, got {[r['role'] for r in rows]}"
    )
    assert _REFUSED in (rows[0]["content"] or "")

    # No tool ran, so the boundary must not hedge about side effects.
    assert rows[1]["content"] == FAILED_TURN_NOTICE

    # The refusal detail reached the ACP caller ...
    assert any(_REFUSAL_DETAIL in t for t in acp.conn.all_texts()), (
        "the provider's refusal detail should reach the ACP client"
    )
    # ... but is not in the boundary, nor anywhere in canonical assistant history.
    assert not any(
        r["role"] == "assistant" and _REFUSAL_DETAIL in (r["content"] or "") for r in acp.rows(sid)
    ), "provider refusal detail must never become canonical assistant history"

    assert acp.durable_tail_role(sid) == "assistant"

    # ── Turn 2: unrelated new request ──
    acp.provider.script = [_ok("Paris.")]
    acp.prompt(sid, _NEW_REQUEST)

    sent_users = acp.provider.user_rows_of_last_request()
    merged = f"{_REFUSED}\n\n{_NEW_REQUEST}"
    assert merged not in sent_users, (
        "the failed turn's user row was merged into the new request.\n"
        f"provider received user rows: {sent_users!r}"
    )
    # The refused turn stays in history — closing it is not erasing it — but it is a
    # CLOSED past turn, not part of the live instruction. The new request is its own row,
    # and it is the last one.
    assert sent_users == [_REFUSED, _NEW_REQUEST], sent_users
    assert sent_users[-1] == _NEW_REQUEST

    sent_roles = [m["role"] for m in acp.provider.requests[-1]["messages"] if m["role"] != "system"]
    assert sent_roles == ["user", "assistant", "user"], (
        "the boundary must separate the failed turn from the new one; "
        f"got {sent_roles}"
    )


# ── Test B — durable boundary contract ────────────────────────────────────────


_DETAIL = "fixture provider failure detail 0e7c"

# (id, sticky spec, detail that must never reach canonical assistant history).
# One parameterization instead of one test per class: every entry is a terminal
# failure that leaves the accepted user row durable, so they share one contract.
_OPEN_TAIL_FAILURE_CLASSES = [
    pytest.param(_REFUSAL, _REFUSAL_DETAIL, id="http200-content-policy-refusal"),
    pytest.param({"kind": "http", "status": 400, "message": _DETAIL},
                 _DETAIL, id="nonretryable-provider-400"),
    pytest.param({"kind": "http", "status": 429, "message": _DETAIL},
                 _DETAIL, id="exhausted-rate-limit"),
    pytest.param({"kind": "http", "status": 503, "message": _DETAIL},
                 _DETAIL, id="exhausted-transient-5xx"),
    pytest.param({"kind": "empty_stream"}, None, id="invalid-response-retry-exhaustion"),
    pytest.param({"kind": "finish", "finish_reason": "length", "content": ""},
                 None, id="first-response-truncated"),
]


@pytest.mark.parametrize("spec, detail", _OPEN_TAIL_FAILURE_CLASSES)
def test_acp_terminal_failure_closes_the_durable_turn(acp, spec, detail):
    """Post-fix invariant, expressed without implementing it.

    For a terminal failure where the accepted user row stays durable:
      1. the durable conversation tail must not remain ``user``;
      2. the closing row must be Hermes-authored, not model output;
      3. provider/model failure detail must not appear in canonical assistant content.

    Parameterized across the failure classes that share this contract (section 4 of the
    matrix). Context overflow is deliberately NOT here — see
    ``test_acp_context_overflow_is_not_a_boundary_class``.
    """
    sid = acp.new_session(cwd=str(acp.db_path.parent))
    acp.provider.sticky = spec  # sticky: retry classes must exhaust, not recover
    acp.prompt(sid, _REFUSED)
    acp.provider.sticky = None

    rows = acp.conversation_rows(sid)
    assert rows and rows[0]["role"] == "user", "the accepted user row must stay durable"

    # (3) provider failure detail is never canonical assistant history
    if detail is not None:
        assert not any(
            r["role"] == "assistant" and detail in (r["content"] or "") for r in acp.rows(sid)
        ), "provider/model failure detail must not be persisted as assistant content"

    # (1) the turn is closed
    assert acp.durable_tail_role(sid) != "user", (
        f"durable tail is still an open user row: {[r['role'] for r in rows]}"
    )

    # (2) the boundary is a Hermes-authored assistant row with visible content
    tail = rows[-1]
    assert tail["role"] == "assistant"
    assert (tail["content"] or "").strip(), "the boundary row must have visible content"


# ── Test C — already-closed idempotence ───────────────────────────────────────


def test_acp_failed_turn_after_assistant_text_adds_no_second_boundary(acp):
    """A failure that already left an assistant row as the tail must not be closed twice.

    Turn 1 succeeds (assistant row durable). Turn 2's provider refuses. The accepted
    user row for turn 2 is durable, so the closer fires — but it must append exactly
    one boundary, and a redelivery of the same terminal result must append none.
    """
    sid = acp.new_session(cwd=str(acp.db_path.parent))

    acp.provider.script = [_ok("Hello there.")]
    acp.prompt(sid, "hi")
    assert acp.durable_tail_role(sid) == "assistant"
    baseline = len(acp.conversation_rows(sid))

    acp.provider.script = [_REFUSAL]
    acp.prompt(sid, _REFUSED)

    rows = acp.conversation_rows(sid)
    roles = [r["role"] for r in rows]
    # user + exactly one boundary on top of the healthy turn
    assert roles == ["user", "assistant", "user", "assistant"], roles
    assert len(rows) == baseline + 2, (
        f"a failed turn must add the user row and exactly one boundary; roles={roles}"
    )

    # Idempotence 1: re-running the CORE closer over the now-closed tail is a no-op.
    # Durable state is the authority, so a redelivery of the same terminal envelope
    # cannot stack a second boundary.
    from agent.conversation_loop import close_durable_failed_turn

    before = acp.rows(sid)
    state = acp.manager.get_session(sid)
    redelivered = {"completed": False, "failed": True,
                   "failure_reason": "content_policy_blocked",
                   "messages": list(state.history)}
    close_durable_failed_turn(state.agent, redelivered)
    assert acp.rows(sid) == before, "closing an already-closed turn must be a no-op"

    # Idempotence 2: the gateway writer from #108033 also no-ops once the core has
    # closed the turn — it reads the same durable tail, so no double boundary.
    from gateway.config import GatewayConfig
    from gateway.run_turn import GatewayTurnMixin
    from gateway.session import AsyncSessionStore, SessionStore

    store = SessionStore(sessions_dir=acp.db_path.parent / "sessions", config=GatewayConfig())
    assert store.transcript_tail_role(sid) != "user", (
        "the gateway's guard must see a closed tail after core closure"
    )
    holder = SimpleNamespace(async_session_store=AsyncSessionStore(store))
    asyncio.run(GatewayTurnMixin._hmwa_close_failed_turn(
        holder, sid, GatewayTurnMixin._FAILED_TURN_NOTICE))
    assert acp.rows(sid) == before, (
        "gateway _hmwa_close_failed_turn must not add a second boundary"
    )


# ── Test D — success unaffected ───────────────────────────────────────────────


def test_acp_successful_turn_gets_no_boundary_row(acp):
    """Pin current healthy behaviour so the fix cannot add boundary rows to good turns."""
    sid = acp.new_session(cwd=str(acp.db_path.parent))

    acp.provider.script = [_ok("Paris.")]
    acp.prompt(sid, _NEW_REQUEST)

    rows = acp.conversation_rows(sid)
    assert [r["role"] for r in rows] == ["user", "assistant"], [r["role"] for r in rows]
    assert rows[1]["content"] == "Paris."
    assert acp.durable_tail_role(sid) == "assistant"

    acp.provider.script = [_ok("Berlin.")]
    acp.prompt(sid, "And Germany?")

    rows = acp.conversation_rows(sid)
    assert [r["role"] for r in rows] == ["user", "assistant", "user", "assistant"], \
        [r["role"] for r in rows]
    assert acp.provider.user_rows_of_last_request() == [_NEW_REQUEST, "And Germany?"]


# ── Section 2 — context overflow is a different contract ──────────────────────


def test_acp_context_overflow_is_not_a_boundary_class(acp):
    """Context overflow must NOT be closed by a generic ``failed=True`` boundary.

    The gateway deliberately writes NOTHING on overflow and then resets the session
    (``_hmwa_persist_turn_transcript`` skip + ``_hmwa_compression_exhaustion_reset``),
    because appending to a session that is already too large reproduces the failure
    forever (#1630 / #9893). This test pins that the two classes are distinguishable
    from the terminal result alone, so a closer can exclude overflow.
    """
    from gateway.run_turn import is_context_overflow_failure_result

    overflow = {"failed": True, "compression_exhausted": True,
                "failure_reason": "context_overflow", "messages": []}
    refusal = {"failed": True, "failure_reason": "content_policy_blocked", "messages": []}

    assert is_context_overflow_failure_result(overflow, 0) is True
    assert is_context_overflow_failure_result(refusal, 0) is False


# ── Section 4 — the remaining failure classes, characterized ──────────────────


def test_acp_interrupt_with_no_assistant_text_closes_the_durable_turn(acp):
    """Interrupt with no assistant text and no tool activity.

    Distinct code path from the terminal-failure classes above: an interrupt DOES reach
    ``finalize_turn``, but ``_close_transcript_tail`` gates its append on
    ``not interrupted`` and ``close_interrupted_tool_sequence`` only fires on a ``tool``
    tail — so before the fix a bare interrupt left the accepted user row as the durable
    tail. The core closer sits past ``finalize_turn`` on the same envelope, so it covers
    this class too.
    """
    import threading as _t

    sid = acp.new_session(cwd=str(acp.db_path.parent))
    acp.provider.script = [{"kind": "hold"}]

    done = _t.Event()
    raised: list[BaseException] = []

    def _run():
        try:
            acp.prompt(sid, _REFUSED)
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            raised.append(exc)
        finally:
            done.set()

    worker = _t.Thread(target=_run, daemon=True)
    worker.start()
    assert acp.provider.received.wait(timeout=20), "fixture provider never saw the request"
    asyncio.run(acp.server.cancel(session_id=sid))
    acp.provider.release.set()
    assert done.wait(timeout=30), "prompt did not finish after cancel"

    # Observed on 6005aa1f and recorded here, but NOT the subject of this test: an
    # interrupted turn can carry ``final_response=None``, and ``_finish_turn`` does
    # ``final_response.startswith(...)`` unguarded (acp_adapter/server.py). Separate
    # defect; it does not change the durable shape asserted below.
    assert not raised or isinstance(raised[0], AttributeError), raised

    rows = acp.conversation_rows(sid)
    assert [r["role"] for r in rows] == ["user", "assistant"], [r["role"] for r in rows]
    assert acp.durable_tail_role(sid) != "user", (
        "an interrupted turn whose accepted user row stayed durable must not leave the "
        f"transcript open; roles={[r['role'] for r in rows]}"
    )


def test_acp_hidden_reasoning_only_turn_leaves_a_blank_assistant_tail(acp):
    """Characterization, not an open-user-tail class.

    A hidden-reasoning-only completion already persists an assistant row, so a
    role-keyed closer correctly skips it. It is recorded here because the row is BLANK:
    a different defect shape, out of scope for failed-turn closure, and the reason this
    class must not be folded into the closer's contract.
    """
    sid = acp.new_session(cwd=str(acp.db_path.parent))
    acp.provider.sticky = {"kind": "reasoning_only", "reasoning": _DETAIL}
    acp.prompt(sid, _REFUSED)
    acp.provider.sticky = None

    rows = acp.conversation_rows(sid)
    assert [r["role"] for r in rows] == ["user", "assistant"], [r["role"] for r in rows]
    assert acp.durable_tail_role(sid) == "assistant", (
        "a role-keyed closer must treat this class as already closed"
    )
    assert (rows[-1]["content"] or "") == "", (
        "pinned: the assistant row is blank — a separate concern from failed-turn closure"
    )
    assert _DETAIL not in (rows[-1]["content"] or "")


def test_acp_context_overflow_keeps_the_user_row_durable_and_the_session_live(acp):
    """Section 2: what context overflow actually does on the ACP path.

    GREEN characterization of the three facts the closer's design depends on:
      1. the agent's turn-start user row IS still durable after the overflow failure;
      2. nothing removes, deactivates or replaces it — the durable tail stays ``user``;
      3. the class is distinguishable from the closer's classes at the result level,
         so a closer can exclude it.

    The gateway's answer is NOT a boundary row: it skips every transcript write
    (``_hmwa_persist_turn_transcript``) and then resets the session
    (``_hmwa_compression_exhaustion_reset``), because appending to an already-oversized
    session is the #1630 growth loop. That is why overflow needs a different repair.
    """
    from gateway.run_turn import is_context_overflow_failure_result

    sid = acp.new_session(cwd=str(acp.db_path.parent))
    acp.provider.sticky = _OVERFLOW
    acp.prompt(sid, _REFUSED)
    acp.provider.sticky = None

    rows = acp.conversation_rows(sid)
    # (1) + (2)
    assert [r["role"] for r in rows] == ["user"], [r["role"] for r in rows]
    assert rows[0]["active"] == 1, "the turn-start user row is not deactivated"
    assert _REFUSED in (rows[0]["content"] or "")
    assert acp.durable_tail_role(sid) == "user"

    # (3)
    assert is_context_overflow_failure_result(
        {"failed": True, "error": "context length exceeded"}, 0) is True
    assert is_context_overflow_failure_result(
        {"failed": True, "failure_reason": "content_policy_blocked", "error": "refused"}, 0) is False


@pytest.mark.xfail(strict=True, reason=(
    "CONTEXT_OVERFLOW_REQUIRES_DIFFERENT_REPAIR: ACP has no equivalent of the gateway's "
    "compression-exhaustion session reset, so an overflow-failed session stays "
    "authoritative and replays the failed request. Out of scope for the failed-turn "
    "closer — appending a boundary would grow a session that is already too large "
    "(#1630). Tracked separately; this test must stay xfail after the closer lands."
))
def test_acp_context_overflow_session_should_not_replay_the_failed_request(acp):
    """The overflow defect, expressed as the invariant a session-rotation fix must meet."""
    sid = acp.new_session(cwd=str(acp.db_path.parent))
    acp.provider.sticky = _OVERFLOW
    acp.prompt(sid, _REFUSED)
    acp.provider.sticky = None

    acp.provider.script = [_ok("Paris.")]
    acp.prompt(sid, _NEW_REQUEST)
    assert acp.provider.user_rows_of_last_request() == [_NEW_REQUEST]


# ── Partial effects — the boundary must not claim "not processed" ─────────────


def test_boundary_hedges_when_a_tool_may_already_have_run():
    """Tool evidence in the turn slice selects the partial-effect notice.

    Unit-level on purpose: the selector is the whole policy, and driving a real tool round
    to a terminal failure would test the tool runner, not the boundary. The evidence shapes
    are exactly the two the gateway keys on — a ``tool`` result row, and an assistant row
    carrying ``tool_calls``.
    """
    from agent.turn_failure_copy import (
        FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE, failed_turn_notice,
    )

    no_effects = [{"role": "user", "content": "hi"}]
    assert failed_turn_notice(no_effects) == FAILED_TURN_NOTICE

    tool_result = [
        {"role": "user", "content": "delete the temp files"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "removed 12 files"},
    ]
    assert failed_turn_notice(tool_result) == PARTIAL_FAILED_TURN_NOTICE

    # An assistant row with tool_calls and no result yet still counts: the call may have
    # left effects even though its result never came back.
    unanswered = [
        {"role": "user", "content": "delete the temp files"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
    ]
    assert failed_turn_notice(unanswered) == PARTIAL_FAILED_TURN_NOTICE

    # Evidence from a PREVIOUS turn must not leak into this turn's notice: the slice
    # starts at the last user row.
    previous_turn_only = [
        {"role": "user", "content": "delete the temp files"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "removed 12 files"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "what is the capital of France?"},
    ]
    assert failed_turn_notice(previous_turn_only) == FAILED_TURN_NOTICE


def test_acp_failed_turn_with_tool_evidence_persists_the_partial_notice(acp):
    """End-to-end through the real closer and the real SessionDB.

    The reachable shape, and the only one: tool activity happened in this turn's LIVE
    messages but nothing after the user row became durable — an incremental flush that
    failed, or a rolled-back envelope projection such as
    ``_get_messages_up_to_last_assistant``. When tool rows DO persist, the durable tail
    is ``tool``, the tail gate no-ops, and no boundary is written at all. The gateway's
    ``_hmwa_close_failed_turn`` has exactly the same reachability, since it gates on the
    same durable tail.
    """
    from agent.conversation_loop import close_durable_failed_turn
    from agent.turn_failure_copy import PARTIAL_FAILED_TURN_NOTICE

    sid = acp.new_session(cwd=str(acp.db_path.parent))
    state = acp.manager.get_session(sid)

    # Durable: the accepted user row only (what _persist_turn_start leaves behind).
    user_row = {"role": "user", "content": "delete the temp files"}
    state.agent._flush_messages_to_session_db([user_row])
    assert acp.durable_tail_role(sid) == "user"

    # Live: the same turn also ran a tool, which never reached the DB.
    messages = [user_row,
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "name": "terminal",
                 "content": "removed 12 files"}]

    close_durable_failed_turn(state.agent, {
        "completed": False, "failed": True, "failure_reason": "overloaded",
        "messages": messages,
    })

    rows = acp.conversation_rows(sid)
    assert rows[-1]["role"] == "assistant"
    assert rows[-1]["content"] == PARTIAL_FAILED_TURN_NOTICE, rows[-1]["content"]
    assert "not processed" not in (rows[-1]["content"] or "")
    # The boundary is in the returned messages too, not only in the DB.
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == PARTIAL_FAILED_TURN_NOTICE


# ── Partial-effect anchoring: synthetic user rows must not end the turn ───────
#
# Hermes appends user-ROLE scaffolding inside a single real user turn: verify-on-stop
# and pre_verify nudges (``agent/turn_stop_gates.py::_continue``), kanban stop nudges,
# length-continuation and dropped-tool-call notices, todo snapshots. A "scan back to the
# last role=user row" boundary starts AFTER that scaffolding and so cannot see tool
# evidence that ran earlier in the same turn — it would claim "not processed" for a turn
# that already deleted files. These pin the anchor to the accepted human turn instead.


def _real_user(text):
    return {"role": "user", "content": text}


def _tool_effect_rows():
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "terminal", "arguments": '{"cmd": "rm -rf build"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "removed 412 files"},
    ]


# Real Hermes scaffolding shapes: the flags ``turn_stop_gates._continue`` stamps, and the
# content prefixes ``agent/conversation_compression._SYNTHETIC_USER_PREFIXES`` knows.
_SYNTHETIC_NUDGES = [
    pytest.param({"role": "user", "content": "Please verify your work before stopping.",
                  "_verification_stop_synthetic": True}, id="verify-on-stop"),
    pytest.param({"role": "user", "content": "Run the pre_verify hook.",
                  "_pre_verify_synthetic": True}, id="pre-verify"),
    pytest.param({"role": "user", "content": "A tool call was dropped; retry it.",
                  "_dropped_toolcall_nudge": True}, id="dropped-tool-call"),
    pytest.param({"role": "user", "content": "[System: Your previous response was truncated. Continue.]"},
                 id="length-continuation"),
    pytest.param({"role": "user", "content": "[Your active task list was preserved across context compression]",
                  "_todo_snapshot_synthetic": True}, id="todo-snapshot"),
    pytest.param({"role": "user", "content": "recovered from an empty response",
                  "_empty_recovery_synthetic": True}, id="empty-response-recovery"),
]


@pytest.mark.parametrize("nudge", _SYNTHETIC_NUDGES)
def test_tool_effect_before_a_synthetic_nudge_still_hedges(nudge):
    """The safety case: a tool ran, THEN Hermes appended internal user-role scaffolding.

    Anchoring on the last ``role=user`` row starts at the nudge, sees no tool evidence and
    falsely reports "not processed" for a turn that already had side effects.
    """
    from agent.turn_failure_copy import PARTIAL_FAILED_TURN_NOTICE, failed_turn_notice

    request = "delete the build directory"
    messages = [_real_user(request), *_tool_effect_rows(), nudge]

    assert failed_turn_notice(messages, request) == PARTIAL_FAILED_TURN_NOTICE, (
        "a tool ran before the synthetic nudge; the boundary must not claim the request "
        "was unprocessed"
    )


@pytest.mark.parametrize("nudge", _SYNTHETIC_NUDGES)
def test_synthetic_nudge_without_tool_effects_reports_unprocessed(nudge):
    """The inverse: scaffolding but no side effects — the plain notice is correct."""
    from agent.turn_failure_copy import FAILED_TURN_NOTICE, failed_turn_notice

    request = "what is the capital of France?"
    messages = [_real_user(request), {"role": "assistant", "content": "Thinking..."}, nudge]

    assert failed_turn_notice(messages, request) == FAILED_TURN_NOTICE


def test_effects_from_a_previous_turn_do_not_leak_into_this_turn():
    """The anchor must also not reach BACKWARD past the accepted turn."""
    from agent.turn_failure_copy import FAILED_TURN_NOTICE, failed_turn_notice

    request = "what is the capital of France?"
    messages = [
        _real_user("delete the build directory"), *_tool_effect_rows(),
        {"role": "assistant", "content": "Done."},
        _real_user(request),
    ]
    assert failed_turn_notice(messages, request) == FAILED_TURN_NOTICE


def test_unresolvable_anchor_fails_safe_towards_hedging():
    """When no accepted turn can be identified, hedge rather than claim "not processed"."""
    from agent.turn_failure_copy import PARTIAL_FAILED_TURN_NOTICE, failed_turn_notice

    orphaned = _tool_effect_rows()  # no user row at all
    assert failed_turn_notice(orphaned, "a request that is not in this list") == \
        PARTIAL_FAILED_TURN_NOTICE


# ── The anchor must survive compression / rewrite / reanchor ─────────────────


def test_anchor_survives_compaction_moving_the_turn_and_appending_a_snapshot():
    """The exact shape ``reanchor_current_turn_user_idx`` documents.

    Compression rebuilds ``messages``, so a pre-rewrite index is meaningless, and it may
    append a todo-snapshot user row AFTER the surviving copy of this turn's message. The
    tool evidence sits between the two. A positional index or a last-user-row scan both
    fail here; reanchoring on the turn's own ``user_message`` does not.
    """
    from agent.turn_failure_copy import PARTIAL_FAILED_TURN_NOTICE, failed_turn_notice

    request = "delete the build directory"
    messages = [
        # Compaction handoff scaffolding now occupies the head of the list.
        {"role": "user", "content": "[conversation summary]", "display_kind": "hidden"},
        {"role": "assistant", "content": "Understood."},
        _real_user(request),          # the surviving copy, at a brand-new index
        *_tool_effect_rows(),
        {"role": "user", "content": "[Your active task list was preserved across context compression]",
         "_todo_snapshot_synthetic": True},
    ]
    assert failed_turn_notice(messages, request) == PARTIAL_FAILED_TURN_NOTICE


def test_anchor_falls_back_to_a_human_row_when_merge_into_tail_rewrote_the_content():
    """Merge-into-tail rewrites the row's content, so the exact-match branch cannot fire.

    ``reanchor_current_turn_user_idx`` then falls back to the last *user-originated* row —
    which can be scaffolding. The walk-back keeps the anchor on a human turn, so the tool
    evidence stays inside the slice.
    """
    from agent.turn_failure_copy import PARTIAL_FAILED_TURN_NOTICE, failed_turn_notice

    messages = [
        {"role": "user", "content": "delete the build directory\n\n[merged summary tail]"},
        *_tool_effect_rows(),
        {"role": "user", "content": "Please verify your work before stopping.",
         "_verification_stop_synthetic": True},
    ]
    # The original text no longer appears verbatim anywhere.
    assert failed_turn_notice(messages, "delete the build directory") == PARTIAL_FAILED_TURN_NOTICE


def test_accepted_turn_slice_starts_at_the_human_row_not_the_scaffolding():
    """Direct assertion on the slice itself, so the anchor is pinned independently of copy."""
    from agent.turn_failure_copy import accepted_turn_slice

    request = "delete the build directory"
    messages = [
        {"role": "user", "content": "[conversation summary]", "display_kind": "hidden"},
        _real_user(request),
        *_tool_effect_rows(),
        {"role": "user", "content": "Please verify your work before stopping.",
         "_verification_stop_synthetic": True},
    ]
    sliced = accepted_turn_slice(messages, request)
    assert sliced[0] is messages[1], "slice must begin at the accepted human row"
    assert len(sliced) == 4
