"""RED regression: standalone ACP leaves a failed turn's user row as the durable tail.

#108033 closed this for the gateway (``gateway/run_turn.py::_hmwa_close_failed_turn``,
keyed on ``SessionDB.latest_conversation_role``). The ACP adapter has no equivalent:
``acp_adapter/server.py::_finish_turn`` never inspects the terminal result's ``failed``
flag, so a turn that ends in a terminal provider failure persists its accepted user row
and stops. The next prompt appends a second user row, ``repair_message_sequence`` merges
the pair into one user instruction, and the provider is asked to act on the refused
request again.

These tests drive the real ``HermesACPAgent.prompt()`` path with a real ``AIAgent``, a
real ``SessionDB`` and a loopback fixture provider. Nothing here is mocked at the
transcript layer: every assertion reads rows back out of SQLite, or reads the request
body the provider actually received.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def test_acp_refusal_leaves_open_user_tail_and_replays_into_next_prompt(acp):
    """Turn 1 is an HTTP-200 content-policy refusal; turn 2 is an unrelated request.

    RED on current main: the durable tail after turn 1 is ``user``, and the provider
    request for turn 2 carries ONE merged user instruction containing the refused
    request. The next provider must see only the new request as the live instruction.
    """
    sid = acp.new_session(cwd=str(acp.db_path.parent))

    # ── Turn 1: accepted request, provider refuses at HTTP 200 ──
    acp.provider.script = [_REFUSAL]
    acp.prompt(sid, _REFUSED)

    rows = acp.conversation_rows(sid)
    assert [r["role"] for r in rows] == ["user"], (
        f"expected only the accepted user row to be durable, got {[r['role'] for r in rows]}"
    )
    assert _REFUSED in (rows[0]["content"] or "")

    # The refusal detail reached the ACP caller ...
    assert any(_REFUSAL_DETAIL in t for t in acp.conn.all_texts()), (
        "the provider's refusal detail should reach the ACP client"
    )
    # ... but was NOT persisted as ordinary assistant content.
    assert not any(
        r["role"] == "assistant" and _REFUSAL_DETAIL in (r["content"] or "") for r in acp.rows(sid)
    ), "provider refusal detail must never become canonical assistant history"

    # The defect, stated as durable state.
    assert acp.durable_tail_role(sid) == "user", (
        "precondition for this RED test: current main leaves the failed turn open"
    )

    # ── Turn 2: unrelated new request ──
    acp.provider.script = [_ok("Paris.")]
    acp.prompt(sid, _NEW_REQUEST)

    sent_users = acp.provider.user_rows_of_last_request()
    merged = f"{_REFUSED}\n\n{_NEW_REQUEST}"
    assert merged not in sent_users, (
        "RED: the failed turn's user row was merged into the new request.\n"
        f"provider received user rows: {sent_users!r}"
    )
    assert sent_users == [_NEW_REQUEST], (
        "the next provider request must carry only the new request as the live user "
        f"instruction; got {sent_users!r}"
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

    # Idempotence: re-running the closer over the now-closed tail is a no-op.
    before = acp.rows(sid)
    from acp_adapter import server as acp_server

    closer = getattr(acp_server.HermesACPAgent, "_close_failed_turn", None)
    assert closer is not None, (
        "no ACP failed-turn closer exists yet; this assertion pins the idempotence "
        "contract the production helper must satisfy"
    )
    asyncio.run(closer(acp.server, sid, "boundary"))
    assert acp.rows(sid) == before, "closing an already-closed turn must be a no-op"


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


def test_acp_interrupt_with_no_assistant_text_leaves_an_open_user_tail(acp):
    """Interrupt with no assistant text and no tool activity.

    Distinct code path from the terminal-failure classes above: an interrupt reaches
    ``finalize_turn``, but ``_close_transcript_tail`` gates its append on
    ``not interrupted`` and ``close_interrupted_tool_sequence`` only fires on a ``tool``
    tail — so a bare interrupt leaves the accepted user row as the durable tail.

    RED on current main.
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
    assert [r["role"] for r in rows] == ["user"], [r["role"] for r in rows]
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
