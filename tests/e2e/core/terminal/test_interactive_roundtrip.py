"""C20 — interactive prompts (clarify / dangerous-command approval) round-trip over the REAL backend.

Class: a tool asks the human something (clarify question, approval for a dangerous command) and the
prompt never shows, shows in the wrong place, is lost on a session switch or on the other window,
the answer never reaches the tool, the approval outcome is ignored (denied command runs / approved
command blocked), or a prompt stays pending after the turn.

Harness: one real ``hermes serve`` per module (``approvals.mode: manual`` — the shipped default,
NOT yolo), Desktop-shaped WebSocket clients that advertise ``client.capabilities
{server_requests: true}``, and a fake provider that issues ``clarify`` / ``terminal(rm -rf <victim>)``
tool calls and records the tool result the agent sends back. The client observes the server→client
``clarify`` / ``approval`` request and answers it through the production paths: a JSON-RPC response
frame (Desktop's channel) or the ``approval.respond`` RPC.

Matrix: {clarify, approval-deny, approval-approve} × {answer at once, answer after switching to
another session and back (request re-delivered via ``session.resume.open_requests``), answer from a
second connection, answer after an abrupt ws drop + reconnect}.
Asserts: prompt visible in bounded time; the answer reaches the tool (next provider request's tool
result carries the clarify answer / the approval outcome; deny ⇒ victim still exists, approve ⇒ it
is gone); nothing is left pending; the turn completes with the model's final reply.
"""

from __future__ import annotations

import json
import re
import sys
import threading

import pytest

from tests.e2e.core._pending_fixes import Gap, expect_gaps
from tests.e2e.core.terminal._gateway_client import Backend, WSClient, etype, poll_until
from tests.fakes.fake_llm_provider import Text, ToolCall

# Not marked ``integration`` (that marker means external services and is deselected by addopts): the backend is
# loopback-only. ``tests/e2e`` is outside default discovery; run with ``scripts/run_tests.sh --include-integration``.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group lifecycle for the spawned backend")

TAG_RE = re.compile(r"\b(CLARIFY|RMRF)-([a-z0-9_]+)\b")
PROMPT_TIMEOUT = 90.0


class PromptLeftPending(AssertionError):
    """The turn finished but a clarify/approval request is still in ``open_requests``."""


class Script:
    """Tool call for the newest user tag; after the tool ran, record its result and reply ``DONE <tag>``."""

    def __init__(self) -> None:
        self.victims: dict[str, str] = {}
        self.tool_results: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def __call__(self, record: dict):
        messages = record["body"].get("messages") or []
        users = [m for m in messages if m.get("role") == "user"]
        text = str((users[-1] if users else {}).get("content") or "")
        match = TAG_RE.search(text)
        if not match:
            return Text("no tag")
        kind, tag = match.groups()
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            with self._lock:
                self.tool_results.setdefault(tag, []).append(str(last.get("content") or ""))
            return Text(f"DONE {tag}")
        if kind == "CLARIFY":
            return ToolCall("clarify", {"question": f"Which colour for {tag}?", "choices": ["red", "blue"]})
        return ToolCall("terminal", {"command": f"rm -rf {self.victims[tag]}"})


@pytest.fixture(scope="module")
def script() -> Script:
    return Script()


@pytest.fixture(scope="module")
def backend(tmp_path_factory, script):
    root = tmp_path_factory.mktemp("c20")
    be = Backend(root, script, extra_config=(
        "approvals:\n  mode: manual\n  timeout: 120\n"
        "clarify:\n  timeout: 120\n"
        "terminal:\n  backend: local\n"))
    be.start()
    try:
        yield be
    finally:
        be.stop()


def _open_requests(result: dict, method: str) -> list[dict]:
    return [r for r in (result.get("open_requests") or []) if r.get("method") == method]


def _pending(conn: WSClient, sid: str) -> list:
    return conn.call("approval.pending", session_id=sid).get("approvals") or []


def _drive(backend: Backend, script: Script, kind: str, how: str, tag: str) -> None:
    victim = backend.root / f"victim-{tag}"
    victim.mkdir()
    (victim / "keep.txt").write_text("precious")
    script.victims[tag] = str(victim)
    method = "clarify" if kind == "clarify" else "approval"

    a = backend.connect(f"A-{tag}")
    created = a.call("session.create", cols=100, source="desktop")
    sid, key = created["session_id"], created["stored_session_id"]
    word = "CLARIFY" if kind == "clarify" else "RMRF"
    submitted = a.call("prompt.submit", session_id=sid, text=f"{word}-{tag} please")
    assert submitted.get("status") == "streaming", submitted

    start = len(a.snapshot())
    request = a.wait_for(lambda f: f.get("method") == method and (f.get("params") or {}).get("session_id") == sid,
                         timeout=PROMPT_TIMEOUT, start=start, what=f"{method} request for {tag}")
    rid = request["id"]
    req_idx = next(i for i, f in enumerate(a.snapshot()) if f is request or f.get("id") == rid)
    assert isinstance(rid, str) and rid.startswith("srq-"), request
    if kind == "clarify":
        assert request["params"].get("question") == f"Which colour for {tag}?", request["params"]
    else:
        assert str(victim) in str(request["params"].get("command")), request["params"]
        assert request["params"].get("request_id"), request["params"]
        assert [p.get("request_id") for p in _pending(a, sid)] == [request["params"]["request_id"]]
    assert victim.exists(), "dangerous command ran before anyone approved it"

    responder: WSClient = a
    if how == "switch":
        other = a.call("session.create", cols=100, source="desktop")
        assert not a.events(other["session_id"], "message.complete")
        back = a.call("session.resume", session_id=key, cols=100)
        assert back["session_id"] == sid, f"switching back re-minted the waiting runtime: {back['session_id']} != {sid}"
        assert [r["id"] for r in _open_requests(back, method)] == [rid], (
            f"switching back lost the pending {method}: {back.get('open_requests')}")
    elif how == "second_conn":
        responder = backend.connect(f"B-{tag}")
        joined = responder.call("session.resume", session_id=key, cols=100)
        assert joined["session_id"] == sid
        assert [r["id"] for r in _open_requests(joined, method)] == [rid], (
            f"second window does not see the pending {method}: {joined.get('open_requests')}")
    elif how == "reconnect":
        a.drop()
        responder = backend.connect(f"A2-{tag}")
        resumed = poll_until(lambda: _resume_ok(responder, key), timeout=PROMPT_TIMEOUT, what="resume after drop")
        assert resumed["session_id"] == sid, "reconnect within grace re-minted the waiting runtime"
        assert [r["id"] for r in _open_requests(resumed, method)] == [rid], (
            f"reconnected client lost the pending {method}: {resumed.get('open_requests')}")

    answer = f"blue-{tag}"
    if kind == "clarify":
        responder.respond(rid, {"answer": answer})
    elif how == "rpc":
        choice = "deny" if kind == "approval_deny" else "once"
        resolved = responder.call("approval.respond", session_id=sid, choice=choice,
                                  request_id=request["params"]["request_id"])
        assert resolved.get("resolved"), resolved
        poll_until(lambda: _pending(responder, sid) == [], timeout=30,
                   what=f"approval.respond to actually settle the pending approval ({tag})")
    else:
        responder.respond(rid, {"choice": "deny" if kind == "approval_deny" else "once"})

    watcher = responder
    done = watcher.wait_for(lambda f: etype(f) == "message.complete" and f["params"].get("session_id") == sid,
                            timeout=PROMPT_TIMEOUT, what=f"turn completion for {tag}")
    assert f"DONE {tag}" in str((done["params"].get("payload") or {}).get("text")), done["params"]

    results = script.tool_results.get(tag) or []
    assert len(results) == 1, f"expected exactly one tool result for {tag}, got {results}"
    result = results[0]
    if kind == "clarify":
        assert answer in result, f"clarify answer never reached the tool: {result[:300]}"
    elif kind == "approval_deny":
        assert victim.exists() and (victim / "keep.txt").exists(), "denied command ran anyway"
        assert "denied by user" in result, f"tool result does not report the denial: {result[:300]}"
    else:
        assert not victim.exists(), f"approved command did not run: {result[:300]}"
        payload = json.loads(result)
        assert payload.get("exit_code") == 0 and "approved by the user" in str(payload.get("approval")), payload

    if responder is not a and not a.dropped:
        # The window that did NOT answer must get the signal that settles its card (Desktop tears the clarify
        # card down on tool.complete / request.cancel and reconciles approvals via approval.pending).
        a.wait_for(lambda f: (f.get("params") or {}).get("session_id") == sid and (
            etype(f) == "tool.complete" or (etype(f) == "request.cancel"
                                            and (f["params"].get("payload") or {}).get("id") == rid)),
            timeout=PROMPT_TIMEOUT, start=req_idx + 1, what=f"card-settling signal on the other window ({tag})")

    # Nothing left waiting on a human, from any surface.
    final = poll_until(lambda: _resume_ok(responder, key), timeout=PROMPT_TIMEOUT, what="final resume")
    if final.get("open_requests"):
        raise PromptLeftPending(f"prompt left pending after the turn: {final.get('open_requests')}")
    assert not final.get("pending_approval"), final.get("pending_approval")
    assert _pending(responder, sid) == []
    for conn in (a, responder):
        if not conn.dropped:
            assert not [f for f in conn.server_requests() if f["params"].get("session_id") == sid
                        and f["id"] != rid], "a second prompt was raised for one tool call"


def _resume_ok(conn: WSClient, key: str) -> dict | None:
    frame = conn.request("session.resume", {"session_id": key, "cols": 100})
    if "error" in frame and frame["error"].get("code") == 4009:
        return None
    assert "error" not in frame, frame
    return frame["result"]


CASES = [
    ("clarify", "immediate"), ("clarify", "switch"), ("clarify", "second_conn"), ("clarify", "reconnect"),
    ("approval_deny", "immediate"), ("approval_deny", "second_conn"), ("approval_deny", "rpc"),
    ("approval_approve", "immediate"), ("approval_approve", "switch"), ("approval_approve", "reconnect"),
]


# Applied only while the probe reproduces it (see _pending_fixes). A race in this cell, so not strict: the RPC
# answer can end the approval wait before the gateway attaches the hook that withdraws the sent request, which
# then stays in ``open_requests`` (red in about 1 in 4 loaded runs). Only ``PromptLeftPending`` is excused.
GAPS = {("approval_deny", "rpc"): Gap(
    120374, "#120374: an approval answered by RPC before the settle hook attaches stays in open_requests",
    raises=PromptLeftPending, strict=False)}


@pytest.mark.parametrize(("kind", "how"), [pytest.param(k, h, id=f"{k}-{h}") for k, h in CASES])
def test_interactive_roundtrip(backend: Backend, script: Script, kind: str, how: str,
                               request: pytest.FixtureRequest) -> None:
    if gap := GAPS.get((kind, how)):
        expect_gaps(request, gap)
    tag = f"{kind}_{how}".lower()
    try:
        _drive(backend, script, kind, how, tag)
    except AssertionError as exc:
        # Keep the class: the gap's xfail excuses only PromptLeftPending.
        raise type(exc)(f"{kind}/{how}: {exc}\n{backend.logs(25)}") from None
