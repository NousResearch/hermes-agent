"""SSE approvals on a real gateway: one turn, one pending approval, two clients.

Refs #145171 (EP-2): rebuilds the manual #134889 "SSE approvals over /v1/chat/completions" check as
a scripted e2e test. A real ``python -m hermes_agent.gateway.run`` serves the API-server platform from the
profile ``.env`` (as a user configures it) in a sandboxed HOME; the model is ``FakeLLMServer``,
which asks for an approval-gated terminal command (``rm -rf`` of a sandbox directory) and answers
once the tool result comes back. ``approvals.mode: manual`` makes the gate the human prompt.

The scenario, run once per module: a streaming client posts ``/v1/chat/completions`` and reads
``approval.request`` off its SSE stream. A second client, on its own connection with the same API
key, polls ``GET /v1/runs/{id}``: it must see the same execution state (``waiting_for_approval``)
and the same pending control (prompt id, generation, command). A reply for a stale generation is
refused and leaves the control pending. The second client then approves; the first stream carries
the model's final answer and ``[DONE]``, the command runs only after the approval (the directory
is still there while pending and after the stale reply, and gone once approved), and a fresh client
sees the run completed with nothing pending.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

from tests.e2e.core.dashboard._helpers import Sandbox, make_sandbox, write_profile_home
from tests.e2e.core.dashboard._issue_helpers import GatewayApiServer
from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux")  # the reaper reads /proc
TURN_TIMEOUT = 90.0
REPLY = "sse-approval-final-" + secrets.token_hex(4)
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def _responder(victim: Path):
    """Ask for the gated command until a tool result comes back, then answer."""
    def respond(record: dict[str, Any]):
        if any(m.get("role") == "tool" for m in record["body"].get("messages") or []):
            return Text(REPLY)
        return ToolCall("terminal", {"command": f"rm -rf {victim}"})
    return respond


class _SseReader:
    """Reads one streaming POST on a background thread, keeping every ``(event, data)`` frame."""

    def __init__(self, client: httpx.Client, url: str, **kw: Any) -> None:
        self.frames: list[tuple[str, Any]] = []
        self.status: int | None = None
        self.error: BaseException | None = None
        self.done = threading.Event()
        self._cond = threading.Condition()
        threading.Thread(target=self._run, args=(client, url, kw), daemon=True, name="sse-reader").start()

    def _run(self, client: httpx.Client, url: str, kw: dict[str, Any]) -> None:
        try:
            with client.stream("POST", url, **kw) as resp:
                self.status = resp.status_code
                event = "message"
                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        event = line[7:].strip()
                    elif line.startswith("data: "):
                        try:
                            data: Any = json.loads(line[6:])
                        except ValueError:
                            data = line[6:]
                        with self._cond:
                            self.frames.append((event, data))
                            self._cond.notify_all()
                        event = "message"
        except Exception as exc:  # reported by wait_for and the cells
            self.error = exc
        finally:
            self.done.set()
            with self._cond:
                self._cond.notify_all()

    def wait_for(self, name: str, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                hit = next((d for e, d in self.frames if e == name), None)
                if hit is not None:
                    return hit
                left = deadline - time.monotonic()
                if left <= 0 or self.done.is_set():
                    raise AssertionError(f"no {name!r} SSE frame: status={self.status} error={self.error!r} "
                                         f"events={[e for e, _ in self.frames][:20]}")
                self._cond.wait(min(left, 0.5))

    def events(self, name: str) -> list[Any]:
        return [d for e, d in self.frames if e == name]

    def text(self) -> str:
        return "".join((d["choices"][0].get("delta") or {}).get("content") or ""
                       for d in self.events("message") if isinstance(d, dict) and d.get("choices"))


def _poll_run(client: httpx.Client, run_id: str, headers: dict[str, str]) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def settled() -> dict[str, Any] | None:
        resp = client.get(f"/v1/runs/{run_id}", headers=headers)
        last.clear()
        last.update(resp.json() if resp.status_code == 200 else {"http_status": resp.status_code})
        return dict(last) if last.get("status") in TERMINAL else None
    try:
        return wait_until(settled, timeout=TURN_TIMEOUT, interval=0.2, what=f"run {run_id} to settle")
    except AssertionError:
        return last  # the cell reports the last state it saw


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("sse-approvals")
    victim = root / "victim-dir"
    victim.mkdir()
    (victim / "keep.txt").write_text("deleted only after approval", encoding="utf-8")
    sb: Sandbox = make_sandbox(root, responder=lambda _p: _responder(victim))
    write_profile_home(sb.profiles["default"], {"approvals": {"mode": "manual", "timeout": 120}})
    srv = sb.profiles["default"].srv
    assert srv is not None
    gw = None
    facts: dict[str, Any] = {}
    try:
        gw = GatewayApiServer(sb, sb.hermes_home, root / "gateway.log")
        auth = {"Authorization": "Bearer " + gw.key}
        timeout = httpx.Timeout(TURN_TIMEOUT)
        with httpx.Client(base_url=gw.base, timeout=timeout, trust_env=False) as streamer, \
                httpx.Client(base_url=gw.base, timeout=timeout, trust_env=False) as viewer:
            stream = _SseReader(streamer, "/v1/chat/completions", headers=auth, json={
                "model": "hermes-agent", "stream": True,
                "messages": [{"role": "user", "content": "clean up the scratch directory"}]})
            event = stream.wait_for("approval.request", TURN_TIMEOUT)
            run_id = event["run_id"]
            facts.update(event=event, victim_while_pending=victim.exists(),
                         viewer_pending=viewer.get(f"/v1/runs/{run_id}", headers=auth).json())
            stale = viewer.post(f"/v1/runs/{run_id}/approval", headers=auth, json={
                "request_id": event["request_id"], "execution_generation": event["execution_generation"] + 1,
                "choice": "once"})
            facts.update(stale=(stale.status_code, stale.json()), victim_after_stale=victim.exists(),
                         viewer_after_stale=viewer.get(f"/v1/runs/{run_id}", headers=auth).json())
            approve = viewer.post(f"/v1/runs/{run_id}/approval", headers=auth, json={
                "request_id": event["request_id"], "execution_generation": event["execution_generation"],
                "choice": "once"})
            facts["approve"] = (approve.status_code, approve.json())
            stream.done.wait(TURN_TIMEOUT)
            facts["stream"] = stream
        # A client that joins after the turn: fresh connection, same key.
        with httpx.Client(base_url=gw.base, timeout=30, trust_env=False) as late:
            facts["late_settled"] = _poll_run(late, run_id, auth)
        facts.update(victim_after=victim.exists(), model_calls=srv.main_requests(), log=gw.log_tail())
        yield facts
    finally:
        if gw is not None:
            gw.stop()
        sb.finish()


def _ctx(s: dict[str, Any]) -> str:
    return f"\n--- gateway log tail ---\n{s['log']}"


def test_stream_client_sees_the_pending_approval(scenario) -> None:
    s = scenario
    ev = s["event"]
    assert s["stream"].status == 200, _ctx(s)
    assert ev["run_id"].startswith("chatcmpl"), ev
    assert ev["command"].startswith("rm -rf ") and ev["command"].endswith("victim-dir"), ev
    assert ev["request_id"] and type(ev["execution_generation"]) is int, ev
    assert set(ev["choices"]) == {"once", "session", "always", "deny"}, ev
    assert s["victim_while_pending"], "the gated command ran before anyone approved it"


def test_second_client_sees_the_same_state_and_pending_control(scenario) -> None:
    s = scenario
    ev, run = s["event"], s["viewer_pending"]
    assert run.get("status") == "waiting_for_approval", run
    assert run.get("run_id") == ev["run_id"] and run.get("execution_generation") == ev["execution_generation"], run
    pending = [p for p in run.get("pending_controls") or [] if p.get("kind") == "approval"]
    assert [(p.get("prompt_id"), p.get("execution_generation"), p.get("command")) for p in pending] == [
        (ev["request_id"], ev["execution_generation"], ev["command"])], run
    assert set(pending[0].get("choices") or []) == set(ev["choices"]), run


def test_stale_generation_reply_is_refused_and_leaves_the_control_pending(scenario) -> None:
    s = scenario
    status, body = s["stale"]
    assert status == 409 and (body.get("error") or {}).get("code") == "stale_generation", s["stale"]
    assert s["victim_after_stale"], "a refused reply still let the command run"
    after = s["viewer_after_stale"]
    assert after.get("status") == "waiting_for_approval", after
    assert [p.get("prompt_id") for p in after.get("pending_controls") or []] == [s["event"]["request_id"]], after


def test_approval_from_the_second_client_resumes_the_streamed_turn(scenario) -> None:
    s = scenario
    status, body = s["approve"]
    assert status == 200 and body.get("status") == "resolved", s["approve"]
    assert body.get("prompt_id") == s["event"]["request_id"], body
    stream = s["stream"]
    assert stream.done.is_set() and stream.error is None, f"stream error={stream.error!r}{_ctx(s)}"
    assert REPLY in stream.text(), f"final answer missing from the stream: {stream.text()!r}{_ctx(s)}"
    assert "[DONE]" in stream.events("message"), "stream never sent [DONE]"
    assert len(stream.events("approval.request")) == 1, stream.events("approval.request")
    assert not s["victim_after"], "the approved command never ran"
    # Exactly two model calls: the tool request, then one answer carrying the tool's result.
    calls = s["model_calls"]
    assert len(calls) == 2, [len(c.get("messages") or []) for c in calls]
    assert sum(m.get("role") == "tool" for m in calls[1].get("messages") or []) == 1, calls[1].get("messages")


def test_late_client_sees_the_run_settled_with_nothing_pending(scenario) -> None:
    run = scenario["late_settled"]
    assert run.get("status") == "completed", run
    # A settled run is answered from the adapter's run record, whose pending approval is ``approval``.
    assert not run.get("pending_controls") and "approval" not in run, run
