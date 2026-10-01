"""Turn-server contract tests with a fake runner (no model, no Hermes agent)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Dict, List

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from litco.assertion import mint_user_assertion, verify_user_assertion
from litco.homes import decode_file_id
from litco.turn_server import TurnContext, TurnOutcome, TurnServer, build_user_message, listen_address

SECRET = "s3cret-host"
MATTER = "matter-123"


class FakeRunner:
    """Scripted runner: emits a few events, optionally blocks until released or interrupted."""

    def __init__(self):
        self.calls: List[TurnContext] = []
        self.block: Dict[str, threading.Event] = {}
        self.running = 0
        self.max_running = 0
        self._lock = threading.Lock()
        self.write_file: str = ""
        self.extra_files: Dict[str, str] = {}      # path under cwd -> text, written during the turn
        self.register: List[dict] = []             # litco_deliver_local calls made during the turn
        self.registered: List[dict] = []

    def run(self, ctx: TurnContext) -> TurnOutcome:
        with self._lock:
            self.calls.append(ctx)
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            ctx.emit("assistant_delta", delta="Looking")
            ctx.emit("tool_started", call={"toolCallId": "c1", "name": "terminal"}, args={"command": "ls"})
            ctx.emit("tool_complete", call={"toolCallId": "c1", "name": "terminal"},
                     result={"status": "ok", "summary": "3 files"})
            gate = self.block.get(ctx.request.session_id)
            if gate is not None:
                while not gate.is_set() and not ctx.interrupted:
                    time.sleep(0.01)
            if ctx.interrupted:
                return TurnOutcome(text="stopped", input_tokens=1, output_tokens=1)
            if self.write_file:
                target = ctx.cwd / "deliverables" / self.write_file
                target.write_text("memo", encoding="utf-8")
            for rel, text in self.extra_files.items():
                target = ctx.cwd / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            if self.register:
                from litco.litkit import tools as T
                from litco.litkit.context import TurnIdentity, turn_scope
                with turn_scope(TurnIdentity(turn_id=ctx.turn_id, matter_id=ctx.request.matter_id, cwd=ctx.cwd)):
                    for args in self.register:
                        self.registered.append(json.loads(T.HANDLERS["litco_deliver_local"](dict(args))))
            ctx.emit("assistant_reset", reason="iteration_committed")
            ctx.emit("assistant_delta", delta="Done.")
            return TurnOutcome(text="Done.", input_tokens=10, output_tokens=4, model_used="fake-model")
        finally:
            with self._lock:
                self.running -= 1


def _headers(**extra):
    return {"X-Host-Secret": SECRET, **extra}


def _body(**overrides):
    body = {"matterId": MATTER, "userId": "u1", "sessionId": "s1", "text": "hello", "attachments": [],
            "channel": "slack", "kind": "channel"}
    body.update(overrides)
    return body


def _parse_sse(raw: str) -> List[dict]:
    events = []
    for block in raw.split("\n\n"):
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "matter"


def _server(runner, home, **kw) -> TurnServer:
    return TurnServer(runner, host_secret=SECRET, matter_id=MATTER, home=home, agent_token="lkm_test", env={}, **kw)


async def _turn(client, body=None, headers=None) -> List[dict]:
    resp = await client.post("/turn", json=body or _body(), headers=headers or _headers())
    assert resp.status == 200, await resp.text()
    return _parse_sse(await resp.text())


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rejects_missing_and_wrong_secret(home):
    async with TestClient(TestServer(_server(FakeRunner(), home).build_app())) as client:
        assert (await client.post("/turn", json=_body())).status == 401
        assert (await client.post("/turn", json=_body(), headers={"X-Host-Secret": "nope"})).status == 401
        assert (await client.post("/interrupt/turn_x", headers={"X-Host-Secret": "nope"})).status == 401
        assert (await client.get("/deliverables/dlv_x")).status == 401


@pytest.mark.asyncio
async def test_unconfigured_secret_refuses_everything(home):
    server = TurnServer(FakeRunner(), host_secret="", matter_id=MATTER, home=home, env={})
    async with TestClient(TestServer(server.build_app())) as client:
        assert (await client.post("/turn", json=_body(), headers={"X-Host-Secret": ""})).status == 401


@pytest.mark.asyncio
async def test_rejects_other_matter_and_bad_body(home):
    async with TestClient(TestServer(_server(FakeRunner(), home).build_app())) as client:
        resp = await client.post("/turn", json=_body(matterId="other"), headers=_headers())
        assert resp.status == 403
        resp = await client.post("/turn", json=_body(channel="fax"), headers=_headers())
        assert resp.status == 400
        resp = await client.post("/turn", json=_body(kind="dm", userId=""), headers=_headers())
        assert resp.status == 400
        resp = await client.post("/turn", data="not json", headers=_headers())
        assert resp.status == 400


def test_assertion_roundtrip_and_failures():
    now = 1_800_000_000_000
    good = mint_user_assertion("u1", MATTER, SECRET, ttl_ms=60_000, now_ms=now)
    assert verify_user_assertion(good, matter_id=MATTER, secret=SECRET, now_ms=lambda: now + 1).user_id == "u1"
    assert verify_user_assertion(good, matter_id=MATTER, secret=SECRET,
                                 now_ms=lambda: now + 61_000).reason == "expired"
    assert verify_user_assertion(good, matter_id="m2", secret=SECRET,
                                 now_ms=lambda: now).reason == "matter_mismatch"
    assert verify_user_assertion(good, matter_id=MATTER, secret="other",
                                 now_ms=lambda: now).reason == "bad_mac"
    tampered = good.replace(".u1.", ".u2.")
    assert verify_user_assertion(tampered, matter_id=MATTER, secret=SECRET, now_ms=lambda: now).reason == "bad_mac"
    assert verify_user_assertion("v1.a.b", matter_id=MATTER, secret=SECRET).reason == "malformed"
    assert verify_user_assertion(None, matter_id=MATTER, secret=SECRET).reason == "missing"


def test_assertion_matches_litkit_wire_format():
    """Same bytes as LitKit's mintUserAssertion: base64url MAC (no padding) over the dotted payload."""
    import base64
    import hashlib
    import hmac

    token = mint_user_assertion("user-1", "matter-1", "k", ttl_ms=300000, now_ms=1700000000000)
    payload = "v1.user-1.matter-1.1700000000000.300000"
    mac = base64.urlsafe_b64encode(hmac.new(b"k", payload.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    assert token == f"{payload}.{mac}"


@pytest.mark.asyncio
async def test_user_assertion_headers(home):
    runner = FakeRunner()
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        good = mint_user_assertion("u1", MATTER, SECRET)
        events = await _turn(client, headers=_headers(**{"X-LitKit-Acting-User": "u1",
                                                         "X-LitKit-User-Assertion": good}))
        assert events[-1]["type"] == "final"
        assert runner.calls[-1].request.acting_user == "u1"

        expired = mint_user_assertion("u1", MATTER, SECRET, ttl_ms=1000, now_ms=int(time.time() * 1000) - 120_000)
        resp = await client.post("/turn", json=_body(), headers=_headers(**{
            "X-LitKit-Acting-User": "u1", "X-LitKit-User-Assertion": expired}))
        assert resp.status == 401
        assert (await resp.json())["error"]["code"] == "assertion_expired"

        resp = await client.post("/turn", json=_body(), headers=_headers(**{
            "X-LitKit-Acting-User": "u2", "X-LitKit-User-Assertion": good}))
        assert resp.status == 401

        resp = await client.post("/turn", json=_body(userId="u9"), headers=_headers(**{
            "X-LitKit-Acting-User": "u1", "X-LitKit-User-Assertion": good}))
        assert resp.status == 403

        resp = await client.post("/turn", json=_body(), headers=_headers(**{"X-LitKit-Acting-User": "u1"}))
        assert resp.status == 401


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sse_frames_in_order(home):
    runner = FakeRunner()
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        resp = await client.post("/turn", json=_body(), headers=_headers())
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        raw = await resp.text()
    events = _parse_sse(raw)
    assert len(events) > 2, raw
    assert [e["type"] for e in events] == [
        "goal_accepted", "assistant_delta", "tool_started", "tool_complete", "assistant_reset",
        "assistant_delta", "final"]
    turn_id = events[0]["turnId"]
    assert all(e["turnId"] == turn_id for e in events)
    assert [e["stepId"] for e in events] == list(range(1, len(events) + 1))
    assert "event: goal_accepted\n" in raw
    assert events[2]["call"] == {"toolCallId": "c1", "name": "terminal"}
    assert events[3]["result"] == {"status": "ok", "summary": "3 files"}
    final = events[-1]
    assert final["text"] == "Done."
    assert final["usage"] == {"inputTokens": 10, "outputTokens": 4}
    assert final["modelUsed"] == "fake-model"
    assert "deliverables" not in final


@pytest.mark.asyncio
async def test_runner_error_is_classified(home):
    class Boom(FakeRunner):
        def run(self, ctx):
            raise RuntimeError("provider exploded")

    async with TestClient(TestServer(_server(Boom(), home).build_app())) as client:
        events = await _turn(client)
    types = [e["type"] for e in events]
    assert types == ["goal_accepted", "error_classified", "final"]
    assert "provider exploded" in events[1]["message"]


# ---------------------------------------------------------------------------
# sessions and concurrency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_sessions_run_concurrently(home):
    runner = FakeRunner()
    runner.block["a"] = threading.Event()
    runner.block["b"] = threading.Event()
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        t1 = asyncio.create_task(_turn(client, _body(sessionId="a")))
        t2 = asyncio.create_task(_turn(client, _body(sessionId="b")))
        for _ in range(300):
            if runner.running == 2:
                break
            await asyncio.sleep(0.01)
        assert runner.running == 2, "different sessionIds must not serialize"
        health = await (await client.get("/health")).json()
        assert health["activeTurns"] == 2
        runner.block["a"].set()
        runner.block["b"].set()
        e1, e2 = await asyncio.gather(t1, t2)
    assert e1[-1]["text"] == "Done." and e2[-1]["text"] == "Done."
    assert runner.max_running == 2


@pytest.mark.asyncio
async def test_same_session_serializes(home):
    runner = FakeRunner()
    runner.block["s1"] = threading.Event()
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        t1 = asyncio.create_task(_turn(client))
        t2 = asyncio.create_task(_turn(client))
        for _ in range(300):
            if runner.running:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        assert runner.running == 1 and len(runner.calls) == 1, "same sessionId must run one turn at a time"
        runner.block["s1"].set()
        await asyncio.gather(t1, t2)
    assert runner.max_running == 1
    assert len(runner.calls) == 2


# ---------------------------------------------------------------------------
# interrupt, budget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interrupt_stops_turn(home):
    runner = FakeRunner()
    runner.block["s1"] = threading.Event()  # never released
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        resp = await client.post("/turn", json=_body(), headers=_headers())
        turn_id = resp.headers["X-Turn-Id"]
        for _ in range(300):
            if runner.running:
                break
            await asyncio.sleep(0.01)
        stop = await client.post(f"/interrupt/{turn_id}", headers=_headers())
        assert stop.status == 202
        events = _parse_sse(await resp.text())
        assert (await client.post("/interrupt/turn_missing", headers=_headers())).status == 404
    types = [e["type"] for e in events]
    assert types[-2:] == ["loop_halted", "final"]
    assert events[-2]["reason"] == "interrupted"
    assert events[-1]["text"] == "stopped"


@pytest.mark.asyncio
async def test_budget_interrupts(home):
    runner = FakeRunner()
    runner.block["s1"] = threading.Event()
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        events = await _turn(client, _body(budgetMs=150))
    assert events[-2]["type"] == "loop_halted"
    assert events[-2]["reason"] == "budget_exhausted"


# ---------------------------------------------------------------------------
# homes, deliverables, attachments, health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_home_dirs_and_deliverables(home):
    runner = FakeRunner()
    runner.write_file = "memo.docx"
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        events = await _turn(client, _body(kind="channel", sessionId="c1"))
        assert runner.calls[-1].cwd == (home / "shared").resolve()
        final = events[-1]
        [item] = final["deliverables"]
        assert item["filename"] == "memo.docx"
        assert item["path"] == "shared/deliverables/memo.docx"
        assert item["mime"].startswith("application/vnd.openxmlformats")
        assert decode_file_id(item["fileId"]) == item["path"]
        got = await client.get(f"/deliverables/{item['fileId']}", headers=_headers())
        assert got.status == 200 and await got.read() == b"memo"

        runner.write_file = "private.txt"
        events = await _turn(client, _body(kind="dm", userId="lawyer/7", sessionId="d1"))
        dm_dir = runner.calls[-1].cwd
        assert dm_dir == (home / "users" / "lawyer_7").resolve()
        assert (dm_dir / "deliverables").is_dir()
        assert [d["path"] for d in events[-1]["deliverables"]] == ["users/lawyer_7/deliverables/private.txt"]

        # an unchanged file is not re-delivered on the next turn
        runner.write_file = ""
        events = await _turn(client, _body(kind="channel", sessionId="c1"))
        assert "deliverables" not in events[-1]

        # ids that escape the deliverables folders are refused
        from litco.homes import encode_file_id
        (home / "secret.txt").write_text("x")
        bad = await client.get(f"/deliverables/{encode_file_id('secret.txt')}", headers=_headers())
        assert bad.status == 404
        bad = await client.get(f"/deliverables/{encode_file_id('shared/../secret.txt')}", headers=_headers())
        assert bad.status == 404


@pytest.mark.asyncio
async def test_attachments_are_fetched_with_agent_token(home):
    seen = {}

    async def serve(request):
        seen["auth"] = request.headers.get("Authorization")
        return web.Response(body=b"%PDF-1.7 test")

    files = web.Application()
    files.router.add_get("/f/{id}", serve)
    runner = FakeRunner()
    async with TestServer(files) as file_server, \
            TestClient(TestServer(_server(runner, home).build_app())) as client:
        url = str(file_server.make_url("/f/1"))
        await _turn(client, _body(attachments=[
            {"fileId": "f1", "mime": "application/pdf", "filename": "complaint.pdf", "url": url},
            {"fileId": "f2", "mime": "text/plain", "filename": "notes.txt"}]))
    ctx = runner.calls[-1]
    assert seen["auth"] == "Bearer lkm_test"
    first, second = ctx.local_attachments
    assert Path(first["path"]).read_bytes() == b"%PDF-1.7 test"
    assert second["path"] is None and second["error"] == "no url supplied"
    message = build_user_message(ctx)
    assert "complaint.pdf" in message and first["path"] in message and "not downloaded" in message


@pytest.mark.asyncio
async def test_health(home):
    async with TestClient(TestServer(_server(FakeRunner(), home).build_app())) as client:
        body = await (await client.get("/health")).json()
    assert body["ok"] is True
    assert body["matterId"] == MATTER
    assert body["activeTurns"] == 0
    assert body["version"]
    assert body["hermesVersion"] and body["hermesVersion"] != "0.0.0"
    assert body["uptimeSeconds"] >= 0


@pytest.mark.asyncio
async def test_health_counts_the_profiles_runnable_cron_jobs(home, tmp_path, monkeypatch):
    """Through the cron package's own store, in whichever profile HERMES_HOME names (A, B, A)."""
    from cron.jobs import create_job, pause_job

    profile_a, profile_b = tmp_path / "hermes-a", tmp_path / "hermes-b"
    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    create_job(prompt="weekly docket check", schedule="every 1h", name="docket")
    paused = create_job(prompt="paused digest", schedule="every 1h", name="digest")
    pause_job(paused["id"])
    monkeypatch.setenv("HERMES_HOME", str(profile_b))
    for n in range(3):
        create_job(prompt=f"b job {n}", schedule="every 2h", name=f"b{n}")

    async def cron_jobs_in(profile):
        monkeypatch.setenv("HERMES_HOME", str(profile))
        async with TestClient(TestServer(_server(FakeRunner(), home).build_app())) as client:
            return (await (await client.get("/health")).json()).get("cronJobs")

    assert await cron_jobs_in(profile_a) == 1      # the paused job does not count
    assert await cron_jobs_in(profile_b) == 3
    assert await cron_jobs_in(profile_a) == 1
    assert await cron_jobs_in(tmp_path / "hermes-empty") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("counter", ["raises", "slow", "not_a_count"])
async def test_health_omits_cron_jobs_when_the_count_is_unknown(home, monkeypatch, counter):
    import litco.turn_server as ts

    def raises():
        raise RuntimeError("Cron database corrupted and unrepairable")

    def slow():
        time.sleep(0.5)
        return 0

    monkeypatch.setattr(ts, "CRON_COUNT_TIMEOUT_SECONDS", 0.05)
    fn = {"raises": raises, "slow": slow, "not_a_count": lambda: None}[counter]
    async with TestClient(TestServer(_server(FakeRunner(), home, cron_counter=fn).build_app())) as client:
        resp = await client.get("/health")
        body = await resp.json()
    assert resp.status == 200 and body["ok"] is True
    assert "cronJobs" not in body


@pytest.mark.asyncio
async def test_scratch_files_do_not_ship(home):
    runner = FakeRunner()
    runner.extra_files = {
        "deliverables/memo.docx": "memo", "deliverables/chart.png": "png", "deliverables/notes.md": "md",
        "deliverables/memo.spec.json": "{}", "deliverables/data.json": "{}", "deliverables/build.tmp": "x",
        "deliverables/~$memo.docx": "lock", "deliverables/.draft.md": "x", "deliverables/.cache/a.pdf": "x",
        "deliverables/script.py": "print()", "work/outline.md": "outside deliverables"}
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        events = await _turn(client, _body(sessionId="c1"))
    names = sorted(d["filename"] for d in events[-1]["deliverables"])
    assert names == ["chart.png", "memo.docx", "notes.md"]
    assert all("deliverableClass" not in d for d in events[-1]["deliverables"])


@pytest.mark.asyncio
async def test_registered_files_ship_with_class(home):
    runner = FakeRunner()
    runner.extra_files = {"work/table.json": "{\"rows\": 1}", "deliverables/letter.docx": "letter",
                          "deliverables/letter.spec.json": "{}"}
    runner.register = [
        {"path": "work/table.json", "name": "privilege-log.json"},                 # copied into deliverables/
        {"path": "deliverables/letter.docx", "deliverableClass": "letter"},
        {"path": "deliverables/not-written-yet.docx"}]
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        events = await _turn(client, _body(kind="dm", userId="u9", sessionId="d9"))
        items = events[-1]["deliverables"]
        assert [(d["filename"], d.get("deliverableClass")) for d in items] == [
            ("privilege-log.json", None), ("letter.docx", "letter")]
        assert items[0]["path"] == "users/u9/deliverables/privilege-log.json"
        got = await client.get(f"/deliverables/{items[0]['fileId']}", headers=_headers())
        assert got.status == 200 and await got.read() == b'{"rows": 1}'
        missing = runner.registered[2]
        assert missing["file_missing"] is True and "does not exist" in missing["error"]
        assert "Write the file first" in missing["error"]

        # registrations belong to one turn: the next turn delivers nothing new
        runner.extra_files, runner.register = {}, []
        events = await _turn(client, _body(kind="dm", userId="u9", sessionId="d9"))
        assert "deliverables" not in events[-1]


def test_deliver_local_needs_a_turn(tmp_path, monkeypatch):
    from litco.litkit import tools as T
    monkeypatch.setenv("LITCO_MATTER_HOME", str(tmp_path))
    out = json.loads(T.HANDLERS["litco_deliver_local"]({"path": "x.md"}))
    assert "only inside a turn" in out["error"]


# ---------------------------------------------------------------------------
# drain (FIRM_AGENT_HOST 3.6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drain_refuses_new_turns_while_running_turns_finish(home):
    runner = FakeRunner()
    gate = runner.block["s1"] = threading.Event()
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        running = await client.post("/turn", json=_body(), headers=_headers())
        for _ in range(300):
            if runner.running:
                break
            await asyncio.sleep(0.01)
        # a turn queued behind the same session's lock is already accepted, so it runs too
        queued = await client.post("/turn", json=_body(text="second"), headers=_headers())
        assert queued.status == 200

        assert (await client.post("/drain")).status == 401
        assert (await client.post("/drain", headers={"X-Host-Secret": "nope"})).status == 401
        drained = await client.post("/drain", headers=_headers())
        assert drained.status == 200
        assert await drained.json() == {"ok": True, "draining": True, "activeTurns": 2}
        assert (await client.post("/drain", headers=_headers())).status == 200  # idempotent

        health = await (await client.get("/health")).json()
        assert (health["draining"], health["state"], health["activeTurns"]) == (True, "draining", 2)
        assert health["ok"] is True  # the drain script treats a failing /health as nothing to drain

        refused = await client.post("/turn", json=_body(sessionId="s2"), headers=_headers())
        assert refused.status == 503
        assert (await refused.json())["error"]["code"] == "draining"
        # the host secret is still checked first
        assert (await client.post("/turn", json=_body(sessionId="s2"))).status == 401

        gate.set()
        first, second = _parse_sse(await running.text()), _parse_sse(await queued.text())
        assert first[-1]["type"] == "final" and first[-1]["text"] == "Done."
        assert second[-1]["type"] == "final" and second[-1]["text"] == "Done."
        health = await (await client.get("/health")).json()
        assert (health["draining"], health["activeTurns"]) == (True, 0)
        assert len(runner.calls) == 2

        resumed = await client.delete("/drain", headers=_headers())
        assert await resumed.json() == {"ok": True, "draining": False, "activeTurns": 0}
        assert (await (await client.get("/health")).json())["state"] == "ready"
        events = await _turn(client, _body(sessionId="s3"))
        assert events[-1]["type"] == "final"


@pytest.mark.asyncio
async def test_interrupt_still_reaches_a_turn_during_drain(home):
    runner = FakeRunner()
    runner.block["s1"] = threading.Event()  # never released
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        resp = await client.post("/turn", json=_body(), headers=_headers())
        turn_id = resp.headers["X-Turn-Id"]
        for _ in range(300):
            if runner.running:
                break
            await asyncio.sleep(0.01)
        await client.post("/drain", headers=_headers())
        assert (await client.post(f"/interrupt/{turn_id}", headers=_headers())).status == 202
        events = _parse_sse(await resp.text())
        assert events[-2]["reason"] == "interrupted"
        assert (await (await client.get("/health")).json())["activeTurns"] == 0


# ---------------------------------------------------------------------------
# slots (FIRM_AGENT_HOST 3.3) and the new turn fields
# ---------------------------------------------------------------------------

def test_listen_address_slot_and_legacy():
    assert listen_address({}) == ("127.0.0.1", 8765)
    assert listen_address({"LITCO_TURN_HOST": "0.0.0.0", "LITCO_TURN_PORT": "9000"}) == ("0.0.0.0", 9000)
    assert listen_address({}, default_host="10.0.0.1", default_port=9100) == ("10.0.0.1", 9100)
    # a slot's supervisor-assigned port wins, and it listens for the tailnet unless told otherwise
    slot = {"LITCO_SLOT_PORT": "8803", "LITCO_TURN_PORT": "8765", "LITCO_MATTER_ID": MATTER}
    assert listen_address(slot) == ("0.0.0.0", 8803)
    assert listen_address({**slot, "LITCO_TURN_HOST": "100.64.0.7"}) == ("100.64.0.7", 8803)
    with pytest.raises(ValueError, match="LITCO_MATTER_ID"):
        listen_address({"LITCO_SLOT_PORT": "8803"})
    for bad in ("0", "70000", "88a"):
        with pytest.raises(ValueError, match="LITCO_SLOT_PORT"):
            listen_address({"LITCO_SLOT_PORT": bad, "LITCO_MATTER_ID": MATTER})
    with pytest.raises(ValueError, match="LITCO_TURN_PORT"):
        listen_address({"LITCO_TURN_PORT": "x"})


@pytest.mark.asyncio
async def test_slot_reads_its_matter_from_env(home):
    env = {"LITCO_HOST_SECRET": SECRET, "LITCO_MATTER_ID": MATTER, "LITCO_SLOT_PORT": "8801",
           "LITCO_MATTER_HOME": str(home)}
    server = TurnServer(FakeRunner(), env=env)
    async with TestClient(TestServer(server.build_app())) as client:
        assert (await (await client.get("/health")).json())["matterId"] == MATTER
        resp = await client.post("/turn", json=_body(matterId="matter-other"), headers=_headers())
        assert resp.status == 403 and (await resp.json())["error"]["code"] == "matter_mismatch"
        assert (await _turn(client))[-1]["type"] == "final"


@pytest.mark.asyncio
async def test_turn_grant_and_shared_memory_are_parsed(home):
    runner = FakeRunner()
    async with TestClient(TestServer(_server(runner, home).build_app())) as client:
        shared = {"firm": ["Cite the record as (Ex. N at p).", {"id": "x", "kind": "convention",
                                                                "content": "  Bluebook,\n  not ALWD.  "}, 7, ""],
                  "person": [{"content": "Short memos."}]}
        await _turn(client, _body(kind="dm", turnGrant="g1.payload.mac", sharedMemory=shared))
        req = runner.calls[-1].request
        assert req.turn_grant == "g1.payload.mac"
        assert req.shared_memory.firm == ("Cite the record as (Ex. N at p).", "Bluebook, not ALWD.")
        assert req.shared_memory.person == ("Short memos.",)
        assert "g1.payload.mac" not in repr(req)

        # malformed optional fields are dropped, not refused
        await _turn(client, _body(turnGrant=["x"], sharedMemory="firm"))
        req = runner.calls[-1].request
        assert req.turn_grant is None and req.shared_memory is None
        await _turn(client, _body(turnGrant="x" * 5000, sharedMemory={"firm": [f"rule {i}" for i in range(40)]}))
        req = runner.calls[-1].request
        assert req.turn_grant is None and len(req.shared_memory.firm) == 12
