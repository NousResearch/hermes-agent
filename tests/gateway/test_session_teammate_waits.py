"""Behaviour contract of the teammate-wait strip's reader (``bot_delivery.list``).

Receipts are the real on-disk shape the canonical owner writes (``gateway/session_bot.py``) under
``<target>/runtime/bot_live_delivery``; the owner join runs against a real ``PendingControls``.
"""
import asyncio
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_controls import AuthorityConnection
from gateway.session_pending_controls import PendingControls
from gateway.session_teammate_waits import bot_delivery_list
from tools import bot_live_delivery
from tools.bot_delivery_status import FAILED_VISIBLE_SECONDS, list_sender_deliveries
from tools.bot_failure_reasons import RUNTIME_OFFLINE
from tools.bot_mode_dm import message_agent_tool


def _profile(root: Path, name: str) -> Path:
    home = root if name == "default" else root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    return home


def _receipt(home: Path, key: str, *, sender: str, status: str, age: float = 0.0, **extra) -> Path:
    root = home / "runtime" / "bot_live_delivery"
    root.mkdir(parents=True, exist_ok=True)
    record = {"delivery_id": key, "profile_home": str(home), "session_id": "chat-1", "principal_id": "p",
              "message": f"Message from 🤖 {sender.title()} (@{sender}): please review the draft",
              "status": status, "admission_id": "adm-" + key,
              "author": {"id": f"bot:{sender}", "name": sender, "is_bot": True}, **extra}
    path = root / f"{key}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("warm_cache", [False, True], ids=["cold", "warm"])
async def test_a_sender_cancel_drops_off_the_gateway_list(tmp_path, monkeypatch, warm_cache):
    root = _profile(tmp_path / "hermes", "default").resolve()
    scribe = _profile(root, "scribe")
    (scribe / "profile.yaml").write_text("ui_meta: {hermes-bots: {}}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    store = SessionStore(root / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, adapters={}, _draining=False)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    cancelled, failed, ambiguous, expired, offline = (digit * 32 for digit in "12345")
    loop = asyncio.get_running_loop()

    async def rpc(connection, method, params):
        frame = await connection.dispatch({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        assert "result" in frame, frame
        return frame["result"]

    def connection(authority):
        return AuthorityConnection(authority, object(), {
            "user_id": "uid:1000", "provider": "local", "profile_id": authority.profile_id,
            "instance_id": authority.instance_id,
            "capabilities": ["session:create", "session:read", "session:submit", "session:control"],
        })

    try:
        owner = await initialize_session_authority(runner, profile_id=str(root), instance_id="target")
        # Hold only execution so the real admission stays queued until the cancel RPC commits.
        monkeypatch.setattr(owner, "_schedule_admitted", lambda *_: None)
        target = connection(owner)
        sender = connection(SimpleNamespace(profile_id=str(scribe), instance_id="sender", events={}, runner=runner))
        delivery = await rpc(target, "bot_relay.deliver", {
            "id": cancelled, "profile": "default", "message": "please review the draft",
            "author": {"id": "bot:scribe", "name": "scribe", "is_bot": True},
        })
        assert delivery["status"] == "queued"
        _receipt(root, failed, sender="scribe", status="failed", reason="rate_limit", error="429")
        _receipt(root, ambiguous, sender="scribe", status="ambiguous")
        _receipt(root, expired, sender="scribe", status="failed", age=FAILED_VISIBLE_SECONDS + 60)
        _receipt(root, offline, sender="scribe", status="queued", admission_id=None)
        offline_error = "Live Bot Chat owner is no longer available"
        legacy = await asyncio.to_thread(bot_live_delivery.cancel_queued_delivery, root, offline,
                                         error=offline_error, reason=RUNTIME_OFFLINE)
        assert legacy["status"] == "cancelled"
        if warm_cache:
            before = await rpc(sender, "bot_delivery.list", {})
            assert {row["delivery_id"]: row["state"] for row in before["deliveries"]}[cancelled] == "queued"

        def route_cancel(home, method, params):
            # Bridge just the transport; the sender tool builds the production request and
            # the real dispatcher validates it and invokes the target owner's cancel handler.
            assert (Path(home), method, params) == (root, "bot_relay.cancel", {
                "id": cancelled, "author_id": "bot:scribe", "profile": "default",
            })
            result = asyncio.run_coroutine_threadsafe(rpc(target, method, params), loop).result(timeout=10)
            assert result == {"delivery_id": cancelled, "status": "cancelled", "receipt_status": "cancelled"}
            return result

        monkeypatch.setattr(bot_live_delivery, "_authority_rpc", route_cancel)
        agent = SimpleNamespace(_session_title_hint="Bot Chat",
                                persistence=SimpleNamespace(scope=SimpleNamespace(profile_dir=scribe)))
        result = json.loads(await asyncio.to_thread(message_agent_tool, "hermes", cancel=cancelled, agent=agent))
        assert (result.get("status"), result.get("delivery_id")) == ("cancelled", cancelled), result
        row = owner.persistence.admissions.get(admission_id=delivery["admission_id"])
        assert (row.status, row.outcome) == ("terminal", "cancelled")
        receipt = await asyncio.to_thread(bot_live_delivery.read_delivery_result, root, cancelled)
        assert receipt["status"] == "cancelled"
        await asyncio.gather(*owner._bot_receipt_tasks)

        for _ in range(2):
            response = await rpc(sender, "bot_delivery.list", {})
            assert response["sender"] == "scribe"
            rows = {row["delivery_id"]: (row["state"], row.get("reason"), row.get("error"))
                    for row in response["deliveries"]}
            assert rows == {
                failed: ("failed", "rate_limit", "429"),
                ambiguous: ("failed", "unknown_outcome", None),
                offline: ("failed", RUNTIME_OFFLINE, offline_error),
            }
    finally:
        store.close_all_db_handles()


def test_lists_only_this_senders_unsettled_deliveries_and_never_writes(tmp_path):
    root = tmp_path / "hermes"
    _profile(root, "default")
    scribe = _profile(root, "scribe")
    _receipt(scribe, "a" * 32, sender="alice", status="queued")
    _receipt(scribe, "b" * 32, sender="alice", status="settled", reply="done")
    _receipt(scribe, "c" * 32, sender="bob", status="queued")
    _receipt(scribe, "d" * 32, sender="alice", status="failed", reason="rate_limit", error="429")
    _receipt(scribe, "e" * 32, sender="alice", status="ambiguous")
    _receipt(scribe, "f" * 32, sender="alice", status="failed", reason="rate_limit", age=FAILED_VISIBLE_SECONDS + 60)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (scribe / "runtime/bot_live_delivery").iterdir()}

    rows = list_sender_deliveries(root, "alice", profiles=[("default", root), ("scribe", scribe)],
                                  live_state=lambda *_: (None, None))

    by_id = {row["delivery_id"][0]: row for row in rows}
    assert set(by_id) == {"a", "d", "e"}  # settled, another sender's and stale failures drop off
    assert by_id["a"]["state"] == "queued" and by_id["a"]["target"] == "scribe"
    assert by_id["a"]["preview"] == "please review the draft"
    assert (by_id["d"]["state"], by_id["d"]["reason"], by_id["d"]["error"]) == ("failed", "rate_limit", "429")
    # An unknown outcome is a failure the user must not resend, and says so.
    assert (by_id["e"]["state"], by_id["e"]["reason"]) == ("failed", "unknown_outcome")
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (scribe / "runtime/bot_live_delivery").iterdir()}
    assert after == before  # read-only: no claim, no lock file, no rewrite


def test_a_poll_reparses_only_receipts_written_since_the_last_one(tmp_path, monkeypatch):
    import tools.bot_delivery_status as status

    root = tmp_path / "hermes"
    _profile(root, "default")
    scribe = _profile(root, "scribe")
    for i in range(5):
        _receipt(scribe, f"{i}" * 32, sender="alice", status="settled", reply="done")
    live = _receipt(scribe, "a" * 32, sender="alice", status="queued")
    reads = []
    real_read = status._read_receipt
    monkeypatch.setattr(status, "_read_receipt", lambda path: reads.append(path.name) or real_read(path))

    def poll():
        return list_sender_deliveries(root, "alice", profiles=[("scribe", scribe)], live_state=lambda *_: (None, None))

    assert [row["state"] for row in poll()] == ["queued"]
    reads.clear()
    assert [row["state"] for row in poll()] == ["queued"]
    assert reads == []  # settled history and the unchanged live receipt are not re-read

    record = json.loads(live.read_text(encoding="utf-8-sig"))
    tmp = live.with_suffix(".tmp")
    tmp.write_text(json.dumps({**record, "status": "failed", "reason": "rate_limit"}), encoding="utf-8")
    os.replace(tmp, live)  # the owner's atomic rewrite
    assert [(row["state"], row["reason"]) for row in poll()] == [("failed", "rate_limit")]
    assert reads == [live.name]


class _Events:
    def __init__(self):
        self.lock = threading.RLock()
        self.published = []

    def publish(self, session_id, payload, event_type=None):
        self.published.append(event_type)


def test_owner_join_reports_working_and_waiting_on_you(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    alice, scribe, critic = _profile(root, "default"), _profile(root, "scribe"), _profile(root, "critic")
    _receipt(scribe, "1" * 32, sender="default", status="queued")
    _receipt(critic, "2" * 32, sender="default", status="queued")
    # Queued behind the parked turn in the same chat: it waits on that turn, not on the user.
    _receipt(scribe, "3" * 32, sender="default", status="queued")
    admissions = {"adm-" + "1" * 32: "started", "adm-" + "2" * 32: "started", "adm-" + "3" * 32: "queued"}
    store = SimpleNamespace(admissions=SimpleNamespace(
        get=lambda *, admission_id: SimpleNamespace(status=admissions[admission_id])))

    events = _Events()
    waiting = PendingControls(events)
    clarify = SimpleNamespace(clarify_id="q1", question="Which draft?", choices=["v1", "v2"], multi_select=False,
                              event=threading.Event())
    waiting.register_clarify("chat-1", 3, clarify)
    published_before = list(events.published)

    def authority(home, controls):
        return SimpleNamespace(profile_id=str(home), persistence=store, retired=False,
                               sessions={"chat-1": SimpleNamespace(controls=controls)})

    authorities = {scribe: authority(scribe, waiting), critic: authority(critic, PendingControls(_Events()))}
    registry = SimpleNamespace(for_home=lambda home: authorities.get(Path(home)))
    runner = SimpleNamespace(session_authorities=registry)
    sender = SimpleNamespace(profile_id=str(alice), retired=False, runner=runner)
    connection = SimpleNamespace(authority=sender, actor=SimpleNamespace(capabilities={"session:read"}))

    result = asyncio.run(bot_delivery_list(connection, None, {}))

    states = {row["delivery_id"][0]: (row["target"], row["state"], row.get("waiting_on")) for row in result["deliveries"]}
    assert result["sender"] == "default"
    assert states == {"1": ("scribe", "waiting_on_you", "clarify"), "2": ("critic", "working", None),
                      "3": ("scribe", "queued", None)}
    assert events.published == published_before  # the poll never settles or re-publishes a prompt
    clarify.event.set()  # the user answered: the same delivery is plain work again
    again = asyncio.run(bot_delivery_list(connection, None, {}))
    assert {row["delivery_id"][0]: row["state"] for row in again["deliveries"]}["1"] == "working"


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a 0o000 file")
def test_a_receipt_that_cannot_be_read_fails_the_poll_instead_of_vanishing(tmp_path):
    root = tmp_path / "hermes"
    _profile(root, "default")
    scribe = _profile(root, "scribe")
    live = _receipt(scribe, "a" * 32, sender="alice", status="queued")
    (scribe / "runtime/bot_live_delivery/notes.json").write_text("not json", encoding="utf-8")

    def poll():
        return list_sender_deliveries(root, "alice", profiles=[("scribe", scribe)], live_state=lambda *_: (None, None))

    assert [row["state"] for row in poll()] == ["queued"]  # a foreign file is still skipped
    record = json.loads(live.read_text(encoding="utf-8-sig"))
    tmp = live.with_suffix(".tmp")
    tmp.write_text(json.dumps({**record, "status": "claimed"}), encoding="utf-8")
    os.replace(tmp, live)
    live.chmod(0)
    try:
        with pytest.raises(PermissionError):
            poll()  # an error the client keeps its last rows through, never an empty answer
    finally:
        live.chmod(0o600)
    assert [row["delivery_id"][0] for row in poll()] == ["a"]
