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

from gateway.session_pending_controls import PendingControls
from gateway.session_teammate_waits import bot_delivery_list
from tools.bot_delivery_status import FAILED_VISIBLE_SECONDS, list_sender_deliveries


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
    monkeypatch.setattr("hermes_state_runtime.get_session_admission",
                        lambda db, *, admission_id: {"status": admissions[admission_id]})

    events = _Events()
    waiting = PendingControls(events)
    clarify = SimpleNamespace(clarify_id="q1", question="Which draft?", choices=["v1", "v2"], multi_select=False,
                              event=threading.Event())
    waiting.register_clarify("chat-1", 3, clarify)
    published_before = list(events.published)

    def authority(home, controls):
        return SimpleNamespace(db=SimpleNamespace(db_path=str(home / "state.db")), retired=False,
                               sessions={"chat-1": SimpleNamespace(controls=controls)})

    authorities = {scribe: authority(scribe, waiting), critic: authority(critic, PendingControls(_Events()))}
    registry = SimpleNamespace(for_home=lambda home: authorities.get(Path(home)))
    runner = SimpleNamespace(session_authorities=registry)
    sender = SimpleNamespace(db=SimpleNamespace(db_path=str(alice / "state.db")), retired=False, runner=runner)
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
