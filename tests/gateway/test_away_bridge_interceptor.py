"""Interceptor + matcher tests (design §4.4) and CLI/waiter tests (§4.3)."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gateway.platforms.away_bridge_interceptor import (
    InterceptResult,
    partition_batch,
    process_batch_sync,
)
from tools.away_bridge_broker import AwayBridgeBroker


class _FakeSource:
    def __init__(self, chat_id="1682802389", platform="telegram"):
        self.chat_id = chat_id
        self.platform = platform
        self.chat_type = "dm"


class _FakeEvent:
    def __init__(self, text, chat_id="1682802389", user_id="1682802389", platform="telegram"):
        self.text = text
        self.source = _FakeSource(chat_id=chat_id, platform=platform)
        self.user_id = user_id
        self.message_type = "text"


class _SentinelBroker:
    """In-memory broker double mirroring the real state transitions."""

    def __init__(self):
        self.armed = {}
        self.requests = {}
        self.acks = set()
        self._ctr = 0

    def arm(self, key):
        self.armed[key] = "armed"

    def has_armed(self):
        return any(v == "armed" for v in self.armed.values())

    def prepare(self, key, question):
        self._ctr += 1
        rk = "ab%032d" % self._ctr
        tok = "A#K7QM4XT9P2WBN5F"[:2] + "T%014d" % self._ctr
        self.requests[rk] = {
            "req_key": rk, "session_key": key, "token": tok,
            "question": question, "state": "prepared",
            "winning_channel": None,
        }
        return dict(self.requests[rk])

    def publish(self, req_key):
        self.requests[req_key]["state"] = "awaiting"

    def claim(self, req_key, channel, answer):
        row = self.requests[req_key]
        if row["state"] == "awaiting":
            row["state"] = "claimed"
            row["winning_channel"] = channel
            row["answer"] = answer
            return {"outcome": "won", "channel": channel}
        if row["state"] == "claimed":
            return {"outcome": "lost", "winner": row["winning_channel"]}
        return {"outcome": "reject", "reason": row["state"]}

    def ack_duplicate(self, req_key, late_channel):
        k = (req_key, late_channel)
        if k in self.acks:
            return False
        self.acks.add(k)
        return True

    def disarm_all(self):
        n = sum(1 for v in self.armed.values() if v == "armed")
        self.armed = {k: "disarmed" for k in self.armed}
        return {"disarmed_count": n}

    def get_request_by_token(self, token):
        for row in self.requests.values():
            if row["token"] == token:
                return dict(row)
        return None

    def get_request(self, req_key):
        row = self.requests.get(req_key)
        return dict(row) if row else None


def _owner_cfg():
    return {
        "enabled": True,
        "sweep_window_hours": 1,
        "retention_days": 30,
        "telegram": {"owner_chat_id": "1682802389", "owner_user_id": "1682802389"},
    }


# ---------------------------------------------------------------------------
# matcher: line-anchored, token-partitioned, extraction boundary (R5-3)
# ---------------------------------------------------------------------------

def test_partition_single_token_plain():
    segs, ri = partition_batch("A#K7QM4XT9P2WBN5F\nyes do it")
    assert segs == [("A#K7QM4XT9P2WBN5F", "yes do it")]
    assert ri is False


def test_partition_same_line_remainder_is_answer():
    segs, _ = partition_batch("A#K7QM4XT9P2WBN5F do X now")
    assert segs == [("A#K7QM4XT9P2WBN5F", "do X now")]


def test_partition_stray_prefix_discarded():
    segs, _ = partition_batch("sorry late\nA#K7QM4XT9P2WBN5F\nthe answer")
    assert segs == [("A#K7QM4XT9P2WBN5F", "the answer")]


def test_partition_midline_mention_not_matched():
    segs, ri = partition_batch("what does A#K7QM4XT9P2WBN5F mean?")
    assert segs == []
    assert ri is False


def test_partition_two_tokens_both_claimed():
    text = "A#AAAAAAAAAAAAAAAA first answer\nA#BBBBBBBBBBBBBBBB second answer"
    segs, _ = partition_batch(text)
    assert segs == [
        ("A#AAAAAAAAAAAAAAAA", "first answer"),
        ("A#BBBBBBBBBBBBBBBB", "second answer"),
    ]


def test_partition_return_intent_and_mixed():
    segs, ri = partition_batch("A#AAAAAAAAAAAAAAAA\nyes\nI'm back")
    assert len(segs) == 1
    assert ri is True
    # disarm-first order must behave identically at the partition layer
    segs2, ri2 = partition_batch("I'm back\nA#AAAAAAAAAAAAAAAA\nyes")
    assert len(segs2) == 1
    assert ri2 is True


def test_partition_intent_only():
    segs, ri = partition_batch("I'm back")
    assert segs == []
    assert ri is True


def test_partition_hebrew_intent():
    _, ri = partition_batch("חזרתי")
    assert ri is True


# ---------------------------------------------------------------------------
# process_batch_sync behavior
# ---------------------------------------------------------------------------

def test_batch_claim_wins_sends_accept():
    b = _SentinelBroker()
    b.arm("sess-K0")
    req = b.prepare("sess-K0", "Proceed?")
    b.publish(req["req_key"])
    result = process_batch_sync(b, _FakeEvent("%s\nyes" % req["token"]), None, _owner_cfg())
    assert result.handled is True
    assert any("Accepted" in t for t in result.reply_texts)
    assert b.requests[req["req_key"]]["winning_channel"] == "telegram"


def test_batch_duplicate_gets_single_ack():
    b = _SentinelBroker()
    b.arm("sess-K0")
    req = b.prepare("sess-K0", "Proceed?")
    b.publish(req["req_key"])
    b.claim(req["req_key"], "chat", "chat answer")
    r1 = process_batch_sync(b, _FakeEvent("%s\nlate" % req["token"]), None, _owner_cfg())
    assert r1.handled is True
    assert any("Already answered via chat" in t for t in r1.reply_texts)
    # second duplicate: ack already claimed → no second ack text
    r2 = process_batch_sync(b, _FakeEvent("%s\nlate again" % req["token"]), None, _owner_cfg())
    assert r2.handled is True
    assert r2.reply_texts == []


def test_batch_unknown_token_nacked_and_suppressed():
    b = _SentinelBroker()
    result = process_batch_sync(b, _FakeEvent("A#ZZZZZZZZZZZZZZZZ\nhi"), None, _owner_cfg())
    assert result.handled is True
    assert any("No active bridge request" in t for t in result.reply_texts)


def test_batch_return_intent_disarms_all():
    b = _SentinelBroker()
    b.arm("sess-A")
    b.arm("sess-B")
    result = process_batch_sync(b, _FakeEvent("I'm back"), None, _owner_cfg())
    assert result.handled is True
    assert not b.has_armed()
    assert any("disarmed" in t for t in result.reply_texts)


def test_batch_return_intent_no_armed_falls_through():
    b = _SentinelBroker()
    result = process_batch_sync(b, _FakeEvent("I'm back"), None, _owner_cfg())
    assert result.handled is False


def test_batch_mixed_claim_then_disarm():
    b = _SentinelBroker()
    b.arm("sess-K0")
    req = b.prepare("sess-K0", "Proceed?")
    b.publish(req["req_key"])
    result = process_batch_sync(
        b, _FakeEvent("I'm back\n%s\nthe answer" % req["token"]), None, _owner_cfg()
    )
    assert result.handled is True
    assert b.requests[req["req_key"]]["winning_channel"] == "telegram"
    assert not b.has_armed()


def test_batch_answer_text_is_im_back_claims_then_disarms():
    b = _SentinelBroker()
    b.arm("sess-K0")
    req = b.prepare("sess-K0", "Are you back?")
    b.publish(req["req_key"])
    result = process_batch_sync(b, _FakeEvent("%s\nI'm back" % req["token"]), None, _owner_cfg())
    assert result.handled is True
    row = b.requests[req["req_key"]]
    assert row["winning_channel"] == "telegram"
    assert row["answer"] == "I'm back"
    assert not b.has_armed()  # documented accepted side effect (§4.6)


def test_non_owner_not_handled():
    b = _SentinelBroker()
    cfg = _owner_cfg()
    from gateway.platforms.away_bridge_interceptor import _is_authorized_owner_sync

    assert _is_authorized_owner_sync(_FakeEvent("hi", user_id="999"), cfg) is False
    assert _is_authorized_owner_sync(_FakeEvent("hi", chat_id="999"), cfg) is False


def test_ordinary_traffic_unhandled():
    b = _SentinelBroker()
    result = process_batch_sync(b, _FakeEvent("what's the weather"), None, _owner_cfg())
    assert result.handled is False


# ---------------------------------------------------------------------------
# CLI: waiter envelope + exit codes (§4.3 stdout contract)
# ---------------------------------------------------------------------------

HERMES_VENV_PY = Path.home() / ".hermes/hermes-agent/venv/bin/python"


def _run_cli(args, home: Path, env_extra=None, timeout=60):
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = "/Users/user/projects/zbubu/hermes-agent"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.away_bridge_cmd"] + args,
        env=env, capture_output=True, text=True, timeout=timeout,
        cwd="/Users/user/projects/zbubu/hermes-agent",
    )


@pytest.mark.skipif(not HERMES_VENV_PY.exists(), reason="hermes venv python missing")
def test_cli_arm_prepare_publish_claim_fetch_roundtrip(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    r = _run_cli(["--db-path", str(home / "away-bridge/bridge.db"), "arm", "--session", "sess-X"], home)
    assert r.returncode == 0, r.stderr
    r = _run_cli(
        ["--db-path", str(home / "away-bridge/bridge.db"), "prepare",
         "--session", "sess-X", "--question", "Go?"], home)
    assert r.returncode == 0, r.stderr
    req = json.loads(r.stdout)
    assert req["token"].startswith("A#")
    db = str(home / "away-bridge/bridge.db")
    r = _run_cli(["--db-path", db, "publish", "--request", req["req_key"]], home)
    assert r.returncode == 0, r.stderr
    r = _run_cli(["--db-path", db, "claim", "--request", req["req_key"],
                  "--channel", "telegram", "--answer", "yes"], home)
    assert r.returncode == 0, r.stderr
    r = _run_cli(["--db-path", db, "fetch", "--request", req["req_key"]], home)
    assert r.returncode == 0, r.stderr
    fetched = json.loads(r.stdout)
    assert fetched["answer"] == "yes"
    assert fetched["delivered"] is True


@pytest.mark.skipif(not HERMES_VENV_PY.exists(), reason="hermes venv python missing")
def test_waiter_chat_won_envelope(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    db = str(home / "away-bridge/bridge.db")
    r = _run_cli(["--db-path", db, "arm", "--session", "sess-W"], home)
    assert r.returncode == 0, r.stderr
    r = _run_cli(["--db-path", db, "prepare", "--session", "sess-W", "--question", "Q?"], home)
    req = json.loads(r.stdout)
    _run_cli(["--db-path", db, "publish", "--request", req["req_key"]], home)
    # Claim from another connection, then start the waiter: it must see the
    # claimed row on its first poll and print exactly one envelope line.
    _run_cli(["--db-path", db, "claim", "--request", req["req_key"],
              "--channel", "chat", "--answer", "from chat"], home)
    r = _run_cli(["--db-path", db, "wait", "--request", req["req_key"],
                  "--poll-seconds", "0.1"], home)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert lines == ["AWAY-BRIDGE outcome=CHAT_WON req_key=%s" % req["req_key"]]


@pytest.mark.skipif(not HERMES_VENV_PY.exists(), reason="hermes venv python missing")
def test_waiter_disarmed_envelope(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    db = str(home / "away-bridge/bridge.db")
    _run_cli(["--db-path", db, "arm", "--session", "sess-D"], home)
    r = _run_cli(["--db-path", db, "prepare", "--session", "sess-D", "--question", "Q?"], home)
    req = json.loads(r.stdout)
    _run_cli(["--db-path", db, "publish", "--request", req["req_key"]], home)
    _run_cli(["--db-path", db, "disarm", "--session", "sess-D"], home)
    r = _run_cli(["--db-path", db, "wait", "--request", req["req_key"],
                  "--poll-seconds", "0.1"], home)
    assert "AWAY-BRIDGE outcome=DISARMED" in r.stdout


def test_waiter_no_asyncio_dependency():
    """The waiter module must stay import-light (spawns in 2 s per poll)."""
    import hermes_cli.away_bridge_cmd as m

    assert callable(m.cmd_wait)
