"""Behaviour pins for the cross-machine relay queue (tools/bot_relay.py).

These freeze today's re-offer / timeout / expiry / sweep edges so a later move onto a shared
inbox primitive can delete this code without changing what senders observe. Time is injected
via ``now=`` and file mtimes; nothing races the wall clock.
"""

from __future__ import annotations

import json
import os

import pytest

from tools import bot_relay

TTL = 900.0
T0 = 1_000_000.0  # fixed epoch for created_at / claim mtimes


def _target():
    return {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
            "connection_label": "", "title": "", "description": ""}


@pytest.fixture()
def base(tmp_path):
    return bot_relay._ensure_dirs(tmp_path)


def _claimed(root, base, *, created_at=T0, claimed_at=T0):
    """Put one unanswered envelope in ``claimed/`` with the given created/claim times."""
    env = bot_relay.enqueue_envelope(root, target=_target(), message="m",
                                     sender_profile="w", sender_handle="w")
    env["created_at"] = int(created_at)
    src = base / bot_relay.OUTBOX_DIR / f"{env['id']}.json"
    dst = base / bot_relay.CLAIMED_DIR / f"{env['id']}.json"
    src.unlink()
    dst.write_text(json.dumps(env), encoding="utf-8")
    os.utime(dst, (claimed_at, claimed_at))
    return env, dst


def _reply(base, env_id):
    path = base / bot_relay.REPLIES_DIR / f"{env_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def test_reoffer_happens_exactly_once_starting_at_reoffer_after(tmp_path, base):
    """An unanswered claim is re-offered only once ``now - claimed_at >= REOFFER_AFTER_SECONDS``, and only once ever."""
    env, path = _claimed(tmp_path, base)
    edge = T0 + bot_relay.REOFFER_AFTER_SECONDS

    assert bot_relay._reoffer_unanswered(tmp_path, base, TTL, edge - 1) == []
    first = bot_relay._reoffer_unanswered(tmp_path, base, TTL, edge)
    assert [e["id"] for e in first] == [env["id"]]
    assert json.loads(path.read_text(encoding="utf-8"))["reoffered_at"] == int(edge)
    # Stamping rewrote the file; even with the claim mtime pushed back again it is never re-offered.
    os.utime(path, (T0, T0))
    assert bot_relay._reoffer_unanswered(tmp_path, base, TTL, edge + 10) == []
    assert _reply(base, env["id"]) is None


def test_past_reply_wait_writes_delivery_timeout_and_never_reoffers(tmp_path, base):
    """Past ``created_at + REPLY_WAIT_SECONDS`` an unanswered claim gets a ``delivery_timeout`` reply and is never handed out."""
    env, _ = _claimed(tmp_path, base)
    late = T0 + bot_relay.REPLY_WAIT_SECONDS + 1

    assert bot_relay._reoffer_unanswered(tmp_path, base, 0, late) == []
    reply = _reply(base, env["id"])
    assert reply is not None
    assert reply["reason"] == "delivery_timeout" and reply["error"] and not reply["reply"]
    # Once replied, the claim is skipped forever — no re-offer and the first reply stands.
    assert bot_relay._reoffer_unanswered(tmp_path, base, 0, late + 10_000) == []
    assert _reply(base, env["id"]) == reply


def test_reply_wait_boundary_is_strict(tmp_path, base):
    """Exactly ``created_at + REPLY_WAIT_SECONDS`` is still inside the window: re-offered, not timed out."""
    env, _ = _claimed(tmp_path, base, created_at=T0, claimed_at=T0)
    edge = T0 + bot_relay.REPLY_WAIT_SECONDS

    assert [e["id"] for e in bot_relay._reoffer_unanswered(tmp_path, base, 0, edge)] == [env["id"]]
    assert _reply(base, env["id"]) is None


def test_reoffer_leg_older_than_ttl_gets_queued_expired(tmp_path, base):
    """A re-offer that sat more than ``ttl`` past ``claim + REOFFER_AFTER_SECONDS`` is refused with ``queued_expired``, never handed out."""
    # Keep created_at recent enough that the REPLY_WAIT bound does not fire first.
    claimed_at = T0
    now = claimed_at + bot_relay.REOFFER_AFTER_SECONDS + TTL + 1
    env, _ = _claimed(tmp_path, base, created_at=now - 10, claimed_at=claimed_at)

    assert bot_relay._reoffer_unanswered(tmp_path, base, TTL, now) == []
    reply = _reply(base, env["id"])
    assert reply is not None
    assert reply["reason"] == "queued_expired" and reply["error"] and not reply["reply"]
    assert bot_relay._reoffer_unanswered(tmp_path, base, TTL, now + 1) == []


def test_reoffer_leg_at_exactly_ttl_is_still_offered(tmp_path, base):
    """``queued_for == ttl`` is not expired (strict ``>``), and ``ttl <= 0`` disables the re-offer expiry."""
    now = T0 + bot_relay.REOFFER_AFTER_SECONDS + TTL
    env, _ = _claimed(tmp_path, base, created_at=now - 10, claimed_at=T0)
    assert [e["id"] for e in bot_relay._reoffer_unanswered(tmp_path, base, TTL, now)] == [env["id"]]

    env2, _ = _claimed(tmp_path, base, created_at=now + 10_000 - 10, claimed_at=T0)
    out = bot_relay._reoffer_unanswered(tmp_path, base, 0, now + 10_000)
    assert [e["id"] for e in out] == [env2["id"]]
    assert _reply(base, env2["id"]) is None


def test_sweep_stale_removes_only_files_older_than_stale_after(tmp_path, base):
    """``_sweep_stale`` unlinks ``*.json`` in outbox/claimed/replies whose mtime is strictly before ``now - STALE_AFTER_SECONDS``."""
    now = T0 + bot_relay.STALE_AFTER_SECONDS + 100
    cutoff = now - bot_relay.STALE_AFTER_SECONDS
    cases = {}
    for sub in (bot_relay.OUTBOX_DIR, bot_relay.CLAIMED_DIR, bot_relay.REPLIES_DIR):
        for label, mtime in (("old", cutoff - 1), ("edge", cutoff), ("fresh", cutoff + 1)):
            path = base / sub / f"{sub}-{label}.json"
            path.write_text("{}", encoding="utf-8")
            os.utime(path, (mtime, mtime))
            cases[path] = label
    other = base / bot_relay.CLAIMED_DIR / "notes.txt"
    other.write_text("x", encoding="utf-8")
    os.utime(other, (0, 0))

    assert bot_relay._sweep_stale(base, now=now) == 3
    for path, label in cases.items():
        assert path.exists() is (label != "old"), path
    assert other.exists()  # non-JSON files are never swept
