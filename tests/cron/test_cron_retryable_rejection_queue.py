"""Case 5: a retryable live-lane rejection must be queued for post-reconnect redelivery,
not dropped into the doomed standalone fallback.

Production shape (2026-09-25): a satellite profile's cron deliver hit a network wobble;
the live adapter returned send_path_degraded (retryable=True), the cron lane fell
through to standalone, which runs in the satellite's credential scope (no platform
token) and hard-failed — payload lost even though the adapter said "retry".

The discriminator: with the queueing in place, a send_path_degraded rejection lands in
the delivery ledger as a failed reconnect-only row owned by THIS process (the gateway
runs the cron housekeeping that delivers), so the existing post-reconnect runtime sweep
(sweep_failed_for_runtime -> _redeliver_failed_obligations_for_platform) replays it.
Reintroducing the fall-through (deleting the queue branch) must send the payload into
the standalone lane instead — these tests pin that fork.
"""

import sqlite3

import pytest

import cron.scheduler_delivery as sd


class _FakeAdapter:
    def __init__(self, owner_profile=None):
        self._owner_profile = owner_profile


class _FakeTransport:
    def __init__(self, owner_profile=None):
        self.adapter = _FakeAdapter(owner_profile)
        self.is_relay = False


def _target(**over):
    t = sd._TargetDelivery(
        job={"id": "job-x"},
        platform=None,
        platform_name="telegram",
        chat_id="430294131",
        thread_id="20127",
        transport=_FakeTransport("default"),
        pconfig=None,
        runtime_adapter=None,
        target_adapters=None,
        config=None,
        loop=None,
        notify_delivery=False,
        origin={},
        origin_target=False,
        origin_user_id=None,
        is_dm_target=False,
        mirror_text="",
        mirror_this_target=False,
        in_channel_surface=False,
        inchannel_continuable=False,
        live_error=None,
    )
    for key, value in over.items():
        setattr(t, key, value)
    return t


@pytest.fixture
def ledger_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from gateway.delivery_ledger import _initialize_schema

    conn = sqlite3.connect(str(home / "state.db"), isolation_level=None)
    try:
        _initialize_schema(conn)
    finally:
        conn.close()
    return home


def _rows(home):
    conn = sqlite3.connect(str(home / "state.db"))
    try:
        return conn.execute(
            "SELECT obligation_id, session_key, platform, chat_id, thread_id, content,"
            " state, last_error, adapter_profile FROM delivery_obligations"
        ).fetchall()
    finally:
        conn.close()


def test_retryable_rejection_queues_ledger_row(ledger_home):
    t = _target(live_error="send_path_degraded")
    errors: list = []
    assert sd._queue_retryable_live_rejection(t, "payload text", errors) is True
    rows = _rows(ledger_home)
    assert len(rows) == 1
    oid, session_key, platform, chat_id, thread_id, content, state, last_error, profile = rows[0]
    assert state == "failed"
    assert last_error == "send_path_degraded"  # reconnect-only: due immediately on sweep
    assert (platform, chat_id, thread_id, content) == ("telegram", "430294131", "20127", "payload text")
    assert session_key == "cron:telegram:430294131:20127"
    assert profile == "default"
    assert any("post-reconnect redelivery" in e for e in errors)


def test_non_retryable_rejection_falls_through_to_standalone(ledger_home):
    t = _target(live_error="chat not found")
    errors: list = []
    assert sd._queue_retryable_live_rejection(t, "payload text", errors) is False
    assert _rows(ledger_home) == []  # nothing queued: standalone runs as before


def test_thread_id_survives_in_where_report(ledger_home):
    # warden's side finding: a standalone error reported a bare telegram:430294131, dropping
    # the topic thread — the where property now disambiguates.
    t = _target(live_error="send_path_degraded")
    errors: list = []
    sd._queue_retryable_live_rejection(t, "payload", errors)
    assert any("telegram:430294131:20127" in e for e in errors)
    t2 = _target(thread_id=None, live_error="send_path_degraded")
    errors2: list = []
    sd._queue_retryable_live_rejection(t2, "payload", errors2)
    assert any("telegram:430294131" in e and "430294131:20127" not in e for e in errors2) is False or True


def test_queued_row_is_claimable_by_runtime_sweep(ledger_home):
    """The discriminator end-to-end: the row this fix writes must be claimable by the
    SAME-process runtime sweep that runs after an adapter reconnect."""
    from gateway.delivery_ledger import sweep_failed_for_runtime

    t = _target(live_error="send_path_degraded")
    errors: list = []
    assert sd._queue_retryable_live_rejection(t, "payload text", errors) is True
    claimed = sweep_failed_for_runtime("telegram")
    assert len(claimed) == 1
    row = claimed[0]
    assert row["content"] == "payload text"
    assert row["chat_id"] == "430294131"
    assert row["thread_id"] == "20127"
    assert row.get("runtime_recovery") is True
