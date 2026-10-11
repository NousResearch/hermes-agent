"""The Slack thread watermark must address the same session-key namespace in both directions.

``_thread_watermark_io`` resolves the session key that owns the watermark metadata. It did not
forward ``chat_type``, so the key defaulted to the ``group:`` namespace while the DM session —
and the ``_has_active_session_for_thread`` guard beside it — lives under ``dm:``. In a DM the
write therefore addressed a key no session owns and the read always returned ``""``, silently
disabling restart rehydration: thread replies posted while the gateway was down were never
recovered.

Both sibling readers of this value carry a comment that ``chat_type`` must come from the event's
``channel_type`` and never be inferred from the channel-ID prefix (MPIM IDs start with ``G``);
this path simply omitted it.
"""
from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig


class _Store:
    """Records metadata reads/writes per session key, like the real store's namespacing."""

    def __init__(self):
        self.data: dict[tuple[str, str], str] = {}
        self.reads: list[str] = []

    def set_session_metadata(self, session_key, meta_key, value):
        self.data[(session_key, meta_key)] = value

    def get_session_metadata(self, session_key, meta_key, default=""):
        self.reads.append(session_key)
        return self.data.get((session_key, meta_key), default)


def _adapter(store):
    from plugins.platforms.slack.adapter import SlackAdapter

    a = object.__new__(SlackAdapter)
    a.config = PlatformConfig(enabled=True, extra={})
    a.platform = Platform.SLACK
    a._session_store = store
    # The key seam the runner normally seeds; mirrors the real namespaces. Signature matches the
    # real method so a drift in how the IO path calls it shows up here.
    def _key(channel_id, thread_ts, user_id, team_id="", *, chat_type="group"):
        return f"agent:main:slack:{chat_type}:{team_id}:{channel_id}:{thread_ts}"

    a._build_thread_session_key = _key
    return a


_DM = dict(channel_id="D1", thread_ts="111.1", user_id="U1", team_id="T1")


def test_dm_watermark_round_trips():
    """A DM write must be visible to a DM read — this is the whole contract."""
    store = _Store()
    a = _adapter(store)

    a._set_thread_watermark(**_DM, watermark_ts="222.2", chat_type="dm")

    assert a._get_thread_watermark(**_DM, chat_type="dm") == "222.2"


def test_dm_watermark_is_stored_under_the_dm_namespace():
    """Not the ``group:`` namespace: that key belongs to no session, so the write is dropped."""
    store = _Store()
    a = _adapter(store)

    a._set_thread_watermark(**_DM, watermark_ts="222.2", chat_type="dm")

    keys = [k for k, _ in store.data]
    assert keys == ["agent:main:slack:dm:T1:D1:111.1"]
    assert not any(":group:" in k for k in keys)


def test_group_watermark_round_trips():
    """The group path was already correct and must stay so."""
    store = _Store()
    a = _adapter(store)
    args = dict(channel_id="C1", thread_ts="111.1", user_id="U1", team_id="T1")

    a._set_thread_watermark(**args, watermark_ts="222.2", chat_type="group")

    assert a._get_thread_watermark(**args, chat_type="group") == "222.2"
    assert [k for k, _ in store.data] == ["agent:main:slack:group:T1:C1:111.1"]


def test_namespaces_do_not_bleed_into_each_other():
    """A DM and a group thread sharing ids keep separate watermarks."""
    store = _Store()
    a = _adapter(store)

    a._set_thread_watermark(**_DM, watermark_ts="dm-ts", chat_type="dm")
    a._set_thread_watermark(**_DM, watermark_ts="group-ts", chat_type="group")

    assert a._get_thread_watermark(**_DM, chat_type="dm") == "dm-ts"
    assert a._get_thread_watermark(**_DM, chat_type="group") == "group-ts"


def test_read_addresses_the_requested_namespace():
    store = _Store()
    a = _adapter(store)

    a._get_thread_watermark(**_DM, chat_type="dm")

    assert store.reads == ["agent:main:slack:dm:T1:D1:111.1"]


@pytest.mark.parametrize("chat_type", ["dm", "group"])
def test_empty_watermark_is_never_written(chat_type):
    """An empty ts would erase a real watermark and re-trigger a full rehydrate."""
    store = _Store()
    a = _adapter(store)

    a._set_thread_watermark(**_DM, watermark_ts="", chat_type=chat_type)

    assert store.data == {}


def test_missing_watermark_reads_as_empty_string():
    assert _adapter(_Store())._get_thread_watermark(**_DM, chat_type="dm") == ""


def test_store_failure_does_not_propagate():
    """A watermark is an optimisation; losing it must not fail the inbound message."""
    class _Broken(_Store):
        def set_session_metadata(self, *a, **k):
            raise RuntimeError("db down")

        def get_session_metadata(self, *a, **k):
            raise RuntimeError("db down")

    a = _adapter(_Broken())
    a._set_thread_watermark(**_DM, watermark_ts="222.2", chat_type="dm")  # must not raise
    assert a._get_thread_watermark(**_DM, chat_type="dm") == ""
