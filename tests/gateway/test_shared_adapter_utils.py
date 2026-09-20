"""Adapters share the three tiny lifecycle utilities in ``gateway.platforms.helpers`` instead of
re-declaring them: ``cancel_task`` (cancel + await, self-cancel safe), ``MessageDeduplicator``
(TTL + size bound) and ``bounded_put`` (insertion-ordered hard cap).
"""

from __future__ import annotations

import asyncio
import inspect
from collections import OrderedDict

import pytest

from gateway.platforms import helpers


def _adapter_classes():
    from gateway.platforms.qqbot.adapter import QQAdapter
    from gateway.platforms.signal import SignalAdapter
    from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
    from plugins.platforms.buzz.adapter import BuzzAdapter
    from plugins.platforms.ntfy.adapter import NtfyAdapter
    from plugins.platforms.photon.adapter import PhotonAdapter
    from plugins.platforms.simplex.adapter import SimplexAdapter
    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter

    return [QQAdapter, SignalAdapter, WhatsAppCloudAdapter, BuzzAdapter, NtfyAdapter,
            PhotonAdapter, SimplexAdapter, WecomCallbackAdapter]


def test_no_adapter_redeclares_the_shared_utilities():
    """The bodies used to live as `_cancel_task` / `_is_duplicate` / `_bounded_put` on each class."""
    for cls in _adapter_classes():
        module = inspect.getmodule(cls)
        for name in ("_cancel_task", "_is_duplicate", "_bounded_put"):
            assert not hasattr(cls, name), (cls.__name__, name)
            assert not hasattr(module, name), (module.__name__, name)


def test_cancel_task_unwinds_and_is_self_cancel_safe():
    async def scenario():
        started = asyncio.Event()

        async def worker():
            started.set()
            await asyncio.sleep(60)

        task = asyncio.create_task(worker())
        await started.wait()
        await helpers.cancel_task(task)
        assert task.cancelled()
        await helpers.cancel_task(None)
        await helpers.cancel_task(task)  # already done: no-op

        async def suicidal():
            await helpers.cancel_task(asyncio.current_task())
            return "survived"

        me = asyncio.create_task(suicidal())
        with pytest.raises(asyncio.CancelledError):
            await me

    asyncio.run(scenario())


def test_bounded_put_refreshes_and_caps():
    store: OrderedDict = OrderedDict()
    for i in range(5):
        helpers.bounded_put(store, f"k{i}", i, cap=3)
    assert list(store) == ["k2", "k3", "k4"]
    helpers.bounded_put(store, "k2", "again", cap=3)  # refresh moves it to the newest slot
    assert list(store) == ["k3", "k4", "k2"]


def test_dedup_sites_keep_their_own_window(monkeypatch):
    from gateway.platforms.qqbot import constants as qq
    from plugins.platforms.photon import adapter as photon
    from plugins.platforms.wecom import callback_adapter as wecom_cb

    import plugins.platforms.line.adapter as line

    windows = {
        "qqbot": (qq.DEDUP_MAX_SIZE, qq.DEDUP_WINDOW_SECONDS),
        "photon": (photon._DEDUP_MAX_SIZE, photon._DEDUP_WINDOW_SECONDS),
        "wecom_callback": (2000, wecom_cb.MESSAGE_DEDUP_TTL_SECONDS),
        "line": (1000, float("inf")),
    }
    # The helper honours ttl exactly: a stale id is admitted again, a fresh one is not.
    now = [1000.0]
    monkeypatch.setattr(helpers.time, "time", lambda: now[0])
    for name, (cap, ttl) in windows.items():
        d = helpers.MessageDeduplicator(max_size=cap, ttl_seconds=ttl)
        assert d.is_duplicate("m") is False and d.is_duplicate("m") is True, name
        now[0] += ttl if ttl != float("inf") else 10 ** 9
        assert d.is_duplicate("m") is (ttl == float("inf")), name
    assert line.MessageDeduplicator is helpers.MessageDeduplicator
