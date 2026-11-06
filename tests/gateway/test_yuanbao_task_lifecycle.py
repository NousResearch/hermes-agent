"""test_yuanbao_task_lifecycle.py - Yuanbao background-task lifecycle.

The reconnect task must be owned by the adapter:

* ``ConnectionManager.schedule_reconnect`` used a bare ``asyncio.create_task``.
  The event loop keeps only a weak reference to such a task, so a pending
  reconnect could be garbage-collected mid-flight — after which the adapter
  never reconnects and the bot is silently offline until the gateway restarts.
  ``YuanbaoAdapter._track_task`` exists for exactly this ("Register a
  fire-and-forget task so it won't be GC'd prematurely") and is already used
  for the inbound pipeline in the same file.
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.yuanbao.adapter import YuanbaoAdapter


def make_config(**kwargs):
    extra = kwargs.pop("extra", {})
    extra.setdefault("app_id", "test_key")
    extra.setdefault("app_secret", "test_secret")
    extra.setdefault("ws_url", "wss://test.example.com/ws")
    extra.setdefault("api_domain", "https://test.example.com")
    return PlatformConfig(extra=extra, **kwargs)


def _adapter() -> YuanbaoAdapter:
    """A real adapter with the status-writing teardown hooks neutralised."""
    adapter = YuanbaoAdapter(make_config())
    adapter._mark_disconnected = lambda: None
    adapter._release_platform_lock = lambda: None
    return adapter


# ---------------------------------------------------------------------------
# schedule_reconnect — the task must be strongly referenced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_reconnect_anchors_task_against_gc():
    """The pending reconnect must be reachable from _background_tasks."""
    adapter = _adapter()
    adapter._running = True
    cm = adapter._connection

    with patch.object(cm, "_reconnect_with_backoff", new_callable=AsyncMock, return_value=True):
        cm.schedule_reconnect()

        anchored = [t for t in adapter._background_tasks if t.get_name() == "yuanbao-reconnect"]
        assert len(anchored) == 1, (
            "reconnect task is not strongly referenced — the loop may GC it mid-flight"
        )

        await asyncio.gather(*anchored)

    # _track_task's done callback releases the reference once it completes.
    assert adapter._background_tasks == set()


@pytest.mark.asyncio
async def test_schedule_reconnect_is_a_noop_when_adapter_stopped():
    """Anchoring must not weaken the existing _running guard."""
    adapter = _adapter()
    adapter._running = False
    cm = adapter._connection

    with patch.object(cm, "_reconnect_with_backoff", new_callable=AsyncMock) as reconnect:
        cm.schedule_reconnect()

    assert adapter._background_tasks == set()
    reconnect.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_reconnect_is_a_noop_while_already_reconnecting():
    """Anchoring must not weaken the existing _reconnecting guard."""
    adapter = _adapter()
    adapter._running = True
    cm = adapter._connection
    cm._reconnecting = True

    with patch.object(cm, "_reconnect_with_backoff", new_callable=AsyncMock) as reconnect:
        cm.schedule_reconnect()

    assert adapter._background_tasks == set()
    reconnect.assert_not_called()
