"""Cron standalone lane keeps a sender's typed ``delivery_unconfirmed`` signal.

A Telegram album whose ``send_media_group`` reply was lost is possibly delivered: the standalone
sender returns success with ``delivery_unconfirmed`` instead of a warning, so a re-run must not
resend it and it is not a delivery error. The live lane records such a target in
``last_delivery_unverified``; the standalone lane must record it the same way instead of reporting
a plain "delivered". Real ``_deliver_result`` → ``_deliver_standalone`` →
``_record_delivery_verification``; only the outermost transport and the job store are faked.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig

UNCONFIRMED = "album possibly delivered (reply lost), so do not resend it"


@pytest.fixture
def standalone_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("cron: {wrap_response: false}\n")
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
    config.platforms[Platform.RELAY] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    update_job = MagicMock()
    monkeypatch.setattr("cron.jobs.update_job", update_job)
    env = SimpleNamespace(calls=[], result={}, update_job=update_job)

    async def standalone(platform, pconfig, chat_id, content, **kwargs):
        env.calls.append((platform.value, chat_id, kwargs.get("thread_id")))
        return dict(env.result)

    monkeypatch.setattr("gateway.outbound.send._send_to_platform", standalone)
    return env


def _job(**extra):
    return {"id": "unconfirmed-job", "deliver": "telegram:-100123:42", **extra}


def test_unconfirmed_only_is_recorded_unverified_not_an_error(standalone_env):
    standalone_env.result = {"success": True, "message_id": "9", "delivery_unconfirmed": UNCONFIRMED}
    job = _job()

    assert _deliver_result(job, "the brief") is None
    assert standalone_env.calls == [("telegram", "-100123", "42")]  # sent once, never resent
    assert job["last_delivery_unverified"] == ["telegram:-100123:42"]
    standalone_env.update_job.assert_called_once_with(
        "unconfirmed-job", {"last_delivery_unverified": ["telegram:-100123:42"]})


def test_clean_send_clears_a_previous_unverified_record(standalone_env):
    standalone_env.result = {"success": True, "message_id": "9"}
    job = _job(last_delivery_unverified=["telegram:-100123:42"])

    assert _deliver_result(job, "the brief") is None
    assert job["last_delivery_unverified"] is None
    standalone_env.update_job.assert_called_once_with(
        "unconfirmed-job", {"last_delivery_unverified": None})


def test_real_warning_stays_an_error_alongside_the_unverified_record(standalone_env):
    standalone_env.result = {
        "success": True, "message_id": "9", "delivery_unconfirmed": UNCONFIRMED,
        "warnings": ["Failed to send media /tmp/b.png: upload rejected"],
    }
    job = _job()

    error = _deliver_result(job, "the brief")
    assert error is not None and "upload rejected" in error
    assert UNCONFIRMED not in error
    assert job["last_delivery_unverified"] == ["telegram:-100123:42"]
    assert len(standalone_env.calls) == 1


@pytest.fixture
def loop_thread():
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(5)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    assert not thread.is_alive()
    loop.close()


def test_relay_failure_still_refuses_the_standalone_fallback(standalone_env, loop_thread):
    standalone_env.result = {"success": True, "message_id": "9", "delivery_unconfirmed": UNCONFIRMED}

    async def relay_send(platform, chat_id, content, metadata=None):
        raise RuntimeError("isolated relay failure")

    async def send(chat_id, content, metadata=None):
        raise RuntimeError("isolated live failure")

    relay = SimpleNamespace(send=send, send_for_platform=relay_send,
                            fronts_platform=lambda p: p == Platform.DISCORD)
    job = {"id": "unconfirmed-job", "deliver": "discord:fixture"}

    error = _deliver_result(job, "the brief", adapters={Platform.RELAY: relay}, loop=loop_thread)
    assert error is not None
    assert standalone_env.calls == []
    assert not job.get("last_delivery_unverified")
