"""Cron delivery targets and home-channel broadcasts per account — #8287.

Cron ``deliver`` strings can address a named bot (``telegram@support:123``),
and gateway home-channel broadcasts (startup/shutdown notices) reach every
account's own home channel, not just the platform default's.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from cron.scheduler import _resolve_single_delivery_target
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig


@pytest.fixture()
def runner(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return gateway_run.GatewayRunner(GatewayConfig())


# ── Cron deliver-string parsing ────────────────────────────────────────────


def test_cron_target_parses_account_scoped_chat():
    target = _resolve_single_delivery_target({}, "telegram@support:123456")
    assert target == {
        "platform": "telegram",
        "chat_id": "123456",
        "thread_id": None,
        "account": "support",
    }


def test_cron_target_account_name_is_lowercased():
    target = _resolve_single_delivery_target({}, "telegram@SUPPORT:123456")
    assert target["account"] == "support"


def test_cron_target_plain_form_carries_no_account():
    target = _resolve_single_delivery_target({}, "telegram:123456")
    assert "account" not in target
    assert target["platform"] == "telegram"


def test_cron_target_account_with_thread():
    target = _resolve_single_delivery_target({}, "telegram@sales:-100777:42")
    assert target["account"] == "sales"
    assert target["chat_id"] == "-100777"
    assert target["thread_id"] == "42"


# ── Home-channel broadcast iteration ───────────────────────────────────────


def test_broadcast_iter_includes_account_homes(runner):
    default_adapter = MagicMock()
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="111", name="Home"),
    )

    support_adapter = MagicMock()
    support_adapter.config = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="222", name="Support Home"),
    )
    runner._account_adapters = {Platform.TELEGRAM: {"support": support_adapter}}

    entries = list(runner._iter_live_adapters_with_home(snapshot=True))
    by_adapter = {id(adapter): home for _p, adapter, home in entries}

    assert len(entries) == 2
    assert by_adapter[id(default_adapter)].chat_id == "111"
    assert by_adapter[id(support_adapter)].chat_id == "222"


def test_broadcast_iter_tolerates_account_without_home(runner):
    runner.adapters = {}
    bare = MagicMock()
    bare.config = PlatformConfig(enabled=True)  # no home_channel
    runner._account_adapters = {Platform.TELEGRAM: {"support": bare}}
    entries = list(runner._iter_live_adapters_with_home())
    assert len(entries) == 1
    assert entries[0][2] is None  # callers skip home-less adapters
