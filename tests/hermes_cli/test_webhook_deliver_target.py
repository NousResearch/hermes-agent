"""A webhook delivery target the gateway cannot dispatch is refused up front, never accepted silently."""

import json
from argparse import Namespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import _DYNAMIC_ROUTES_FILENAME, WebhookAdapter
from hermes_cli.webhook import _load_subscriptions, _mutate_subscriptions, webhook_command


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.webhook._is_webhook_enabled", lambda: True)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)


def _args(**kwargs):
    base = dict(webhook_action="subscribe", name="", prompt="", events="", description="", skills="",
                deliver="log", deliver_chat_id="", secret="", route_profile=None, payload="", script="")
    return Namespace(**{**base, **kwargs})


def test_subscribe_rejects_a_target_with_a_chat_suffix(capsys):
    assert webhook_command(_args(name="bad", deliver="telegram/all")) == 1
    assert "Unknown delivery target 'telegram/all'" in capsys.readouterr().out
    assert "bad" not in _load_subscriptions()


@pytest.mark.parametrize("target", ["log", "github_comment", "telegram"])
def test_subscribe_accepts_dispatchable_targets(target):
    webhook_command(_args(name="ok", deliver=target, deliver_chat_id="123"))
    assert _load_subscriptions()["ok"]["deliver"] == target


def test_subscribe_accepts_a_plugin_platform_registered_by_discovery(monkeypatch):
    discovered = []
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: discovered.append(True))
    monkeypatch.setattr("gateway.platform_registry.platform_registry.is_registered",
                        lambda target: bool(discovered) and target == "custom_chat")
    webhook_command(_args(name="plugin", deliver="custom_chat"))
    assert _load_subscriptions()["plugin"]["deliver"] == "custom_chat"


def test_list_flags_a_persisted_unknown_target(capsys):
    _mutate_subscriptions(lambda subs: subs.update(
        broken={"secret": "dynamic-secret", "prompt": "t", "deliver": "telegram/all"}))
    webhook_command(_args(webhook_action="list"))
    out = capsys.readouterr().out
    assert "Deliver: telegram/all (INVALID)" in out
    assert "--deliver telegram --deliver-chat-id <CHAT_ID>" in out


def test_gateway_skips_a_dynamic_route_with_an_unknown_target(tmp_path, caplog):
    (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(json.dumps({
        "broken": {"secret": "dynamic-secret", "prompt": "t", "deliver": "telegram/all"},
        "good": {"secret": "dynamic-secret", "prompt": "t", "deliver": "telegram"}}))
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "global"}))
    adapter._reload_dynamic_routes()
    assert "broken" not in adapter._routes and "good" in adapter._routes
    assert "unknown delivery target 'telegram/all'" in caplog.text


@pytest.mark.parametrize("bad", [[], {"chat": 1}, None])
def test_list_flags_a_non_string_target_instead_of_crashing(bad, capsys):
    _mutate_subscriptions(lambda subs: subs.update(
        odd={"secret": "dynamic-secret", "prompt": "t", "deliver": bad},
        fine={"secret": "dynamic-secret", "prompt": "t", "deliver": "telegram"}))
    webhook_command(_args(webhook_action="list"))
    out = capsys.readouterr().out
    assert "(INVALID)" in out and "fine" in out


def test_list_repair_hint_keeps_the_rest_of_the_route(capsys):
    _mutate_subscriptions(lambda subs: subs.update(
        broken={"secret": "dynamic-secret", "prompt": "t", "deliver": "telegram/all", "deliver_only": True}))
    webhook_command(_args(webhook_action="list"))
    out = capsys.readouterr().out
    # subscribe replaces the whole route, so the hint must say to resupply the existing options.
    assert "--deliver-only" in out and "replaces the whole route" in out
