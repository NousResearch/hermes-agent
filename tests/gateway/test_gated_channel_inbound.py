"""A profile never answers in its own slack.post_gated_channels (education
design 5b): the channel accepts no free text from the bot, so inbound there
is treated like an ignored channel."""
from types import SimpleNamespace

import hermes_cli.config as hc
from gateway.run import GatewayRunner


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False, platforms={})
    return runner


def test_gated_channel_and_its_threads_ignored(monkeypatch):
    monkeypatch.setattr(hc, "read_raw_config", lambda: {"slack": {"post_gated_channels": ["CREL0001"]}})
    r = _runner()
    assert r._slack_source_gated(SimpleNamespace(chat_id="CREL0001"))
    assert r._slack_source_ignored(SimpleNamespace(chat_id="CREL0001"))
    assert not r._slack_source_gated(SimpleNamespace(chat_id="CHOME001"))


def test_no_policy_no_change(monkeypatch):
    monkeypatch.setattr(hc, "read_raw_config", lambda: {})
    assert not _runner()._slack_source_gated(SimpleNamespace(chat_id="CREL0001"))
