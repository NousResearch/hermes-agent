"""A profile never answers in its own slack.post_gated_channels (education
design 5b): the channel accepts no free text from the bot, so inbound there
is treated like an ignored channel."""
from types import SimpleNamespace

import agent.boundary_config as bc
from gateway.run import GatewayRunner


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False, platforms={})
    return runner


def test_gated_channel_and_its_threads_ignored(monkeypatch):
    monkeypatch.setattr(bc, "read_boundary_config", lambda home=None: {"slack": {"post_gated_channels": ["CREL0001"]}})
    r = _runner()
    assert r._slack_source_gated(SimpleNamespace(chat_id="CREL0001"))
    assert r._slack_source_ignored(SimpleNamespace(chat_id="CREL0001"))
    assert not r._slack_source_gated(SimpleNamespace(chat_id="CHOME001"))


def test_no_policy_no_change(monkeypatch):
    monkeypatch.setattr(bc, "read_boundary_config", lambda home=None: {})
    assert not _runner()._slack_source_gated(SimpleNamespace(chat_id="CREL0001"))


def test_check_failure_ignores(monkeypatch):
    def boom(home=None):
        raise RuntimeError("x")
    monkeypatch.setattr(bc, "read_boundary_config", boom)
    assert _runner()._slack_source_gated(SimpleNamespace(chat_id="CREL0001"))
