"""Tests for the AI_AGENT / HERMES_AGENT harness-attribution env vars.

Port of earendil-works/pi#7493: entry points advertise the agent harness to
child processes via the cross-agent ``AI_AGENT`` standard plus a
Hermes-specific marker, without clobbering an outer harness.
"""

import os

from hermes_cli.main import _advertise_agent_env


class TestAdvertiseAgentEnv:
    def test_sets_both_vars_when_unset(self, monkeypatch):
        monkeypatch.delenv("AI_AGENT", raising=False)
        monkeypatch.delenv("HERMES_AGENT", raising=False)
        _advertise_agent_env()
        assert os.environ["AI_AGENT"] == "hermes"
        assert os.environ["HERMES_AGENT"] == "true"

    def test_does_not_clobber_outer_harness(self, monkeypatch):
        monkeypatch.setenv("AI_AGENT", "pi")
        monkeypatch.delenv("HERMES_AGENT", raising=False)
        _advertise_agent_env()
        assert os.environ["AI_AGENT"] == "pi"
        assert os.environ["HERMES_AGENT"] == "true"

    def test_idempotent(self, monkeypatch):
        monkeypatch.delenv("AI_AGENT", raising=False)
        monkeypatch.delenv("HERMES_AGENT", raising=False)
        _advertise_agent_env()
        _advertise_agent_env()
        assert os.environ["AI_AGENT"] == "hermes"
        assert os.environ["HERMES_AGENT"] == "true"
