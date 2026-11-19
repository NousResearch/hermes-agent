"""Regression test: /config must show the LIVE agent credential.

HermesCLI.__init__ seeds ``self.api_key`` from OPENAI_API_KEY /
OPENROUTER_API_KEY env vars before provider resolution runs. On any
non-OpenAI provider (Nous, Anthropic, ...) the constructor value is a
different vendor's key than the one actually used for requests, so
``/config`` displayed e.g. an ``sk-proj-...`` OpenAI key next to a Nous
base URL. ``show_config`` must prefer ``self.agent.api_key`` when an
agent exists.
"""

from datetime import datetime
from types import SimpleNamespace

from cli import HermesCLI


def _run_show_config(stand_in, capsys):
    HermesCLI.show_config(stand_in)
    return capsys.readouterr().out


def _make_stand_in(cli_key, agent_key):
    agent = SimpleNamespace(api_key=agent_key) if agent_key is not None else None
    return SimpleNamespace(
        api_key=cli_key,
        agent=agent,
        model="test/model",
        base_url="https://example.invalid/v1",
        enabled_toolsets=[],
        max_turns=5,
        verbose=False,
        session_start=datetime(2026, 8, 19, 12, 0, 0),
    )


class TestShowConfigCredentialSource:
    def test_prefers_live_agent_key(self, capsys):
        out = _run_show_config(
            _make_stand_in(cli_key="sk-proj-WRONGVENDORKEY1234", agent_key="nous-REALKEY-abcdef9876"),
            capsys,
        )
        assert "nous-REA" in out
        assert "sk-proj-" not in out

    def test_falls_back_to_cli_key_without_agent(self, capsys):
        out = _run_show_config(
            _make_stand_in(cli_key="sk-proj-CONSTRUCTORKEY5678", agent_key=None),
            capsys,
        )
        assert "sk-proj-" in out
