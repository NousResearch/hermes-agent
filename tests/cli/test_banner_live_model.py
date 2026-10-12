"""show_banner() must display the agent's LIVE model, not the CLI's launch-time self.model.

self.model is set once at CLI startup and never updated; agent.model updates on provider
fallback and mid-session /model switches (the same distinction the status bar already makes
in cli_status_bar_mixin._get_status_bar_snapshot: "Prefer the agent's model name — it updates
on fallback; self.model never changes."). Before this fix, show_banner() read self.model
directly, so a session that fell back from e.g. a local model to a cloud model (or vice versa)
would show the stale, pre-fallback model in the banner forever while the status bar correctly
tracked the switch -- the two surfaces visibly disagreed about what model was actually running.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def _isolate(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))


@pytest.fixture
def cli_obj(_isolate):
    """A minimal HermesCLI instance whose self.model and agent.model deliberately differ."""
    with patch("cli.load_cli_config", return_value={
        "display": {"tool_progress": "new"},
        "terminal": {},
    }), patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.build_welcome_banner"):
        from cli import HermesCLI
        obj = HermesCLI.__new__(HermesCLI)
        obj.model = "launch-time-model"  # set once at startup; must NOT win over agent.model
        obj.enabled_toolsets = ["hermes-core"]
        obj.compact = False
        obj.console = MagicMock()
        obj.session_id = None
        obj.api_key = "test"
        obj.base_url = ""
        obj.provider = "test"
        obj._provider_source = None
        obj.agent = SimpleNamespace(
            model="fallback-model",  # what the agent actually switched to mid-session
            context_compressor=SimpleNamespace(context_length=None),
        )
        return obj


def test_show_banner_uses_live_agent_model_not_stale_self_model(cli_obj):
    """The full (non-compact) banner path must pass agent.model, matching the status bar."""
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.build_welcome_banner") as mock_banner:
        cli_obj.show_banner()

    assert mock_banner.called, "build_welcome_banner was not invoked"
    _, kwargs = mock_banner.call_args
    assert kwargs.get("model") == "fallback-model", (
        "banner must show the agent's live model (post-fallback), not the stale "
        f"launch-time self.model; got {kwargs.get('model')!r}")


def test_show_banner_falls_back_to_self_model_when_agent_has_none(cli_obj):
    """No live agent.model yet (e.g. agent not constructed) -> self.model is still the answer."""
    cli_obj.agent = SimpleNamespace(context_compressor=SimpleNamespace(context_length=None))

    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.build_welcome_banner") as mock_banner:
        cli_obj.show_banner()

    _, kwargs = mock_banner.call_args
    assert kwargs.get("model") == "launch-time-model"
