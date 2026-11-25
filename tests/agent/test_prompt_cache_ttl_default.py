"""The shipped ``prompt_caching.cache_ttl`` default is ``auto``: with no explicit setting a session a
person types into (CLI, Telegram, ...) starts on 5m and moves to the 1h Anthropic cache tier at its first
pause over five minutes; a machine-paced one (cron, subagent, the API server, the skill curator) stays on 5m.
"""

from pathlib import Path

import hermes_yaml as yaml
import pytest

from agent.prompt_caching import AUTO_CACHE_TTL, auto_cache_ttl_for_source

_TEMPLATE = Path(__file__).resolve().parents[2] / "cli-config.yaml.example"


def test_shipped_default_and_seeded_template_are_auto():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["prompt_caching"]["cache_ttl"] == AUTO_CACHE_TTL
    template = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    assert "cache_ttl" not in (template.get("prompt_caching") or {}), (
        "installers copy the template into config.yaml, so any value there pins every fresh install to it")


def test_background_platform_labels_are_machine_paced():
    # The API server's agents carry platform "api_server" and the curator's "curator"; neither has a
    # person between calls, so auto keeps them on the cheaper 5m write tier.
    assert auto_cache_ttl_for_source("api_server") == "5m"
    assert auto_cache_ttl_for_source("curator") == "5m"


@pytest.mark.parametrize(
    "platform, side_agent, expected",
    [("cli", False, True), ("tui", False, True), ("desktop", False, True),
     ("acp", False, True), ("telegram", False, True), ("cron", False, False), ("subagent", False, False),
     ("api_server", False, False), ("cli", True, False), ("tui", True, False)],
)
def test_unconfigured_agent_resolves_tier_from_its_platform(tmp_path, monkeypatch, platform, side_agent, expected):
    """A real AIAgent on a config.yaml with no ``prompt_caching`` section at all. A side agent
    (CLI/TUI ``/background``, ``/btw``) runs one finite task under its parent's transport label."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "config.yaml").write_text("model:\n  default: anthropic/claude-sonnet-4.6\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        session_id=f"default-{platform}-{side_agent}", platform=platform, side_agent=side_agent, api_key="k", base_url="https://openrouter.ai/api/v1",
        provider="openrouter", api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", quiet_mode=True,
        skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    try:
        # Every session starts on 5m; only a person's chat is marked to upgrade to 1h at its first pause.
        assert (agent._cache_ttl, bool(getattr(agent, "_cache_ttl_adaptive", False))) == ("5m", expected)
        assert agent._use_prompt_caching and not agent._cache_disabled
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()
