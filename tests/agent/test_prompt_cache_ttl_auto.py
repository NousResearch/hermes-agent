"""``prompt_caching.cache_ttl: auto`` picks the Anthropic cache tier by who paces the session.

The 1h tier writes at 2x base (5m: 1.25x) and only pays off across >5-minute pauses, i.e.
for a person; machine-paced sessions (subagent, cron, oneshot, webhook, ...) never collect the
retention. ``auto`` resolves once per session in ``_init_prompt_cache_config``.
"""

from agent.prompt_caching import MACHINE_PACED_SOURCES, auto_cache_ttl_for_source


def test_auto_tier_follows_session_pace():
    assert auto_cache_ttl_for_source("cli") == "1h"
    assert auto_cache_ttl_for_source("telegram") == "1h"
    assert auto_cache_ttl_for_source(None) == "1h", "an unknown/blank source is a person until proven otherwise"
    for source in MACHINE_PACED_SOURCES:
        assert auto_cache_ttl_for_source(source) == "5m", source


def test_real_agent_resolves_auto_from_its_platform(tmp_path, monkeypatch):
    """A real AIAgent under ``cache_ttl: auto``: a CLI session lands on 1h, a cron session on 5m, and
    caching itself stays enabled on both (auto is a tier choice, never a disable)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    (tmp_path / "config.yaml").write_text(
        "prompt_caching:\n  cache_ttl: auto\nmodel:\n  default: anthropic/claude-sonnet-4.6\n", encoding="utf-8")
    from run_agent import AIAgent
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
              api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", quiet_mode=True,
              skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    interactive = AIAgent(session_id="p-cli", platform="cli", **kw)
    scheduled = AIAgent(session_id="p-cron", platform="cron", **kw)
    try:
        assert interactive._cache_ttl == "1h"
        assert scheduled._cache_ttl == "5m"
        assert interactive._use_prompt_caching and scheduled._use_prompt_caching
        assert not interactive._cache_disabled and not scheduled._cache_disabled
    finally:
        for a in (interactive, scheduled):
            close = getattr(a, "close", None)
            if callable(close):
                close()
