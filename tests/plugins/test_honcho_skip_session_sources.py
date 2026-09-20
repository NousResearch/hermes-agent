"""``skipSessionSources`` must keep autonomous kanban-worker transcripts out of Honcho entirely."""

from __future__ import annotations

from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider


class _FakeHonchoConfig(SimpleNamespace):
    raw: dict = {}
    host: str = "hermes"

    def resolve_session_name(self, **kwargs):
        return "test-session"


def _configured_hybrid_config(**overrides) -> _FakeHonchoConfig:
    cfg = _FakeHonchoConfig(
        enabled=True,
        api_key=None,
        base_url="http://127.0.0.1:8000",
        recall_mode="hybrid",
        init_on_session_start=False,
        injection_frequency="every-turn",
        context_cadence=1,
        dialectic_cadence=1,
        query_rewrite=False,
        first_turn_base_wait=3.0,
        first_turn_dialectic_wait=2.0,
        dialectic_depth=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        reasoning_level_cap="high",
        context_tokens=None,
        message_max_chars=25000,
        session_strategy="per-directory",
        skip_session_sources=frozenset(),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_kanban_session_source_skips_initialization(monkeypatch):
    """HERMES_SESSION_SOURCE=kanban with skipSessionSources=["kanban"] never starts a Honcho session."""
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    cfg = _configured_hybrid_config(skip_session_sources=frozenset({"kanban"}))
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    provider = HonchoMemoryProvider()
    provider.initialize("session-1", platform="cli")

    assert provider._cron_skipped is True
    assert provider._config is None
    assert provider._writes_enabled() is False
    assert provider.system_prompt_block() == ""


def test_kanban_session_source_without_config_entry_still_ingests(monkeypatch):
    """Default skipSessionSources is empty, so upstream behaviour (ingest everything) is unchanged."""
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    cfg = _configured_hybrid_config(skip_session_sources=frozenset())
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    provider = HonchoMemoryProvider()
    provider.initialize("session-1", platform="cli")

    assert provider._cron_skipped is False
    assert provider._config is cfg


def test_interactive_cli_session_still_ingests_with_skip_configured(monkeypatch):
    """An interactive session (no HERMES_SESSION_SOURCE=kanban) is unaffected by the skip list."""
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    cfg = _configured_hybrid_config(skip_session_sources=frozenset({"kanban"}))
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: cfg,
    )

    provider = HonchoMemoryProvider()
    provider.initialize("session-1", platform="cli")

    assert provider._cron_skipped is False
    assert provider._config is cfg
