"""LOCAL PATCH 28 regression tests: _reload_dm_topics_from_config must honor
the top-level ``telegram:`` config layout, not only ``platforms.telegram``.

Poison scenario these lock in: with a top-level layout, a message into an
ad-hoc (non-configured) topic triggers the hot-reload; the old code read only
``platforms.telegram.extra.dm_topics`` (empty), then CLEARED
``_dm_topics_config`` — silently dropping skill bindings for every operator
topic until the next gateway restart.
"""

import types
from unittest.mock import MagicMock, patch

import pytest


class _FakeAdapter:
    """Minimal stand-in exposing the reload method under test."""

    def __init__(self):
        self.name = "telegram-test"
        self._dm_topics_config: list = []
        self._dm_topic_chat_ids: set = set()
        self._dm_topics: dict = {}
        self._reload = None  # bound later from the real adapter

    def reload(self):
        assert self._reload is not None
        return self._reload()


TOP_LEVEL_CFG = {
    "telegram": {
        "extra": {
            "dm_topics": [
                {
                    "chat_id": 8806247320,
                    "topics": [
                        {"name": "BaciScout", "thread_id": 13099,
                         "skill": ["project-baciscout", "autonomous-project-orchestrator"]},
                        {"name": "BaciCheck", "thread_id": 25664,
                         "skill": ["project-bacicheck", "autonomous-project-orchestrator"]},
                    ],
                }
            ]
        }
    }
}

NESTED_CFG = {
    "platforms": {
        "telegram": {
            "extra": {
                "dm_topics": [
                    {
                        "chat_id": 111,
                        "topics": [{"name": "Nested", "thread_id": 222, "skill": "s"}],
                    }
                ]
            }
        }
    }
}

BOTH_CFG = {
    "telegram": {"extra": {"dm_topics": [
        {"chat_id": 1, "topics": [{"name": "TopWins", "thread_id": 2, "skill": "top"}]},
    ]}},
    "platforms": {"telegram": {"extra": {"dm_topics": [
        {"chat_id": 3, "topics": [{"name": "NestedLoses", "thread_id": 4, "skill": "nested"}]},
    ]}}},
}


def _make_adapter(monkeypatch, cfg):
    from plugins.platforms.telegram import adapter as telegram_adapter

    fake = _FakeAdapter()
    fake._reload = telegram_adapter.TelegramAdapter._reload_dm_topics_from_config.__get__(fake)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: cfg,
    )
    return fake


class TestReloadDmTopicsTopLevelLayout:
    def test_top_level_layout_is_honored_not_cleared(self, monkeypatch):
        """The poison scenario: top-level layout must NOT clear topic config."""
        fake = _make_adapter(monkeypatch, TOP_LEVEL_CFG)
        # simulate state loaded at startup
        fake._dm_topics_config = TOP_LEVEL_CFG["telegram"]["extra"]["dm_topics"]
        fake._dm_topics = {"8806247320:BaciScout": 13099}

        fake.reload()

        assert fake._dm_topics_config, "topic config was cleared on top-level layout (poison bug)"
        assert fake._dm_topic_chat_ids == {"8806247320"}

    def test_top_level_layout_caches_new_thread_ids(self, monkeypatch):
        fake = _make_adapter(monkeypatch, TOP_LEVEL_CFG)
        fake._dm_topics = {}

        fake.reload()

        assert fake._dm_topics.get("8806247320:BaciCheck") == 25664
        assert fake._dm_topics.get("8806247320:BaciScout") == 13099

    def test_skill_lookup_survives_adhoc_topic_reload(self, monkeypatch):
        """End-to-end poison repro: after reload, _get_dm_topic_info must still
        return the skill for a cached topic (old code returned name-only)."""
        from plugins.platforms.telegram import adapter as telegram_adapter

        fake = _make_adapter(monkeypatch, TOP_LEVEL_CFG)
        fake._dm_topics_config = TOP_LEVEL_CFG["telegram"]["extra"]["dm_topics"]
        fake._dm_topics = {"8806247320:BaciCheck": 25664}

        info = telegram_adapter.TelegramAdapter._get_dm_topic_info(
            fake, chat_id="8806247320", thread_id="25664"
        )
        assert isinstance(info, dict)
        assert info.get("skill"), (
            "topic resolved without skill after hot-reload (poison bug): %r" % (info,)
        )


class TestReloadDmTopicsNestedLayout:
    def test_nested_layout_still_works(self, monkeypatch):
        fake = _make_adapter(monkeypatch, NESTED_CFG)
        fake._dm_topics = {}

        fake.reload()

        assert "111:Nested" in fake._dm_topics
        assert fake._dm_topic_chat_ids == {"111"}


class TestReloadDmTopicsPrecedence:
    def test_top_level_wins_when_both_present(self, monkeypatch):
        fake = _make_adapter(monkeypatch, BOTH_CFG)
        fake._dm_topics = {}

        fake.reload()

        assert fake._dm_topics.get("1:TopWins") == 2
        assert "3:NestedLoses" not in fake._dm_topics


class TestReloadDmTopicsEmpty:
    def test_truly_empty_config_still_clears(self, monkeypatch):
        """Legitimate clear behavior is preserved: config with NO topics
        anywhere must still clear in-memory state."""
        fake = _make_adapter(monkeypatch, {})
        fake._dm_topics_config = [{"chat_id": 9, "topics": []}]
        fake._dm_topics = {"9:Old": 99}

        fake.reload()

        assert fake._dm_topics_config == []
        assert fake._dm_topic_chat_ids == set()
