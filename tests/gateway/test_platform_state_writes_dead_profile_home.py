"""Platform state writers bound to a profile home that ``hermes profile delete``/``rename``
already removed skip their write instead of recreating the home — and still write for a live one.

Discord's slash-command sync pacing state and the Google Chat bot-id cache are both best-effort:
the refusal must not raise into the adapter's sync/identity path."""

from __future__ import annotations

import json

import pytest

from gateway.config import PlatformConfig


def _discord_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    return DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))


def _google_chat_adapter():
    from plugins.platforms.google_chat.adapter import GoogleChatAdapter

    cfg = PlatformConfig(enabled=True)
    cfg.extra.update({
        "project_id": "test-project",
        "subscription_name": "projects/test-project/subscriptions/test-sub",
        "service_account_json": "/tmp/fake-sa.json",
    })
    return GoogleChatAdapter(cfg)


@pytest.mark.parametrize("move", ("delete", "rename"))
def test_discord_sync_state_refuses_a_dead_profile_home(tmp_path, monkeypatch, move):
    from tests.fixtures.dead_profile_home import dead_named_home

    home = dead_named_home(tmp_path / "dead", move)
    monkeypatch.setenv("HERMES_HOME", str(home))

    _discord_adapter()._write_command_sync_state({"app": {"fingerprint": "f"}})

    assert not home.exists()


@pytest.mark.parametrize("move", ("delete", "rename"))
def test_google_chat_bot_id_cache_refuses_a_dead_profile_home(tmp_path, monkeypatch, move):
    from tests.fixtures.dead_profile_home import dead_named_home

    home = dead_named_home(tmp_path / "dead", move)
    monkeypatch.setenv("HERMES_HOME", str(home))

    _google_chat_adapter()._save_cached_bot_id("bot-1")

    assert not home.exists()


def test_both_writers_still_persist_for_a_live_profile_home(tmp_path, monkeypatch):
    home = tmp_path / "live" / ".hermes" / "profiles" / "b"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    _discord_adapter()._write_command_sync_state({"app": {"fingerprint": "f"}})
    _google_chat_adapter()._save_cached_bot_id("bot-1")

    state = json.loads((home / "gateway" / "discord_command_sync_state.json").read_text(encoding="utf-8"))
    assert state == {"app": {"fingerprint": "f"}}
    assert json.loads((home / "google_chat_bot_id.json").read_text(encoding="utf-8")) == {"bot_user_id": "bot-1"}