"""Tests for the ``pinPeerName`` config flag (#14984).

By default, when Hermes runs under a gateway (Telegram, Discord, Slack, ...)
it passes the platform-native user ID as ``runtime_user_peer_name`` into
``HonchoSessionManager``.  That ID wins over any configured ``peer_name``
so multi-user bots scope memory per user.

For a single-user personal deployment where the user connects over multiple
platforms, that default forks memory into one Honcho peer per platform
(Telegram UID, Discord snowflake, Slack user ID, ...).  The user asked for
an opt-in knob that pins the user peer to ``peer_name`` from ``honcho.json``
so the same person's memory stays unified regardless of which platform the
turn arrived on — ``hosts.<host>.pinPeerName: true`` (or root-level
``pinPeerName: true``).

These tests exercise both the config parsing (``client.py::from_global_config``)
and the resolution order (``session.py::get_or_create``).  We stub the
Honcho API calls so we can assert the chosen ``user_peer_id`` without
touching the network.
"""

import json
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestPinPeerNameConfigParsing:
    def test_default_is_false(self):
        """Default preserves existing behaviour — multi-user bots unaffected."""
        config = HonchoClientConfig()
        assert config.pin_peer_name is False

    def test_root_level_true(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "pinPeerName": True,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is True
        assert config.peer_name == "Igor"

    def test_host_block_true(self, tmp_path, monkeypatch):
        """Host-level flag works the same as root-level."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "hosts": {
                "hermes": {"pinPeerName": True},
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is True

    def test_host_block_overrides_root(self, tmp_path, monkeypatch):
        """Host block wins over root — matches how every other flag behaves."""
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "pinPeerName": True,
            "hosts": {
                "hermes": {"pinPeerName": False},
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is False, (
            "host-level pinPeerName=false must override root-level true, the "
            "same way every other flag in this config is resolved"
        )

    def test_explicit_false_parses(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "peerName": "Igor",
            "pinPeerName": False,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.pin_peer_name is False


# ---------------------------------------------------------------------------
# Peer resolution (the actual bug fix)
# ---------------------------------------------------------------------------


def _patch_manager_for_resolution_test(mgr: HonchoSessionManager) -> None:
    """Stub out the Honcho client so ``get_or_create`` doesn't try to talk
    to the network — we only care about the user_peer_id chosen before
    those calls happen.
    """
    fake_peer = MagicMock()
    mgr._get_or_create_peer = MagicMock(return_value=fake_peer)
    mgr._get_or_create_honcho_session = MagicMock(
        return_value=(MagicMock(), [])
    )


class TestPeerResolutionOrder:
    """Matrix of (runtime_id, pin_peer_name, peer_name) → expected user_peer_id."""

    def _config(self, *, peer_name: str | None, pin_peer_name: bool) -> HonchoClientConfig:
        # The test doesn't need auth / Honcho — disable the provider so
        # the manager doesn't try to open a real client.
        return HonchoClientConfig(
            api_key="test-key",
            peer_name=peer_name,
            pin_peer_name=pin_peer_name,
            enabled=False,
            write_frequency="turn",  # avoid spawning the async writer thread
        )

    def test_runtime_wins_when_pin_is_false(self):
        """Regression guard: default behaviour must stay unchanged.
        Multi-user bots rely on the platform-native ID winning."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name="Igor", pin_peer_name=False),
            runtime_user_peer_name="86701400",  # e.g. Telegram UID
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "86701400", (
            "pin_peer_name=False is the multi-user default — the gateway's "
            "platform-native user ID must win so each user gets their own "
            "peer scope.  If this regresses, every Telegram/Discord/Slack "
            "bot immediately merges memory across users."
        )

    def test_config_wins_when_pin_is_true(self):
        """The #14984 fix: single-user deployments opt into config pinning."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name="Igor", pin_peer_name=True),
            runtime_user_peer_name="86701400",  # Telegram pushes this in
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "Igor", (
            "With pinPeerName=true the user's configured peer_name must "
            "beat the platform-native runtime ID so memory stays unified "
            "across Telegram/Discord/Slack for the same person."
        )

    def test_pin_noop_when_peer_name_missing(self):
        """Safety: pinPeerName alone (no peer_name) must not silently drop
        the runtime identity.  Without a configured peer_name there's
        nothing to pin to — fall back to runtime as before."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name=None, pin_peer_name=True),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "86701400", (
            "pin_peer_name=True with no peer_name set must not strip the "
            "runtime ID — otherwise the user peer would collapse to the "
            "session-key fallback and lose per-user scoping entirely"
        )

    def test_runtime_missing_falls_back_to_peer_name(self):
        """CLI-mode (no gateway runtime identity) uses config peer_name —
        this path was already correct but the refactor shouldn't break it."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name="Igor", pin_peer_name=False),
            runtime_user_peer_name=None,
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("cli:local")
        assert session.user_peer_id == "Igor"

    def test_everything_missing_falls_back_to_session_key(self):
        """Deepest fallback: no runtime identity, no peer_name, no pin.
        Must still produce a deterministic peer_id from the session key."""
        # Config with no peer_name and default pin_peer_name=False
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name=None, pin_peer_name=False),
            runtime_user_peer_name=None,
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:123")
        assert session.user_peer_id == "user-telegram-123"

    def test_pin_does_not_affect_assistant_peer(self):
        """The flag only pins the USER peer — the assistant peer continues
        to come from ``ai_peer`` and must not be touched."""
        cfg = HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=True,
            ai_peer="hermes-assistant",
            enabled=False,
            write_frequency="turn",
        )
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=cfg,
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "Igor"
        assert session.assistant_peer_id == "hermes-assistant"


class TestCrossPlatformMemoryUnification:
    """The user-visible outcome of the #14984 fix: the same physical user
    talking to Hermes via Telegram AND Discord should land on ONE peer
    (not two) when pinPeerName is opted in.
    """

    def _config_pinned(self) -> HonchoClientConfig:
        return HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=True,
            enabled=False,
            write_frequency="turn",
        )

    def test_telegram_and_discord_collapse_to_one_peer_when_pinned(self):
        """Single-user deployment: Telegram UID and Discord snowflake
        both resolve to the same configured peer_name."""
        # Telegram turn
        mgr_telegram = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config_pinned(),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr_telegram)
        telegram_session = mgr_telegram.get_or_create("telegram:86701400")

        # Discord turn (separate manager instance — simulates a fresh
        # platform-adapter invocation)
        mgr_discord = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config_pinned(),
            runtime_user_peer_name="1348750102029926454",
        )
        _patch_manager_for_resolution_test(mgr_discord)
        discord_session = mgr_discord.get_or_create("discord:1348750102029926454")

        assert telegram_session.user_peer_id == "Igor"
        assert discord_session.user_peer_id == "Igor"
        assert telegram_session.user_peer_id == discord_session.user_peer_id, (
            "cross-platform memory unification is the whole point of "
            "pinPeerName — both platforms must land on the same Honcho peer"
        )

    def test_multiuser_default_keeps_platforms_separate(self):
        """Negative control: with pinPeerName=false (the default), two
        different platform IDs must produce two different peers so
        multi-user bots don't merge users."""
        cfg = HonchoClientConfig(
            api_key="k",
            peer_name="Igor",
            pin_peer_name=False,
            enabled=False,
            write_frequency="turn",
        )
        mgr_a = HonchoSessionManager(
            honcho=MagicMock(), config=cfg, runtime_user_peer_name="user_a",
        )
        mgr_b = HonchoSessionManager(
            honcho=MagicMock(), config=cfg, runtime_user_peer_name="user_b",
        )
        _patch_manager_for_resolution_test(mgr_a)
        _patch_manager_for_resolution_test(mgr_b)

        sess_a = mgr_a.get_or_create("telegram:a")
        sess_b = mgr_b.get_or_create("telegram:b")

        assert sess_a.user_peer_id == "user_a"
        assert sess_b.user_peer_id == "user_b"
        assert sess_a.user_peer_id != sess_b.user_peer_id, (
            "multi-user default MUST keep users separate — a regression "
            "here would silently merge unrelated users' memory"
        )
