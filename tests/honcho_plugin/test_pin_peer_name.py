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

import hashlib
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


class TestRuntimePeerMappingConfigParsing:
    def test_defaults_are_empty(self):
        config = HonchoClientConfig()
        assert config.user_peer_aliases == {}
        assert config.runtime_peer_prefix == ""

    def test_root_level_aliases_and_prefix_parse(self, tmp_path):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "userPeerAliases": {
                " 86701400 ": " Igor ",
                "": "ignored",
                "empty-value": " ",
                "null-value": None,
            },
            "runtimePeerPrefix": "telegram_",
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)

        assert config.user_peer_aliases == {"86701400": "Igor"}
        assert config.runtime_peer_prefix == "telegram_"

    def test_host_aliases_override_root_aliases_as_whole_map(self, tmp_path):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "userPeerAliases": {"root-user": "root-peer"},
            "hosts": {
                "hermes": {
                    "userPeerAliases": {"host-user": "host-peer"},
                },
            },
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)

        assert config.user_peer_aliases == {"host-user": "host-peer"}

    def test_host_empty_aliases_disable_root_aliases(self, tmp_path):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "userPeerAliases": {"root-user": "root-peer"},
            "hosts": {
                "hermes": {
                    "userPeerAliases": {},
                },
            },
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)

        assert config.user_peer_aliases == {}

    def test_host_empty_prefix_disables_root_prefix(self, tmp_path):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "runtimePeerPrefix": "telegram_",
            "hosts": {
                "hermes": {
                    "runtimePeerPrefix": "",
                },
            },
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)

        assert config.runtime_peer_prefix == ""

    def test_malformed_alias_config_is_ignored(self, tmp_path):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "userPeerAliases": ["not", "a", "map"],
        }))

        config = HonchoClientConfig.from_global_config(config_path=config_file)

        assert config.user_peer_aliases == {}


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

    def _config(
        self,
        *,
        peer_name: str | None,
        pin_peer_name: bool,
        user_peer_aliases: dict[str, str] | None = None,
        runtime_peer_prefix: str = "",
        session_peer_prefix: bool = False,
    ) -> HonchoClientConfig:
        # The test doesn't need auth / Honcho — disable the provider so
        # the manager doesn't try to open a real client.
        return HonchoClientConfig(
            api_key="test-key",
            peer_name=peer_name,
            pin_peer_name=pin_peer_name,
            user_peer_aliases=user_peer_aliases or {},
            runtime_peer_prefix=runtime_peer_prefix,
            session_peer_prefix=session_peer_prefix,
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

    def test_alias_wins_for_known_runtime_id(self):
        """Known platform IDs can preserve an existing stable Honcho peer."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                user_peer_aliases={"86701400": "Igor"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "Igor"

    def test_unknown_runtime_id_uses_prefix(self):
        """Unknown gateway users stay isolated but become platform-scoped."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "telegram_86701400"

    def test_prefixed_runtime_id_hashes_when_sanitization_is_lossy(self):
        """Generated prefixed IDs avoid merges caused by lossy sanitization."""
        raw_peer_id = "telegram_user:42"
        expected_hash = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()[:8]
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="user:42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:user:42")
        assert session.user_peer_id == f"telegram_user-42-{expected_hash}"

    def test_prefixed_runtime_id_hashes_when_it_collides_with_peer_name(self):
        """Unknown generated peers should not silently merge into peerName."""
        raw_peer_id = "telegram_86701400"
        expected_hash = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()[:8]
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="telegram_86701400",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == f"telegram_86701400-{expected_hash}"

    def test_prefixed_runtime_id_hashes_when_it_collides_with_alias_target(self):
        """Unknown generated peers should not silently merge into alias targets."""
        raw_peer_id = "telegram_86701400"
        expected_hash = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()[:8]
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={"known-user": "telegram_86701400"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == f"telegram_86701400-{expected_hash}"

    def test_prefixed_runtime_id_extends_hash_when_short_hash_collides(self):
        raw_peer_id = "telegram_86701400"
        digest = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={
                    "known-user": "telegram_86701400",
                    "reserved-user": f"telegram_86701400-{digest[:8]}",
                },
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == f"telegram_86701400-{digest[:12]}"

    def test_alias_value_is_sanitized_after_selection(self):
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={"86701400": "Alice Smith!"},
            ),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "Alice-Smith-"

    def test_alias_keys_match_raw_runtime_id_before_sanitization(self):
        """Alias selection is exact on platform IDs before Honcho ID cleanup."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={
                    "user:42": "raw-match",
                    "user-42": "sanitized-match",
                },
            ),
            runtime_user_peer_name="user:42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:user:42")
        assert session.user_peer_id == "raw-match"

    def test_session_peer_prefix_is_orthogonal_to_runtime_peer_prefix(self):
        """sessionPeerPrefix scopes session IDs; runtimePeerPrefix scopes user peers."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=False,
                runtime_peer_prefix="telegram_",
                session_peer_prefix=True,
            ),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "telegram_86701400"
        assert session.honcho_session_id == "telegram-86701400"

    def test_config_wins_when_pin_is_true(self):
        """The #14984 fix: single-user deployments opt into config pinning."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name="Igor",
                pin_peer_name=True,
                user_peer_aliases={"86701400": "Alias"},
                runtime_peer_prefix="telegram_",
            ),
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
        nothing to pin to — fall through to runtime mapping."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=True,
                user_peer_aliases={"86701400": "Igor"},
                runtime_peer_prefix="telegram_",
            ),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "Igor"

    def test_pin_noop_without_peer_name_or_mapping_preserves_runtime(self):
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(peer_name=None, pin_peer_name=True),
            runtime_user_peer_name="86701400",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("telegram:86701400")
        assert session.user_peer_id == "86701400"

    def test_alt_runtime_id_can_match_alias_without_changing_raw_fallback(self):
        """Stable alternate IDs can map known users while primary ID fallback stays unchanged."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={"union-user": "Igor"},
                runtime_peer_prefix="feishu_",
            ),
            runtime_user_peer_name="open-id",
            runtime_user_peer_name_alt="union-user",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:chat")
        assert session.user_peer_id == "Igor"

    def test_alt_runtime_id_does_not_replace_primary_prefix_fallback(self):
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=self._config(
                peer_name=None,
                pin_peer_name=False,
                user_peer_aliases={"other-union": "Igor"},
                runtime_peer_prefix="feishu_",
            ),
            runtime_user_peer_name="open-id",
            runtime_user_peer_name_alt="union-user",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:chat")
        assert session.user_peer_id == "feishu_open-id"

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
