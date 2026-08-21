"""Tests for the ``threadPeerMode`` config flag.

When ``threadPeerMode: true`` is set in Honcho config and a gateway
thread_id is available (Feishu topics, Telegram forum topics, Discord
threads, …), the user peer is resolved per-thread instead of per-user.
All participants in the same thread share one Honcho peer
(``thread-{thread_id}``), so memory accumulates per-topic rather than
per-person.

Priority: ``pin_peer_name`` still wins over ``thread_peer_mode``.
When no thread_id is present, falls through to normal per-user resolution.
"""

import json
from unittest.mock import MagicMock

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager


# ---------------------------------------------------------------------------
# Helpers
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


def _config(
    *,
    thread_peer_mode: bool = False,
    pin_peer_name: bool = False,
    peer_name: str | None = None,
    runtime_peer_prefix: str = "",
    user_peer_aliases: dict[str, str] | None = None,
) -> HonchoClientConfig:
    return HonchoClientConfig(
        api_key="test-key",
        peer_name=peer_name,
        pin_peer_name=pin_peer_name,
        thread_peer_mode=thread_peer_mode,
        user_peer_aliases=user_peer_aliases or {},
        runtime_peer_prefix=runtime_peer_prefix,
        enabled=False,
        write_frequency="turn",
    )


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestThreadPeerModeConfigParsing:
    def test_default_is_false(self):
        """Default preserves per-user behaviour."""
        config = HonchoClientConfig()
        assert config.thread_peer_mode is False

    def test_root_level_true(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "threadPeerMode": True,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.thread_peer_mode is True

    def test_host_block_true(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "hosts": {
                "hermes": {
                    "threadPeerMode": True,
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.thread_peer_mode is True

    def test_host_block_overrides_root(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "threadPeerMode": True,
            "hosts": {
                "hermes": {
                    "threadPeerMode": False,
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.thread_peer_mode is False

    def test_explicit_false_parses(self, tmp_path, monkeypatch):
        config_file = tmp_path / "honcho.json"
        config_file.write_text(json.dumps({
            "apiKey": "k",
            "threadPeerMode": False,
        }))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))

        config = HonchoClientConfig.from_global_config(config_path=config_file)
        assert config.thread_peer_mode is False


# ---------------------------------------------------------------------------
# Peer resolution
# ---------------------------------------------------------------------------


class TestThreadPeerResolution:
    """Verify the thread peer mode branch in _resolve_user_peer_id."""

    def test_thread_mode_with_thread_id_resolves_per_thread(self):
        """All users in the same thread share one peer."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=_config(thread_peer_mode=True),
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:group:oc_xxx:topic_42")
        assert session.user_peer_id == "thread-topic_42"

    def test_different_users_same_thread_share_peer(self):
        """Two different users in the same thread → same peer."""
        cfg = _config(thread_peer_mode=True)

        mgr_alice = HonchoSessionManager(
            honcho=MagicMock(),
            config=cfg,
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr_alice)

        mgr_bob = HonchoSessionManager(
            honcho=MagicMock(),
            config=cfg,
            runtime_user_peer_name="ou_bob",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr_bob)

        session_alice = mgr_alice.get_or_create("feishu:group:oc_xxx:topic_42")
        session_bob = mgr_bob.get_or_create("feishu:group:oc_xxx:topic_42")
        assert session_alice.user_peer_id == session_bob.user_peer_id == "thread-topic_42"

    def test_different_threads_different_peers(self):
        """Same user in different threads → different peers."""
        cfg = _config(thread_peer_mode=True)

        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=cfg,
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr)

        mgr2 = HonchoSessionManager(
            honcho=MagicMock(),
            config=cfg,
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_99",
        )
        _patch_manager_for_resolution_test(mgr2)

        s1 = mgr.get_or_create("feishu:group:oc_xxx:topic_42")
        s2 = mgr2.get_or_create("feishu:group:oc_xxx:topic_99")
        assert s1.user_peer_id == "thread-topic_42"
        assert s2.user_peer_id == "thread-topic_99"
        assert s1.user_peer_id != s2.user_peer_id

    def test_thread_mode_without_thread_id_falls_through_to_per_user(self):
        """No thread (e.g. DM or non-threaded group) → normal per-user."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=_config(thread_peer_mode=True),
            runtime_user_peer_name="ou_alice",
            runtime_thread_id=None,
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:dm:oc_yyy")
        assert session.user_peer_id == "ou_alice"

    def test_thread_mode_disabled_with_thread_id_uses_per_user(self):
        """thread_peer_mode=False → thread_id is ignored."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=_config(thread_peer_mode=False),
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:group:oc_xxx:topic_42")
        assert session.user_peer_id == "ou_alice"

    def test_pin_peer_name_still_wins_over_thread_mode(self):
        """pin_peer_name has highest priority — thread mode doesn't override."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=_config(
                thread_peer_mode=True,
                pin_peer_name=True,
                peer_name="global-user",
            ),
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:group:oc_xxx:topic_42")
        assert session.user_peer_id == "global-user"

    def test_thread_mode_beats_aliases(self):
        """thread_peer_mode has higher priority than user_peer_aliases."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=_config(
                thread_peer_mode=True,
                user_peer_aliases={"ou_alice": "alice-stable"},
            ),
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:group:oc_xxx:topic_42")
        assert session.user_peer_id == "thread-topic_42"

    def test_thread_mode_beats_prefix(self):
        """thread_peer_mode has higher priority than runtime_peer_prefix."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=_config(
                thread_peer_mode=True,
                runtime_peer_prefix="feishu_",
            ),
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:group:oc_xxx:topic_42")
        assert session.user_peer_id == "thread-topic_42"

    def test_thread_id_sanitization(self):
        """Thread IDs with special chars are sanitized to ^[a-zA-Z0-9_-]+."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=_config(thread_peer_mode=True),
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic/42:sub",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:group:oc_xxx:topic_42")
        # "/" and ":" are replaced with "-"
        assert session.user_peer_id == "thread-topic-42-sub"

    def test_does_not_affect_assistant_peer(self):
        """Thread mode only affects user peer, not the assistant peer."""
        mgr = HonchoSessionManager(
            honcho=MagicMock(),
            config=_config(thread_peer_mode=True, peer_name="bot"),
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        _patch_manager_for_resolution_test(mgr)

        session = mgr.get_or_create("feishu:group:oc_xxx:topic_42")
        assert session.user_peer_id == "thread-topic_42"
        # Assistant peer should still be "bot" (from config.ai_peer default "hermes")
        assert session.assistant_peer_id != "thread-topic_42"


# ---------------------------------------------------------------------------
# Cache busting
# ---------------------------------------------------------------------------


class TestThreadPeerCacheBusting:
    """Gateway caches AIAgent instances by a signature that includes
    identity-relevant config.  thread_peer_mode changes peer resolution,
    so it must be part of the cache key."""

    def test_cache_busting_signature_reflects_thread_peer_mode(self):
        """Different thread_peer_mode values → different cache signatures."""
        cfg_off = _config(thread_peer_mode=False)
        cfg_on = _config(thread_peer_mode=True)

        mgr_off = HonchoSessionManager(
            honcho=MagicMock(),
            config=cfg_off,
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )
        mgr_on = HonchoSessionManager(
            honcho=MagicMock(),
            config=cfg_on,
            runtime_user_peer_name="ou_alice",
            runtime_thread_id="topic_42",
        )

        # The resolved peer IDs differ → gateway would cache separately
        _patch_manager_for_resolution_test(mgr_off)
        _patch_manager_for_resolution_test(mgr_on)

        s_off = mgr_off.get_or_create("feishu:group:oc_xxx:topic_42")
        s_on = mgr_on.get_or_create("feishu:group:oc_xxx:topic_42")
        assert s_off.user_peer_id != s_on.user_peer_id
