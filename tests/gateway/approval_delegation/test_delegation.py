"""Tests for gateway.approval_delegation — native approval delegation."""

import asyncio
import threading
import time
from unittest.mock import patch, MagicMock

import pytest


# ── Config tests ────────────────────────────────────────────────────────


class TestDelegationConfig:
    """Test config loading and admin lookup."""

    def _reset_config(self):
        """Reset the cached config between tests."""
        import gateway.approval_delegation as mod
        mod._delegation_config = None

    def test_delegation_disabled_by_default(self):
        """Delegation is disabled when config has no delegation section."""
        self._reset_config()
        with patch("hermes_cli.config.load_config", return_value={}):
            from gateway.approval_delegation import is_delegation_enabled
            assert is_delegation_enabled() is False

    def test_delegation_enabled(self):
        """Delegation is enabled when config says so."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1"}
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import is_delegation_enabled, get_admins
            assert is_delegation_enabled() is True
            admins = get_admins()
            assert len(admins) == 1
            assert admins[0]["platform"] == "feishu"
            assert admins[0]["user_id"] == "admin1"

    def test_admin_user_detection(self):
        """is_admin_user correctly identifies admins."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1"},
                        {"platform": "weixin", "user_id": "admin2"},
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import is_admin_user
            assert is_admin_user("feishu", "admin1") is True
            assert is_admin_user("weixin", "admin2") is True
            assert is_admin_user("feishu", "regular_user") is False
            assert is_admin_user("telegram", "admin1") is False

    def test_admin_chat_id_defaults_to_user_id(self):
        """chat_id defaults to user_id when not specified."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1"}
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import get_admins
            admins = get_admins()
            assert admins[0]["chat_id"] == "admin1"

    def test_admin_explicit_chat_id(self):
        """chat_id can be explicitly set."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1", "chat_id": "oc_123"}
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import get_admins
            admins = get_admins()
            assert admins[0]["chat_id"] == "oc_123"

    def test_multiple_admins(self):
        """Multiple admins are loaded correctly."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1"},
                        {"platform": "weixin", "user_id": "admin2"},
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import get_admins
            admins = get_admins()
            assert len(admins) == 2
            assert admins[0]["platform"] == "feishu"
            assert admins[1]["platform"] == "weixin"


# ── Delegation state tests ──────────────────────────────────────────────


class TestDelegationState:
    """Test delegation registration, resolution, and cleanup."""

    def setup_method(self):
        """Clear delegation state before each test."""
        from gateway.approval_delegation import clear_all_delegations
        clear_all_delegations()

    def test_register_and_resolve(self):
        """Can register and resolve a delegation."""
        from gateway.approval_delegation import register_delegation, resolve_delegation

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_123",
            session_key="sk_abc",
            user_platform="weixin",
            user_chat_id="wx_user",
            command="rm -rf /",
            description="destructive delete",
        )

        entry = resolve_delegation("feishu", "oc_123")
        assert entry is not None
        assert entry["session_key"] == "sk_abc"
        assert entry["user_platform"] == "weixin"
        assert entry["user_chat_id"] == "wx_user"
        assert entry["command"] == "rm -rf /"

    def test_resolve_nonexistent(self):
        """Resolving a nonexistent delegation returns None."""
        from gateway.approval_delegation import resolve_delegation
        assert resolve_delegation("feishu", "nonexistent") is None

    def test_clear_delegation(self):
        """Clearing a delegation removes it."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation, clear_delegation,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_123",
            session_key="sk_abc",
            user_platform="weixin",
            user_chat_id="wx_user",
        )

        clear_delegation("feishu", "oc_123", session_key="sk_abc")
        assert resolve_delegation("feishu", "oc_123") is None

    def test_stale_delegation_expires(self):
        """Delegations older than TTL are automatically pruned."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation, _DELEGATION_TTL,
            _delegation_map,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_123",
            session_key="sk_abc",
            user_platform="weixin",
            user_chat_id="wx_user",
        )

        # Manually age the entry
        _delegation_map["feishu:oc_123"]["sk_abc"]["created_at"] = time.monotonic() - _DELEGATION_TTL - 1

        assert resolve_delegation("feishu", "oc_123") is None

    def test_concurrent_delegations_to_same_admin(self):
        """Multiple concurrent delegations to the same admin coexist."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_admin",
            session_key="sk_1",
            user_platform="weixin",
            user_chat_id="wx_1",
            command="cmd1",
        )
        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_admin",
            session_key="sk_2",
            user_platform="telegram",
            user_chat_id="tg_2",
            command="cmd2",
        )

        # resolve_delegation returns the most recent
        entry = resolve_delegation("feishu", "oc_admin")
        assert entry is not None
        assert entry["session_key"] == "sk_2"

    def test_concurrent_delegations_independent_clear(self):
        """Clearing one delegation doesn't affect others."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation, clear_delegation,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_admin",
            session_key="sk_1",
            user_platform="weixin",
            user_chat_id="wx_1",
        )
        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_admin",
            session_key="sk_2",
            user_platform="telegram",
            user_chat_id="tg_2",
        )

        clear_delegation("feishu", "oc_admin", session_key="sk_1")

        # sk_2 should still be there
        entry = resolve_delegation("feishu", "oc_admin")
        assert entry is not None
        assert entry["session_key"] == "sk_2"

    def test_clear_all_delegations(self):
        """clear_all_delegations removes everything."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation, clear_all_delegations,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_111",
            session_key="sk_1",
            user_platform="weixin",
            user_chat_id="wx_1",
        )
        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_222",
            session_key="sk_2",
            user_platform="telegram",
            user_chat_id="tg_2",
        )

        clear_all_delegations()
        assert resolve_delegation("feishu", "oc_111") is None
        assert resolve_delegation("feishu", "oc_222") is None

    def test_multiple_admins_different_platforms(self):
        """Delegations to admins on different platforms are independent."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_feishu",
            session_key="sk_1",
            user_platform="weixin",
            user_chat_id="wx_1",
        )
        register_delegation(
            admin_platform="telegram",
            admin_chat_id="tg_admin",
            session_key="sk_2",
            user_platform="weixin",
            user_chat_id="wx_2",
        )

        e1 = resolve_delegation("feishu", "oc_feishu")
        e2 = resolve_delegation("telegram", "tg_admin")
        assert e1["session_key"] == "sk_1"
        assert e2["session_key"] == "sk_2"


class TestAdminIdentityGateDetection:
    """N1 (adversarial review): delegation button UX must only route to
    adapters whose click callbacks enforce the admin identity gate.

    Upstream made ``send_exec_approval`` a base template method that always
    accepts ``admin_user_id`` and renders via ``_send_exec_approval_prompt``,
    so a signature probe no longer distinguishes enforcing adapters from ones
    that ignore the field. Enforcement is now an explicit opt-in marker
    (``_enforces_delegation_admin_identity``); anything without it — including
    test doubles, duck types, and upstream-native qqbot/whatsapp buttons —
    falls back to typed /approve (is_admin_user). Fail-closed on detection
    errors either way.
    """

    def _gate(self, cls):
        from gateway.run_turn_runner import TurnRunner
        return TurnRunner._adapter_has_admin_identity_gate(cls)

    def test_plain_class_has_no_gate(self):
        class Bare:
            async def send_exec_approval(self, chat_id, admin_user_id=None):
                pass
        assert self._gate(Bare) is False

    def test_marker_class_has_gate(self):
        class Marked:
            _enforces_delegation_admin_identity = True
        assert self._gate(Marked) is True

    def test_no_send_exec_approval_is_fail_closed(self):
        class NoBtn:
            pass
        assert self._gate(NoBtn) is False

    def test_instance_and_class_both_accepted(self):
        class Marked:
            _enforces_delegation_admin_identity = True
        assert self._gate(Marked) is True
        assert self._gate(Marked()) is True

    def test_base_default_is_no_gate(self):
        from gateway.platforms.base import BasePlatformAdapter
        assert self._gate(BasePlatformAdapter) is False

    def test_qqbot_and_whatsapp_native_buttons_gated_out(self):
        # Upstream-native renderers never set the marker: they ignore
        # prompt.admin_user_id, so delegation must not send them buttons.
        from gateway.platforms.qqbot.adapter import QQAdapter
        from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
        assert self._gate(QQAdapter) is False
        assert self._gate(WhatsAppCloudAdapter) is False

    def test_delegation_adapters_keep_their_gate(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter
        from plugins.platforms.slack.adapter import SlackAdapter
        from plugins.platforms.feishu.adapter import FeishuAdapter
        from plugins.platforms.wecom.adapter import WeComAdapter
        from plugins.platforms.discord.adapter import DiscordAdapter
        from plugins.platforms.matrix.adapter import MatrixAdapter
        from plugins.platforms.teams.adapter import TeamsAdapter
        for cls in (TelegramAdapter, SlackAdapter, FeishuAdapter, WeComAdapter,
                    DiscordAdapter, MatrixAdapter, TeamsAdapter):
            assert self._gate(cls) is True, cls.__name__
            assert cls.supports_delegation_admin_gate() is True, cls.__name__


class TestDelegatedAwaySelfApproveGate:
    """Adversarial-review finding (first-principles pass over the whole
    delegation feature): the admin redirect re-routes the *notification* but
    the agent thread keeps blocking on its LOCAL queue entry, and the
    delegation map is keyed by the admin's chat — so the local typed-/approve
    path used to let the regular user self-resolve the approval that was
    delegated away from them.  ``_handle_approve_command`` now gates on
    ``is_session_delegated_away``; these tests lock the gate AND its edges
    (redirect-failed fallback keeps self-approve; admins keep authority;
    /deny stays open)."""

    CFG = {"approvals": {"delegation": {"enabled": True, "admins": [
        {"platform": "feishu", "user_id": "admin1", "chat_id": "oc_admin"},
        {"platform": "weixin", "user_id": "o9admin", "chat_id": "o9admin"}]}}}

    def _reset(self):
        import gateway.approval_delegation as ad
        ad._delegation_config = None
        ad._delegation_map.clear()
        from tools import approval as mod
        for d in (mod._gateway_queues, mod._gateway_notify_cbs, mod._session_approved,
                  mod._permanent_approved, mod._pending):
            d.clear()

    def _isolated_home(self, tmp_path, monkeypatch):
        # i18n catalog and PM env scans read the home at call time; point both
        # the home and the runtime dir at the fixture so the real-home I/O
        # guard never fires (this repo lives under ~/.hermes/scratch).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
        import agent.i18n as i18n
        i18n.reset_language_cache()

    # ── map-level ────────────────────────────────────────────────────────

    def test_is_session_delegated_away_hit_and_miss(self):
        self._reset()
        with patch("hermes_cli.config.load_config", return_value=self.CFG):
            from gateway.approval_delegation import (
                register_delegation, is_session_delegated_away)
            register_delegation(admin_platform="feishu", admin_chat_id="oc_admin",
                                session_key="sk_u", user_platform="weixin", user_chat_id="u1")
            assert is_session_delegated_away("sk_u") is True
            assert is_session_delegated_away("sk_other") is False
            assert is_session_delegated_away("") is False

    def test_is_session_delegated_away_expires(self):
        self._reset()
        with patch("hermes_cli.config.load_config", return_value=self.CFG):
            from gateway.approval_delegation import (
                register_delegation, is_session_delegated_away)
            import gateway.approval_delegation as ad
            register_delegation(admin_platform="feishu", admin_chat_id="oc_admin",
                                session_key="sk_u", user_platform="weixin", user_chat_id="u1")
            ad._delegation_map["feishu:oc_admin"]["sk_u"]["created_at"] = \
                time.monotonic() - ad._DELEGATION_TTL - 1
            assert is_session_delegated_away("sk_u") is False

    # ── handler-level regression (the PoC itself) ────────────────────────

    SK = "agent:main:weixin:dm:o9user"

    def _runner_for(self, user_id):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.platforms.event import MessageEvent
        from gateway.session import SessionSource
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={Platform.WEIXIN: PlatformConfig(enabled=True)})
        runner.adapters = {}
        runner._pending_approvals = {}
        runner._background_tasks = set()
        runner.session_store = MagicMock()
        runner._running_agents = {}
        source = SessionSource(platform=Platform.WEIXIN, user_id=user_id,
                               chat_id="o9user", user_name="u", chat_type="dm")
        event = MessageEvent(text="/approve", source=source, message_id="m1")
        return runner, event

    def _queue(self):
        from tools import approval as mod
        unlock = threading.Event()
        class E:
            result = None
            data = {"command": "rm -rf /tmp/evil", "description": "dangerous"}
        entry = E()
        entry.event = unlock  # resolve_gateway_approval sets .result then .event.set()
        mod._gateway_queues[self.SK] = [entry]
        return unlock, entry

    def test_user_cannot_self_approve_delegated_session(self, tmp_path, monkeypatch):
        self._reset()
        self._isolated_home(tmp_path, monkeypatch)
        import gateway.approval_delegation as ad
        with patch("hermes_cli.config.load_config", return_value=self.CFG):
            ad.register_delegation(admin_platform="feishu", admin_chat_id="oc_admin",
                                   session_key=self.SK, user_platform="weixin",
                                   user_chat_id="o9user")
            unlock, entry = self._queue()
            runner, event = self._runner_for("o9user")
            with patch.object(runner, "_session_key_for_source", return_value=self.SK):
                reply = asyncio.run(runner._handle_approve_command(event))
            assert unlock.is_set() is False, "delegated approval must NOT self-resolve"
            assert entry.result is None
            assert reply  # user gets a refusal, not silence

    def test_admin_participant_can_approve_in_user_chat(self, tmp_path, monkeypatch):
        # An admin who IS a participant of the delegated user's chat keeps
        # approve authority (the gate exempts is_admin_user senders).
        self._reset()
        self._isolated_home(tmp_path, monkeypatch)
        import gateway.approval_delegation as ad
        with patch("hermes_cli.config.load_config", return_value=self.CFG):
            ad.register_delegation(admin_platform="feishu", admin_chat_id="oc_admin",
                                   session_key=self.SK, user_platform="weixin",
                                   user_chat_id="o9user")
            unlock, entry = self._queue()
            runner, event = self._runner_for("o9admin")  # weixin admin in user chat
            with patch.object(runner, "_session_key_for_source", return_value=self.SK):
                asyncio.run(runner._handle_approve_command(event))
            assert unlock.is_set() is True

    def test_redirect_failed_session_keeps_self_approve(self, tmp_path, monkeypatch):
        # Delegation enabled but this session has NO live entry (all admins
        # unreachable → fell back to the user): normal self-approve UX holds.
        self._reset()
        self._isolated_home(tmp_path, monkeypatch)
        with patch("hermes_cli.config.load_config", return_value=self.CFG):
            unlock, entry = self._queue()
            runner, event = self._runner_for("o9user")
            with patch.object(runner, "_session_key_for_source", return_value=self.SK):
                asyncio.run(runner._handle_approve_command(event))
            assert unlock.is_set() is True, "fallback sessions must still self-approve"

    def test_user_can_still_deny_own_delegated_session(self, tmp_path, monkeypatch):
        # Refusing is always safe: the deny path must not gain the admin gate.
        self._reset()
        self._isolated_home(tmp_path, monkeypatch)
        import gateway.approval_delegation as ad
        with patch("hermes_cli.config.load_config", return_value=self.CFG):
            ad.register_delegation(admin_platform="feishu", admin_chat_id="oc_admin",
                                   session_key=self.SK, user_platform="weixin",
                                   user_chat_id="o9user")
            unlock, entry = self._queue()
            runner, event = self._runner_for("o9user")
            event.text = "/deny"
            with patch.object(runner, "_session_key_for_source", return_value=self.SK):
                asyncio.run(runner._handle_deny_command(event))
            assert entry.result == "deny"
