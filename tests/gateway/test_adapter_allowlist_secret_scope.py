"""Allowlist / gate reads must honor the active profile secret scope.

These adapter-level reads sit in front of gateway authz. A bare
``os.getenv`` under ``gateway.multiplex_profiles`` would borrow the
default profile's allowlist (or allow-all flag) for a secondary
profile. Same shape as the Matrix recovery-key fix (#69090) and the
Slack app-token pattern (#59739).
"""
from types import SimpleNamespace

import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.signal import SignalAdapter
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
from gateway.platforms.whatsapp_common import _get_wsecret as whatsapp_secret
from gateway.session import SessionSource
from plugins.platforms.email.adapter import EmailAdapter
from plugins.platforms.email.adapter import _get_secret as email_secret
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.adapter import _apply_yaml_config
from plugins.platforms.matrix.adapter import _startup_env_secret as matrix_secret
from gateway.platforms.signal import _startup_env_secret as signal_secret


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _signal_event(sender: str) -> MessageEvent:
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=SimpleNamespace(value="signal"),
            chat_id=sender,
            user_id=sender,
        ),
    )


class TestMatrixYamlAllowlistsSurviveScopedMiss:
    def test_yaml_hook_seeds_extra_and_skips_env_under_multiplex(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!default:example.org")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            seeded = _apply_yaml_config(
                {},
                {
                    "allowed_users": ["@operator:example.org"],
                    "allowed_rooms": ["!private:example.org"],
                },
            )
            assert seeded["allowed_users"] == "@operator:example.org"
            assert seeded["allowed_rooms"] == "!private:example.org"
            import os
            assert os.environ["MATRIX_ALLOWED_USERS"] == "@default:example.org"
            assert os.environ["MATRIX_ALLOWED_ROOMS"] == "!default:example.org"
        finally:
            ss.reset_secret_scope(token)

    def test_adapter_uses_yaml_extra_not_default_profile_env(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!default:example.org")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            seeded = _apply_yaml_config(
                {},
                {
                    "allowed_users": ["@operator:example.org"],
                    "allowed_rooms": ["!private:example.org"],
                },
            )
            config = PlatformConfig(enabled=True)
            config.extra = {
                "homeserver": "https://example.org",
                **seeded,
            }
            adapter = MatrixAdapter(config)
            assert adapter._allowed_user_ids == {"@operator:example.org"}
            assert adapter._allowed_rooms == {"!private:example.org"}
        finally:
            ss.reset_secret_scope(token)

    def test_scoped_env_value_wins_when_extra_absent(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"MATRIX_ALLOWED_USERS": "@second:example.org"})
        try:
            assert matrix_secret("MATRIX_ALLOWED_USERS") == "@second:example.org"
            config = PlatformConfig(enabled=True)
            config.extra = {"homeserver": "https://example.org"}
            adapter = MatrixAdapter(config)
            assert adapter._allowed_user_ids == {"@second:example.org"}
        finally:
            ss.reset_secret_scope(token)

    def test_scoped_env_beats_yaml_extra(self, monkeypatch):
        """Named-profile secret must outrank YAML extra and process env."""
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!default:example.org")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(
            {
                "MATRIX_ALLOWED_USERS": "@second:example.org",
                "MATRIX_ALLOWED_ROOMS": "!second:example.org",
            }
        )
        try:
            config = PlatformConfig(enabled=True)
            config.extra = {
                "homeserver": "https://example.org",
                "allowed_users": "@operator:example.org",
                "allowed_rooms": "!private:example.org",
            }
            adapter = MatrixAdapter(config)
            assert adapter._allowed_user_ids == {"@second:example.org"}
            assert adapter._allowed_rooms == {"!second:example.org"}
        finally:
            ss.reset_secret_scope(token)

    def test_yaml_extra_beats_unscoped_process_env(self, monkeypatch):
        """Without a named-profile scope, YAML extra still beats process env."""
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!default:example.org")
        config = PlatformConfig(enabled=True)
        config.extra = {
            "homeserver": "https://example.org",
            "allowed_users": "@operator:example.org",
            "allowed_rooms": "!private:example.org",
        }
        adapter = MatrixAdapter(config)
        assert adapter._allowed_user_ids == {"@operator:example.org"}
        assert adapter._allowed_rooms == {"!private:example.org"}

    def test_unscoped_startup_still_reads_environ(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        assert matrix_secret("MATRIX_ALLOWED_USERS") == "@default:example.org"

    def test_named_profile_yaml_survives_load_gateway_config(self, tmp_path, monkeypatch):
        """YAML-only Matrix allowlists must reach MatrixAdapter via extra.

        Reviewer asked for this exact path: a named profile configured only
        in config.yaml, load_gateway_config() under that profile's runtime
        scope, then construct MatrixAdapter and assert the effective
        _allowed_rooms / _allowed_user_ids.
        """
        from gateway.config import Platform, load_gateway_config
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        home = tmp_path / "profiles" / "operator"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "matrix:\n"
            "  allowed_users:\n"
            "    - \"@operator:example.org\"\n"
            "  allowed_rooms:\n"
            "    - \"!private:example.org\"\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@default:example.org")
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!default:example.org")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        home_token = set_hermes_home_override(str(home))
        try:
            cfg = load_gateway_config()
            extra = cfg.platforms[Platform.MATRIX].extra
            assert extra.get("allowed_users") == "@operator:example.org"
            assert extra.get("allowed_rooms") == "!private:example.org"
            import os
            assert os.environ["MATRIX_ALLOWED_USERS"] == "@default:example.org"
            assert os.environ["MATRIX_ALLOWED_ROOMS"] == "!default:example.org"
            adapter = MatrixAdapter(cfg.platforms[Platform.MATRIX])
            assert adapter._allowed_user_ids == {"@operator:example.org"}
            assert adapter._allowed_rooms == {"!private:example.org"}
        finally:
            reset_hermes_home_override(home_token)
            ss.reset_secret_scope(token)


class TestSignalReactionsFailClosedOnScopedMiss:
    def test_helper_scoped_miss_is_empty_not_star(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+155****0001")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            assert signal_secret("SIGNAL_ALLOWED_USERS", "") == ""
        finally:
            ss.reset_secret_scope(token)

    def test_adapter_reactions_closed_when_profile_has_no_allowlist(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+155****0001")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            config = PlatformConfig(enabled=True)
            config.extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}
            adapter = SignalAdapter(config)
            assert adapter.dm_allow_from == set()
            event = _signal_event("+155****9999")
            assert adapter._reactions_enabled(event) is False
        finally:
            ss.reset_secret_scope(token)

    def test_explicit_star_still_opens_reactions(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "*")
        config = PlatformConfig(enabled=True)
        config.extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}
        adapter = SignalAdapter(config)
        assert "*" in adapter.dm_allow_from
        assert adapter._reactions_enabled(_signal_event("+155****9999")) is True

    def test_scoped_profile_allowlist_is_used(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+155****0001")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SIGNAL_ALLOWED_USERS": "+155****0002"})
        try:
            config = PlatformConfig(enabled=True)
            config.extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}
            adapter = SignalAdapter(config)
            assert adapter.dm_allow_from == {"+155****0002"}
            assert adapter._reactions_enabled(_signal_event("+155****0002")) is True
            assert adapter._reactions_enabled(_signal_event("+155****0001")) is False
        finally:
            ss.reset_secret_scope(token)


class TestEmailAndWhatsAppAdapterGates:
    def test_email_allow_all_does_not_borrow_environ(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        monkeypatch.delenv("EMAIL_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("EMAIL_ALLOWED_USERS", raising=False)
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"EMAIL_ADDRESS": "b@example.com"})
        try:
            assert email_secret("GATEWAY_ALLOW_ALL_USERS", "") == ""
            config = PlatformConfig(enabled=True)
            adapter = EmailAdapter(config)
            assert adapter._allow_all_senders() is False
            assert adapter._allowlist_in_effect() is False
        finally:
            ss.reset_secret_scope(token)

    def test_whatsapp_open_dm_does_not_borrow_environ(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"WHATSAPP_ALLOWED_USERS": "x"})
        try:
            assert (whatsapp_secret("GATEWAY_ALLOW_ALL_USERS", default="") or "") == ""

            class _Host(WhatsAppBehaviorMixin):
                name = "whatsapp"

            assert _Host()._open_dm_opted_in() is False
        finally:
            ss.reset_secret_scope(token)


class TestGatewayAuthorizationDoesNotBorrowEnviron:
    def _runner(self):
        from gateway.config import GatewayConfig, Platform
        from gateway.run import GatewayRunner
        from unittest.mock import AsyncMock, MagicMock

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform.SIGNAL: PlatformConfig(enabled=True)})
        runner.adapters = {
            Platform.SIGNAL: SimpleNamespace(
                send=AsyncMock(),
                enforces_own_access_policy=False,
                authorization_is_upstream=False,
            )
        }
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        runner.pairing_stores = {}
        runner._profile_adapters = {}
        return runner

    def test_scoped_miss_does_not_borrow_process_allowlist(self, monkeypatch):
        from gateway.config import Platform

        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+155****0001")
        monkeypatch.delenv("SIGNAL_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            runner = self._runner()
            source = SessionSource(
                platform=Platform.SIGNAL,
                chat_id="+155****0001",
                user_id="+155****0001",
                chat_type="dm",
            )
            assert runner._is_user_authorized(source) is False
        finally:
            ss.reset_secret_scope(token)

    def test_scoped_profile_allowlist_authorizes_only_that_profile(self, monkeypatch):
        from gateway.config import Platform

        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+155****0001")
        monkeypatch.delenv("SIGNAL_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SIGNAL_ALLOWED_USERS": "+155****0002"})
        try:
            runner = self._runner()
            allowed = SessionSource(
                platform=Platform.SIGNAL,
                chat_id="+155****0002",
                user_id="+155****0002",
                chat_type="dm",
            )
            borrowed = SessionSource(
                platform=Platform.SIGNAL,
                chat_id="+155****0001",
                user_id="+155****0001",
                chat_type="dm",
            )
            assert runner._is_user_authorized(allowed) is True
            assert runner._is_user_authorized(borrowed) is False
        finally:
            ss.reset_secret_scope(token)


class TestSignalScopedSecretBeatsYamlExtra:
    def test_scoped_env_wins_over_conflicting_extra(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+155****0001")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SIGNAL_ALLOWED_USERS": "+155****0002"})
        try:
            config = PlatformConfig(enabled=True)
            config.extra = {
                "http_url": "http://localhost:8080",
                "account": "+155****4567",
                "allowed_users": "+155****0003",
            }
            adapter = SignalAdapter(config)
            assert adapter.dm_allow_from == {"+155****0002"}
            assert adapter._reactions_enabled(_signal_event("+155****0002")) is True
            assert adapter._reactions_enabled(_signal_event("+155****0003")) is False
        finally:
            ss.reset_secret_scope(token)


class TestMatrixSiblingAllowAllIsScoped:
    def test_invite_does_not_borrow_process_allow_all(self, monkeypatch):
        import asyncio
        from unittest.mock import MagicMock

        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            config = PlatformConfig(enabled=True)
            config.extra = {"homeserver": "https://example.org"}
            adapter = MatrixAdapter(config)
            adapter._allowed_user_ids = set()
            scheduled = []
            adapter._schedule_invite_join = lambda *args, **kwargs: scheduled.append((args, kwargs))
            event = SimpleNamespace(
                room_id="!evil:example.org",
                sender="@stranger:example.org",
                content=SimpleNamespace(is_direct=True),
            )
            asyncio.run(adapter._on_invite(event))
            assert scheduled == []
        finally:
            ss.reset_secret_scope(token)

    def test_free_rooms_and_ignore_patterns_do_not_borrow_environ(self, monkeypatch):
        monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", "!default:example.org")
        monkeypatch.setenv("MATRIX_IGNORE_USER_PATTERNS", "@spam:example.org")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            config = PlatformConfig(enabled=True)
            config.extra = {"homeserver": "https://example.org"}
            adapter = MatrixAdapter(config)
            assert adapter._free_rooms == set()
            assert adapter._ignored_user_patterns == []
        finally:
            ss.reset_secret_scope(token)


class TestSiblingAllowAllDoesNotBorrowEnviron:
    def test_weixin_open_dm_does_not_borrow_environ(self, monkeypatch):
        from gateway.platforms.weixin import WeixinAdapter

        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            adapter = object.__new__(WeixinAdapter)
            adapter._dm_policy = "open"
            assert adapter._open_dm_opted_in() is False
        finally:
            ss.reset_secret_scope(token)

    def test_wecom_open_dm_does_not_borrow_environ(self, monkeypatch):
        from plugins.platforms.wecom.adapter import WeComAdapter

        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            adapter = object.__new__(WeComAdapter)
            assert adapter._open_dm_opted_in() is False
        finally:
            ss.reset_secret_scope(token)

    def test_qqbot_open_dm_does_not_borrow_environ(self, monkeypatch):
        from gateway.platforms.qqbot.adapter import QQAdapter

        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            adapter = object.__new__(QQAdapter)
            assert adapter._open_dm_opted_in() is False
        finally:
            ss.reset_secret_scope(token)

    def test_yuanbao_open_dm_does_not_borrow_environ(self, monkeypatch):
        from gateway.platforms.yuanbao import AccessPolicy

        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
        try:
            policy = AccessPolicy("open", [], "open", [])
            assert policy._open_dm_opted_in() is False
        finally:
            ss.reset_secret_scope(token)
