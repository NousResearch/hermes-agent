"""Tests for TelegramMiniAppProvider and its register(ctx) entry point."""

import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

import plugins.dashboard_auth.telegram_miniapp as miniapp_plugin
from hermes_cli.dashboard_auth import TokenPrincipal, assert_protocol_compliance
from hermes_cli.dashboard_auth import token_auth

BOT_TOKEN = "123456:AAFakeTestTokenNotReal-abcdefghijklmno"


def _build_init_data(fields: dict, *, bot_token: str = BOT_TOKEN) -> str:
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    computed_hash = hmac.new(
        secret_key, check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    all_fields = {**fields, "hash": computed_hash}
    return "&".join(f"{k}={quote(str(v), safe='')}" for k, v in all_fields.items())


def _init_data_for(user_id: int, *, bot_token: str = BOT_TOKEN) -> str:
    fields = {
        "auth_date": str(int(time.time())),
        "query_id": "AAFakeQueryId",
        "user": json.dumps({"id": user_id, "first_name": "Test"}),
    }
    return _build_init_data(fields, bot_token=bot_token)


def _pairing_store(*, paired: bool):
    return SimpleNamespace(is_approved=lambda *_a, **_kw: paired)


def _write_machine_env(text: str) -> None:
    """TELEGRAM_DASHBOARD_ADMIN_USERS is read fresh from the machine's .env
    file per call (tiers.py's _dashboard_env_get), not os.environ -- write
    the file, not the process env. See test_telegram_miniapp_tiers.py's
    identical helper for the full rationale.
    """
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(text)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_DASHBOARD_ADMIN_USERS", "TELEGRAM_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)
    token_auth.clear_token_routes()
    yield
    token_auth.clear_token_routes()


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


class TestProvider:
    def test_protocol_compliance(self):
        assert_protocol_compliance(miniapp_plugin.TelegramMiniAppProvider)

    def test_supports_token_flag(self):
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        assert p.supports_token is True

    def test_is_non_interactive(self):
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        assert p.supports_session is False

    def test_construction_rejects_empty_bot_token(self):
        with pytest.raises(ValueError):
            miniapp_plugin.TelegramMiniAppProvider(
                bot_token="", pairing_store=_pairing_store(paired=True)
            )

    def test_verify_token_accepts_paired_user(self):
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        principal = p.verify_token(token=_init_data_for(42))
        assert isinstance(principal, TokenPrincipal)
        assert principal.principal == "telegram:42"
        assert principal.provider == "telegram-miniapp"
        assert principal.scopes == ("dashboard:read",)

    def test_verify_token_rejects_unpaired_user(self):
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=False)
        )
        assert p.verify_token(token=_init_data_for(42)) is None

    def test_verify_token_rejects_bad_hmac(self):
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        assert p.verify_token(token=_init_data_for(42, bot_token="wrong:token")) is None

    def test_verify_token_rejects_empty(self):
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        assert p.verify_token(token="") is None

    def test_verify_token_admin_scope_when_admin_listed(self):
        _write_machine_env("TELEGRAM_DASHBOARD_ADMIN_USERS=42\n")
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        principal = p.verify_token(token=_init_data_for(42))
        assert principal.scopes == ("dashboard:read", "dashboard:admin")

    def test_verify_token_paired_scope_when_not_admin(self):
        _write_machine_env("TELEGRAM_DASHBOARD_ADMIN_USERS=99\n")
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        principal = p.verify_token(token=_init_data_for(42))
        assert principal.scopes == ("dashboard:read",)

    def test_verify_session_returns_none_not_raises(self):
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        assert p.verify_session(access_token="anything") is None

    def test_interactive_methods_raise(self):
        p = miniapp_plugin.TelegramMiniAppProvider(
            bot_token=BOT_TOKEN, pairing_store=_pairing_store(paired=True)
        )
        with pytest.raises(NotImplementedError):
            p.start_login(redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.complete_login(code="c", state="s", code_verifier="v", redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.refresh_session(refresh_token="r")


# ---------------------------------------------------------------------------
# _load_config_section() — the config surface itself, not just register()'s
# use of it. Every TestRegister/TestRegisterRoutes test below monkeypatches
# this function out, so it needs its own direct coverage of the actual
# config.yaml traversal and its fail-closed exception handling.
# ---------------------------------------------------------------------------


class TestLoadConfigSection:
    def test_missing_section_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {"dashboard": {}}
        )
        assert miniapp_plugin._load_config_section() == {}

    def test_missing_dashboard_key_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
        assert miniapp_plugin._load_config_section() == {}

    def test_reads_configured_section(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"dashboard": {"telegram_miniapp": {"enabled": True, "max_age_seconds": 60}}},
        )
        assert miniapp_plugin._load_config_section() == {
            "enabled": True,
            "max_age_seconds": 60,
        }

    def test_non_dict_section_fails_closed_to_empty_dict(self, monkeypatch):
        """A user who writes a scalar where a mapping is expected must not
        crash plugin discovery — treated the same as "not configured".
        """
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"dashboard": {"telegram_miniapp": "oops_a_string"}},
        )
        assert miniapp_plugin._load_config_section() == {}

    def test_load_config_raising_fails_closed_to_empty_dict(self, monkeypatch):
        """A broken/unreadable config.yaml must not prevent the rest of
        plugin discovery from proceeding — this plugin just stays disabled.
        """

        def _boom():
            raise OSError("config.yaml is not readable")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        assert miniapp_plugin._load_config_section() == {}


# ---------------------------------------------------------------------------
# register() entry point
# ---------------------------------------------------------------------------


class TestRegister:
    def test_skips_when_disabled_by_default(self, monkeypatch):
        monkeypatch.setattr(miniapp_plugin, "_load_config_section", lambda: {})
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
        ctx = MagicMock()
        miniapp_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "enabled" in miniapp_plugin.LAST_SKIP_REASON

    def test_skips_and_fails_closed_when_no_bot_token(self, monkeypatch):
        monkeypatch.setattr(miniapp_plugin, "_load_config_section", lambda: {"enabled": True})
        ctx = MagicMock()
        miniapp_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "TELEGRAM_BOT_TOKEN" in miniapp_plugin.LAST_SKIP_REASON

    def test_registers_when_enabled_with_bot_token(self, monkeypatch):
        monkeypatch.setattr(miniapp_plugin, "_load_config_section", lambda: {"enabled": True})
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
        ctx = MagicMock()
        miniapp_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(provider, miniapp_plugin.TelegramMiniAppProvider)
        assert miniapp_plugin.LAST_SKIP_REASON == ""

    def test_registered_provider_uses_configured_max_age(self, monkeypatch):
        monkeypatch.setattr(
            miniapp_plugin,
            "_load_config_section",
            lambda: {"enabled": True, "max_age_seconds": 60},
        )
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
        ctx = MagicMock()
        miniapp_plugin.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert provider._max_age_seconds == 60

    def test_registered_provider_rejects_when_pairing_store_denies(self, monkeypatch):
        """register() wires a real PairingStore; sanity-check it's consulted."""
        monkeypatch.setattr(miniapp_plugin, "_load_config_section", lambda: {"enabled": True})
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
        ctx = MagicMock()
        miniapp_plugin.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        # A random unpaired/unlisted user id must not verify against the real store.
        assert provider.verify_token(token=_init_data_for(918273645)) is None


# ---------------------------------------------------------------------------
# register() route wiring — the explicit allowlist, not a blanket prefix
# ---------------------------------------------------------------------------


class TestRegisterRoutes:
    def _register(self, monkeypatch):
        monkeypatch.setattr(miniapp_plugin, "_load_config_section", lambda: {"enabled": True})
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
        miniapp_plugin.register(MagicMock())

    def test_skipped_registration_registers_no_routes(self, monkeypatch):
        monkeypatch.setattr(miniapp_plugin, "_load_config_section", lambda: {})
        miniapp_plugin.register(MagicMock())
        assert token_auth.is_token_route("/api/status") is False

    @pytest.mark.parametrize(
        "path",
        [
            "/api/status",
            "/api/skills",
            "/api/skills/content",
            "/api/cron/jobs",
            "/api/cron/delivery-targets",
            "/api/cron/blueprints",
            "/api/sessions",
        ],
    )
    def test_registers_expected_literal_routes(self, monkeypatch, path):
        self._register(monkeypatch)
        assert token_auth.is_token_route(path) is True

    def test_registers_session_id_shapes_via_regex(self, monkeypatch):
        self._register(monkeypatch)
        assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3d4") is True
        assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3d4/messages") is True
        # 6-hex branch/thread session id shape must also match.
        assert token_auth.is_token_route("/api/sessions/20260702_143022_a1b2c3") is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/sessions/stats",
            "/api/sessions/empty/count",
            "/api/sessions/bulk-delete",
            "/api/sessions/prune",
        ],
    )
    def test_does_not_register_sibling_literal_routes(self, monkeypatch, path):
        """The exact gap flagged during review: a blanket prefix would sweep
        these destructive/broad-read routes in. This plugin must not.

        /api/sessions/search used to be in this list too -- it's now
        deliberately registered (admin-tier gated at the handler level via
        _require_dashboard_admin, see search_sessions's own docstring) for
        the Mini App's Sessions search box, which the design only ever
        shows to admins. See test_registers_sessions_search_admin_only below.
        """
        self._register(monkeypatch)
        assert token_auth.is_token_route(path) is False

    def test_registers_sessions_search_read_only(self, monkeypatch):
        """Dispatch-eligible for GET/HEAD only -- authorization itself
        (admin-tier only) is enforced in the handler, proven end-to-end in
        tests/hermes_cli/test_web_server.py, not here.
        """
        self._register(monkeypatch)
        assert token_auth.is_token_route_method_allowed("/api/sessions/search", "GET") is True
        assert token_auth.is_token_route_method_allowed("/api/sessions/search", "POST") is False

    def test_does_not_register_cron_job_detail_or_skills_hub_routes(self, monkeypatch):
        """Scope discipline: only what spec Task #6 named, nothing further."""
        self._register(monkeypatch)
        assert token_auth.is_token_route("/api/cron/jobs/some-job-id") is False
        assert token_auth.is_token_route("/api/skills/hub/search") is False

    @pytest.mark.parametrize(
        "path",
        [
            "/api/status",
            "/api/skills",
            "/api/skills/content",
            "/api/cron/jobs",
            "/api/cron/delivery-targets",
            "/api/cron/blueprints",
            "/api/sessions",
        ],
    )
    def test_registered_literal_routes_are_read_only(self, monkeypatch, path):
        """The read-only guarantee (spec §1). Several of these paths ALSO
        mount a mutating handler on the same path (POST /api/skills, PUT
        /api/skills/content, POST /api/cron/jobs). The plugin must register
        them token-authable for GET/HEAD ONLY, so a paired Mini App principal
        can never authenticate a write verb (create-skill is agent-executed,
        i.e. code execution) that shares the path with a registered reader.
        """
        self._register(monkeypatch)
        assert token_auth.is_token_route_method_allowed(path, "GET") is True
        assert token_auth.is_token_route_method_allowed(path, "HEAD") is True
        for verb in ("POST", "PUT", "PATCH", "DELETE"):
            assert token_auth.is_token_route_method_allowed(path, verb) is False

    def test_registered_session_routes_get_head_always_dispatch_eligible(self, monkeypatch):
        """GET/HEAD are always dispatch-eligible for the session-id regex,
        for every tier.
        """
        self._register(monkeypatch)
        sid = "/api/sessions/20260702_143022_a1b2c3d4"
        assert token_auth.is_token_route_method_allowed(sid, "GET") is True
        msgs = "/api/sessions/20260702_143022_a1b2c3d4/messages"
        assert token_auth.is_token_route_method_allowed(msgs, "GET") is True
        assert token_auth.is_token_route_method_allowed(msgs, "POST") is False

    def test_registered_session_routes_now_also_allow_patch_delete_dispatch(self, monkeypatch):
        """PATCH (archive) + DELETE are now dispatch-eligible on the same
        session-id regex, for the Users/Sessions admin actions this Mini App
        build adds.

        Dispatch-eligible is NOT the same as authorized -- this only proves
        the token-auth seam will let a presented token be evaluated for
        these verbs at all; the handler-level admin gate
        (_require_dashboard_admin) is what actually rejects a non-admin
        paired token, and is proven separately by the full-stack tests
        against the real handlers (tests/hermes_cli/test_web_server.py).
        """
        self._register(monkeypatch)
        sid = "/api/sessions/20260702_143022_a1b2c3d4"
        assert token_auth.is_token_route_method_allowed(sid, "PATCH") is True
        assert token_auth.is_token_route_method_allowed(sid, "DELETE") is True
        # .../messages never got PATCH/DELETE -- only the base session-id
        # regex did, matching the two real handlers (archive/delete apply
        # to the session, not to its message list).
        msgs = "/api/sessions/20260702_143022_a1b2c3d4/messages"
        assert token_auth.is_token_route_method_allowed(msgs, "PATCH") is False
        assert token_auth.is_token_route_method_allowed(msgs, "DELETE") is False

    def test_registers_new_admin_mutating_routes(self, monkeypatch):
        """The admin-tier mutating routes this build adds are all
        dispatch-eligible for exactly their intended method, cron job
        detail-edit and skill create/edit remain untouched."""
        self._register(monkeypatch)

        cron_run = "/api/cron/jobs/abcdef123456/trigger"
        assert token_auth.is_token_route_method_allowed(cron_run, "POST") is True
        # Job delete IS registered (DELETE only) -- job detail-edit (GET/PUT)
        # is NOT a prefix registration, so it must stay unregistered even
        # though DELETE on the exact same path shape now is.
        assert token_auth.is_token_route_method_allowed("/api/cron/jobs/abcdef123456", "DELETE") is True
        assert token_auth.is_token_route_method_allowed("/api/cron/jobs/abcdef123456", "GET") is False
        assert token_auth.is_token_route_method_allowed("/api/cron/jobs/abcdef123456", "PUT") is False
        assert token_auth.is_token_route("/api/cron/jobs/abcdef123456/runs") is False

        assert token_auth.is_token_route_method_allowed("/api/skills/toggle", "PUT") is True
        # Skill create/edit (agent-executed) must never become token-authable.
        assert token_auth.is_token_route("/api/skills") is True
        assert token_auth.is_token_route_method_allowed("/api/skills", "POST") is False
        assert token_auth.is_token_route_method_allowed("/api/skills/content", "PUT") is False

        assert token_auth.is_token_route_method_allowed("/api/telegram/allowlist", "GET") is True
        assert token_auth.is_token_route_method_allowed("/api/telegram/allowlist", "POST") is True
        assert (
            token_auth.is_token_route_method_allowed("/api/telegram/allowlist/224918330", "DELETE")
            is True
        )

        assert token_auth.is_token_route_method_allowed("/api/gateway/restart", "POST") is True
        assert token_auth.is_token_route_method_allowed("/api/hermes/update", "POST") is True
        assert token_auth.is_token_route_method_allowed("/api/miniapp/me", "GET") is True

        # Session resume: a distinct sibling path from the existing
        # PATCH/DELETE session-id registration, POST only.
        resume_path = "/api/sessions/20260702_143022_a1b2c3d4/resume"
        assert token_auth.is_token_route_method_allowed(resume_path, "POST") is True
        assert token_auth.is_token_route_method_allowed(resume_path, "GET") is False

        # Logs: the pre-existing desktop log-viewer endpoint, GET only.
        assert token_auth.is_token_route_method_allowed("/api/logs", "GET") is True
        assert token_auth.is_token_route_method_allowed("/api/logs", "POST") is False
