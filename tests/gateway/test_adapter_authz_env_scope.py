"""Scope-aware authorization reads for the QQBot / Teams / Email adapters.

Complements ``test_qqbot_scope_paths.py`` (the gateway authz choke point) and
``test_platform_authz_scope.py``: those pin ``_auth_env`` /
``_platform_gate_env``, but three ADAPTER-internal authorization gates still
read the allow-all / allowlist env vars with bare ``os.getenv`` — the
cross-profile leak class of #72348 / #93522 / #93705:

- ``gateway/platforms/qqbot/adapter.py`` ``_open_dm_opted_in`` — the DM intake
  opt-in for ``dm_policy: open``. A secondary profile must not inherit the
  DEFAULT profile's ``GATEWAY_ALLOW_ALL_USERS`` via ``os.environ``.
- ``plugins/platforms/teams/adapter.py`` ``_on_card_action`` — the ONLY gate
  before ``resolve_gateway_approval`` executes. A secondary profile must not
  inherit the DEFAULT profile's ``TEAMS_ALLOWED_USERS`` /
  ``TEAMS_ALLOW_ALL_USERS``.
- ``plugins/platforms/email/adapter.py`` ``_allow_all_senders`` /
  ``_allowlist_in_effect`` / the dispatch guard — a secondary profile must not
  inherit the DEFAULT profile's ``GATEWAY_ALLOW_ALL_USERS`` /
  ``GATEWAY_ALLOWED_USERS``.

Each site is authorization config (the sharpest edge of the #72348 class):
a leaked allow-all fails OPEN and admits senders / clickers the secondary
profile never opted into. The existing scoped helpers already in each file
(``_resolve_qq_secret`` / ``_get_scoped_secret`` / ``_get_secret``) are the fix
— they return the default on a scoped miss and keep the ``os.environ`` read
only on the unscoped default-profile path.

Single-profile deployments (no scope, multiplex off) keep the legacy
``os.environ`` behavior.
"""

import pytest

from agent import secret_scope as ss

_TRUTHY = {"true", "1", "yes"}


@pytest.fixture(autouse=True)
def _reset_scope_state(monkeypatch):
    for key in (
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "QQ_ALLOW_ALL_USERS",
        "QQ_DM_POLICY",
        "TEAMS_ALLOWED_USERS",
        "TEAMS_ALLOW_ALL_USERS",
        "EMAIL_ALLOWED_USERS",
        "EMAIL_ALLOW_ALL_USERS",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
    ):
        monkeypatch.delenv(key, raising=False)
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestQqbotIntakeOptInScope:
    """``QQAdapter._open_dm_opted_in`` must honor the installed profile scope."""

    @staticmethod
    def _adapter():
        from gateway.config import PlatformConfig
        from gateway.platforms.qqbot.adapter import QQAdapter

        return QQAdapter(PlatformConfig(enabled=True, extra={"dm_policy": "open"}))

    def test_scope_miss_does_not_inherit_gateway_allow_all(self, monkeypatch):
        # The DEFAULT profile opted in via os.environ; the secondary
        # profile's installed scope has no opt-in. Intake must fail closed.
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            assert adapter._is_dm_intake_allowed("user-1") is False
        finally:
            ss.reset_secret_scope(tok)

    def test_scoped_opt_in_is_honored(self, monkeypatch):
        # os.environ has NO opt-in; the secondary profile's own scope does.
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"GATEWAY_ALLOW_ALL_USERS": "true"})
        try:
            assert adapter._is_dm_intake_allowed("user-1") is True
        finally:
            ss.reset_secret_scope(tok)

    def test_scoped_platform_opt_in_is_honored(self):
        # The per-platform QQ_ALLOW_ALL_USERS flag, scoped-only.
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"QQ_ALLOW_ALL_USERS": "true"})
        try:
            assert adapter._is_dm_intake_allowed("user-1") is True
        finally:
            ss.reset_secret_scope(tok)

    def test_unscoped_single_profile_keeps_environ(self, monkeypatch):
        # Multiplex inactive, no scope: legacy os.environ read preserved.
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._adapter()
        assert adapter._is_dm_intake_allowed("user-1") is True


class TestTeamsCardActionScope:
    """``_on_card_action``'s authorization gate must honor the profile scope.

    Drives the REAL handler end-to-end with stub SDK response classes (the
    botbuilder SDK is optional at import time — the adapter module's
    ``InvokeResponse`` / response classes are ``None`` without it) and a
    patched ``tools.approval``. The security-relevant assertion: when the
    secondary profile's scope has no authorization keys,
    ``resolve_gateway_approval`` must never run — on main the gate reads
    bare ``os.getenv`` and the DEFAULT profile's opt-in authorizes the click.
    """

    @staticmethod
    def _stub_response_classes(monkeypatch):
        """Give the adapter module usable InvokeResponse / card classes."""
        import plugins.platforms.teams.adapter as tmod

        class _InvokeResponse:
            def __init__(self, status, body):
                self.status = status
                self.body = body

        class _MsgBody:
            def __init__(self, value):
                self.value = value

        class _CardBody:
            def __init__(self, value):
                self.value = value

        class _AdaptiveCard:
            def __init__(self):
                self.body = None

            def with_version(self, v):
                return self

            def with_body(self, b):
                self.body = b
                return self

        class _TextBlock:
            def __init__(self, text, **kw):
                self.text = text

        monkeypatch.setattr(tmod, "InvokeResponse", _InvokeResponse)
        monkeypatch.setattr(tmod, "AdaptiveCardActionMessageResponse", _MsgBody)
        monkeypatch.setattr(tmod, "AdaptiveCardActionCardResponse", _CardBody)
        monkeypatch.setattr(tmod, "AdaptiveCard", _AdaptiveCard)
        monkeypatch.setattr(tmod, "TextBlock", _TextBlock)

    @staticmethod
    def _make_ctx(clicker_id, action, session_key, cmd="rm -rf /tmp/x", desc="cleanup"):
        """Stub ActivityContext shaped like a card Action.Execute click."""
        from types import SimpleNamespace

        data = {
            "hermes_action": action,
            "session_key": session_key,
            "cmd": cmd,
            "desc": desc,
        }
        activity = SimpleNamespace(
            value=SimpleNamespace(
                action=SimpleNamespace(data=data),
            ),
            from_=SimpleNamespace(aad_object_id=clicker_id, id=clicker_id),
        )
        return SimpleNamespace(activity=activity)

    @pytest.mark.asyncio
    async def test_scope_miss_does_not_inherit_primary_allow_all(self, monkeypatch):
        # DEFAULT profile: TEAMS_ALLOW_ALL_USERS=true in os.environ.
        # Secondary profile scope: no authorization keys. The click must be
        # default-denied — resolve_gateway_approval must NEVER run.
        import plugins.platforms.teams.adapter as tmod

        self._stub_response_classes(monkeypatch)
        monkeypatch.setenv("TEAMS_ALLOW_ALL_USERS", "true")
        resolved = []
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval", lambda k: True, raising=False
        )
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda k, c: resolved.append((k, c)),
            raising=False,
        )
        adapter = object.__new__(tmod.TeamsAdapter)
        ctx = self._make_ctx("stranger-1", "approve_once", "telegram:12345")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            resp = await adapter._on_card_action(ctx)
        finally:
            ss.reset_secret_scope(tok)
        assert resolved == []  # the click was denied, nothing executed
        assert "require TEAMS_ALLOWED_USERS" in resp.body.value

    @pytest.mark.asyncio
    async def test_scope_miss_does_not_inherit_primary_allowlist(self, monkeypatch):
        # DEFAULT profile allowlists stranger-1; the secondary profile's
        # scope has no allowlist — the same clicker must stay denied there
        # even though the env allowlist names him.
        import plugins.platforms.teams.adapter as tmod

        self._stub_response_classes(monkeypatch)
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "stranger-1")
        resolved = []
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval", lambda k: True, raising=False
        )
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda k, c: resolved.append((k, c)),
            raising=False,
        )
        adapter = object.__new__(tmod.TeamsAdapter)
        ctx = self._make_ctx("stranger-1", "approve_once", "telegram:12345")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            resp = await adapter._on_card_action(ctx)
        finally:
            ss.reset_secret_scope(tok)
        assert resolved == []
        # Denied — either the "not configured" default-deny or the explicit
        # "not authorized" branch; both mean the click was refused.
        assert (
            "require TEAMS_ALLOWED_USERS" in resp.body.value
            or "Not authorized" in resp.body.value
        )

    @pytest.mark.asyncio
    async def test_scoped_allowlist_authorizes_click(self, monkeypatch):
        # The secondary profile's OWN scope allowlists the clicker — the
        # approval resolves, with no env value present at all.
        import plugins.platforms.teams.adapter as tmod

        self._stub_response_classes(monkeypatch)
        resolved = []
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval", lambda k: True, raising=False
        )
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda k, c: resolved.append((k, c)),
            raising=False,
        )
        adapter = object.__new__(tmod.TeamsAdapter)
        ctx = self._make_ctx("clicker-9", "approve_once", "telegram:12345")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"TEAMS_ALLOWED_USERS": "clicker-9"})
        try:
            await adapter._on_card_action(ctx)
        finally:
            ss.reset_secret_scope(tok)
        assert resolved == [("telegram:12345", "once")]  # authorized + executed

    @pytest.mark.asyncio
    async def test_unscoped_single_profile_keeps_environ(self, monkeypatch):
        # Multiplex inactive, no scope: the default profile's own env
        # allowlist authorizes the click (legacy behavior preserved).
        import plugins.platforms.teams.adapter as tmod

        self._stub_response_classes(monkeypatch)
        monkeypatch.setenv("TEAMS_ALLOWED_USERS", "clicker-1")
        resolved = []
        monkeypatch.setattr(
            "tools.approval.has_blocking_approval", lambda k: True, raising=False
        )
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda k, c: resolved.append((k, c)),
            raising=False,
        )
        adapter = object.__new__(tmod.TeamsAdapter)
        ctx = self._make_ctx("clicker-1", "approve_once", "telegram:12345")
        await adapter._on_card_action(ctx)
        assert resolved == [("telegram:12345", "once")]


class TestEmailDispatchGuardScope:
    """``_allow_all_senders`` / ``_allowlist_in_effect`` must honor the scope."""

    @staticmethod
    def _adapter():
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True))

    def test_allow_all_senders_does_not_inherit_gateway_flag(self, monkeypatch):
        # DEFAULT profile set GATEWAY_ALLOW_ALL_USERS=true in os.environ; the
        # secondary profile's scope has no opt-in. _allow_all_senders must
        # return False — otherwise the From:-authentication gate is skipped
        # on the secondary profile (GHSA-rxqh-5572-8m77 class).
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            assert adapter._allow_all_senders() is False
        finally:
            ss.reset_secret_scope(tok)

    def test_allow_all_senders_honors_scoped_opt_in(self):
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"EMAIL_ALLOW_ALL_USERS": "true"})
        try:
            assert adapter._allow_all_senders() is True
        finally:
            ss.reset_secret_scope(tok)

    def test_allowlist_in_effect_does_not_inherit_gateway_allowlist(self, monkeypatch):
        # GATEWAY_ALLOWED_USERS in os.environ belongs to the DEFAULT
        # profile; a secondary profile whose scope has no allowlist must
        # not treat one as "in effect".
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "owner@example.com")
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            assert adapter._allowlist_in_effect() is False
        finally:
            ss.reset_secret_scope(tok)

    def test_allowlist_in_effect_honors_scoped_email_allowlist(self):
        adapter = self._adapter()
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"EMAIL_ALLOWED_USERS": "beta@example.com"})
        try:
            assert adapter._allowlist_in_effect() is True
        finally:
            ss.reset_secret_scope(tok)

    def test_unscoped_single_profile_keeps_environ(self, monkeypatch):
        # Multiplex inactive, no scope: legacy os.environ behavior preserved.
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = self._adapter()
        assert adapter._allow_all_senders() is True
