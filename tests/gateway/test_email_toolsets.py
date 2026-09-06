"""Tool isolation for email under the review-first policy.

Under ``review_first`` EVERY email-triggered turn — trusted, allowlisted
senders included — must run with exactly ZERO callable tools, resolver
exceptions must fail closed to zero, proxy-mode delegation must be refused
for zero-tool turns, and the trust/zero-tool flags must be unforgeable over
the wire. Untrusted senders additionally get a session key segregated from
the trusted sender's session.
"""

import asyncio
import os
import unittest
from unittest.mock import patch

from gateway.run import GatewayRunner
from gateway.session import Platform as SessionPlatform
from gateway.session import SessionSource, build_session_key
from hermes_cli.tools_config import _get_platform_tools

ARMEN = "armenlsuny@gmail.com"
STRANGER = "someone@example.org"

_BASE_ENV = {
    "EMAIL_ADDRESS": "jordan@goodgravel.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
    "EMAIL_HOME_ADDRESS": ARMEN,
    "EMAIL_ALLOW_ALL_USERS": "true",
}

BASE_CONFIG = {"platform_toolsets": {"email": ["web", "vision"]}}


def _make_adapter(**extra):
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter

    merged = {
        "outbound_policy": "review_first",
        "auto_send_authenticated_senders": [ARMEN],
    }
    merged.update(extra)
    with patch.dict(os.environ, _BASE_ENV, clear=False):
        return EmailAdapter(PlatformConfig(enabled=True, extra=merged))


def _make_runner(adapter):
    gr = object.__new__(GatewayRunner)
    gr._adapter_for_source = lambda source: adapter
    return gr


def _email_source(**kwargs):
    defaults = dict(
        platform=SessionPlatform.EMAIL,
        chat_id=STRANGER,
        chat_type="dm",
        user_id=STRANGER,
    )
    defaults.update(kwargs)
    return SessionSource(**defaults)


def _msg_data(sender, authenticated, message_id="<m1@x>"):
    return {
        "sender_addr": sender,
        "sender_name": "Sender",
        "subject": "Hello",
        "message_id": message_id,
        "in_reply_to": "",
        "body": "hi",
        "attachments": [],
        "date": "",
        "sender_authenticated": authenticated,
        "auth_reason": "dmarc=pass" if authenticated else "authentication failed",
    }


class TestEmailToolsetsForSource(unittest.TestCase):
    def test_direct_policy_returns_none(self):
        adapter = _make_adapter(outbound_policy="")
        self.assertIsNone(adapter.toolsets_for_source(_email_source()))

    def test_trusted_source_gets_zero_tools_too(self):
        """Review-first: even the authenticated allowlisted sender's turns
        are tool-free — email content must never trigger tools."""
        adapter = _make_adapter()
        src = _email_source(chat_id=ARMEN, user_id=ARMEN)
        src.email_sender_trusted = True
        self.assertEqual(adapter.toolsets_for_source(src), [])

    def test_untrusted_source_gets_zero_tools(self):
        adapter = _make_adapter()
        self.assertEqual(adapter.toolsets_for_source(_email_source()), [])

    def test_configured_sender_toolsets_are_ignored(self):
        """The legacy untrusted_sender_toolsets knob is a widening lever and
        is ignored under review_first."""
        adapter = _make_adapter(untrusted_sender_toolsets=["web", "terminal"])
        self.assertEqual(adapter.toolsets_for_source(_email_source()), [])
        src = _email_source(chat_id=ARMEN, user_id=ARMEN)
        src.email_sender_trusted = True
        self.assertEqual(adapter.toolsets_for_source(src), [])

    def test_fail_closed_contract_declared_under_review_first(self):
        self.assertTrue(_make_adapter().toolsets_override_fail_closed)
        self.assertFalse(
            _make_adapter(outbound_policy="").toolsets_override_fail_closed
        )


class TestTrustFlagsForgeProof(unittest.TestCase):
    def test_to_dict_omits_security_flags(self):
        src = _email_source()
        src.email_sender_trusted = True
        src.email_zero_tools = True
        d = src.to_dict()
        self.assertNotIn("email_sender_trusted", d)
        self.assertNotIn("email_zero_tools", d)

    def test_from_dict_ignores_forged_flags(self):
        data = _email_source().to_dict()
        data["email_sender_trusted"] = True
        data["email_zero_tools"] = False
        restored = SessionSource.from_dict(data)
        self.assertFalse(restored.email_sender_trusted)
        self.assertFalse(restored.email_zero_tools)


class TestDispatchStampsTrust(unittest.TestCase):
    def _dispatch(self, adapter, msg_data):
        captured = []

        async def handle(event):
            captured.append(event)

        # Stub handle_message directly: the base-class implementation spawns background session
        # tasks, and this class tests dispatch stamping, not the session machinery.
        adapter.handle_message = handle
        with patch.dict(os.environ, _BASE_ENV, clear=False):
            asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured), 1)
        return captured[0].source

    def test_authenticated_allowlisted_sender_is_trusted_but_toolfree(self):
        adapter = _make_adapter()
        source = self._dispatch(adapter, _msg_data(ARMEN, True))
        self.assertTrue(source.email_sender_trusted)
        self.assertTrue(source.email_zero_tools)
        self.assertIsNone(source.thread_id)

    def test_forged_allowlisted_sender_is_untrusted_and_segregated(self):
        adapter = _make_adapter()
        source = self._dispatch(adapter, _msg_data(ARMEN, False))
        self.assertFalse(source.email_sender_trusted)
        self.assertTrue(source.email_zero_tools)
        self.assertEqual(source.thread_id, "untrusted")

    def test_stranger_is_untrusted_even_when_authenticated(self):
        adapter = _make_adapter()
        source = self._dispatch(adapter, _msg_data(STRANGER, True))
        self.assertFalse(source.email_sender_trusted)
        self.assertTrue(source.email_zero_tools)
        self.assertEqual(source.thread_id, "untrusted")

    def test_direct_policy_marks_trusted_without_zero_tools(self):
        adapter = _make_adapter(outbound_policy="")
        source = self._dispatch(adapter, _msg_data(STRANGER, False))
        self.assertTrue(source.email_sender_trusted)
        self.assertFalse(source.email_zero_tools)
        self.assertIsNone(source.thread_id)

    def test_untrusted_session_key_differs_from_trusted(self):
        adapter = _make_adapter()
        trusted_src = self._dispatch(adapter, _msg_data(ARMEN, True, "<t@x>"))
        forged_src = self._dispatch(adapter, _msg_data(ARMEN, False, "<f@x>"))
        self.assertNotEqual(
            build_session_key(trusted_src), build_session_key(forged_src)
        )
        self.assertIn(ARMEN, build_session_key(forged_src))
        self.assertTrue(build_session_key(forged_src).endswith(":untrusted"))


class TestResolverZeroToolContract(unittest.TestCase):
    def test_zero_tools_flag_wins_before_any_adapter(self):
        """The stamped source flag resolves to [] even with NO adapter —
        a missing adapter can never widen the boundary to defaults."""
        gr = _make_runner(None)
        src = _email_source()
        src.email_zero_tools = True
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, src, "email"
        )
        self.assertEqual(res, [])

    def test_zero_tools_flag_wins_over_raising_adapter(self):
        adapter = _make_adapter()
        adapter.toolsets_for_source = lambda source: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        gr = _make_runner(adapter)
        src = _email_source()
        src.email_zero_tools = True
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, src, "email"
        )
        self.assertEqual(res, [])

    def test_adapter_exception_fails_closed_for_email(self):
        """Even WITHOUT the source flag (e.g. a restored source), an
        exception from the email adapter's override fails closed to zero
        tools via toolsets_override_fail_closed — never back to defaults."""
        adapter = _make_adapter()
        adapter.toolsets_for_source = lambda source: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        gr = _make_runner(adapter)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _email_source(), "email"
        )
        self.assertEqual(res, [])

    def test_trusted_email_source_resolves_to_zero_tools(self):
        adapter = _make_adapter()
        gr = _make_runner(adapter)
        src = _email_source(chat_id=ARMEN, user_id=ARMEN)
        src.email_sender_trusted = True
        src.email_zero_tools = True
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, src, "email"
        )
        self.assertEqual(res, [])

    def test_empty_override_bypasses_platform_tools_readds(self):
        """_get_platform_tools re-adds MCP servers / non-configurable
        toolsets even for an empty list — the explicit-empty contract must
        bypass it entirely."""
        adapter = _make_adapter()
        gr = _make_runner(adapter)
        cfg = {
            "platform_toolsets": {"email": ["web", "vision"]},
            "mcp": {"servers": {"some_mcp": {"enabled": True}}},
        }
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, cfg, _email_source(), "email"
        )
        self.assertEqual(res, [])

    def test_direct_policy_email_uses_platform_resolution(self):
        adapter = _make_adapter(outbound_policy="")
        gr = _make_runner(adapter)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _email_source(), "email"
        )
        self.assertEqual(res, sorted(_get_platform_tools(BASE_CONFIG, "email")))

    def test_nonempty_override_contract_still_validated(self):
        """Generic resolver contract (webhook-style adapters): a non-empty
        override is validated through _get_platform_tools."""

        class _Stub:
            def toolsets_for_source(self, source):
                return ["web", "discord_admin"]

        gr = _make_runner(_Stub())
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _email_source(), "email"
        )
        expected = sorted(
            _get_platform_tools(
                {"platform_toolsets": {"email": ["web", "discord_admin"]}}, "email"
            )
        )
        self.assertEqual(res, expected)
        self.assertNotIn("discord_admin", res)

    def test_raising_adapter_without_contract_falls_back(self):
        """Adapters that do NOT declare fail-closed keep the legacy
        fallback-to-defaults behavior (webhook invariant)."""

        class _Stub:
            def toolsets_for_source(self, source):
                raise RuntimeError("boom")

        gr = _make_runner(_Stub())
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, _email_source(), "email"
        )
        self.assertEqual(res, sorted(_get_platform_tools(BASE_CONFIG, "email")))

    def test_user_config_not_mutated(self):
        class _Stub:
            def toolsets_for_source(self, source):
                return ["web"]

        gr = _make_runner(_Stub())
        cfg = {"platform_toolsets": {"email": ["vision"]}}
        GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, cfg, _email_source(), "email"
        )
        self.assertEqual(cfg["platform_toolsets"]["email"], ["vision"])


class TestFinalToolAssembly(unittest.TestCase):
    def test_empty_toolsets_produce_zero_tool_definitions(self):
        """The exact-zero-tools invariant must hold at the assembly point
        that builds AIAgent.tools (agent_init -> get_tool_definitions), not
        only at the resolver: an explicit empty enabled_toolsets yields an
        empty schema list for the model call."""
        from model_tools import get_tool_definitions

        tools = get_tool_definitions(
            enabled_toolsets=[], disabled_toolsets=None, quiet_mode=True
        )
        self.assertEqual(tools, [])

    def test_none_toolsets_are_not_zero(self):
        """Sanity for the invariant above: None means unrestricted — the
        zero-tool boundary depends on [] and None staying distinct here."""
        from model_tools import get_tool_definitions

        tools = get_tool_definitions(
            enabled_toolsets=None, disabled_toolsets=None, quiet_mode=True
        )
        self.assertTrue(tools)


class TestProxyModeIsolation(unittest.TestCase):
    def test_flagged_source_denied_by_delegation_guard(self):
        gr = _make_runner(_make_adapter())
        src = _email_source()
        src.email_zero_tools = True
        self.assertFalse(gr._proxy_delegation_allowed(src))

    def test_restored_source_still_denied_proxy_delegation(self):
        """Startup auto-resume regression: SessionSource serialization
        deliberately drops the wire-invisible flags, so a restored
        review_first email source arrives with email_zero_tools=False. The
        delegation guard must still refuse it by resolving the effective
        toolsets through the adapter's zero-tool override."""
        restored = SessionSource.from_dict(_email_source().to_dict())
        self.assertFalse(restored.email_zero_tools)
        gr = _make_runner(_make_adapter())
        self.assertFalse(gr._proxy_delegation_allowed(restored))

    def test_absent_adapter_denies_proxy_delegation_for_email(self):
        """An email source whose adapter is not (yet) registered must never
        delegate: the adapter is the only authority on the outbound policy, so
        its absence cannot default to a tool-bearing proxy turn."""
        gr = _make_runner(None)
        self.assertFalse(gr._proxy_delegation_allowed(_email_source()))
        restored = SessionSource.from_dict(_email_source().to_dict())
        self.assertFalse(gr._proxy_delegation_allowed(restored))

    def test_absent_adapter_local_resolution_fails_closed_for_email(self):
        """Reviewer-blocker regression (PR #103977): denying the proxy must not fall
        through to a tool-bearing LOCAL run. A restored source (wire flags dropped by
        serialization) with no live adapter and a config full of email tools resolves
        to zero toolsets — TestFinalToolAssembly proves [] means zero schemas at the
        AIAgent assembly point, so the local fallback is tool-free too."""
        gr = _make_runner(None)
        restored = SessionSource.from_dict(_email_source().to_dict())
        self.assertFalse(restored.email_zero_tools)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_CONFIG, restored, "email"
        )
        self.assertEqual(res, [])

    def test_absent_adapter_keeps_defaults_for_other_platforms(self):
        """The unresolved-authority rule is email-scoped: other platforms with a
        missing adapter keep the legacy default resolution."""
        gr = _make_runner(None)
        src = SessionSource(
            platform=SessionPlatform.TELEGRAM, chat_id="123", chat_type="dm"
        )
        cfg = {"platform_toolsets": {"telegram": ["web"]}}
        res = GatewayRunner._resolve_enabled_toolsets_for_source(gr, cfg, src, "telegram")
        self.assertEqual(res, sorted(_get_platform_tools(cfg, "telegram")))

    def test_raising_adapter_denies_proxy_delegation(self):
        adapter = _make_adapter()
        adapter.toolsets_for_source = lambda source: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        gr = _make_runner(adapter)
        restored = SessionSource.from_dict(_email_source().to_dict())
        self.assertFalse(gr._proxy_delegation_allowed(restored))

    def test_direct_policy_source_allows_proxy_delegation(self):
        gr = _make_runner(_make_adapter(outbound_policy=""))
        self.assertTrue(gr._proxy_delegation_allowed(_email_source()))

    def test_proxy_branch_consults_the_guard(self):
        """The proxy-mode short-circuit in _run_agent_inner must delegate
        only when the guard allows it (source-level assertion of the
        wiring, kept in sync with gateway/run_turn.py)."""
        import inspect

        from gateway import run as run_mod

        src_text = inspect.getsource(run_mod.GatewayRunner._run_agent_inner)
        self.assertIn(
            "self._get_proxy_url() and self._proxy_delegation_allowed(source)",
            src_text,
        )


if __name__ == "__main__":
    unittest.main()
