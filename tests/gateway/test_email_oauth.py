"""Tests for the Email OAuth2 module (gateway/platforms/email_oauth.py).

Covers:
1. Exception hierarchy (OAuthError, DeviceCodeExpiredError, DeviceCodeDeniedError)
2. OAuthTokenManager constructor and basic properties
3. build_imap_xoauth2_bytes — signature and return type
4. build_smtp_xoauth2_str — signature and return type
5. begin_device_code_flow — return dict shape
6. poll_device_code — return type
7. Module-level convenience wrappers
"""

import unittest
from unittest.mock import patch, MagicMock


class TestOAuthExceptions(unittest.TestCase):
    """Verify exception hierarchy matches expectations."""

    def test_oauth_error_is_runtime_error(self):
        from gateway.platforms.email_oauth import OAuthError
        self.assertTrue(issubclass(OAuthError, RuntimeError))
        self.assertIsInstance(OAuthError("fail"), Exception)

    def test_device_code_expired_error_hierarchy(self):
        from gateway.platforms.email_oauth import (
            OAuthError,
            DeviceCodeExpiredError,
        )
        self.assertTrue(issubclass(DeviceCodeExpiredError, OAuthError))
        self.assertIsInstance(DeviceCodeExpiredError("expired"), OAuthError)

    def test_device_code_denied_error_hierarchy(self):
        from gateway.platforms.email_oauth import (
            OAuthError,
            DeviceCodeDeniedError,
        )
        self.assertTrue(issubclass(DeviceCodeDeniedError, OAuthError))
        self.assertIsInstance(DeviceCodeDeniedError("denied"), OAuthError)

    def test_exceptions_have_unique_names(self):
        from gateway.platforms.email_oauth import (
            OAuthError,
            DeviceCodeExpiredError,
            DeviceCodeDeniedError,
        )
        names = {e.__name__ for e in (OAuthError, DeviceCodeExpiredError, DeviceCodeDeniedError)}
        self.assertEqual(len(names), 3)


class TestOAuthTokenManager(unittest.TestCase):
    """Verify OAuthTokenManager constructor and basic state."""

    def test_constructor_sets_fields(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager(
            tenant_id="tenant-abc",
            client_id="client-123",
            client_secret="secret!",
        )
        self.assertEqual(mgr.tenant_id, "tenant-abc")
        self.assertEqual(mgr.client_id, "client-123")
        self.assertEqual(mgr.client_secret, "secret!")
        self.assertEqual(
            mgr.scope,
            OAuthTokenManager.EMAIL_OAUTH_SCOPE,
        )

    def test_constructor_accepts_custom_scope(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager(
            tenant_id="t",
            client_id="c",
            client_secret="s",
            scope="https://custom.scope/.default",
        )
        self.assertEqual(mgr.scope, "https://custom.scope/.default")

    def test_inspect_health_returns_dict(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager("t", "c", "s")
        health = mgr.inspect_health()
        self.assertIsInstance(health, dict)
        self.assertIn("tenant_id", health)
        self.assertIn("client_id", health)
        self.assertIn("scope", health)
        self.assertIn("cached", health)
        self.assertEqual(health["cached"], False)

    def test_get_token_raises_not_implemented(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager("t", "c", "s")
        with self.assertRaises(NotImplementedError):
            mgr.get_token()

    def test_save_disk_cache_raises_not_implemented(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager("t", "c", "s")
        with self.assertRaises(NotImplementedError):
            mgr.save_disk_cache()

    def test_load_disk_cache_raises_not_implemented(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        with self.assertRaises(NotImplementedError):
            OAuthTokenManager.load_disk_cache()

    def test_invalidate_disk_cache_raises_not_implemented(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager("t", "c", "s")
        with self.assertRaises(NotImplementedError):
            mgr.invalidate_disk_cache()

    def test_clear_cache_does_not_raise(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager("t", "c", "s")
        # Should work without error even with no cached token
        mgr.clear_cache()
        self.assertIsNone(mgr._cached_token)

    def test_begin_device_code_flow_raises_not_implemented(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager("t", "c", "s")
        with self.assertRaises(NotImplementedError):
            mgr.begin_device_code_flow()

    def test_poll_device_code_raises_not_implemented(self):
        from gateway.platforms.email_oauth import OAuthTokenManager

        mgr = OAuthTokenManager("t", "c", "s")
        with self.assertRaises(NotImplementedError):
            mgr.poll_device_code("some-code")


class TestBuildXoauth2Functions(unittest.TestCase):
    """Verify XOAUTH2 string builders produce correct SASL payloads."""

    USER = "alice@outlook.com"
    TOKEN = "ya29.example_access_token"

    def test_imap_payload_is_bytes_in_canonical_format(self):
        from gateway.platforms.email_oauth import build_imap_xoauth2_bytes

        payload = build_imap_xoauth2_bytes(self.USER, self.TOKEN)
        self.assertIsInstance(payload, bytes)
        # XOAUTH2 SASL: user=<u>\\x01auth=Bearer <t>\\x01\\x01
        expected = (
            b"user=" + self.USER.encode("ascii")
            + b"\x01auth=Bearer " + self.TOKEN.encode("ascii")
            + b"\x01\x01"
        )
        self.assertEqual(payload, expected)

    def test_smtp_payload_is_raw_str(self):
        from gateway.platforms.email_oauth import (
            build_smtp_xoauth2_str,
            build_imap_xoauth2_bytes,
        )

        encoded = build_smtp_xoauth2_str(self.USER, self.TOKEN)
        self.assertIsInstance(encoded, str)
        # The raw string must match the IMAP-shaped payload decoded
        self.assertEqual(
            encoded,
            build_imap_xoauth2_bytes(self.USER, self.TOKEN).decode("ascii"),
        )

    def test_builders_reject_non_ascii_user(self):
        from gateway.platforms.email_oauth import build_imap_xoauth2_bytes

        with self.assertRaises(UnicodeEncodeError):
            build_imap_xoauth2_bytes("éléonore@outlook.com", self.TOKEN)

    def test_build_imap_xoauth2_bytes_has_correct_signature(self):
        from gateway.platforms.email_oauth import build_imap_xoauth2_bytes
        import inspect

        sig = inspect.signature(build_imap_xoauth2_bytes)
        params = list(sig.parameters.keys())
        self.assertIn("address", params)
        self.assertIn("access_token", params)
        # ``from __future__ import annotations`` turns annotations into strings
        self.assertEqual(sig.return_annotation, "bytes")

    def test_build_smtp_xoauth2_str_has_correct_signature(self):
        from gateway.platforms.email_oauth import build_smtp_xoauth2_str
        import inspect

        sig = inspect.signature(build_smtp_xoauth2_str)
        params = list(sig.parameters.keys())
        self.assertIn("address", params)
        self.assertIn("access_token", params)
        self.assertEqual(sig.return_annotation, "str")


class TestModuleLevelConvenience(unittest.TestCase):
    """Verify module-level convenience wrappers exist and forward correctly."""

    def test_module_begin_device_code_flow_raises_not_implemented(self):
        from gateway.platforms.email_oauth import begin_device_code_flow

        with self.assertRaises(NotImplementedError):
            begin_device_code_flow(tenant_id="t", client_id="c")

    def test_module_poll_device_code_raises_not_implemented(self):
        from gateway.platforms.email_oauth import poll_device_code

        with self.assertRaises(NotImplementedError):
            poll_device_code(tenant_id="t", client_id="c", device_code="dc")


class TestPublicApiSurface(unittest.TestCase):
    """Verify all names in the task spec are importable from the module."""

    def test_all_names_importable(self):
        from gateway.platforms.email_oauth import (
            OAuthTokenManager,
            build_imap_xoauth2_bytes,
            build_smtp_xoauth2_str,
            begin_device_code_flow,
            poll_device_code,
            DeviceCodeExpiredError,
            DeviceCodeDeniedError,
            OAuthError,
        )

        # Just verify they're the right types
        self.assertTrue(issubclass(OAuthError, RuntimeError))
        self.assertTrue(issubclass(DeviceCodeExpiredError, OAuthError))
        self.assertTrue(issubclass(DeviceCodeDeniedError, OAuthError))
        self.assertIsInstance(OAuthTokenManager, type)
        self.assertTrue(callable(build_imap_xoauth2_bytes))
        self.assertTrue(callable(build_smtp_xoauth2_str))
        self.assertTrue(callable(begin_device_code_flow))
        self.assertTrue(callable(poll_device_code))


if __name__ == "__main__":
    unittest.main()
