"""Unit tests for credential storage + precedence (stdlib only)."""
import importlib
import os
import tempfile
import unittest

from plugins.platforms.linq import auth as _linq_auth


class AuthTest(unittest.TestCase):
    def setUp(self):
        # Point ~/.hermes at a throwaway dir for the duration of each test.
        self._tmp = tempfile.TemporaryDirectory()
        self._home = self._tmp.name
        self._old_home = os.environ.get("HOME")
        os.environ["HOME"] = self._home
        for var in ("LINQ_API_TOKEN", "LINQ_FROM_PHONE"):
            os.environ.pop(var, None)
        importlib.reload(_linq_auth)
        self.auth = _linq_auth

    def tearDown(self):
        if self._old_home is not None:
            os.environ["HOME"] = self._old_home
        for var in ("LINQ_API_TOKEN", "LINQ_FROM_PHONE"):
            os.environ.pop(var, None)
        self._tmp.cleanup()

    def test_store_and_load_roundtrip(self):
        self.auth.store_credentials("tok-123", from_phone="+15551112222")
        self.assertEqual(self.auth.load_token(), "tok-123")
        self.assertEqual(self.auth.load_from_phone(), "+15551112222")
        self.assertEqual(self.auth.load_credentials(), ("tok-123", "+15551112222"))

    def test_env_token_wins_over_file(self):
        self.auth.store_credentials("file-token", from_phone="+1999")
        os.environ["LINQ_API_TOKEN"] = "env-token"
        os.environ["LINQ_FROM_PHONE"] = "+1888"
        self.assertEqual(self.auth.load_token(), "env-token")
        self.assertEqual(self.auth.load_from_phone(), "+1888")

    def test_missing_returns_none(self):
        self.assertIsNone(self.auth.load_token())
        self.assertIsNone(self.auth.load_from_phone())

    def test_auth_file_is_chmod_600(self):
        self.auth.store_credentials("tok")
        path = self.auth._auth_json_path()
        self.assertTrue(path.exists())
        mode = oct(path.stat().st_mode & 0o777)
        self.assertEqual(mode, "0o600")

    def test_summary_emits_without_leaking_token(self):
        self.auth.store_credentials("super-secret-token", from_phone="+15551112222")
        lines = []
        self.auth.print_credential_summary(lines.append)
        joined = "\n".join(lines)
        self.assertIn("api token", joined)
        self.assertNotIn("super-secret-token", joined)


if __name__ == "__main__":
    unittest.main()
