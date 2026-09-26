"""Send-only/no-auth email enablement at the config + dashboard layer.

Complements the adapter tests: env config must ENABLE email with just
``EMAIL_ADDRESS`` + ``EMAIL_SMTP_HOST`` (no password / IMAP), and the dashboard
setup metadata must list only those two as required. Kept in a dedicated file so
the send-only PR adds no hunks to the shared ``test_config.py``.
"""

import os
import unittest
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, _apply_env_overrides


def _email_platform(env):
    config = GatewayConfig()
    with patch.dict(os.environ, env, clear=True):
        _apply_env_overrides(config)
    return config.platforms[Platform.EMAIL]


class TestEmailEnvEnable(unittest.TestCase):
    def test_minimal_address_plus_smtp_enables_send_only(self):
        pc = _email_platform({
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        })
        self.assertTrue(pc.enabled)
        self.assertEqual(pc.extra.get("address"), "hermes@test.com")
        self.assertEqual(pc.extra.get("smtp_host"), "smtp.test.com")

    def test_full_imap_smtp_config_still_enables(self):
        pc = _email_platform({
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        })
        self.assertTrue(pc.enabled)
        self.assertEqual(pc.extra.get("imap_host"), "imap.test.com")

    def test_smtp_username_flows_into_extra(self):
        pc = _email_platform({
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SMTP_USERNAME": "relay-login@test.com",
        })
        self.assertTrue(pc.enabled)
        self.assertEqual(pc.extra.get("smtp_username"), "relay-login@test.com")

    def test_address_without_smtp_does_not_enable(self):
        config = GatewayConfig()
        with patch.dict(os.environ, {"EMAIL_ADDRESS": "hermes@test.com"}, clear=True):
            _apply_env_overrides(config)
        # A failed credential gate leaves the platform unprovisioned rather than enabled.
        self.assertNotIn(Platform.EMAIL, config.platforms)


class TestDashboardRequiredEnv(unittest.TestCase):
    def test_only_address_and_smtp_required(self):
        from hermes_cli.web_server_messaging import _PLATFORM_OVERRIDES

        entry = _PLATFORM_OVERRIDES["email"]
        self.assertEqual(set(entry["required_env"]), {"EMAIL_ADDRESS", "EMAIL_SMTP_HOST"})
        # Password / IMAP remain available to fill in, but are not required.
        for var in ("EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_USERNAME"):
            self.assertIn(var, entry["env_vars"])


if __name__ == "__main__":
    unittest.main()
