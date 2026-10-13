"""Host vs named-profile platform maps for the Channels page."""

import unittest
from pathlib import Path

from gateway.status import messaging_platforms_for_home


class MessagingPlatformsForHomeTest(unittest.TestCase):
    def test_host_home_keeps_bare_whatsapp_and_hides_other_profiles(self):
        runtime = {
            "platforms": {
                "whatsapp": {"state": "disconnected"},
                "api_server": {"state": "connected"},
                "cmp_market:relay": {"state": "disabled"},
            }
        }
        shown = messaging_platforms_for_home(runtime, Path("/opt/data"))
        self.assertEqual(set(shown), {"whatsapp", "api_server"})
        self.assertEqual(shown["whatsapp"]["state"], "disconnected")

    def test_named_profile_sees_only_its_prefixed_rows(self):
        runtime = {
            "platforms": {
                "whatsapp": {"state": "disconnected"},
                "cmp_market:relay": {"state": "connected"},
            }
        }
        shown = messaging_platforms_for_home(runtime, Path("/opt/data/profiles/cmp_market"))
        self.assertEqual(set(shown), {"relay"})
        self.assertEqual(shown["relay"]["state"], "connected")


if __name__ == "__main__":
    unittest.main()
