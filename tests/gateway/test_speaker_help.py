import unittest

from gateway.slash_commands_hardware import _SPEAKER_USAGE


class SpeakerHelpTests(unittest.TestCase):
    def test_speaker_help_is_mobile_readable(self):
        help_text = _SPEAKER_USAGE
        self.assertIn("*Speaker commands*", help_text)
        self.assertIn("• `/speaker status` — show the speaker state", help_text)
        self.assertIn("`random` · `longest-title` · `shortest-title` · `newest` · `oldest`", help_text)
        self.assertNotIn("status|stop|fm|npc|haunted|roulette|hn|podcast|oracle", help_text)
        self.assertNotIn("  status                         ", help_text)


if __name__ == "__main__":
    unittest.main()
