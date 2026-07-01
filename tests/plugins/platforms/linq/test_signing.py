"""Unit tests for the gateway-free signing/parsing helpers.

These use only the standard library so they run anywhere — no Hermes runtime,
httpx, or aiohttp required.
"""
import hashlib
import hmac
import time
import unittest

from plugins.platforms.linq import signing


def _sign(secret: str, ts: int, body: bytes) -> str:
    msg = f"{ts}.".encode("utf-8") + body
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


class VerifySignatureTest(unittest.TestCase):
    def setUp(self):
        self.secret = "s3cr3t"
        self.body = b'{"event_type":"message.received"}'
        self.now = 1_700_000_000

    def test_valid_signature(self):
        ts = self.now
        sig = _sign(self.secret, ts, self.body)
        self.assertTrue(
            signing.verify_signature(
                body=self.body, timestamp_header=str(ts), signature_header=sig,
                signing_secret=self.secret, now=self.now,
            )
        )

    def test_sha256_prefix_accepted(self):
        ts = self.now
        sig = "sha256=" + _sign(self.secret, ts, self.body)
        self.assertTrue(
            signing.verify_signature(
                body=self.body, timestamp_header=str(ts), signature_header=sig,
                signing_secret=self.secret, now=self.now,
            )
        )

    def test_wrong_secret_rejected(self):
        ts = self.now
        sig = _sign("other", ts, self.body)
        self.assertFalse(
            signing.verify_signature(
                body=self.body, timestamp_header=str(ts), signature_header=sig,
                signing_secret=self.secret, now=self.now,
            )
        )

    def test_tampered_body_rejected(self):
        ts = self.now
        sig = _sign(self.secret, ts, self.body)
        self.assertFalse(
            signing.verify_signature(
                body=b"tampered", timestamp_header=str(ts), signature_header=sig,
                signing_secret=self.secret, now=self.now,
            )
        )

    def test_stale_timestamp_rejected(self):
        ts = self.now - 10_000
        sig = _sign(self.secret, ts, self.body)
        self.assertFalse(
            signing.verify_signature(
                body=self.body, timestamp_header=str(ts), signature_header=sig,
                signing_secret=self.secret, now=self.now,
            )
        )

    def test_missing_fields_rejected(self):
        self.assertFalse(
            signing.verify_signature(
                body=self.body, timestamp_header="", signature_header="x",
                signing_secret=self.secret, now=self.now,
            )
        )
        self.assertFalse(
            signing.verify_signature(
                body=self.body, timestamp_header=str(self.now), signature_header="x",
                signing_secret="", now=self.now,
            )
        )

    def test_non_numeric_timestamp_rejected(self):
        self.assertFalse(
            signing.verify_signature(
                body=self.body, timestamp_header="not-a-number",
                signature_header="x", signing_secret=self.secret, now=self.now,
            )
        )

    def test_default_now_uses_wall_clock(self):
        ts = int(time.time())
        sig = _sign(self.secret, ts, self.body)
        self.assertTrue(
            signing.verify_signature(
                body=self.body, timestamp_header=str(ts), signature_header=sig,
                signing_secret=self.secret,
            )
        )


class MentionGatingTest(unittest.TestCase):
    def test_defaults_match_hermes_wake_words(self):
        pats = signing.compile_mention_patterns(None)
        self.assertTrue(signing.message_matches_mention("hey hermes, what's up", pats))
        self.assertTrue(signing.message_matches_mention("@hermes agent ping", pats))
        self.assertFalse(signing.message_matches_mention("just chatting", pats))

    def test_clean_strips_leading_wake_word_only(self):
        pats = signing.compile_mention_patterns(None)
        self.assertEqual(signing.clean_mention_text("hermes do the thing", pats), "do the thing")
        # A later occurrence is left intact.
        self.assertEqual(
            signing.clean_mention_text("tell hermes hello", pats), "tell hermes hello"
        )

    def test_custom_patterns_from_json_string(self):
        pats = signing.compile_mention_patterns('["@?amos\\\\b"]')
        self.assertTrue(signing.message_matches_mention("amos ping", pats))
        self.assertFalse(signing.message_matches_mention("hermes ping", pats))

    def test_custom_patterns_from_csv_string(self):
        pats = signing.compile_mention_patterns("bot, assistant")
        self.assertTrue(signing.message_matches_mention("assistant help", pats))

    def test_invalid_regex_skipped(self):
        pats = signing.compile_mention_patterns(["(", "valid"])
        # The bad pattern is skipped; the good one still compiles.
        self.assertEqual(len(pats), 1)


class ParsingTest(unittest.TestCase):
    def test_extract_text_joins_text_parts(self):
        parts = [
            {"type": "text", "value": "hello"},
            {"type": "media", "url": "https://x/y.jpg"},
            {"type": "text", "value": "world"},
        ]
        self.assertEqual(signing.extract_text(parts), "hello\nworld")

    def test_extract_media_returns_urls_and_mime(self):
        parts = [
            {"type": "media", "url": "https://x/y.jpg", "mime_type": "image/jpeg"},
            {"type": "media", "filename": "no-url.pdf"},  # dropped: no url
            {"type": "text", "value": "hi"},
        ]
        media = signing.extract_media(parts)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0]["url"], "https://x/y.jpg")
        self.assertEqual(media[0]["mime_type"], "image/jpeg")

    def test_is_group_chat_heuristics(self):
        self.assertTrue(signing.is_group_chat({"is_group": True}))
        self.assertTrue(signing.is_group_chat({"group_id": "g1"}))
        self.assertTrue(signing.is_group_chat({"participants": ["a", "b", "c"]}))
        self.assertTrue(signing.is_group_chat({"chat_type": "group"}))
        self.assertFalse(signing.is_group_chat({"from": "+15551112222"}))
        self.assertFalse(signing.is_group_chat({"participants": ["a", "b"]}))

    def test_coerce_helpers(self):
        self.assertEqual(signing.coerce_port("8790", 1), 8790)
        self.assertEqual(signing.coerce_port(None, 8790), 8790)
        self.assertEqual(signing.coerce_port("nope", 8790), 8790)
        self.assertTrue(signing.coerce_bool("yes"))
        self.assertTrue(signing.coerce_bool(True))
        self.assertFalse(signing.coerce_bool("off"))
        self.assertTrue(signing.coerce_bool(None, default=True))


if __name__ == "__main__":
    unittest.main()
