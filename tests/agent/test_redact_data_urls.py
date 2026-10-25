"""Inline base64 data-URL payloads must survive redaction byte-identical (#136388).

vision_analyze embeds images as ``data:<mime>;base64,…`` parts in tool messages; every
boundary that rewrites message content with ``redact_sensitive_text(force=True)`` (request
dumps, compaction, egress sweeps) must not corrupt them — one masked span inside the base64
bricks every later image request with a non-retryable 400. Same class as the CDP binary
payloads (#94138), but fixed in the redactor itself so every boundary is covered.
"""

import json

from agent.redact import redact_for_egress, redact_sensitive_text

# Hand-built payloads (deterministic, no image codec needed) that plant the exact
# substrings the secret passes key on: the JWT gate ``eyJ`` mid-stream, the Fernet
# prefix ``gAAAA`` after a ``/`` (#94138's trigger), and ``=`` padding followed by
# more base64 (the assignment-pass trigger).
_JPEG_URL = (
    "data:image/jpeg;base64,"
    + "A" * 40 + "eyJhbGciOiJIUzI1NiJ9" + "B" * 60
    + "/gAAAA" + "C" * 60 + "+D" * 20 + "=="
)
_PNG_URL = "data:image/png;base64," + "iVBORw0KGgo" + "eyJ" + "nS/CXSjqHj" * 9 + "="

# Fake fixture body, assembled so no usable credential literal sits in source.
_FAKE_KEY = "sk-ant-api03-" + "F4kE" + "KeyBody" + "0123456789" + "abcdefXYZ"


class TestInlineDataUrlPassthrough:

    def test_image_payloads_pass_through_byte_identical(self):
        for url in (_JPEG_URL, _PNG_URL):
            assert redact_sensitive_text(url, force=True) == url
            assert redact_for_egress(url) == url  # gateway/A2A egress sweep

    def test_url_credentials_mode_keeps_payload_intact(self):
        out = redact_sensitive_text(_JPEG_URL, force=True, redact_url_credentials=True)
        assert out == _JPEG_URL

    def test_serialized_tool_message_round_trip(self):
        """The shape of the real boundary: the whole message list is JSON-serialized and
        force-redacted in one string (request dumps, compressor, memory egress)."""
        message = {
            "role": "tool", "tool_name": "vision_analyze",
            "content": [
                {"type": "text", "text": "Image loaded into your context."},
                {"type": "image_url", "image_url": {"url": _JPEG_URL}},
            ],
        }
        serialized = json.dumps(message)
        out = redact_sensitive_text(serialized, force=True)
        assert _JPEG_URL in out
        assert json.loads(out) == message

    def test_payload_boundary_stops_where_base64_does(self):
        """Text after the payload is ordinary content and keeps full redaction."""
        text = _PNG_URL + " then the key " + _FAKE_KEY
        out = redact_sensitive_text(text, force=True)
        assert _PNG_URL in out
        assert _FAKE_KEY not in out


class TestFailClosedAroundTheExemption:

    def test_bare_jwt_next_to_data_url_still_redacted(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        out = redact_sensitive_text(f"token {jwt} and {_PNG_URL}", force=True)
        assert jwt not in out
        assert _PNG_URL in out

    def test_non_base64_data_url_keeps_full_redaction(self):
        """Only ``;base64,`` media payloads are exempt — an inline ``data:text/plain``
        URL is free text and a secret inside it must still be masked."""
        url = f"data:text/plain,see {_FAKE_KEY} here"
        out = redact_sensitive_text(url, force=True)
        assert _FAKE_KEY not in out

    def test_bare_base64_outside_a_data_url_still_redacted(self):
        """The reporter's standalone repro (a bare base64 blob with no ``data:`` header)
        cannot be exempted: it is indistinguishable from a real JWT, so the mask stays.
        The fix scopes the passthrough to the structured ``data:…;base64,`` span only."""
        blob = "A" * 40 + "eyJhbGciOiJIUzI1NiJ9" + "B" * 60
        out = redact_sensitive_text(blob, force=True)
        assert out != blob
