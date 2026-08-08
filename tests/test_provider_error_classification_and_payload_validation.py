"""Regression tests for the 2026-08-07 provider-failure incident.

Two independent defects, both reproduced here before the fixes landed:

DEFECT 1 (root cause) — ``_ensure_leading_user_turn`` in
``agent/anthropic_adapter.py`` prepended a literal ``" "`` text block when
compression stranded an assistant message at index 0. Anthropic rejects
whitespace-only text blocks, so it traded one HTTP 400 for another and wedged
the session: every subsequent turn re-sent the same poisoned array.

DEFECT 2 (why it cost a morning) — ``_gateway_provider_error_reply`` in
``gateway/run.py`` had no branch for malformed-request 4xx, so a payload bug,
a billing notice and an invalid API key ALL rendered as the same string:
"The model provider failed after retries." That string was wrong three ways
(the provider was fine, nothing was retried, and the actionable field was
hidden).

Runs under pytest, or standalone: ``python3 tests/test_...py``
(pytest is not currently installed in this checkout).
"""

import json
import os
import re
import sys
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --------------------------------------------------------------------------
# Load the two units under test WITHOUT importing their heavyweight packages.
# gateway/run.py is ~26k lines and pulls the whole agent stack; agent/
# anthropic_adapter.py needs the anthropic SDK. Both are unnecessary here —
# we extract the specific functions by exec'ing the relevant source region.
# --------------------------------------------------------------------------

def _load_gateway_error_classifier():
    """Exec just the provider-error classifier block out of gateway/run.py."""
    path = os.path.join(REPO_ROOT, "gateway", "run.py")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    start = src.index("_GATEWAY_PROVIDER_POLICY_RE = re.compile(")
    end = src.index("_GATEWAY_SECRET_PATTERNS = (")
    block_a = src[start:end]

    start_b = src.index("_GATEWAY_HTTP_STATUS_RE = re.compile(")
    end_b = src.index("_GATEWAY_PROVIDER_ERROR_SHAPE_RE = re.compile(")
    block_b = src[start_b:end_b]

    mod = types.ModuleType("_gw_err")
    mod.__dict__["re"] = re
    exec(compile(block_a + "\n" + block_b, path, "exec"), mod.__dict__)
    return mod


def _load_anthropic_validator():
    """Import the real adapter module.

    Previously this exec'd a source slice to avoid importing the anthropic
    SDK. That coupled the test to exact symbol names and broke on rebase.
    Import the module for real — it exercises the actual code path, which is
    what we want anyway.
    """
    import agent.anthropic_adapter as mod

    return mod


GW = _load_gateway_error_classifier()
AV = _load_anthropic_validator()

FALLBACK_SUBSTRING = "failed after retries"

# The exact body Anthropic returned, from the saved request dump
# request_dump_20260806_125912_f692bbae_20260807_095517_988018.json
REAL_400_PAYLOAD = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'messages: text content blocks must "
    "contain non-whitespace text'}, 'request_id': "
    "'req_011CdoafYLfRyFsdyHCy1hZ9'}"
)

# Real billing 400 seen across 5+ sessions on 2026-08-02
REAL_BILLING_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'Third-party apps now draw from your "
    "extra usage, not your plan limits. Add more at "
    "claude.ai/settings/usage and keep going.'}}"
)

REAL_INVALID_KEY = (
    "Error code: 401 - {'type': 'error', 'error': {'type': "
    "'authentication_error', 'message': 'invalid x-api-key'}}"
)

REAL_USAGE_LIMIT = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'You have reached your specified API "
    "usage limits. You will regain access on 2026-09-01 at 00:00 UTC.'}}"
)


class TestGatewayProviderErrorClassifier(unittest.TestCase):
    """DEFECT 2 — every 4xx must surface status + provider message."""

    def test_malformed_payload_400_is_not_the_generic_fallback(self):
        """The incident case: must NOT read 'failed after retries'."""
        reply = GW._gateway_provider_error_reply(REAL_400_PAYLOAD)
        self.assertNotIn(FALLBACK_SUBSTRING, reply)

    def test_malformed_payload_400_surfaces_status_type_and_message(self):
        reply = GW._gateway_provider_error_reply(REAL_400_PAYLOAD)
        self.assertIn("400", reply)
        self.assertIn("invalid_request_error", reply)
        self.assertIn("text content blocks must contain non-whitespace text", reply)

    def test_malformed_payload_400_states_it_is_terminal(self):
        """A 400 is deterministic — the reply must not imply retries happened."""
        reply = GW._gateway_provider_error_reply(REAL_400_PAYLOAD)
        self.assertIn("terminal", reply.lower())
        self.assertIn("not retried", reply.lower())

    def test_billing_400_is_distinguishable_from_payload_400(self):
        """Billing and payload 400s rendered identically before the fix."""
        billing = GW._gateway_provider_error_reply(REAL_BILLING_400)
        payload = GW._gateway_provider_error_reply(REAL_400_PAYLOAD)
        self.assertNotEqual(billing, payload)
        self.assertNotIn(FALLBACK_SUBSTRING, billing)
        self.assertIn("extra usage", billing)

    def test_usage_limit_400_surfaces_the_regain_date(self):
        reply = GW._gateway_provider_error_reply(REAL_USAGE_LIMIT)
        self.assertNotIn(FALLBACK_SUBSTRING, reply)
        self.assertIn("2026-09-01", reply)

    def test_invalid_api_key_is_classified_as_auth_and_says_so(self):
        reply = GW._gateway_provider_error_reply(REAL_INVALID_KEY)
        self.assertNotIn(FALLBACK_SUBSTRING, reply)
        self.assertIn("authentication", reply.lower())
        self.assertIn("invalid x-api-key", reply)

    def test_all_three_error_classes_render_differently(self):
        """The core regression: one useless string for three distinct faults."""
        replies = {
            GW._gateway_provider_error_reply(REAL_400_PAYLOAD),
            GW._gateway_provider_error_reply(REAL_BILLING_400),
            GW._gateway_provider_error_reply(REAL_INVALID_KEY),
        }
        self.assertEqual(len(replies), 3, "error classes must not collapse")

    def test_5xx_still_mentions_retries(self):
        """Server errors ARE retried — that wording is correct for them."""
        reply = GW._gateway_provider_error_reply(
            "Error code: 529 - {'type': 'error', 'error': "
            "{'type': 'overloaded_error', 'message': 'Overloaded'}}"
        )
        self.assertIn("529", reply)
        self.assertIn("after retries", reply)

    def test_status_extraction_variants(self):
        for text, expected in [
            ("HTTP 400: bad", 400),
            ("Error code: 429 - {}", 429),
            ("status_code=503", 503),
            ("no status here", None),
            ("HTTP 999 nonsense", None),
        ]:
            self.assertEqual(GW._extract_gateway_http_status(text), expected, text)

    def test_provider_message_is_length_capped(self):
        long_msg = "x" * 5000
        text = "Error code: 400 - {'error': {'message': '%s'}}" % long_msg
        out = GW._extract_gateway_provider_message(text)
        self.assertLessEqual(len(out), GW._GATEWAY_PROVIDER_MSG_MAX + 1)

    def _fake_credentials(self):
        """Credential-shaped fixtures assembled at RUNTIME.

        These are fake, but they must match the real credential SHAPES or the
        redaction regexes are not actually exercised. Written as concatenated
        fragments so no complete credential-shaped literal exists in the
        source: GitHub push protection (secret scanning) rejects the push
        otherwise, and a scanner cannot tell a test fixture from a live key.
        """
        return [
            "sk-" + "ant-api03-" + "A" * 24 + "xyz",
            "gh" + "p_" + "B" * 36,
            "ey" + "JhbGciOiJIUzI1NiJ9." + "C" * 20 + "." + "D" * 16,
            "xo" + "xb-" + "1234567890-" + "E" * 24,
            "AK" + "IA" + "F" * 16,
            "hf" + "_" + "G" * 34,
        ]

    def test_no_api_key_material_reaches_the_reply(self):
        """Credentials must never be surfaced, even when embedded in the body."""
        secret = self._fake_credentials()[0]
        text = (
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': 'bad request'}, "
            "'key': '%s'}" % secret
        )
        reply = GW._gateway_provider_error_reply(text)
        self.assertNotIn(secret, reply)

    def test_secret_echoed_inside_the_provider_message_is_redacted(self):
        """Regression: surfacing the provider message leaked the credential.

        Providers echo the offending key back inside ``message`` (Anthropic
        returns "invalid x-api-key <key>"). The first version of this fix
        surfaced ``message`` verbatim and leaked real key material to chat —
        caught only by replaying credential-shaped values through the
        classifier. Redaction now happens inside
        ``_extract_gateway_provider_message`` rather than being delegated to
        the caller, because one of the two call sites passes the RAW body.
        """
        templates = [
            "Error code: 401 - {'type':'error','error':{'type':"
            "'authentication_error','message':'invalid x-api-key %s'}}",
            "Error code: 400 - {'type':'error','error':{'type':"
            "'invalid_request_error','message':'token %s rejected'}}",
            "API call failed: HTTP 403 - Authorization: Bearer %s",
        ]
        for secret in self._fake_credentials():
            for tmpl in templates:
                reply = GW._gateway_provider_error_reply(tmpl % secret)
                self.assertNotIn(
                    secret, reply,
                    "credential leaked to chat: %s…" % secret[:8],
                )

    def test_status_is_retained_while_secrets_are_stripped(self):
        """Both properties must hold at once — one must not defeat the other."""
        reply = GW._gateway_provider_error_reply(
            "API call failed after 3 retries: HTTP 401 Unauthorized — "
            "Authorization: Bearer sk-ABCDEF0123456789abcdef"
        )
        self.assertIn("401", reply)
        self.assertNotIn("sk-ABCDEF", reply)


class TestScrubAndValidateLayering(unittest.TestCase):
    """DEFECT 1 — the two-layer guard.

    Upstream's ``_scrub_blank_text_blocks`` REPAIRS blank text blocks
    (including tool_result inner content and cache_control relocation).
    ``validate_anthropic_messages`` runs AFTER it and FAILS LOUDLY on what
    repair cannot fix. These tests pin the division of labour: if the
    validator ever fires on something the scrub already repaired, that is a
    layering bug, not a reason to loosen the validator.
    """

    def test_leading_user_turn_placeholder_is_non_whitespace(self):
        """The literal ' ' placeholder is what caused the incident."""
        self.assertTrue(AV._EMPTY_TEXT_PLACEHOLDER.strip())

    def test_scrub_repairs_the_incident_payload_and_gate_accepts_it(self):
        """messages[0] = {'text': ' '} — the exact captured defect."""
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": " "}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        AV._scrub_blank_text_blocks(msgs)
        self.assertTrue(msgs[0]["content"][0]["text"].strip())
        AV.validate_anthropic_messages(msgs)  # must not raise

    def test_scrub_drops_blank_block_alongside_others(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "   "},
                {"type": "text", "text": "real content"},
            ],
        }]
        AV._scrub_blank_text_blocks(msgs)
        self.assertEqual(len(msgs[0]["content"]), 1)
        self.assertEqual(msgs[0]["content"][0]["text"], "real content")
        AV.validate_anthropic_messages(msgs)

    def test_scrub_never_empties_a_message(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "\t\n  "}]}]
        AV._scrub_blank_text_blocks(msgs)
        self.assertGreater(len(msgs[0]["content"]), 0)
        AV.validate_anthropic_messages(msgs)

    def test_end_to_end_assistant_first_history_is_valid(self):
        """Compression strands an assistant turn at index 0 — the trigger."""
        _system, result = AV.convert_messages_to_anthropic([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "[Context compaction summary] work"},
            {"role": "user", "content": "continue"},
        ])
        self.assertEqual(result[0]["role"], "user")
        self.assertTrue(result[0]["content"][0]["text"].strip())

    def test_blank_block_reaching_the_gate_is_reported_as_a_layering_bug(self):
        """If the scrub is skipped, the gate must fail loudly, not repair."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": " "}]}]
        with self.assertRaises(AV.AnthropicMessageValidationError) as cm:
            AV.validate_anthropic_messages(msgs)
        self.assertIn("LAYERING bug", str(cm.exception))


class TestValidatorCatchesUnrepairable(unittest.TestCase):
    """The four classes the scrub does not cover — must raise, never guess."""

    def test_empty_content_array_raises_with_index(self):
        with self.assertRaises(AV.AnthropicMessageValidationError) as cm:
            AV.validate_anthropic_messages([{"role": "user", "content": []}])
        self.assertIn("messages[0]", str(cm.exception))
        self.assertIn("empty", str(cm.exception).lower())

    def test_whitespace_only_string_content_raises(self):
        with self.assertRaises(AV.AnthropicMessageValidationError) as cm:
            AV.validate_anthropic_messages([{"role": "user", "content": "   "}])
        self.assertIn("whitespace-only", str(cm.exception))

    def test_orphaned_tool_result_raises_and_names_the_block(self):
        msgs = [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_ORPHANED",
                "content": "result",
            }],
        }]
        with self.assertRaises(AV.AnthropicMessageValidationError) as cm:
            AV.validate_anthropic_messages(msgs)
        self.assertIn("orphaned tool_result", str(cm.exception))
        self.assertIn("messages[0].content[0]", str(cm.exception))

    def test_matched_tool_result_passes(self):
        AV.validate_anthropic_messages([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_OK", "name": "t", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_OK", "content": "r"},
            ]},
        ])

    def test_consecutive_same_role_raises_with_both_indices(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
        ]
        with self.assertRaises(AV.AnthropicMessageValidationError) as cm:
            AV.validate_anthropic_messages(msgs)
        self.assertIn("consecutive same-role", str(cm.exception))
        self.assertIn("messages[1]", str(cm.exception))

    def test_message_with_no_content_at_all_raises(self):
        with self.assertRaises(AV.AnthropicMessageValidationError):
            AV.validate_anthropic_messages([{"role": "user"}])

    def test_invalid_role_raises(self):
        with self.assertRaises(AV.AnthropicMessageValidationError) as cm:
            AV.validate_anthropic_messages(
                [{"role": "system", "content": [{"type": "text", "text": "x"}]}]
            )
        self.assertIn("invalid role", str(cm.exception))

    def test_clean_array_passes_silently(self):
        AV.validate_anthropic_messages([
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ])


class TestAgainstRealCapturedPayload(unittest.TestCase):
    """Drive the validator with the actual failing request from disk."""

    DUMP = os.path.expanduser(
        "~/.hermes/sessions/"
        "request_dump_20260806_125912_f692bbae_20260807_095517_988018.json"
    )

    def test_real_poisoned_payload_is_repaired_then_accepted(self):
        """Scrub repairs the captured defect; the gate then accepts it."""
        if not os.path.exists(self.DUMP):
            self.skipTest("request dump not present on this machine")
        with open(self.DUMP, "r", encoding="utf-8") as fh:
            dump = json.load(fh)
        body = dump["request"]["body"]
        if isinstance(body, str):
            body = json.loads(body)
        msgs = body["messages"]

        # Precondition: the captured array really does contain the defect.
        empties = [
            i for i, m in enumerate(msgs)
            if isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "text"
                and not b.get("text", "").strip()
                for b in m["content"]
            )
        ]
        self.assertEqual(empties, [0], "expected the empty block at messages[0]")

        # Layer 1 repairs it.
        AV._scrub_blank_text_blocks(msgs)
        # Layer 2 accepts the repaired array — it must NOT fire here.
        AV.validate_anthropic_messages(msgs)

        # Post-condition: no empty text block survives anywhere.
        for m in msgs:
            if isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "text":
                        self.assertTrue(b["text"].strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
