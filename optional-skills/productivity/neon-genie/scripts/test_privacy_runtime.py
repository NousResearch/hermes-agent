#!/usr/bin/env python3
"""Focused deterministic checks for the repository privacy boundary."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from privacy_runtime import (
    CONTRACT_VERSION,
    assert_envelope_privacy,
    assert_no_raw_secrets,
    build_consent_record,
    build_external_action,
    default_privacy_context,
    enumerate_external_actions,
    learning_ledger_privacy_fields,
    prepare_egress,
    privacy_egress_check,
    privacy_findings,
    privacy_receipt_fields,
    redact_payload,
    resolve_privacy_config_from_brief,
    sanitize_error_message,
    validate_consent_record,
    validate_external_action,
)


class PrivacyRuntimeTests(unittest.TestCase):
    def test_local_only_blocks_external_action(self):
        self.assertEqual(
            privacy_egress_check(
                "public market size", "example.com", "research", default_privacy_context()
            ),
            "BLOCK",
        )

    def test_public_research_allows_when_explicit(self):
        c = default_privacy_context(mode="external_research_allowed")
        self.assertEqual(
            privacy_egress_check("public market size", "example.com", "public research", c),
            "ALLOW",
        )

    def test_secret_never_egresses_or_appears_in_finding(self):
        secret = "sk_abcdefghijklmnopqrstuvwxyz123456"
        self.assertEqual(
            privacy_egress_check(
                secret,
                "example.com",
                "search",
                default_privacy_context(mode="external_research_allowed"),
            ),
            "BLOCK",
        )
        findings_blob = str(privacy_findings(secret))
        self.assertNotIn(secret, findings_blob)
        prep = prepare_egress(
            secret,
            "example.com",
            "search",
            default_privacy_context(mode="external_research_allowed"),
        )
        self.assertEqual(prep["decision"], "BLOCK")
        self.assertNotIn(secret, json.dumps(prep))

    def test_unknown_destination_or_purpose_blocks(self):
        c = default_privacy_context(mode="external_research_allowed")
        self.assertEqual(privacy_egress_check("public", None, "research", c), "BLOCK")
        self.assertEqual(privacy_egress_check("public", "example.com", None, c), "BLOCK")
        self.assertEqual(privacy_egress_check("public", "", "research", c), "BLOCK")
        self.assertEqual(privacy_egress_check("public", "example.com", "", c), "BLOCK")

    def test_local_envelope_rejects_actions(self):
        c = default_privacy_context()
        c["external_actions"].append({"egress_decision": "ALLOW"})
        with self.assertRaises(ValueError):
            assert_envelope_privacy(c)

    def test_explicit_consent_for_private_customer_lists(self):
        payload = (
            "customer list export: alice@example.com bob@example.com "
            "carol@example.com dana@example.com"
        )
        c = default_privacy_context(mode="external_research_allowed")
        self.assertEqual(
            privacy_egress_check(payload, "crm.example.com", "crm enrichment research", c),
            "REQUEST_CONSENT",
        )
        consent = build_consent_record(
            purpose="crm enrichment research",
            categories_allowed=["private_email_list", "private_customer_list"],
            destinations=["crm.example.com"],
            issued_at="2026-01-01T00:00:00Z",
        )
        c2 = default_privacy_context(mode="external_research_allowed", consents=[consent])
        decision = privacy_egress_check(
            payload, "crm.example.com", "crm enrichment research", c2
        )
        self.assertEqual(decision, "REDACT_THEN_ALLOW")

    def test_redaction_path_and_public_query_preservation(self):
        payload = (
            "public market size for developer tools 2026 "
            "contact alice@example.com for notes"
        )
        c = default_privacy_context(mode="external_research_allowed")
        decision = privacy_egress_check(payload, "example.com", "public market research", c)
        self.assertEqual(decision, "REDACT_THEN_ALLOW")
        prep = prepare_egress(
            payload,
            "example.com",
            "public market research",
            c,
            recorded_at="2026-07-31T12:00:00Z",
        )
        self.assertEqual(prep["decision"], "REDACT_THEN_ALLOW")
        self.assertIsNotNone(prep["safe_query"])
        self.assertIn("public market size", prep["safe_query"])
        self.assertIn("developer tools", prep["safe_query"])
        self.assertNotIn("alice@example.com", prep["safe_query"])
        self.assertIn("[EMAIL_REDACTED]", prep["safe_query"])
        self.assertFalse(prep["redaction"]["private_source_persisted"])
        self.assertNotIn(payload, json.dumps(prep["external_action"]))

    def test_redact_payload_strips_secrets_without_echo(self):
        secret = "sk_abcdefghijklmnopqrstuvwxyz123456"
        payload = f"search public pricing {secret}"
        red = redact_payload(payload)
        self.assertNotIn(secret, red["safe_query"])
        self.assertIn("public pricing", red["safe_query"])
        self.assertTrue(red["redaction_applied"])
        self.assertFalse(red["private_source_persisted"])
        self.assertNotIn("payload", red)

    def test_receipt_external_action_enumeration(self):
        action = build_external_action(
            provider="host",
            tool_class="web_search",
            destination="example.com",
            purpose="public market research",
            source_class="public_research",
            classification="public",
            decision="ALLOW",
            redaction_status="not_required",
            recorded_at="2026-07-31T12:00:00Z",
            safe_query="public market size developer tools",
        )
        c = default_privacy_context(
            mode="external_research_allowed", external_actions=[action]
        )
        enumerated = enumerate_external_actions(c)
        self.assertEqual(len(enumerated), 1)
        for field in (
            "provider",
            "tool_class",
            "destination",
            "purpose",
            "source_class",
            "classification",
            "decision",
            "redaction_status",
            "recorded_at",
        ):
            self.assertIn(field, enumerated[0])
            self.assertTrue(enumerated[0][field])
        fields = privacy_receipt_fields(c)
        self.assertEqual(fields["external_actions"], enumerated)
        assert_envelope_privacy(c)

    def test_raw_secret_non_disclosure_in_errors_logs_receipts(self):
        secret = "ghp_abcdefghijklmnopqrstuvwx"
        msg = f"egress failed for token {secret}"
        safe = sanitize_error_message(msg)
        self.assertNotIn(secret, safe)
        with self.assertRaises(ValueError) as ctx:
            assert_no_raw_secrets(json.dumps({"note": secret}), context="receipt")
        self.assertNotIn(secret, str(ctx.exception))
        # findings must not include the secret value
        self.assertNotIn(secret, str(privacy_findings(secret)))

    def test_unknown_retention_assertions(self):
        c = default_privacy_context()
        self.assertEqual(c["unknowns"]["host_providers"], "NOT_COMPUTABLE")
        self.assertEqual(c["unknowns"]["provider_retention"], "NOT_COMPUTABLE")
        self.assertEqual(c["unknowns"]["provider_processing"], "NOT_COMPUTABLE")
        self.assertEqual(c["retention"]["host_provider_retention"], "NOT_COMPUTABLE")
        fields = privacy_receipt_fields(c)
        self.assertIn("NOT_COMPUTABLE", fields["retention_statement"])
        self.assertEqual(fields["privacy_unknowns"]["provider_retention"], "NOT_COMPUTABLE")

    def test_local_learning_ledger_disclosure(self):
        c = default_privacy_context()
        self.assertTrue(c["learning_ledger"]["auto_apply_forbidden"])
        self.assertFalse(c["learning_ledger"]["ships_with_skill"])
        disclosure = c["learning_ledger"]["disclosure"]
        self.assertIn("local", disclosure.lower())
        self.assertIn("auto-applied", disclosure.lower())
        ll = learning_ledger_privacy_fields(c)
        self.assertTrue(ll["learning_local_only"])
        self.assertTrue(ll["learning_auto_apply_forbidden"])
        self.assertEqual(ll["host_provider_retention"], "NOT_COMPUTABLE")
        self.assertEqual(ll["privacy_mode"], "local_only")

    def test_external_action_unknown_purpose_destination_blocking(self):
        c = default_privacy_context(mode="external_research_allowed")
        self.assertEqual(
            privacy_egress_check("public query", None, "research", c), "BLOCK"
        )
        self.assertEqual(
            privacy_egress_check("public query", "example.com", None, c), "BLOCK"
        )
        with self.assertRaises(ValueError):
            build_external_action(
                provider="host",
                tool_class="web_search",
                destination="",
                purpose="research",
                source_class="public_research",
                classification="public",
                decision="ALLOW",
                redaction_status="not_required",
                recorded_at="2026-07-31T12:00:00Z",
            )
        with self.assertRaises(ValueError):
            build_external_action(
                provider="host",
                tool_class="web_search",
                destination="example.com",
                purpose="",
                source_class="public_research",
                classification="public",
                decision="ALLOW",
                redaction_status="not_required",
                recorded_at="2026-07-31T12:00:00Z",
            )

    def test_external_action_requires_full_record(self):
        with self.assertRaises(ValueError) as ctx:
            validate_external_action({"provider": "host", "decision": "ALLOW"})
        self.assertIn("NG-PRIVACY-037", str(ctx.exception))
        with self.assertRaises(ValueError):
            build_external_action(
                provider="host",
                tool_class="web_search",
                destination="example.com",
                purpose="research",
                source_class="public_research",
                classification="public",
                decision="BLOCK",  # cannot record blocked as external action
                redaction_status="blocked",
                recorded_at="2026-07-31T12:00:00Z",
            )
        with self.assertRaises(ValueError) as ctx2:
            build_external_action(
                provider="host",
                tool_class="web_search",
                destination="example.com",
                purpose="research",
                source_class="public_research",
                classification="public",
                decision="ALLOW",
                redaction_status="not_required",
                recorded_at="not-a-timestamp",
            )
        self.assertIn("NG-PRIVACY-035", str(ctx2.exception))

    def test_purpose_bound_consent_never_global_disable(self):
        with self.assertRaises(ValueError) as ctx:
            validate_consent_record(
                {
                    "consent_id": "x",
                    "scope": "global_disable",
                    "purpose": "anything",
                    "categories_allowed": ["all"],
                    "source_class": "operator",
                    "issued_at": "2026-01-01T00:00:00Z",
                }
            )
        self.assertIn("NG-PRIVACY-022", str(ctx.exception))
        with self.assertRaises(ValueError):
            validate_consent_record(
                {
                    "consent_id": "x",
                    "scope": "disable_privacy",
                    "purpose": "anything",
                    "categories_allowed": ["all"],
                    "source_class": "operator",
                    "issued_at": "2026-01-01T00:00:00Z",
                }
            )
        ok = build_consent_record(
            purpose="public market research for pricing band",
            categories_allowed=["public_query"],
            issued_at="2026-01-01T00:00:00Z",
        )
        self.assertEqual(ok["scope"], "purpose_bound")

    def test_brief_privacy_config_flows_deterministically(self):
        brief = {
            "research": {"enabled": True, "offline": False},
            "privacy": {
                "mode": "external_research_allowed",
                "approved_domains": ["example.com"],
                "approved_tool_classes": ["web_search"],
                "consents": [
                    build_consent_record(
                        purpose="public market research",
                        categories_allowed=["public_query"],
                        destinations=["example.com"],
                        issued_at="2026-01-01T00:00:00Z",
                        consent_id="consent_test_001",
                    )
                ],
            },
        }
        ctx = resolve_privacy_config_from_brief(brief, artifact_path="out/neon-genie/test")
        self.assertEqual(ctx["mode"], "external_research_allowed")
        self.assertEqual(ctx["egress"]["approved_domains"], ["example.com"])
        self.assertEqual(ctx["contract_version"], CONTRACT_VERSION)
        self.assertEqual(len(ctx["consents"]), 1)
        fields = privacy_receipt_fields(ctx)
        self.assertEqual(fields["privacy_mode"], "external_research_allowed")
        self.assertEqual(fields["privacy"]["consents"][0]["consent_id"], "consent_test_001")

    def test_offline_research_forces_local_only(self):
        brief = {
            "research": {"enabled": True, "offline": True},
            "privacy": {"mode": "external_research_allowed"},
        }
        ctx = resolve_privacy_config_from_brief(brief)
        self.assertEqual(ctx["mode"], "local_only")
        brief2 = {"research": {"enabled": False}, "privacy": {"mode": "external_research_allowed"}}
        self.assertEqual(resolve_privacy_config_from_brief(brief2)["mode"], "local_only")

    def test_brief_rejects_global_privacy_disable(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_privacy_config_from_brief({"privacy": {"mode": "local_only", "disable": True}})
        self.assertIn("NG-PRIVACY-051", str(ctx.exception))

    def test_yaml_brief_privacy_section(self):
        yaml_text = """
request_id: "privacy-brief-001"
research:
  enabled: true
  offline: false
privacy:
  mode: external_research_allowed
  approved_domains:
    - example.com
  approved_tool_classes:
    - web_search
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "brief.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            ctx = resolve_privacy_config_from_brief(path, artifact_path=str(Path(tmp) / "out"))
            self.assertEqual(ctx["mode"], "external_research_allowed")
            self.assertIn("example.com", ctx["egress"]["approved_domains"])

    def test_unsupported_contract_version_errors(self):
        c = default_privacy_context()
        c["contract_version"] = "9.9.9"
        with self.assertRaises(ValueError) as ctx:
            assert_envelope_privacy(c)
        self.assertIn("NG-PRIVACY-011", str(ctx.exception))

    def test_prepare_egress_allow_builds_action(self):
        c = default_privacy_context(mode="external_research_allowed")
        prep = prepare_egress(
            "public market size developer tools",
            "example.com",
            "public market research",
            c,
            recorded_at="2026-07-31T12:00:00Z",
        )
        self.assertEqual(prep["decision"], "ALLOW")
        self.assertIsNotNone(prep["external_action"])
        self.assertEqual(prep["external_action"]["decision"], "ALLOW")
        self.assertEqual(prep["external_action"]["recorded_at"], "2026-07-31T12:00:00Z")


if __name__ == "__main__":
    unittest.main()
