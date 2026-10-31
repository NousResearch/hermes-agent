#!/usr/bin/env python3
"""Unit tests for privacy_preflight (stdlib only)."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import privacy_preflight as pp  # noqa: E402


def test_api_key_blocks_egress() -> None:
    text = "deploy with sk-abc123DEF456ghi789jkl012mno345pqr678stu901vwx"
    r = pp.preflight(text)
    assert r["safe_for_egress"] is False
    assert "credentials" in r["blocked_categories"]
    assert "sk-abc123" not in r["redacted_text"] or "[REDACTED" in r["redacted_text"]
    print("PASS: api key blocks egress")


def test_pem_private_key_blocks() -> None:
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF6P\n-----END RSA PRIVATE KEY-----"
    r = pp.preflight(text)
    assert r["safe_for_egress"] is False
    assert "secrets" in r["blocked_categories"]
    print("PASS: pem blocks egress")


def test_bearer_token_blocks() -> None:
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aaa.bbb"
    r = pp.preflight(text)
    assert r["safe_for_egress"] is False
    print("PASS: bearer blocks egress")


def test_benign_product_copy_safe() -> None:
    text = (
        "We sell an audio continuity tool for podcast editors. "
        "Pricing is unknown. Buyer is missing. Research public competitors."
    )
    r = pp.preflight(text)
    assert r["safe_for_egress"] is True
    assert r["blocked_categories"] == []
    print("PASS: benign product copy safe")


def test_password_in_url_blocks() -> None:
    text = "postgres://admin:SuperSecret99@db.example.com:5432/app"
    r = pp.preflight(text)
    assert r["safe_for_egress"] is False
    assert "passwords_connection" in r["blocked_categories"] or "credentials" in r["blocked_categories"]
    print("PASS: connection string blocks")


def main() -> int:
    test_api_key_blocks_egress()
    test_pem_private_key_blocks()
    test_bearer_token_blocks()
    test_benign_product_copy_safe()
    test_password_in_url_blocks()
    print("PASS: all privacy_preflight tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
