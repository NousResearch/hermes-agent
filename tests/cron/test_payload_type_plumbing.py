"""Tests for cron payload_type plumbing (TKT-0033)."""

import cron.scheduler as scheduler


def test_extract_payload_type_default_returns_text_markdown():
    job = {}
    assert scheduler._extract_payload_type(job) == "text/markdown"


def test_extract_payload_type_explicit_text_html():
    job = {"payload_type": "text/html"}
    assert scheduler._extract_payload_type(job) == "text/html"


def test_extract_payload_type_invalid_coerces_to_text_markdown():
    job = {"payload_type": "xml"}
    assert scheduler._extract_payload_type(job) == "text/markdown"
