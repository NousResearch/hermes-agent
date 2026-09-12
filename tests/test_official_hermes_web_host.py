"""Official Hermes web-host identity: exact nousresearch.com (or subdomain) only."""

from pathlib import Path

import pytest

from hermes_constants import is_official_hermes_web_host


def test_official_docs_host_is_official():
    assert is_official_hermes_web_host("https://hermes-agent.nousresearch.com/docs/") is True


def test_lookalike_other_tld_is_not_official():
    assert is_official_hermes_web_host("https://hermes-agent.ai/") is False


def test_suffix_spoof_is_not_official():
    assert is_official_hermes_web_host("https://hermes-agent.nousresearch.com.evil.test/") is False


def test_faq_names_official_site():
    faq = Path("website/docs/reference/faq.md").read_text(encoding="utf-8")
    assert "### What is the official Hermes Agent website?" in faq
    assert "https://hermes-agent.nousresearch.com" in faq


@pytest.mark.parametrize(
    "url_or_host",
    [
        "hermes-agent.nousresearch.com",
        "https://hermes-agent.nousresearch.com/docs/",
        "https://portal.nousresearch.com",
        "https://nousresearch.com",
        "www.nousresearch.com",
        "HTTPS://HERMES-AGENT.NOUSRESEARCH.COM/DOCS/",
        "Hermes-Agent.NousResearch.com",
        "https://hermes-agent.nousresearch.com.",
        "www.nousresearch.com.",
        "https://hermes-agent.nousresearch.com:443",
        "https://user:pass@portal.nousresearch.com/docs/",
    ],
)
def test_official_hosts_are_exact_or_dot_suffix(url_or_host):
    assert is_official_hermes_web_host(url_or_host) is True


@pytest.mark.parametrize(
    "url_or_host",
    [
        "hermes-agent.ai",
        "https://hermes-agent.ai/",
        "hermes-agent.nousresearch.com.evil.test",
        "https://hermes-agent.nousresearch.com.evil.test/",
        "https://proxy.test/hermes-agent.nousresearch.com",
        "notnousresearch.com",
        "evil-nousresearch.com",
        "",
        None,
        "127.0.0.1",
        "https://192.168.1.1/",
        "https://[::1]/",
        "medium.com",
        "   ",
    ],
)
def test_non_official_hosts_are_false(url_or_host):
    assert is_official_hermes_web_host(url_or_host) is False


def test_none_does_not_raise():
    assert is_official_hermes_web_host(None) is False
