"""Regression tests for #59465 — ``no_proxy`` must honor CIDR / single-IP entries.

``urllib.request.proxy_bypass_environment()`` (which underlies
``_get_proxy_for_base_url``) only matches exact hostnames / domain suffixes.
CIDR ranges such as ``10.0.0.0/24`` and bare single-IP entries like
``192.168.1.5`` are silently ignored, so a custom provider whose
``base_url`` lives inside the operator's intended exclusion subnet gets
routed through the unrelated HTTP proxy and fails with an empty 503.

These tests pin the extended behavior added by
:func:`agent.process_bootstrap._no_proxy_matches_no_proxy_env`:

* CIDR subnet — host inside the subnet bypasses the proxy.
* CIDR subnet — host outside the subnet still uses the proxy.
* Single-IP ``no_proxy`` entry — bypass only for that exact IP.
* Plain hostname entry — continues to match (no regression on #14966).
* Mixed entries — both hostnames and CIDR apply.
"""
from __future__ import annotations

import pytest

from agent.process_bootstrap import (
    _get_proxy_for_base_url,
    _no_proxy_matches_no_proxy_env,
)


PROXY_ENV_KEYS = (
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
    "https_proxy", "http_proxy", "all_proxy",
    "NO_PROXY", "no_proxy",
)


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch):
    """Wipe every proxy-related env var before each test."""
    for key in PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def test_cidr_subnet_bypasses_proxy_for_inside_target():
    """``no_proxy=10.0.0.0/24`` excludes ``10.0.0.5`` (inside the subnet)."""
    import os
    os.environ["HTTPS_PROXY"] = "http://corp:8080"
    os.environ["NO_PROXY"] = "10.0.0.0/24"

    assert _get_proxy_for_base_url("http://10.0.0.5:4001/v1") is None


def test_cidr_subnet_does_not_bypass_outside_target():
    """``no_proxy=10.0.0.0/24`` does NOT exclude ``10.0.1.5`` (outside)."""
    import os
    os.environ["HTTPS_PROXY"] = "http://corp:8080"
    os.environ["NO_PROXY"] = "10.0.0.0/24"

    assert _get_proxy_for_base_url("http://10.0.1.5:4001/v1") == "http://corp:8080"


def test_single_ip_bypasses_exact_match():
    """``no_proxy=192.168.1.5`` excludes only ``192.168.1.5``."""
    import os
    os.environ["HTTPS_PROXY"] = "http://corp:8080"
    os.environ["NO_PROXY"] = "192.168.1.5"

    assert _get_proxy_for_base_url("http://192.168.1.5:4001/v1") is None
    # An adjacent IP on the same /24 is NOT covered by a single-IP entry.
    assert _get_proxy_for_base_url("http://192.168.1.6:4001/v1") == "http://corp:8080"


def test_plain_hostname_bypasses_no_regression():
    """Plain hostname ``no_proxy=example.com`` still bypasses (no regression)."""
    import os
    os.environ["HTTPS_PROXY"] = "http://corp:8080"
    os.environ["NO_PROXY"] = "example.com"

    assert _get_proxy_for_base_url("https://api.example.com/v1") is None


def test_mixed_hostname_and_cidr_entries_both_apply():
    """Mixed ``no_proxy`` entries — hostnames via stdlib, CIDR via this fix."""
    import os
    os.environ["HTTPS_PROXY"] = "http://corp:8080"
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,10.10.10.0/24"

    # Hostname matches
    assert _get_proxy_for_base_url("http://localhost:11434/v1") is None
    assert _get_proxy_for_base_url("http://127.0.0.1:11434/v1") is None
    # CIDR match
    assert _get_proxy_for_base_url("http://10.10.10.101:4001/v1") is None
    # Outside the CIDR — proxy still applies.
    assert _get_proxy_for_base_url("http://10.10.11.5:4001/v1") == "http://corp:8080"
    # Public host — proxy still applies.
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://corp:8080"


def test_lowercase_no_proxy_also_recognized():
    """``no_proxy`` (lowercase) must be honored just like ``NO_PROXY``."""
    import os
    os.environ["HTTPS_PROXY"] = "http://corp:8080"
    os.environ["no_proxy"] = "10.0.0.0/24"

    assert _get_proxy_for_base_url("http://10.0.0.5:4001/v1") is None
    assert _get_proxy_for_base_url("http://10.0.1.5:4001/v1") == "http://corp:8080"


def test_ipv6_cidr_bypasses_ipv6_target():
    """IPv6 CIDR ranges also work — e.g. ``2001:db8::/32``."""
    import os
    os.environ["HTTPS_PROXY"] = "http://corp:8080"
    os.environ["NO_PROXY"] = "2001:db8::/32"

    assert _get_proxy_for_base_url("http://[2001:db8::5]:4001/v1") is None
    # IPv4 target against an IPv6 CIDR — mismatch, proxy applies.
    assert _get_proxy_for_base_url("http://10.0.0.5:4001/v1") == "http://corp:8080"


def test_malformed_cidr_entry_does_not_break_other_entries():
    """A bad CIDR entry must not torpedo the rest of the ``no_proxy`` list."""
    import os
    os.environ["HTTPS_PROXY"] = "http://corp:8080"
    os.environ["NO_PROXY"] = "not-an-ip/99,10.0.0.0/24"

    # Valid CIDR still wins after the malformed entry.
    assert _get_proxy_for_base_url("http://10.0.0.5:4001/v1") is None
    # Unrelated target — proxy still applies.
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://corp:8080"


def test_helper_direct_no_proxy_returns_true_for_cidr_match():
    """Direct unit-test of the helper for the inside-CIDR case."""
    import os
    os.environ["NO_PROXY"] = "10.0.0.0/24"

    assert _no_proxy_matches_no_proxy_env("10.0.0.5") is True
    assert _no_proxy_matches_no_proxy_env("10.0.1.5") is False


def test_helper_preserves_localhost_match():
    """Direct unit-test — helper still honors localhost/stdlib semantics."""
    import os
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"

    assert _no_proxy_matches_no_proxy_env("localhost") is True
    assert _no_proxy_matches_no_proxy_env("127.0.0.1") is True
    assert _no_proxy_matches_no_proxy_env("10.0.0.5") is False
