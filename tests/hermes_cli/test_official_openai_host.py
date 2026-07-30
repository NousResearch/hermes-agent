"""Security + parity contract for ``is_official_openai_host``.

One predicate decides "is this endpoint OpenAI's official API surface?"
for every lane that branches on it: transport mandates
(``host_mandated_api_mode``), URL auto-detection in the runtime resolver,
model-catalog filtering, and live-listing authority. OpenAI's documented
data-residency hosts (``us.api.openai.com``, ``eu.api.openai.com``, and any
future ``<region>.api.openai.com``) are the same API surface as the
canonical host and must match; lookalike/spoof hosts must not (#32243).
"""

from __future__ import annotations

import pytest

from hermes_cli.providers import is_official_openai_host


class TestOfficialHosts:
    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1",
            "https://api.openai.com",
            "https://us.api.openai.com/v1",
            "https://eu.api.openai.com/v1",
            "https://US.api.OpenAI.com/v1",  # case-insensitive hostname
            "https://in.api.openai.com/v1",  # future regional variants
        ],
    )
    def test_official_hosts_match(self, url):
        assert is_official_openai_host(url) is True


class TestSpoofRejection:
    @pytest.mark.parametrize(
        "url",
        [
            # Lookalike host suffix: registrable domain is attacker.test.
            "https://api.openai.com.attacker.test/v1",
            "https://us.api.openai.com.attacker.test/v1",
            # Path-segment spoofing: host is proxy.test.
            "https://proxy.test/api.openai.com/v1",
            "https://proxy.test/us.api.openai.com/v1",
            # Prefix tricks that are NOT dot-separated subdomains of
            # api.openai.com (fooapi.openai.com is an openai.com host but
            # not the official API host family this predicate is scoped to).
            "https://evilapi.openai.com.attacker.test/v1",
            "https://fooapi.openai.com/v1",
            # Unrelated hosts.
            "https://openrouter.ai/api/v1",
            "https://api.anthropic.com/v1",
            "",
        ],
    )
    def test_spoof_and_unrelated_hosts_rejected(self, url):
        assert is_official_openai_host(url) is False
