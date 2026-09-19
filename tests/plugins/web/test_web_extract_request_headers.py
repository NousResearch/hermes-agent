"""Tests for config-driven per-host request headers on web_extract.

Covers the ``web.request_headers`` resolution contract in
``tools.web_tools`` and its wiring into the Firecrawl provider:

  - host scoping (a credential configured for one host never leaks to
    another),
  - ``${VAR}`` expansion against the environment so secrets stay in
    ``.env`` rather than ``config.yaml``,
  - graceful degradation when the installed Firecrawl client predates the
    ``headers`` parameter.
"""

import pytest

import tools.web_tools as web_tools
from plugins.web.firecrawl.provider import (
    _KeylessFirecrawlClient,
    _scrape_with_optional_headers,
)


@pytest.fixture
def web_config(monkeypatch):
    """Install a fake ``web:`` config section."""

    def _install(cfg):
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: cfg)

    return _install


class TestHostMatching:
    def test_exact_host_matches(self):
        assert web_tools._host_matches("api.github.com", "api.github.com")

    def test_different_host_does_not_match(self):
        assert not web_tools._host_matches("api.github.com", "evil.com")

    def test_dot_prefix_matches_subdomain_and_apex(self):
        assert web_tools._host_matches(".example.com", "docs.example.com")
        assert web_tools._host_matches(".example.com", "example.com")

    def test_dot_prefix_does_not_match_suffix_impostor(self):
        # notexample.com must not match .example.com
        assert not web_tools._host_matches(".example.com", "notexample.com")

    def test_bare_wildcard_is_not_a_glob(self):
        # A bare "*" must not match everything — that would broadcast
        # credentials to every scraped host.
        assert not web_tools._host_matches("*", "api.github.com")

    def test_empty_pattern_or_host_never_matches(self):
        assert not web_tools._host_matches("", "api.github.com")
        assert not web_tools._host_matches("api.github.com", "")


class TestResolveRequestHeaders:
    def test_returns_headers_for_matching_host(self, web_config, monkeypatch):
        monkeypatch.setenv("TEST_GH_TOKEN", "ghp_secret")
        web_config(
            {
                "request_headers": {
                    "api.github.com": {"Authorization": "Bearer ${TEST_GH_TOKEN}"}
                }
            }
        )
        headers = web_tools.resolve_request_headers(
            "https://api.github.com/repos/o/r/pulls/1"
        )
        assert headers == {"Authorization": "Bearer ghp_secret"}

    def test_does_not_leak_headers_to_other_hosts(self, web_config, monkeypatch):
        monkeypatch.setenv("TEST_GH_TOKEN", "ghp_secret")
        web_config(
            {
                "request_headers": {
                    "api.github.com": {"Authorization": "Bearer ${TEST_GH_TOKEN}"}
                }
            }
        )
        assert web_tools.resolve_request_headers("https://evil.example.com/") == {}

    def test_unset_env_var_drops_header_rather_than_sending_literal(
        self, web_config, monkeypatch
    ):
        monkeypatch.delenv("TEST_MISSING_TOKEN", raising=False)
        web_config(
            {
                "request_headers": {
                    "api.github.com": {
                        "Authorization": "Bearer ${TEST_MISSING_TOKEN}"
                    }
                }
            }
        )
        headers = web_tools.resolve_request_headers("https://api.github.com/x")
        assert headers == {}

    def test_literal_value_without_env_ref_is_passed_through(self, web_config):
        web_config(
            {"request_headers": {"api.github.com": {"X-Custom": "static-value"}}}
        )
        headers = web_tools.resolve_request_headers("https://api.github.com/x")
        assert headers == {"X-Custom": "static-value"}

    def test_absent_config_returns_empty(self, web_config):
        web_config({})
        assert web_tools.resolve_request_headers("https://api.github.com/x") == {}

    def test_malformed_config_does_not_raise(self, web_config):
        web_config({"request_headers": "not-a-dict"})
        assert web_tools.resolve_request_headers("https://api.github.com/x") == {}

    def test_unparseable_url_returns_empty(self, web_config):
        web_config({"request_headers": {"api.github.com": {"X": "y"}}})
        assert web_tools.resolve_request_headers("not a url") == {}


class TestKeylessClientScrapePayload:
    def test_headers_go_in_payload_not_to_firecrawl_api(self, monkeypatch):
        captured = {}

        def fake_post(self, path, payload, extra_headers=None):
            captured["path"] = path
            captured["payload"] = payload
            captured["extra_headers"] = extra_headers
            return {"data": {}}

        monkeypatch.setattr(_KeylessFirecrawlClient, "_post", fake_post)
        client = _KeylessFirecrawlClient()
        client.scrape(
            url="https://api.github.com/x",
            formats=["markdown"],
            headers={"Authorization": "Bearer tok"},
        )

        # Target-site headers ride in the request BODY; they must not be
        # sent as headers to the Firecrawl API itself.
        assert captured["payload"]["headers"] == {"Authorization": "Bearer tok"}
        assert captured["extra_headers"] is None

    def test_no_headers_key_when_none_configured(self, monkeypatch):
        captured = {}

        def fake_post(self, path, payload, extra_headers=None):
            captured["payload"] = payload
            return {"data": {}}

        monkeypatch.setattr(_KeylessFirecrawlClient, "_post", fake_post)
        _KeylessFirecrawlClient().scrape(
            url="https://example.com", formats=["markdown"]
        )
        assert "headers" not in captured["payload"]


class TestScrapeWithOptionalHeaders:
    def test_passes_headers_when_client_accepts_them(self):
        seen = {}

        class Client:
            def scrape(self, *, url, formats, headers=None):
                seen["headers"] = headers
                return {"ok": True}

        result = _scrape_with_optional_headers(
            Client(),
            {
                "url": "https://api.github.com/x",
                "formats": ["markdown"],
                "headers": {"Authorization": "Bearer tok"},
            },
        )
        assert result == {"ok": True}
        assert seen["headers"] == {"Authorization": "Bearer tok"}

    def test_retries_without_headers_on_legacy_client(self):
        calls = []

        class LegacyClient:
            def scrape(self, **kwargs):
                calls.append(kwargs)
                if "headers" in kwargs:
                    raise TypeError("unexpected keyword argument 'headers'")
                return {"ok": True}

        result = _scrape_with_optional_headers(
            LegacyClient(),
            {
                "url": "https://api.github.com/x",
                "formats": ["markdown"],
                "headers": {"Authorization": "Bearer tok"},
            },
        )
        # Degrades to an unauthenticated scrape rather than failing outright.
        assert result == {"ok": True}
        assert len(calls) == 2
        assert "headers" not in calls[1]

    def test_typeerror_without_headers_is_not_swallowed(self):
        class BrokenClient:
            def scrape(self, **kwargs):
                raise TypeError("genuine signature error")

        # No headers configured: the TypeError is a real bug and must
        # propagate rather than being masked by the retry path.
        with pytest.raises(TypeError):
            _scrape_with_optional_headers(
                BrokenClient(),
                {"url": "https://example.com", "formats": ["markdown"]},
            )
