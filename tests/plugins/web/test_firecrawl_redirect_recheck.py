"""Firecrawl extract re-checks the website policy against the URL it was redirected to.

Firecrawl reports the requested URL as ``sourceURL`` (REST) / ``source_url`` (v4 SDK ``model_dump``, used
by the direct-key and Tool Gateway paths) and the post-redirect URL as ``url``. Reading the requested URL
made the re-check a no-op, so a public redirector returned a blocklisted page.
"""

import asyncio
import json

import pytest

from plugins.web.firecrawl import provider as firecrawl_provider
from tools import website_policy

REQUESTED = "https://redirector.example/go"


class _SdkDocument:
    """Stand-in for firecrawl v4 ``Document``: ``_to_plain_object`` reads it via ``model_dump``."""

    def __init__(self, final_url: str):
        self._final_url = final_url

    def model_dump(self):
        return {
            "markdown": "# landed",
            "metadata": {"title": "Landed", "url": self._final_url, "source_url": REQUESTED, "status_code": 200},
        }


def _rest_payload(final_url: str) -> dict:
    return {"success": True, "data": {"markdown": "# landed", "metadata": {"title": "Landed", "url": final_url, "sourceURL": REQUESTED, "statusCode": 200}}}


SHAPES = {"sdk": _SdkDocument, "rest": _rest_payload}


@pytest.fixture
def scrape(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"security": {"website_blocklist": {"enabled": True, "domains": ["blocked.example"]}}}))
    monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda url: website_policy.check_website_access(url, config_path=config))
    monkeypatch.setattr(firecrawl_provider, "is_safe_url", lambda url: True)

    def _run(shape: str, final_url: str) -> dict:
        class _Client:
            def scrape(self, **_kwargs):
                return SHAPES[shape](final_url)

        monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda _capability=None: _Client())
        return asyncio.run(firecrawl_provider._scrape_one(REQUESTED, ["markdown"], "markdown"))

    return _run


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_redirect_to_blocklisted_host_is_refused(scrape, shape):
    entry = scrape(shape, "https://blocked.example/page")

    assert entry["url"] == "https://blocked.example/page"
    assert entry["content"] == ""
    assert entry["blocked_by_policy"]["host"] == "blocked.example"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_allowed_redirect_keeps_content_and_requested_url_for_the_cache(scrape, shape):
    entry = scrape(shape, "https://allowed.example/page")

    assert not entry.get("error")
    assert entry["url"] == "https://allowed.example/page"
    assert entry["content"] == "# landed"
    # web_tools_extract files a redirected page under metadata.sourceURL (the requested URL).
    assert entry["metadata"]["sourceURL"] == REQUESTED
