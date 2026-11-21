"""Keyless Exa web_extract returns the whole page, and says so when Exa cut it.

The hosted ``web_fetch_exa`` tool cuts each page at ``maxCharacters`` (3,000 when omitted) and marks
nothing, so a cut page read as a complete one. Regression for #146558.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from plugins.web import keyless_mcp
from tools import web_tools

# Identifiable text on both sides of web_fetch_exa's 3,000-char default. The astral emoji count as
# two UTF-16 code units each, as Exa counts ``maxCharacters``.
PAGE = ("intro \U0001F600 " * 100 + "MARKER_BEFORE_3000 " + "filler " * 450
        + "MARKER_PAST_3000 " + "filler " * 400 + "MARKER_PAGE_END")
WEB_KEYS = ("EXA_API_KEY", "PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "TAVILY_API_KEY",
            "KEENABLE_API_KEY", "BRAVE_SEARCH_API_KEY", "SEARXNG_URL", "TOOL_GATEWAY_USER_TOKEN")


def _exa_cut(text: str, max_characters: int) -> str:
    """Exa's cut: the first ``max_characters`` UTF-16 code units, then trailing whitespace trimmed."""
    return text.encode("utf-16-le")[:2 * max_characters].decode("utf-16-le", errors="ignore").rstrip()


def _web_fetch_exa_post(url, **kwargs):
    """``requests.post`` to mcp.exa.ai as deployed: an SSE ``tools/call`` answer whose text is a
    ``# Title`` / ``URL:`` header and the page cut at ``maxCharacters`` (3,000 when omitted), unmarked."""
    params = kwargs["json"]["params"]
    assert url == keyless_mcp.EXA_MCP_URL and params["name"] == "web_fetch_exa"
    args = params["arguments"]
    text = f"# Fixture\nURL: {args['urls'][0]}\n\n" + _exa_cut(PAGE, args.get("maxCharacters", 3000))
    envelope = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": text}]}}
    return SimpleNamespace(status_code=200, headers={"Content-Type": "text/event-stream"},
                           content=f"event: message\ndata: {json.dumps(envelope)}\n\n".encode())


@pytest.fixture
def keyless_exa_extract(monkeypatch):
    """``web_extract`` through real plugin discovery, with ``web.backend: exa`` in the isolated
    HERMES_HOME's config.yaml, no web key, and only the network (Exa's endpoint, SSRF DNS) faked."""
    for key in WEB_KEYS:
        monkeypatch.delenv(key, raising=False)
    (get_hermes_home() / "config.yaml").write_text("web:\n  backend: exa\n")
    monkeypatch.setattr("requests.post", _web_fetch_exa_post)

    async def _public(_url, **_kwargs):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _public)

    def _run(char_limit: int) -> str:
        raw = asyncio.run(web_tools.web_extract_tool(["https://t.test/page"], char_limit=char_limit))
        return json.loads(raw)["results"][0]["content"]

    return _run


def test_text_past_3000_chars_reaches_the_model(keyless_exa_extract):
    assert len(_exa_cut(PAGE, 3000)) < PAGE.index("MARKER_PAST_3000")
    content = keyless_exa_extract(char_limit=20_000)
    assert "MARKER_PAST_3000" in content
    assert content.endswith("MARKER_PAGE_END")
    assert "[Exa returned only" not in content


def test_page_exa_cut_is_flagged_to_the_model(keyless_exa_extract, monkeypatch):
    """Exa's cut lands short of the request in Python characters (UTF-16 counting, trimmed
    whitespace); the notice must still lead the model's text, beside Hermes' own footer."""
    monkeypatch.setattr(keyless_mcp, "_EXA_FETCH_MAX_CHARACTERS", 5000)
    assert 5000 - len(_exa_cut(PAGE, 5000)) > keyless_mcp._EXA_CUT_SLACK  # a len() check would miss it
    content = keyless_exa_extract(char_limit=2000)
    assert content.startswith("[Exa returned only the first 5,000 characters of this page;")
    assert "[TRUNCATED]" in content
    assert "MARKER_PAGE_END" not in content
