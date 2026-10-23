"""Request shape of the Exa provider under the optional ``web.exa`` config.

A recording client sits in the provider's SDK client slot, so each test sees exactly the
arguments ``search`` / ``get_contents`` would hand ``exa-py``. Its ``get_contents`` honours
``text.max_characters`` the way Exa does, so the web_extract case sees what a capped page becomes.
"""
from __future__ import annotations

import asyncio
import json
import types

import pytest


class _RecordingExa:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.page = "body"

    def search(self, *args, **kwargs):
        self.calls.append(("search", args, kwargs))
        hit = types.SimpleNamespace(url="https://example.com", title="t", highlights=["h"])
        return types.SimpleNamespace(results=[hit])

    def get_contents(self, *args, **kwargs):
        self.calls.append(("get_contents", args, kwargs))
        text = kwargs.get("text")
        cap = text.get("max_characters") if isinstance(text, dict) else None
        page = types.SimpleNamespace(url=args[0][0], title="t", text=self.page[:cap])
        return types.SimpleNamespace(results=[page])


@pytest.fixture
def exa(monkeypatch: pytest.MonkeyPatch):
    """Keyed Exa with a recording client; ``exa.config`` is the ``web`` config section."""
    from tools import web_tools

    client = _RecordingExa()
    client.config = {}
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    monkeypatch.setattr("plugins.web.keyless_mcp.keyless_enabled", lambda: False)
    monkeypatch.setattr(web_tools, "_exa_client", ("exa-test", client), raising=False)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: client.config)
    return client


def _search(exa, **exa_cfg) -> dict:
    from plugins.web.exa.provider import ExaWebSearchProvider

    exa.config = {"exa": exa_cfg} if exa_cfg else {}
    result = ExaWebSearchProvider().search("q", 5)
    assert result["success"], result
    name, args, kwargs = exa.calls[-1]
    assert (name, args) == ("search", ("q",))
    return kwargs


def _extract(exa, web_cfg: dict | None = None, **exa_cfg) -> dict:
    from plugins.web.exa.provider import ExaWebSearchProvider

    exa.config = {**(web_cfg or {}), **({"exa": exa_cfg} if exa_cfg else {})}
    pages = ExaWebSearchProvider().extract(["https://example.com"])
    assert pages[0]["content"] and "body".startswith(pages[0]["content"]), pages
    name, args, kwargs = exa.calls[-1]
    assert (name, args) == ("get_contents", (["https://example.com"],))
    return kwargs


def test_unconfigured_requests_are_unchanged(exa):
    assert _search(exa) == {"num_results": 5, "contents": {"highlights": True}}
    assert _extract(exa) == {"text": True}


@pytest.mark.parametrize("search_type", ["lite", "fast", "deep-lite", "deep"])
def test_search_type_is_sent(exa, search_type):
    assert _search(exa, type=search_type) == {
        "num_results": 5, "contents": {"highlights": True}, "type": search_type,
    }


@pytest.mark.parametrize("search_type", ["auto", "neural-typo", 3])
def test_auto_or_unknown_search_type_sends_no_type(exa, search_type):
    assert "type" not in _search(exa, type=search_type)


def test_dynamic_highlights_sends_the_beta(exa):
    kwargs = _search(exa, dynamic_highlights=True)
    assert kwargs["contents"] == {"highlights": {"dynamic": True}}
    assert kwargs["betas"] == ["dynamic-highlights-2026-08-28"]


def test_max_age_hours_reaches_search_and_extract(exa):
    assert _search(exa, max_age_hours=24)["contents"] == {"highlights": True, "max_age_hours": 24}
    assert _extract(exa, max_age_hours=24) == {"text": True, "max_age_hours": 24}


def test_lite_search_drops_max_age_hours(exa):
    """Exa rejects contents.maxAgeHours on lite searches; the rest of the request still goes out."""
    assert _search(exa, type="lite", max_age_hours=24) == {
        "num_results": 5, "contents": {"highlights": True}, "type": "lite",
    }


@pytest.mark.parametrize("configured", [None, "auto", "lite", "deep-lite"])
def test_depth_overrides_the_configured_type(exa, configured):
    """standard keeps web.exa.type; quick is lite (dropping max_age_hours, which lite rejects); deep is deep."""
    from plugins.web.exa.provider import ExaWebSearchProvider

    exa.config = {"exa": {"max_age_hours": 24, **({"type": configured} if configured else {})}}
    sent = {}
    for depth in ("standard", "quick", "deep"):
        assert ExaWebSearchProvider().search("q", 5, depth=depth)["success"]
        sent[depth] = exa.calls[-1][2]
    assert sent["standard"] == _search(exa, max_age_hours=24, **({"type": configured} if configured else {}))
    assert sent["quick"] == {"num_results": 5, "contents": {"highlights": True}, "type": "lite"}
    assert sent["deep"] == {"num_results": 5, "contents": {"highlights": True, "max_age_hours": 24}, "type": "deep"}


@pytest.mark.parametrize("bad", [721, -2, "24", True])
def test_invalid_max_age_hours_is_ignored(exa, bad):
    assert _search(exa, max_age_hours=bad)["contents"] == {"highlights": True}


@pytest.mark.parametrize("cap", [1, 12000, 1_000_000])
def test_extract_cap_is_sent_as_configured(exa, cap):
    assert _extract(exa, extract_max_characters=cap) == {"text": {"max_characters": cap}}


@pytest.mark.parametrize("bad", [0, 1_000_001, "12000", True])
def test_extract_cap_outside_exa_range_keeps_full_text(exa, bad):
    assert _extract(exa, extract_max_characters=bad) == {"text": True}


def _web_extract(monkeypatch, url: str, char_limit: int | None = None) -> str:
    from plugins.web.exa.provider import ExaWebSearchProvider
    from tools import web_tools

    async def _safe(_url: str) -> bool:
        return True

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "async_is_safe_url", _safe)
    monkeypatch.setattr("agent.web_search_registry.get_provider", lambda name: ExaWebSearchProvider())
    result = json.loads(asyncio.run(web_tools.web_extract_tool([url], "markdown", char_limit=char_limit)))
    return result["results"][0]["content"]


def test_extract_cap_cuts_the_page_before_web_extract_truncation(exa, monkeypatch):
    """Without the cap, web_extract keeps the page tail in its head+tail window. With it, Exa returns
    only the first ``cap`` characters, so the tail is gone even from a later, larger char_limit
    (served from the stored, already-capped copy)."""
    exa.page = "\n".join(f"para {i} " + "y" * 80 for i in range(3000))
    exa.config = {"extract_backend": "exa"}
    assert "para 2999 " in _web_extract(monkeypatch, "https://example.com/full")

    exa.config = {"extract_backend": "exa", "exa": {"extract_max_characters": 20000}}
    first = _web_extract(monkeypatch, "https://example.com/capped")
    assert exa.calls[-1][2] == {"text": {"max_characters": 20000}}
    assert "para 0 " in first and "[TRUNCATED]" in first and "para 2999 " not in first

    calls = len(exa.calls)
    wider = _web_extract(monkeypatch, "https://example.com/capped", char_limit=500_000)
    assert len(exa.calls) == calls
    assert "para 2999 " not in wider and "[TRUNCATED]" not in wider
