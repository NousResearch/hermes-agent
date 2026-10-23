"""Request shape of the Exa provider under the optional ``web.exa`` config.

A recording client sits in the provider's SDK client slot, so each test sees exactly the
arguments ``search`` / ``get_contents`` would hand ``exa-py``.
"""
from __future__ import annotations

import types

import pytest


class _RecordingExa:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def search(self, *args, **kwargs):
        self.calls.append(("search", args, kwargs))
        hit = types.SimpleNamespace(url="https://example.com", title="t", highlights=["h"])
        return types.SimpleNamespace(results=[hit])

    def get_contents(self, *args, **kwargs):
        self.calls.append(("get_contents", args, kwargs))
        page = types.SimpleNamespace(url="https://example.com", title="t", text="body")
        return types.SimpleNamespace(results=[page])


@pytest.fixture
def exa(monkeypatch: pytest.MonkeyPatch):
    """Keyed Exa with a recording client; ``exa.config`` is the ``web`` config section."""
    import tools.web_tools as web_tools

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
    assert pages[0]["content"] == "body", pages
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


@pytest.mark.parametrize("bad", [721, -2, "24", True])
def test_invalid_max_age_hours_is_ignored(exa, bad):
    assert _search(exa, max_age_hours=bad)["contents"] == {"highlights": True}


def test_extract_cap_never_undercuts_the_extract_budget(exa):
    assert _extract(exa, extract_max_characters=12000) == {"text": {"max_characters": 15000}}
    assert _extract(exa, {"extract_char_limit": 8000}, extract_max_characters=12000) == {
        "text": {"max_characters": 12000},
    }
    assert _extract(exa, extract_max_characters=40000) == {"text": {"max_characters": 40000}}
