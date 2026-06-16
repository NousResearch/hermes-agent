"""Tests for the multi-search web provider."""

from __future__ import annotations

import json

from plugins.web.multi_search.provider import _normalize_wsp_result


def test_wsp_normalizer_accepts_double_encoded_result_string() -> None:
    """web-search-prime may wrap its JSON array as a JSON string twice."""
    payload = json.dumps(
        {
            "result": json.dumps(
                json.dumps(
                    [
                        {
                            "title": "OpenAI Platform",
                            "link": "https://platform.openai.com/",
                            "content": "API docs and developer tools.",
                            "refer": "ref_1",
                        }
                    ]
                )
            )
        }
    )

    result = _normalize_wsp_result(json.loads(payload), 5)

    assert result == {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "OpenAI Platform",
                    "url": "https://platform.openai.com/",
                    "description": "API docs and developer tools.",
                    "position": 1,
                }
            ]
        },
    }


def test_web_tools_accepts_configured_multi_search_backend(monkeypatch) -> None:
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "multi-search"})

    assert web_tools._get_backend() == "multi-search"


def test_web_tools_reports_multi_search_available_with_tavily_or_exa(
    monkeypatch,
) -> None:
    from tools import web_tools

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    assert web_tools._is_backend_available("multi-search") is False

    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test")
    assert web_tools._is_backend_available("multi-search") is True

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    assert web_tools._is_backend_available("multi-search") is True


def test_web_tools_accepts_configured_multi_search_search_backend(
    monkeypatch,
) -> None:
    from tools import web_tools

    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test")
    monkeypatch.setattr(
        web_tools,
        "_load_web_config",
        lambda: {"search_backend": "multi-search"},
    )

    assert web_tools._get_search_backend() == "multi-search"
