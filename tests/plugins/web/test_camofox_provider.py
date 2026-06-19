from __future__ import annotations

from typing import Any, Dict, List

import pytest


@pytest.fixture(autouse=True)
def _clean_camofox_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAMOFOX_URL", raising=False)


class TestCamofoxAvailability:
    def test_unavailable_without_camofox_url(self) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider

        assert CamofoxWebSearchProvider().is_available() is False

    def test_available_with_camofox_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377/")

        assert CamofoxWebSearchProvider().is_available() is True


class TestDuckDuckGoSnapshotParser:
    def test_parses_representative_camofox_duckduckgo_snapshot(self) -> None:
        from plugins.web.camofox.provider import parse_duckduckgo_snapshot

        snapshot = """
- heading "Hermes Agent - GitHub" [level=2]
  - link "Hermes Agent - GitHub" [e4]
    - /url: https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2FNousResearch%2Fhermes-agent&rut=abc
  - text: github.com/NousResearch/hermes-agent
  - text: Hermes Agent is an open-source terminal and messaging agent.
- heading "Docs | Hermes Agent" [level=2]
  - link "Docs | Hermes Agent" [e8]
    - /url: //duckduckgo.com/l/?uddg=https%3A%2F%2Fhermes-agent.nousresearch.com%2Fdocs%2F&rut=def
  - link "hermes-agent.nousresearch.com/docs/"
  - text: User guide, CLI commands, providers, and tools.
"""

        assert parse_duckduckgo_snapshot(snapshot, limit=5) == [
            {
                "title": "Hermes Agent - GitHub",
                "url": "https://github.com/NousResearch/hermes-agent",
                "description": "Hermes Agent is an open-source terminal and messaging agent.",
                "position": 1,
            },
            {
                "title": "Docs | Hermes Agent",
                "url": "https://hermes-agent.nousresearch.com/docs/",
                "description": "User guide, CLI commands, providers, and tools.",
                "position": 2,
            },
        ]

    def test_parser_respects_limit_and_uses_url_as_title_fallback(self) -> None:
        from plugins.web.camofox.provider import parse_duckduckgo_snapshot

        snapshot = """
- heading "" [level=2]
  - link ""
    - /url: https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com
  - text: Example description.
- heading "Second" [level=2]
  - link "Second"
    - /url: https://duckduckgo.com/l/?uddg=https%3A%2F%2Fsecond.example
  - text: Second description.
"""

        assert parse_duckduckgo_snapshot(snapshot, limit=1) == [
            {
                "title": "https://example.com",
                "url": "https://example.com",
                "description": "Example description.",
                "position": 1,
            }
        ]


class TestCamofoxSearch:
    def test_search_prefers_dom_evaluate_results_and_closes_tab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        calls: List[Any] = []

        monkeypatch.setattr(
            "tools.camofox_client.camofox_create_tab",
            lambda user_id, session_key, **kwargs: calls.append(("create", user_id, session_key, kwargs))
            or {"tabId": "tab-search"},
        )
        monkeypatch.setattr(
            mod,
            "camofox_tab_navigate",
            lambda user_id, tab_id, url, **kwargs: calls.append(("navigate", user_id, tab_id, url, kwargs))
            or {"url": url},
        )
        monkeypatch.setattr(
            mod,
            "camofox_tab_evaluate",
            lambda user_id, tab_id, expression, **kwargs: calls.append(("evaluate", user_id, tab_id, expression, kwargs))
            or {
                "result": [
                    {
                        "title": "Hermes Agent",
                        "url": "https://example.com/hermes",
                        "description": "Readable result from DOM.",
                    },
                    {
                        "title": "Ignored",
                        "url": "https://example.com/ignored",
                        "description": "Past limit.",
                    },
                ]
            },
        )
        monkeypatch.setattr(
            mod,
            "camofox_tab_snapshot",
            lambda *args, **kwargs: pytest.fail("search should not use snapshot parser when DOM evaluate succeeds"),
        )
        monkeypatch.setattr(
            "tools.camofox_client.camofox_close_tab",
            lambda user_id, tab_id, **kwargs: calls.append(("close", user_id, tab_id, kwargs)),
        )

        result = CamofoxWebSearchProvider().search("hermes agent", limit=1)

        assert result == {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Hermes Agent",
                        "url": "https://example.com/hermes",
                        "description": "Readable result from DOM.",
                        "position": 1,
                    }
                ]
            },
        }
        assert [call[0] for call in calls] == ["create", "navigate", "evaluate", "close"]
        assert "html.duckduckgo.com/html/?q=hermes+agent" in calls[1][3]

    def test_search_falls_back_to_snapshot_parser_when_evaluate_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        closed: List[str] = []
        monkeypatch.setattr("tools.camofox_client.camofox_create_tab", lambda *args, **kwargs: {"tabId": "tab-search"})
        monkeypatch.setattr(mod, "camofox_tab_navigate", lambda *args, **kwargs: {"url": "ok"})
        monkeypatch.setattr(mod, "camofox_tab_evaluate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("old server")))
        monkeypatch.setattr(
            mod,
            "camofox_tab_snapshot",
            lambda *args, **kwargs: {
                "snapshot": (
                    '- heading "Fallback Result" [level=2]\n'
                    '  - link "Fallback Result"\n'
                    '    - /url: https://duckduckgo.com/l/?uddg=https%3A%2F%2Ffallback.example\n'
                    '  - text: Snapshot summary.'
                )
            },
        )
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", lambda user_id, tab_id, **kwargs: closed.append(tab_id))

        result = CamofoxWebSearchProvider().search("fallback", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "title": "Fallback Result",
                "url": "https://fallback.example",
                "description": "Snapshot summary.",
                "position": 1,
            }
        ]
        assert closed == ["tab-search"]


class TestCamofoxExtract:
    def test_extract_returns_readable_text_shape_with_metadata_and_closes_tab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        calls: List[Any] = []
        monkeypatch.setattr(
            "tools.camofox_client.camofox_create_tab",
            lambda user_id, session_key, url=None, **kwargs: calls.append(("create", user_id, session_key, url, kwargs))
            or {"tabId": "tab-extract"},
        )
        monkeypatch.setattr(
            mod,
            "camofox_tab_evaluate",
            lambda user_id, tab_id, expression, **kwargs: calls.append(("evaluate", user_id, tab_id, expression, kwargs))
            or {
                "result": {
                    "url": "https://example.com/final",
                    "title": "Example Title",
                    "content": "Example Title\n\nReadable body text.",
                }
            },
        )
        monkeypatch.setattr(
            mod,
            "camofox_tab_snapshot",
            lambda *args, **kwargs: pytest.fail("extract should use readable DOM text before accessibility snapshot fallback"),
        )
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", lambda user_id, tab_id, **kwargs: calls.append(("close", user_id, tab_id, kwargs)))

        result = CamofoxWebSearchProvider().extract(["https://example.com"])

        assert result == [
            {
                "url": "https://example.com/final",
                "title": "Example Title",
                "content": "Example Title\n\nReadable body text.",
                "raw_content": "Example Title\n\nReadable body text.",
                "metadata": {
                    "sourceURL": "https://example.com/final",
                    "title": "Example Title",
                    "backend": "camofox",
                    "content_type": "readable_text",
                },
            }
        ]
        assert [call[0] for call in calls] == ["create", "evaluate", "close"]
        assert calls[0][3] == "https://example.com"

    def test_extract_truncates_content_to_max_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setattr(mod, "_MAX_EXTRACT_CHARS", 12)
        monkeypatch.setattr("tools.camofox_client.camofox_create_tab", lambda *args, **kwargs: {"tabId": "tab-extract"})
        monkeypatch.setattr(
            mod,
            "camofox_tab_evaluate",
            lambda *args, **kwargs: {"result": {"url": "https://example.com", "title": "Title", "content": "0123456789abcdef"}},
        )
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", lambda *args, **kwargs: None)

        result = CamofoxWebSearchProvider().extract(["https://example.com"])

        assert result[0]["content"] == "0123456789ab"
        assert result[0]["raw_content"] == "0123456789ab"

    def test_extract_uses_url_title_fallback_when_dom_title_is_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setattr("tools.camofox_client.camofox_create_tab", lambda *args, **kwargs: {"tabId": "tab-extract"})
        monkeypatch.setattr(
            mod,
            "camofox_tab_evaluate",
            lambda *args, **kwargs: {"result": {"url": "https://example.com/final", "title": "", "content": "Body"}},
        )
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", lambda *args, **kwargs: None)

        result = CamofoxWebSearchProvider().extract(["https://example.com"])

        assert result[0]["url"] == "https://example.com/final"
        assert result[0]["title"] == "https://example.com/final"
        assert result[0]["metadata"]["title"] == "https://example.com/final"

    def test_extract_returns_per_url_error_and_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        created: List[str] = []
        monkeypatch.setattr(
            "tools.camofox_client.camofox_create_tab",
            lambda user_id, session_key, url=None, **kwargs: created.append(url or "")
            or ({"tabId": "bad"} if url and "bad" in url else {"tabId": "good"}),
        )

        def fake_evaluate(user_id: str, tab_id: str, expression: str, **kwargs: Any) -> Dict[str, Any]:
            if tab_id == "bad":
                raise RuntimeError("boom")
            return {"result": {"url": "https://good.example", "title": "Good", "content": "Body"}}

        closed: List[str] = []
        monkeypatch.setattr(mod, "camofox_tab_evaluate", fake_evaluate)
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", lambda user_id, tab_id, **kwargs: closed.append(tab_id))

        result = CamofoxWebSearchProvider().extract(["https://bad.example", "https://good.example"])

        assert result[0] == {
            "url": "https://bad.example",
            "title": "",
            "content": "",
            "raw_content": "",
            "error": "Camofox extract failed: boom",
        }
        assert result[1]["url"] == "https://good.example"
        assert created == ["https://bad.example", "https://good.example"]
        assert closed == ["bad", "good"]

    def test_extract_marks_each_url_interrupted_without_creating_tabs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: True)
        monkeypatch.setattr("tools.camofox_client.camofox_create_tab", lambda *args, **kwargs: pytest.fail("interrupted extract should not create tabs"))

        result = CamofoxWebSearchProvider().extract(["https://one.example", "https://two.example"])

        assert result == [
            {"url": "https://one.example", "title": "", "content": "", "raw_content": "", "error": "Interrupted"},
            {"url": "https://two.example", "title": "", "content": "", "raw_content": "", "error": "Interrupted"},
        ]

    def test_extract_falls_back_to_accessibility_snapshot_when_evaluate_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        import plugins.web.camofox.provider as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setattr("tools.camofox_client.camofox_create_tab", lambda *args, **kwargs: {"tabId": "tab-extract"})
        monkeypatch.setattr(mod, "camofox_tab_evaluate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("old server")))
        monkeypatch.setattr(
            mod,
            "camofox_tab_snapshot",
            lambda *args, **kwargs: {"url": "https://example.com/final", "snapshot": '- heading "Snapshot Title"\n- text: Snapshot body'},
        )
        monkeypatch.setattr("tools.camofox_client.camofox_close_tab", lambda *args, **kwargs: None)

        result = CamofoxWebSearchProvider().extract(["https://example.com"])

        assert result == [
            {
                "url": "https://example.com/final",
                "title": "Snapshot Title",
                "content": '- heading "Snapshot Title"\n- text: Snapshot body',
                "raw_content": '- heading "Snapshot Title"\n- text: Snapshot body',
                "metadata": {
                    "sourceURL": "https://example.com/final",
                    "title": "Snapshot Title",
                    "backend": "camofox",
                    "content_type": "accessibility_snapshot",
                },
            }
        ]


class TestCamofoxSetupSchema:
    def test_setup_schema_describes_local_url(self) -> None:
        from plugins.web.camofox.provider import CamofoxWebSearchProvider

        schema = CamofoxWebSearchProvider().get_setup_schema()

        assert schema["name"] == "Camofox"
        assert schema["badge"] == "local · no key"
        assert schema["env_vars"][0]["key"] == "CAMOFOX_URL"
        assert schema["env_vars"][0]["default"] == "http://localhost:9377"
