"""Tests for the Yandex Cloud Search web provider plugin."""

from __future__ import annotations

import pytest

from plugins.web.yandex.provider import (
    YandexWebSearchProvider,
    parse_yandex_xml_results,
)


_SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<response>
  <results>
    <grouping>
      <group>
        <doc>
          <url>https://example.com/a</url>
          <title>Example A</title>
          <passages><passage>Snippet A</passage></passages>
        </doc>
      </group>
      <group>
        <doc>
          <url>https://example.com/b</url>
          <title>Example B</title>
          <passages><passage>Snippet B</passage></passages>
        </doc>
      </group>
    </grouping>
  </results>
</response>
"""


class TestParseYandexXmlResults:
    def test_parses_docs_into_rows(self) -> None:
        rows = parse_yandex_xml_results(_SAMPLE_XML, limit=5)
        assert len(rows) == 2
        assert rows[0]["url"] == "https://example.com/a"
        assert rows[0]["title"] == "Example A"
        assert rows[0]["description"] == "Snippet A"
        assert rows[0]["domain"] == "example.com"

    def test_respects_limit(self) -> None:
        rows = parse_yandex_xml_results(_SAMPLE_XML, limit=1)
        assert len(rows) == 1
        assert rows[0]["url"] == "https://example.com/a"


class TestYandexWebSearchProvider:
    def test_is_available_requires_both_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = YandexWebSearchProvider()
        monkeypatch.delenv("YANDEX_CLOUD_API_KEY", raising=False)
        monkeypatch.delenv("YANDEX_CLOUD_FOLDER_ID", raising=False)
        assert provider.is_available() is False

        monkeypatch.setenv("YANDEX_CLOUD_API_KEY", "key")
        assert provider.is_available() is False

        monkeypatch.setenv("YANDEX_CLOUD_FOLDER_ID", "folder")
        assert provider.is_available() is True

    def test_search_returns_error_when_unconfigured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("YANDEX_CLOUD_API_KEY", raising=False)
        monkeypatch.delenv("YANDEX_CLOUD_FOLDER_ID", raising=False)
        provider = YandexWebSearchProvider()
        result = provider.search("test", limit=5)
        assert result["success"] is False
        assert "YANDEX_CLOUD_API_KEY" in result["error"]

    def test_capability_flags(self) -> None:
        provider = YandexWebSearchProvider()
        assert provider.name == "yandex"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False
