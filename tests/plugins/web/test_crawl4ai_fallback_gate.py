"""crawl4ai Firecrawl 폴백 OFF 스위치 테스트.

invest 도메인처럼 Firecrawl이 금지된 역할은 ``CRAWL4AI_DISABLE_FALLBACK=1``로
폴백을 끈다. 끄면 실패 URL은 Firecrawl로 넘어가지 않고 backend="none" 에러로
남아야 한다(도메인 차단 룰 준수). 켜져 있으면(기본) 실패 URL은 Firecrawl 폴백을
탄다.

dev 스킬 원칙: 실제 provider 모듈을 import하고 _read_local만 스텁한다 — 폴백
게이트 분기와 결과 shape를 동시에 검증.
"""
from __future__ import annotations

import asyncio

import pytest

from plugins.web.crawl4ai import provider as c4


def _fail_read(url: str):
    """로컬 읽기 실패를 흉내(서비스 다운)."""
    return {"success": False, "failure_code": "service_down", "markdown": ""}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_fallback_disabled_skips_firecrawl(monkeypatch: pytest.MonkeyPatch) -> None:
    """DISABLE=1 → Firecrawl 미호출, backend="none" 에러로 남는다."""
    monkeypatch.setenv("CRAWL4AI_DISABLE_FALLBACK", "1")
    monkeypatch.setattr(c4, "_read_local", _fail_read)

    # Firecrawl이 절대 호출되면 안 됨 — 호출 시 폭발하는 스파이를 심는다.
    import plugins.web.firecrawl.provider as fc_mod

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("Firecrawl must not be constructed when fallback disabled")

    monkeypatch.setattr(fc_mod, "FirecrawlWebSearchProvider", _Boom)

    prov = c4.Crawl4aiWebSearchProvider()
    out = _run(prov.extract(["https://example.com/blocked"]))

    assert len(out) == 1
    assert out[0]["backend"] == "none"
    assert "firecrawl_fallback_disabled" in out[0]["error"]
    assert out[0]["content"] == ""


def test_fallback_enabled_uses_firecrawl(monkeypatch: pytest.MonkeyPatch) -> None:
    """config가 명시적 opt-in(extract_fallback=firecrawl) → 실패 URL은 Firecrawl 폴백."""
    monkeypatch.delenv("CRAWL4AI_DISABLE_FALLBACK", raising=False)
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda *a, **k: {"web": {"extract_fallback": "firecrawl"}})
    monkeypatch.setattr(c4, "_read_local", _fail_read)

    import plugins.web.firecrawl.provider as fc_mod

    called = {"n": 0}

    class _StubFC:
        def __init__(self, *a, **k):
            pass

        async def extract(self, urls, **kwargs):
            called["n"] += 1
            return [{"url": u, "title": "", "content": "FC", "raw_content": "FC"} for u in urls]

    monkeypatch.setattr(fc_mod, "FirecrawlWebSearchProvider", _StubFC)

    prov = c4.Crawl4aiWebSearchProvider()
    out = _run(prov.extract(["https://example.com/blocked"]))

    assert called["n"] == 1
    assert len(out) == 1
    assert out[0]["backend"] == "firecrawl(fallback)"
    assert out[0]["content"] == "FC"


def test_env_falsey_with_optin_keeps_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """config opt-in(firecrawl) 상태에서 env 0/빈값은 폴백을 끄지 않고, env truthy만 강제 차단."""
    monkeypatch.delenv("CRAWL4AI_DISABLE_FALLBACK", raising=False)
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda *a, **k: {"web": {"extract_fallback": "firecrawl"}})
    monkeypatch.setenv("CRAWL4AI_DISABLE_FALLBACK", "0")
    assert c4._fallback_disabled() is False
    monkeypatch.setenv("CRAWL4AI_DISABLE_FALLBACK", "")
    assert c4._fallback_disabled() is False
    monkeypatch.setenv("CRAWL4AI_DISABLE_FALLBACK", "1")
    assert c4._fallback_disabled() is True


def test_config_extract_fallback_none_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """프로파일 config ``web.extract_fallback: none`` → env 없이도 폴백 차단."""
    monkeypatch.delenv("CRAWL4AI_DISABLE_FALLBACK", raising=False)
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda *a, **k: {"web": {"extract_fallback": "none"}})
    assert c4._fallback_disabled() is True


def test_config_missing_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """fail-closed: extract_fallback 키 없음 → 폴백 차단. firecrawl 명시만 opt-in."""
    monkeypatch.delenv("CRAWL4AI_DISABLE_FALLBACK", raising=False)
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda *a, **k: {"web": {}})
    assert c4._fallback_disabled() is True          # 키 없음 → 차단
    monkeypatch.setattr(hc, "load_config", lambda *a, **k: {"web": {"extract_fallback": "firecrawl"}})
    assert c4._fallback_disabled() is False          # 명시 opt-in → 허용


def test_config_parse_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex 마일스톤 검증 회귀: config parse 실패로 DEFAULT_CONFIG(web/키 없음)면 차단.

    invest 도메인이 config 손상 시 Firecrawl로 새지 않도록 fail-closed 보장."""
    monkeypatch.delenv("CRAWL4AI_DISABLE_FALLBACK", raising=False)
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda *a, **k: {"some": "default"})  # web 키 자체 없음
    assert c4._fallback_disabled() is True
    monkeypatch.setattr(hc, "load_config", lambda *a, **k: {})                    # 완전 빈 dict
    assert c4._fallback_disabled() is True


def test_config_load_exception_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex 마일스톤 검증 회귀: load_config 예외 → 폴백 차단(fail-closed)."""
    monkeypatch.delenv("CRAWL4AI_DISABLE_FALLBACK", raising=False)
    import hermes_cli.config as hc

    def _boom(*a, **k):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(hc, "load_config", _boom)
    assert c4._fallback_disabled() is True
