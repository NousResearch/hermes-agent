"""Crawl4AI 로컬 읽기 제공자 (extract 전용) + Firecrawl 폴백.

3-LLM 합의(2026-06-05) 채택안 "B 변형"의 읽기 계층:
  읽기 = Crawl4AI(M4 로컬 서비스, 무료) → 실패 시 → Firecrawl(폴백).

워커는 하드 샌드박스(browser·terminal 차단)라 Chromium을 직접 못 띄운다.
대신 localhost의 ``hermes-crawl4ai-read`` 서비스(``crawl4ai-service/server.py``)가
렌더링을 수행하고, 이 제공자가 HTTP로 호출한다.

핵심(합의 #1): **실패 판정**. 서비스가 captcha_detected·empty_body·login_wall·
timeout·provider_blocked·provider_429 등 ``failure_code``를 붙여 주면, 이 제공자가
그 URL만 Firecrawl로 폴백한다. HTTP 200 + CAPTCHA 페이지를 본문으로 저장하는
환각을 막는다. 각 결과에 ``backend``/``failure_code``를 기록해 운영에서 한국 소스
실제 성공률을 학습한다. 검색은 미지원 — brave-free 담당.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

_SERVICE_URL = os.environ.get("CRAWL4AI_SERVICE_URL", "http://127.0.0.1:8787")
_READ_TIMEOUT_S = float(os.environ.get("CRAWL4AI_READ_TIMEOUT", "35"))
_ACCESS_LOG = os.path.expanduser("~/.hermes/crawl4ai-service/access.log")


def _log(line: str) -> None:
    try:
        with open(_ACCESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _read_local(url: str) -> Dict[str, Any]:
    """로컬 Crawl4AI 서비스 /read 호출(blocking). 다운 시 service_down."""
    body = json.dumps({"url": url, "timeout_s": _READ_TIMEOUT_S}).encode("utf-8")
    req = urllib.request.Request(
        _SERVICE_URL + "/read", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_READ_TIMEOUT_S + 15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"success": False, "failure_code": "service_down", "markdown": "",
                "_err": str(getattr(e, "reason", e))}
    except Exception as e:
        return {"success": False, "failure_code": f"client_error:{type(e).__name__}",
                "markdown": ""}


class Crawl4aiWebSearchProvider(WebSearchProvider):
    """extract 전용. Crawl4AI 로컬 우선, 실패 URL만 Firecrawl 폴백."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI (local) + Firecrawl fallback"

    def is_available(self) -> bool:
        # 네트워크 호출 금지(등록 시점 실행). 서비스 URL이 설정돼 있으면 가용으로 본다.
        # 실제 서비스 다운은 호출 시점에 service_down → Firecrawl 폴백으로 처리.
        return bool(_SERVICE_URL)

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        fallback: List[str] = []
        fail_reason: Dict[str, str] = {}

        for url in urls:
            r = await asyncio.to_thread(_read_local, url)
            if r.get("success"):
                md = r.get("markdown", "")
                results.append({
                    "url": url, "title": "", "content": md, "raw_content": md,
                    "backend": "crawl4ai",
                })
                _log(f"{int(time.time())}\tcrawl4ai\tOK\t{r.get('length',0)}\t{url}")
            else:
                code = r.get("failure_code", "unknown")
                fallback.append(url)
                fail_reason[url] = code
                _log(f"{int(time.time())}\tcrawl4ai\tFAIL:{code}\t0\t{url}")

        if fallback:
            try:
                from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider
                fc = FirecrawlWebSearchProvider()
                fc_results = await fc.extract(fallback, **kwargs)
                for fr in fc_results:
                    u = fr.get("url", "")
                    fr["backend"] = "firecrawl(fallback)"
                    fr["crawl4ai_failure"] = fail_reason.get(u, "?")
                    ok = "OK" if not fr.get("error") else f"ERR:{fr.get('error')}"
                    _log(f"{int(time.time())}\tfirecrawl-fb\t{ok}\t{len(fr.get('content','') or '')}\t{u}")
                    results.append(fr)
            except Exception as e:
                for u in fallback:
                    results.append({
                        "url": u, "title": "", "content": "", "raw_content": "",
                        "error": f"crawl4ai_failed:{fail_reason.get(u)};firecrawl_fallback_error:{type(e).__name__}",
                        "backend": "none",
                    })
        return results
