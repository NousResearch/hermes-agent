"""Crawl4ai extract provider — self-hosted, Bearer token auth."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx
from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url

logger = logging.getLogger(__name__)


def _crawl4ai_url() -> str:
    """Return CRAWL4AI_URL from config.yaml, then Hermes env, then process env."""
    # 1. Check config.yaml web.crawl4ai_url
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        val = cfg.get("web", {}).get("crawl4ai_url", "").strip()
        if val:
            return val.rstrip("/")
    except Exception:
        pass
    
    # 2. Fallback to Hermes config-aware env (get_env_value checks .env files)
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value("CRAWL4AI_URL")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("CRAWL4AI_URL", "")
    return (val or "").strip().rstrip("/")


def _crawl4ai_token() -> str:
    """Return CRAWL4AI_API_TOKEN from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value("CRAWL4AI_API_TOKEN")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("CRAWL4AI_API_TOKEN", "")
    return (val or "").strip()


class Crawl4aiWebExtractProvider(WebSearchProvider):
    """Crawl4ai extraction provider using /md endpoint."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4ai (self-hosted)"

    def is_available(self) -> bool:
        return bool(_crawl4ai_url() and _crawl4ai_token())

    def supports_search(self) -> bool:
        return False  # Use SearXNG for search

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        format = kwargs.get("format", "markdown")
        results: List[Dict[str, Any]] = []

        base_url = _crawl4ai_url()
        token = _crawl4ai_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            for url in urls:
                # SSRF check
                if not is_safe_url(url):
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": "Blocked: URL targets a private or internal network address",
                    })
                    continue

                try:
                    logger.info("Crawl4ai extracting: %s", url)
                    response = await client.post(
                        f"{base_url}/md",
                        json={"url": url},
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Response shape: {"url": "...", "markdown": "...", "success": true}
                    markdown = data.get("markdown", "")
                    if isinstance(markdown, dict):
                        markdown = markdown.get("raw_markdown", "") or markdown.get("markdown", "")

                    title = self._extract_title(markdown, url)

                    results.append({
                        "url": url,
                        "title": title,
                        "content": markdown,
                        "raw_content": markdown,
                        "metadata": {},
                    })

                except httpx.HTTPStatusError as e:
                    logger.warning("Crawl4ai HTTP error for %s: %s", url, e)
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"Crawl4ai HTTP {e.response.status_code}: {e.response.text[:200]}",
                    })
                except Exception as e:  # noqa: BLE001
                    logger.warning("Crawl4ai extraction error for %s: %s", url, e)
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"Crawl4ai extraction failed: {e}",
                    })

        return results

    def _extract_title(self, markdown: str, fallback_url: str) -> str:
        if not markdown:
            return fallback_url
        for line in markdown.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return fallback_url

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "self-hosted",
            "tag": "Uses your self-hosted crawl4ai instance at CRAWL4AI_URL",
            "env_vars": [
                {
                    "key": "CRAWL4AI_URL",
                    "prompt": "Crawl4ai instance URL (e.g., https://crawl4ai.docker.packetflood.net)",
                    "url": "https://github.com/unclecode/crawl4ai",
                },
                {
                    "key": "CRAWL4AI_API_TOKEN",
                    "prompt": "Crawl4ai JWT API token",
                    "url": "",
                },
            ],
        }