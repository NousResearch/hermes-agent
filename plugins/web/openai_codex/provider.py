"""OpenAI Codex web search provider — ChatGPT Pro/Plus OAuth, no API key.

Routes ``web_search`` through the OpenAI Codex backend
(``chatgpt.com/backend-api/codex``) using the Responses-API server-side
``web_search`` tool. Authenticated with the Hermes-managed Codex OAuth
(``hermes auth add openai-codex``) — no separate API key and no per-call
web-search billing for ChatGPT Pro/Plus subscribers.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider
from agent.auxiliary_client import (
    _read_codex_access_token,
    _codex_cloudflare_headers,
    _CODEX_AUX_BASE_URL,
)

DEFAULT_MODEL = "gpt-5.5"
_JSON_RE = re.compile(r'\{.*"results".*\}', re.DOTALL)

_SCHEMA = (
    '{"results": [{"title": "string", "url": "https://...", '
    '"description": "1-2 sentence summary"}]}'
)


class OpenAICodexWebSearchProvider(WebSearchProvider):
    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex Web Search (ChatGPT Pro OAuth)"

    def is_available(self) -> bool:
        try:
            return bool(_read_codex_access_token())
        except Exception:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            token = _read_codex_access_token()
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": str(exc)}
        if not token:
            return {
                "success": False,
                "error": "No Codex OAuth credentials. Run `hermes auth add openai-codex`.",
            }
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            limit = 5

        headers = _codex_cloudflare_headers(token)
        headers["Authorization"] = "Bearer " + token
        headers["Content-Type"] = "application/json"

        prompt = (
            "Use the web_search tool to find current information for the query "
            "below, then respond with ONLY a single JSON object (no prose, no "
            "markdown fences) matching this schema:\n" + _SCHEMA + "\n"
            f"Return at most {limit} results, ordered by relevance, with absolute "
            'https:// URLs. If none, return {"results": []}.\n\nQuery: ' + query
        )
        payload = {
            "model": DEFAULT_MODEL,
            "instructions": (
                "You are a web search backend. Use the web_search tool, then "
                "output only the requested JSON object."
            ),
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "tools": [{"type": "web_search"}],
            "stream": True,
            "store": False,
        }
        req = urllib.request.Request(
            _CODEX_AUX_BASE_URL + "/responses",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read().decode()
        except urllib.error.HTTPError as exc:
            return {
                "success": False,
                "error": f"Codex web search HTTP {exc.code}: {exc.read().decode()[:200]}",
            }
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": str(exc)}

        text = ""
        for line in raw.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except Exception:
                continue
            if event.get("type") == "response.output_text.delta":
                text += event.get("delta", "")

        return {"success": True, "data": {"web": self._parse_results(text, limit)}}

    @staticmethod
    def _parse_results(text: str, limit: int) -> List[Dict[str, Any]]:
        match = _JSON_RE.search(text)
        rows: List[Any] = []
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    rows = obj.get("results") or []
            except Exception:
                rows = []
        out: List[Dict[str, Any]] = []
        for i, row in enumerate(rows[:limit]):
            if not isinstance(row, dict):
                continue
            url = str(row.get("url", "")).strip()
            if not url:
                continue
            out.append(
                {
                    "title": str(row.get("title", "")),
                    "url": url,
                    "description": str(row.get("description", "")),
                    "position": i + 1,
                }
            )
        return out

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "fields": [],
            "notes": (
                "Authenticate with `hermes auth add openai-codex` (ChatGPT "
                "Pro/Plus). No API key required."
            ),
        }
