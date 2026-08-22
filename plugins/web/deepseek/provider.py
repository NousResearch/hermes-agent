"""DeepSeek Web Search — plugin form.

Routes ``web_search`` tool calls through DeepSeek's server-side ``web_search``
tool on the Responses API. DeepSeek runs the searching server-side; we ask
the model (``deepseek-v4-flash``) to return the top results as structured
JSON so we hand back the same ``{title, url, description, position}`` rows
every other Hermes web provider produces.

Reference: https://api-docs.deepseek.com/guides/responses_api

Config keys this provider responds to::

    web:
      search_backend: "deepseek"      # explicit per-capability
      backend: "deepseek"             # shared fallback

Optional knobs (under ``web.deepseek`` in ``config.yaml``)::

    web:
      deepseek:
        model: "deepseek-v4-flash"    # Responses API currently only supports deepseek-v4-flash
        base_url: "https://api.deepseek.com"
        timeout: 90                   # seconds (default 90)

Auth: reads ``DEEPSEEK_API_KEY`` via the config-aware env lookup
(:func:`agent.web_search_provider.get_provider_env`), so a key stored in
``~/.hermes/.env`` is picked up without being exported to the process env.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider, get_provider_env

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT = 90

# Match the JSON object the model is asked to emit. Tolerates leading/trailing
# prose since reasoning models occasionally narrate before the JSON block
# even when explicitly asked not to.
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_deepseek_web_config() -> Dict[str, Any]:
    """Read ``web.deepseek`` from config.yaml (returns {} on miss)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web_section = cfg.get("web") if isinstance(cfg, dict) else None
        ds_section = web_section.get("deepseek") if isinstance(web_section, dict) else None
        return ds_section if isinstance(ds_section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load web.deepseek config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class DeepSeekWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by DeepSeek's server-side Web Search tool.

    Sends a structured prompt to DeepSeek with ``tools=[{"type": "web_search"}]``
    enabled and asks it to return the top *limit* results as JSON. Falls back
    to the Responses API ``citations`` list if the model ignores the JSON
    schema instruction.

    No extract capability — pair with Tavily / Firecrawl / Exa for
    ``web_extract`` if you need page content.

    Trust model
    -----------
    Same caveat as web-xai: this backend is an LLM in a trench coat. DeepSeek
    decides which URLs to surface and generates titles/descriptions itself,
    so callers that pipe untrusted text directly into ``web_search`` should
    treat returned URLs as model-generated links — validate before fetching.
    """

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def display_name(self) -> str:
        return "DeepSeek Web Search (V4 Flash)"

    def is_available(self) -> bool:
        """Cheap availability probe — just checks for the API key."""
        return bool(get_provider_env("DEEPSEEK_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    # -- Search -----------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a DeepSeek-backed web search.

        Returns ``{"success": True, "data": {"web": [{title, url, description, position}, ...]}}``
        on success, ``{"success": False, "error": str}`` on failure.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except Exception:  # noqa: BLE001 — interrupt module is best-effort
            pass

        api_key = get_provider_env("DEEPSEEK_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": (
                    "DEEPSEEK_API_KEY environment variable not set. "
                    "Get your API key at https://platform.deepseek.com"
                ),
            }

        # Clamp limit to the same range the caller (web_search_tool) accepts.
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 100))

        cfg = _load_deepseek_web_config()
        model = cfg.get("model") if isinstance(cfg.get("model"), str) else DEFAULT_MODEL
        model = model.strip() or DEFAULT_MODEL
        base_url = cfg.get("base_url") if isinstance(cfg.get("base_url"), str) else DEFAULT_BASE_URL
        base_url = base_url.strip().rstrip("/") or DEFAULT_BASE_URL

        try:
            timeout = float(cfg.get("timeout", DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT

        payload: Dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": self._build_prompt(query, limit)}],
            "tools": [{"type": "web_search"}],
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            import httpx
        except ImportError:
            return {
                "success": False,
                "error": "httpx is not installed (required for DeepSeek web search)",
            }

        logger.info(
            "DeepSeek web search via %s: '%s' (limit=%d, model=%s)",
            base_url, query, limit, model,
        )

        try:
            resp = httpx.post(
                f"{base_url}/responses",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            body = ""
            try:
                body = exc.response.text[:300] if exc.response is not None else ""
            except Exception:  # noqa: BLE001
                body = ""
            logger.warning("DeepSeek web search HTTP %d: %s", status, body)
            return {
                "success": False,
                "error": f"DeepSeek web search returned HTTP {status}: {body}".rstrip(),
            }
        except httpx.RequestError as exc:
            logger.warning("DeepSeek web search request error: %s", exc)
            return {"success": False, "error": f"Could not reach DeepSeek: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepSeek web search bad JSON: %s", exc)
            return {
                "success": False,
                "error": "Could not parse DeepSeek Responses API reply as JSON",
            }

        # DeepSeek's Responses surface sometimes returns HTTP 200 with an
        # error envelope (model overloaded, content-policy refusal, etc.).
        api_error = data.get("error") if isinstance(data, dict) else None
        if isinstance(api_error, dict):
            err_msg = (
                api_error.get("message")
                or api_error.get("code")
                or "unknown error"
            )
            logger.warning("DeepSeek web search returned error envelope: %s", err_msg)
            return {"success": False, "error": f"DeepSeek returned an error: {err_msg}"}

        web_results = self._extract_results(data, limit=limit)
        if not web_results:
            # Successful call, just no usable rows — return success with an
            # empty list so the model can decide whether to retry.
            return {"success": True, "data": {"web": []}}

        return {"success": True, "data": {"web": web_results}}

    # -- Prompt + parsing -------------------------------------------------

    @staticmethod
    def _build_prompt(query: str, limit: int) -> str:
        """Compose the prompt that asks DeepSeek to act as a search engine.

        We deliberately ask for a JSON object (not bare array) so we can
        match it cheaply with ``_JSON_BLOCK_RE``; we explicitly forbid
        prose, markdown fences, and inline-citation links to keep the
        payload parseable.
        """
        return (
            "Use the web_search tool to find current information for the query below, "
            "then respond with ONLY a single JSON object — no prose, no markdown "
            "fences, no inline citation links — matching this exact schema:\n\n"
            '{"results": [{"title": "string", "url": "string", '
            '"description": "1-2 sentence summary"}]}\n\n'
            f"Return at most {limit} results, ordered by relevance, with absolute "
            'https:// URLs. If no usable results exist, return '
            '{"results": []}.\n\n'
            f"Query: {query}"
        )

    @classmethod
    def _extract_results(
        cls,
        response_data: Dict[str, Any],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Pull a ``[{title, url, description, position}, ...]`` list out of a
        Responses-API reply.

        Strategy:

        1. Walk ``output[*].content[*].text`` for ``output_text`` blocks and
           try to parse the first JSON object that has a ``results`` list.
        2. If the JSON path fails, fall back to the message annotations
           (``url_citation`` entries) — every annotation carries a URL and
           a ``title`` (citation number); we pair those URLs with surrounding
           text from the message body as a best-effort description.
        3. Last-ditch: the top-level ``citations`` list (URLs only).
        """
        text_blocks, annotations = cls._collect_output_text(response_data)

        # Primary path: parse the JSON object the model was asked for.
        for block in text_blocks:
            parsed = cls._try_parse_json_results(block, limit=limit)
            if parsed:
                return parsed

        # Secondary path: derive results from message annotations + raw text.
        if annotations:
            joined_text = "\n".join(text_blocks)
            annotation_results = cls._results_from_annotations(
                annotations, joined_text, limit=limit,
            )
            if annotation_results:
                return annotation_results

        # Last-ditch: raw citations list (no titles or descriptions).
        citations = response_data.get("citations") or []
        if isinstance(citations, list):
            return [
                {
                    "title": "",
                    "url": str(u),
                    "description": "",
                    "position": i + 1,
                }
                for i, u in enumerate(citations[:limit])
                if isinstance(u, str) and u.strip()
            ]

        return []

    @staticmethod
    def _collect_output_text(
        response_data: Dict[str, Any],
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Return (text_blocks, annotations) extracted from ``response.output``."""
        text_blocks: List[str] = []
        annotations: List[Dict[str, Any]] = []
        output = response_data.get("output")
        if not isinstance(output, list):
            return text_blocks, annotations

        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for chunk in content:
                if not isinstance(chunk, dict) or chunk.get("type") != "output_text":
                    continue
                text = chunk.get("text")
                if isinstance(text, str) and text.strip():
                    text_blocks.append(text)
                chunk_annotations = chunk.get("annotations")
                if isinstance(chunk_annotations, list):
                    for ann in chunk_annotations:
                        if isinstance(ann, dict):
                            annotations.append(ann)
        return text_blocks, annotations

    @staticmethod
    def _try_parse_json_results(
        text: str,
        *,
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse a JSON object with a ``results`` array out of ``text``.

        Returns the normalized result list on success, ``None`` when the
        block has no valid JSON object or no ``results`` key. Tolerates
        leading/trailing prose because reasoning models sometimes prefix a
        short narration even when told not to.
        """
        # Try the whole string first — cheapest path when the model obeys.
        candidates = [text]
        match = _JSON_BLOCK_RE.search(text)
        if match and match.group(0) != text:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(parsed, dict):
                continue
            results = parsed.get("results")
            if not isinstance(results, list):
                continue
            normalized: List[Dict[str, Any]] = []
            for row in results[:limit]:
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url", "")).strip()
                if not url:
                    continue
                normalized.append(
                    {
                        "title": str(row.get("title", "")).strip(),
                        "url": url,
                        "description": str(row.get("description", "")).strip(),
                        # Renumber from the kept results, not the raw input
                        # index, so a dropped malformed row doesn't leave a
                        # gap in the positions handed back to the agent.
                        "position": len(normalized) + 1,
                    }
                )
            if normalized:
                return normalized
        return None

    @staticmethod
    def _results_from_annotations(
        annotations: List[Dict[str, Any]],
        joined_text: str,
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Best-effort fallback when JSON parsing fails.

        Uses each ``url_citation`` annotation's ``url`` and slices ~200
        characters of surrounding text as the description.
        """
        seen: set[str] = set()
        results: List[Dict[str, Any]] = []
        for ann in annotations:
            if ann.get("type") != "url_citation":
                continue
            url = str(ann.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)

            description = ""
            start = ann.get("start_index")
            end = ann.get("end_index")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(joined_text):
                window_start = max(0, start - 200)
                description = joined_text[window_start:start].strip()
                if len(description) > 200:
                    description = description[-200:].strip()

            results.append(
                {
                    "title": "",
                    "url": url,
                    "description": description,
                    "position": len(results) + 1,
                }
            )
            if len(results) >= limit:
                break
        return results

    # -- Setup picker -----------------------------------------------------

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "DeepSeek Web Search (V4 Flash)",
            "badge": "paid",
            "tag": (
                "Server-side web search via DeepSeek's Responses API — "
                "uses the existing DEEPSEEK_API_KEY."
            ),
            "env_vars": [
                {
                    "key": "DEEPSEEK_API_KEY",
                    "prompt": "DeepSeek API key",
                    "url": "https://platform.deepseek.com",
                },
            ],
        }
