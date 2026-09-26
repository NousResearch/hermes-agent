"""xAI Web Search — search-only provider backed by Grok's server-side ``web_search`` tool on the
Responses API (https://docs.x.ai/developers/tools/web-search); Grok is asked for structured JSON
so rows match every other Hermes web provider. Config: ``web.backend: "xai"``; optional ``web.xai``:
``model`` (default grok-build-0.1), ``allowed_domains`` / ``excluded_domains`` (max 5, mutually
exclusive), ``timeout`` (default 90s). Auth: Grok OAuth via ``hermes auth``, else XAI_API_KEY.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from plugins.web._common import BaseWebSearchProvider, search_fail as _fail, search_ok, setup_schema, title_hit as _row
from tools.xai_http import has_xai_credentials, hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "grok-build-0.1"
DEFAULT_TIMEOUT = 90
_MAX_DOMAIN_FILTERS = 5  # xAI hard cap on allowed_domains / excluded_domains

# Tolerates leading/trailing prose — reasoning models occasionally narrate before the JSON block.
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _load_xai_web_config() -> Dict[str, Any]:
    """Read ``web.xai`` from config.yaml (returns {} on miss)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        for key in ("web", "xai"):
            cfg = cfg.get(key) if isinstance(cfg, dict) else None
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load web.xai config: %s", exc)
        return {}


def _coerce_domain_list(value: Any) -> List[str]:
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:_MAX_DOMAIN_FILTERS] if isinstance(value, list) else []


def _coerce(cast, value: Any, default: Any) -> Any:
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


class XAIWebSearchProvider(BaseWebSearchProvider):
    """Sends a structured prompt with ``tools=[{"type": "web_search"}]`` and parses the JSON Grok
    returns; falls back to message annotations, then ``citations``. Trust model: Grok *generates*
    the URLs/titles/descriptions and is steerable by the query text — validate before fetching."""

    NAME = "xai"
    DISPLAY_NAME = "xAI Web Search (Grok)"

    def is_available(self) -> bool:
        """Cheap probe (env var OR auth-store tokens). Deliberately NOT
        ``resolve_xai_http_credentials``: must never refresh tokens or take the
        auth-store lock, since this runs on every ``hermes tools`` repaint."""
        return has_xai_credentials()

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            from tools.interrupt import is_interrupted
            if is_interrupted():
                return _fail("Interrupted")
        except Exception:  # noqa: BLE001 — interrupt module is best-effort
            pass
        creds = resolve_xai_http_credentials()
        api_key = str(creds.get("api_key") or "").strip()
        base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")
        if not api_key:
            return _fail("No xAI credentials found. Run `hermes auth` to sign in with xAI Grok OAuth, or set XAI_API_KEY.")
        # Same clamp range as web_search_tool so explicit limits aren't downgraded.
        limit = max(1, min(_coerce(int, limit, 5), 100))
        cfg = _load_xai_web_config()
        model = (cfg["model"].strip() if isinstance(cfg.get("model"), str) else "") or DEFAULT_MODEL
        web_search_tool = self._web_search_tool(cfg)
        if web_search_tool is None:
            # xAI rejects this combo — surface a clear error rather than an API 400.
            return _fail("web.xai.allowed_domains and web.xai.excluded_domains cannot both be set (xAI restriction).")
        # include=no_inline_citations keeps the JSON block clean; URLs come from annotations/citations.
        payload: Dict[str, Any] = {"model": model, "input": [{"role": "user", "content": self._build_prompt(query, limit)}], "tools": [web_search_tool], "include": ["no_inline_citations"]}
        try:
            import httpx  # noqa: F401 — availability probe
        except ImportError:
            return _fail("httpx is not installed (required for xAI web search)")
        logger.info("xAI web search via %s: '%s' (limit=%d, model=%s)", base_url, query, limit, model)
        data, error = self._post_responses(
            base_url, payload, api_key, _coerce(float, cfg.get("timeout", DEFAULT_TIMEOUT), DEFAULT_TIMEOUT),
            is_oauth_path=(creds.get("provider") == "xai-oauth"),
        )
        if error:
            return error
        # xAI sometimes returns HTTP 200 with an error envelope (overloaded, refusal);
        # without this check we'd report success-with-no-rows and mask a real failure.
        api_error = data.get("error") if isinstance(data, dict) else None
        if isinstance(api_error, dict):
            err_msg = api_error.get("message") or api_error.get("code") or "unknown error"
            logger.warning("xAI web search returned error envelope: %s", err_msg)
            return _fail(f"xAI returned an error: {err_msg}")
        # Empty list on 0 hits is a success (matches brave-free / exa).
        return search_ok(self._extract_results(data, limit=limit))

    @staticmethod
    def _web_search_tool(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """``web_search`` tool spec with optional domain filters; None when both
        allowed and excluded are set (xAI rejects the combination)."""
        filters = {k: _coerce_domain_list(cfg.get(k)) for k in ("allowed_domains", "excluded_domains")}
        filters = {k: v for k, v in filters.items() if v}
        if len(filters) == 2:
            return None
        return {"type": "web_search", "filters": filters} if filters else {"type": "web_search"}

    @staticmethod
    def _post_responses(base_url: str, payload: Dict[str, Any], api_key: str, timeout: float, *, is_oauth_path: bool) -> tuple[Any, Optional[Dict[str, Any]]]:
        """POST ``/responses`` → ``(parsed_json, None)`` or ``(None, failure_envelope)``.

        Two attempts: on a first-call 401 with OAuth creds, force-refresh once and retry
        (opaque tokens the resolver can't pre-check; mid-window revocation/rotation).
        XAI_API_KEY creds can't be refreshed, so they skip the retry rather than burn quota.
        """
        import httpx
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": hermes_xai_user_agent()}
        def _refreshed_key() -> str:
            """New bearer after a 401, or "" when refresh fails / returns the same token (retry would be pointless)."""
            try:
                key = str(resolve_xai_http_credentials(force_refresh=True, api_key_hint=api_key).get("api_key") or "").strip()
                return key if key != api_key else ""
            except Exception as refresh_exc:  # noqa: BLE001
                logger.warning("xAI web search OAuth refresh after 401 failed: %s", refresh_exc)
                return ""

        resp = None
        for attempt in range(2):
            try:
                resp = httpx.post(f"{base_url}/responses", headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 401 and attempt == 0 and is_oauth_path:
                    logger.info("xAI web search got 401 on first attempt; forcing OAuth refresh and retrying once.")
                    if new_key := _refreshed_key():
                        api_key, headers["Authorization"] = new_key, f"Bearer {new_key}"
                        continue
                try:
                    body = exc.response.text[:300] if exc.response is not None else ""
                except Exception:
                    body = ""
                logger.warning("xAI web search HTTP %d: %s", status, body)
                return None, _fail(f"xAI web search returned HTTP {status}: {body}".rstrip())
            except httpx.RequestError as exc:
                logger.warning("xAI web search request error: %s", exc)
                return None, _fail(f"Could not reach xAI: {exc}")
        if resp is None:
            return None, _fail("xAI web search produced no response")
        try:
            return resp.json(), None
        except Exception as exc:  # noqa: BLE001
            logger.warning("xAI web search bad JSON: %s", exc)
            return None, _fail("Could not parse xAI Responses API reply as JSON")

    @staticmethod
    def _build_prompt(query: str, limit: int) -> str:
        """Ask for a JSON *object* (cheap to match with ``_JSON_BLOCK_RE``) and forbid
        prose/fences/inline citations to keep the payload parseable."""
        return (
            "Use the web_search tool to find current information for the query below, then respond with ONLY a single "
            "JSON object — no prose, no markdown fences, no inline citation links — matching this exact schema:\n\n"
            '{"results": [{"title": "string", "url": "string", "description": "1-2 sentence summary"}]}\n\n'
            f'Return at most {limit} results, ordered by relevance, with absolute https:// URLs. If no usable results exist, return '
            '{"results": []}.\n\n'
            f"Query: {query}"
        )

    @classmethod
    def _extract_results(cls, response_data: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
        """Rows in order of preference: (1) the JSON object in ``output_text`` blocks,
        (2) ``url_citation`` annotations paired with surrounding text, (3) the raw
        ``citations`` list. (2) only short-circuits when it yields rows, so future
        annotation types don't mask real data in ``citations``."""
        text_blocks, annotations = cls._collect_output_text(response_data)
        parsed = next((p for p in (cls._try_parse_json_results(b, limit=limit) for b in text_blocks) if p), None)
        if parsed or (annotations and (parsed := cls._results_from_annotations(annotations, "\n".join(text_blocks), limit=limit))):
            return parsed
        citations = response_data.get("citations") or []
        return [_row("", str(u), "", i + 1) for i, u in enumerate(citations[:limit]) if isinstance(u, str) and u.strip()] if isinstance(citations, list) else []

    @staticmethod
    def _collect_output_text(response_data: Dict[str, Any]) -> tuple[List[str], List[Dict[str, Any]]]:
        """(text_blocks, annotations) from ``response.output`` message chunks."""
        output = response_data.get("output")
        chunks = [
            chunk
            for item in (output if isinstance(output, list) else [])
            if isinstance(item, dict) and item.get("type") == "message" and isinstance(item.get("content"), list)
            for chunk in item["content"]
            if isinstance(chunk, dict) and chunk.get("type") == "output_text"
        ]
        text_blocks = [c["text"] for c in chunks if isinstance(c.get("text"), str) and c["text"].strip()]
        annotations = [a for c in chunks if isinstance(c.get("annotations"), list) for a in c["annotations"] if isinstance(a, dict)]
        return text_blocks, annotations

    @staticmethod
    def _try_parse_json_results(text: str, *, limit: int) -> Optional[List[Dict[str, Any]]]:
        """Parse a JSON object with a ``results`` array out of ``text``; None when absent.
        Whole string first, then the regex-matched block (reasoning models prefix narration)."""
        match = _JSON_BLOCK_RE.search(text)
        for candidate in [text] + ([match.group(0)] if match and match.group(0) != text else []):
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            results = parsed.get("results") if isinstance(parsed, dict) else None
            if not isinstance(results, list):
                continue
            normalized: List[Dict[str, Any]] = []
            for row in results[:limit]:
                url = str(row.get("url", "")).strip() if isinstance(row, dict) else ""
                if url:
                    # Renumber from kept rows so a dropped malformed row leaves no gap.
                    normalized.append(_row(str(row.get("title", "")).strip(), url, str(row.get("description", "")).strip(), len(normalized) + 1))
            if normalized:
                return normalized
        return None

    @staticmethod
    def _results_from_annotations(annotations: List[Dict[str, Any]], joined_text: str, *, limit: int) -> List[Dict[str, Any]]:
        """Fallback rows from ``url_citation`` annotations: URL plus ~200 chars of
        preceding text as the description (the annotation title is just a number)."""
        seen: set[str] = set()
        results: List[Dict[str, Any]] = []
        for ann in annotations:
            url = str(ann.get("url", "")).strip() if ann.get("type") == "url_citation" else ""
            if not url or url in seen:
                continue
            seen.add(url)
            description = ""
            start, end = ann.get("start_index"), ann.get("end_index")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(joined_text):
                description = joined_text[max(0, start - 200):start].strip()
                if len(description) > 200:
                    description = description[-200:].strip()
            results.append(_row("", url, description, len(results) + 1))
            if len(results) >= limit:
                break
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        # Auth resolution is delegated to the shared ``xai_grok`` post_setup hook
        # (same one image_gen.xai / tts.xai use) for a consistent OAuth-or-key prompt.
        return setup_schema(
            "xAI Web Search (Grok)", "paid",
            "Agentic web search via Grok's web_search tool — uses xAI Grok OAuth or XAI_API_KEY.", post_setup="xai_grok",
        )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'WebSearchProvider': ('agent.web_search_provider', 'WebSearchProvider'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
