"""OpenAI Codex OAuth web-search/extract provider.

This provider talks directly to the ChatGPT/Codex Responses endpoint with the
same Hermes-managed ``openai-codex`` OAuth credentials used by the Codex model
provider and image-generation plugin. The hosted Codex endpoint exposes a
single ``web_search`` tool; asking it to open a concrete URL emits an
``open_page`` action inside that web-search call, so both Hermes
``web_search`` and ``web_extract`` map to the same hosted tool type.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.auxiliary_client import _codex_cloudflare_headers, _read_codex_access_token
from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TIMEOUT = 120.0
DEFAULT_EXTRACT_MAX_CHARS = 12000


def _has_codex_oauth_token() -> bool:
    """Cheap probe for locally-stored OpenAI Codex OAuth credentials.

    This intentionally does **not** call :func:`_read_codex_access_token`: that
    helper may consult the credential pool and can perform heavier auth-store
    logic. Provider availability checks run during plugin registration and
    ``hermes tools`` repaints, so they must stay side-effect-free and avoid
    refresh/network paths.
    """
    try:
        from hermes_constants import get_hermes_home

        auth_path = get_hermes_home() / "auth.json"
        if not auth_path.exists():
            return False
        store = json.loads(auth_path.read_text())
        if not isinstance(store, dict):
            return False

        providers = store.get("providers")
        state = providers.get("openai-codex") if isinstance(providers, dict) else None
        tokens = state.get("tokens") if isinstance(state, dict) else None
        token = tokens.get("access_token") if isinstance(tokens, dict) else None
        if str(token or "").strip():
            return True

        pool = store.get("credential_pool")
        entries = pool.get("openai-codex") if isinstance(pool, dict) else None
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and str(entry.get("access_token") or "").strip():
                    return True
    except Exception:
        return False
    return False


def _load_codex_web_config() -> Dict[str, Any]:
    """Load ``web.openai-codex`` / ``web.openai_codex`` config.

    The hyphenated key mirrors the provider id and works with
    ``hermes config set web.openai-codex.model ...``; the underscored alias is
    accepted for users editing YAML by hand. When no web-specific model/base URL
    is configured, reuse the main ``model`` section if it is also targeting
    ``openai-codex``.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = {}

    if not isinstance(cfg, dict):
        cfg = {}
    web_cfg = cfg.get("web") if isinstance(cfg.get("web"), dict) else {}
    specific: Dict[str, Any] = {}
    if isinstance(web_cfg, dict):
        for key in ("openai-codex", "openai_codex"):
            candidate = web_cfg.get(key)
            if isinstance(candidate, dict):
                specific.update(candidate)

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    if isinstance(model_cfg, dict) and str(model_cfg.get("provider") or "").strip() == "openai-codex":
        specific.setdefault("model", model_cfg.get("default"))
        specific.setdefault("base_url", model_cfg.get("base_url"))

    return specific


def _coerce_positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result <= 0 or result == float("inf"):
        return default
    return result


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


class OpenAICodexWebSearchProvider(WebSearchProvider):
    """Web search + URL extraction through OpenAI Codex OAuth hosted web tools."""

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex Web (OAuth)"

    def is_available(self) -> bool:
        return _has_codex_oauth_token()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except Exception:
            pass

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 100))

        cfg = _load_codex_web_config()
        prompt = self._build_search_prompt(query, limit)
        instructions = (
            "Use the hosted web_search tool to perform web search. Return ONLY "
            "a single JSON object matching this schema, with no markdown fences "
            "or prose: {\"results\":[{\"title\":\"string\",\"url\":\"string\","
            "\"description\":\"1-2 sentence summary\"}]}."
        )

        ok, payload_or_error = self._call_codex_responses(
            prompt=prompt,
            instructions=instructions,
            cfg=cfg,
        )
        if not ok:
            return {"success": False, "error": str(payload_or_error)}

        results = self._extract_search_results(payload_or_error, limit=limit)
        return {"success": True, "data": {"web": results}}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": str(url), "title": "", "content": "", "error": "Interrupted"}
                    for url in urls
                ]
        except Exception:
            pass

        cfg = _load_codex_web_config()
        extract_format = str(kwargs.get("format") or "markdown").strip() or "markdown"
        max_chars = _coerce_positive_int(
            cfg.get("extract_max_chars") or cfg.get("max_extract_chars"),
            DEFAULT_EXTRACT_MAX_CHARS,
        )
        results: List[Dict[str, Any]] = []
        for raw_url in urls:
            url = str(raw_url or "").strip()
            if not url:
                continue
            results.append(
                self._extract_one_url(
                    url,
                    cfg=cfg,
                    extract_format=extract_format,
                    max_chars=max_chars,
                )
            )
        return results

    def _extract_one_url(
        self,
        url: str,
        *,
        cfg: Dict[str, Any],
        extract_format: str,
        max_chars: int,
    ) -> Dict[str, Any]:
        instructions = (
            "Use the hosted web_search tool to open the requested URL. Return "
            "ONLY compact JSON matching {\"title\":\"string\",\"content\":\"string\"}. "
            "Do not include markdown fences or prose outside the JSON object."
        )
        prompt = (
            f"Open {url} and extract the main page content as {extract_format}. "
            f"Return at most {max_chars} characters of useful content. If the "
            "page cannot be opened or has no readable main content, return "
            "{\"title\":\"\",\"content\":\"\"}."
        )
        ok, payload_or_error = self._call_codex_responses(
            prompt=prompt,
            instructions=instructions,
            cfg=cfg,
        )
        if not ok:
            return {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": str(payload_or_error),
                "metadata": {"provider": self.name, "action": "open_page", "format": extract_format},
            }

        title, content = self._extract_page_json(payload_or_error)
        if len(content) > max_chars:
            content = content[:max_chars].rstrip()
        action = self._first_hosted_action(payload_or_error) or "open_page"
        base = {
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": {"provider": self.name, "action": action, "format": extract_format},
        }
        if not content.strip():
            base["error"] = "No content returned by OpenAI Codex hosted open_page action"
        return base

    def _call_codex_responses(
        self,
        *,
        prompt: str,
        instructions: str,
        cfg: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        token = _read_codex_access_token()
        if not token:
            return (
                False,
                "No OpenAI Codex OAuth token found. Run `hermes auth add openai-codex`.",
            )

        try:
            import httpx
        except ImportError:
            return False, "httpx is not installed (required for OpenAI Codex web tools)"

        model = str(cfg.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        base_url = str(cfg.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL
        timeout = _coerce_positive_float(cfg.get("timeout"), DEFAULT_TIMEOUT)

        headers = _codex_cloudflare_headers(token)
        headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
        )
        payload: Dict[str, Any] = {
            "model": model,
            "store": False,
            "instructions": instructions,
            "input": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search"}],
            "stream": True,
        }

        try:
            with httpx.stream(
                "POST",
                f"{base_url}/responses",
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as resp:
                if getattr(resp, "status_code", 0) >= 400:
                    body = ""
                    try:
                        raw = resp.read()
                        body = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                    except Exception:
                        body = str(getattr(resp, "text", "") or "")
                    body = body[:500]
                    return False, f"OpenAI Codex web returned HTTP {resp.status_code}: {body}".rstrip()
                return True, self._parse_sse_response(resp.iter_lines())
        except httpx.RequestError as exc:
            logger.warning("OpenAI Codex web request error: %s", exc)
            return False, f"Could not reach OpenAI Codex web endpoint: {exc}"
        except Exception as exc:
            logger.warning("OpenAI Codex web call failed: %s", exc)
            return False, f"OpenAI Codex web call failed: {exc}"

    @staticmethod
    def _build_search_prompt(query: str, limit: int) -> str:
        return (
            "Search the web for the query below using the hosted web_search tool. "
            "Return ONLY JSON matching {\"results\":[{\"title\":\"string\","
            "\"url\":\"string\",\"description\":\"1-2 sentence summary\"}]}. "
            f"Return at most {limit} results, ordered by relevance, with absolute URLs. "
            "If no usable results exist, return {\"results\":[]}.\n\n"
            f"Query: {query}"
        )

    @staticmethod
    def _parse_sse_response(lines: Iterable[Any]) -> Dict[str, Any]:
        event_type: Optional[str] = None
        data_lines: List[str] = []
        events: List[Dict[str, Any]] = []
        output_items: List[Dict[str, Any]] = []
        completed_texts: List[str] = []
        deltas: List[str] = []

        def flush() -> None:
            nonlocal event_type, data_lines
            if not data_lines:
                event_type = None
                return
            raw = "\n".join(data_lines).strip()
            current_event = event_type
            event_type = None
            data_lines = []
            if not raw or raw == "[DONE]":
                return
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"type": current_event or "message", "raw": raw}
            if current_event and isinstance(payload, dict) and "type" not in payload:
                payload["type"] = current_event
            if not isinstance(payload, dict):
                return
            events.append(payload)

            ptype = payload.get("type")
            if ptype == "response.output_text.delta":
                delta = payload.get("delta")
                if isinstance(delta, str):
                    deltas.append(delta)
            if ptype == "response.output_item.done":
                item = payload.get("item")
                if isinstance(item, dict):
                    output_items.append(item)
                    completed_texts.extend(_text_blocks_from_item(item))
            if ptype == "response.completed":
                response = payload.get("response")
                output = response.get("output") if isinstance(response, dict) else None
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict):
                            output_items.append(item)
                            completed_texts.extend(_text_blocks_from_item(item))

        for raw_line in lines:
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
            if line == "":
                flush()
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        flush()

        text = "\n".join(t for t in completed_texts if t.strip()).strip()
        if not text:
            text = "".join(deltas).strip()
        return {"events": events, "output": output_items, "text": text}

    @classmethod
    def _extract_search_results(cls, response: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
        for text in cls._candidate_texts(response):
            parsed = cls._parse_json_object(text)
            if not isinstance(parsed, dict):
                continue
            rows = parsed.get("results")
            if not isinstance(rows, list):
                continue
            normalized: List[Dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url") or "").strip()
                if not url:
                    continue
                normalized.append(
                    {
                        "title": str(row.get("title") or "").strip(),
                        "url": url,
                        "description": str(row.get("description") or "").strip(),
                        "position": len(normalized) + 1,
                    }
                )
                if len(normalized) >= limit:
                    break
            if normalized:
                return normalized
        return []

    @classmethod
    def _extract_page_json(cls, response: Dict[str, Any]) -> Tuple[str, str]:
        for text in cls._candidate_texts(response):
            parsed = cls._parse_json_object(text)
            if isinstance(parsed, dict):
                title = str(parsed.get("title") or "").strip()
                content = str(parsed.get("content") or parsed.get("markdown") or parsed.get("text") or "").strip()
                if title or content:
                    return title, content
        text = str(response.get("text") or "").strip()
        return "", text

    @staticmethod
    def _candidate_texts(response: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        text = response.get("text")
        if isinstance(text, str) and text.strip():
            candidates.append(text)
        for item in response.get("output") or []:
            if isinstance(item, dict):
                candidates.extend(_text_blocks_from_item(item))
        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique: List[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        return unique

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        stripped = text.strip()
        starts = [0] if stripped.startswith("{") else []
        starts.extend(i for i, ch in enumerate(stripped) if ch == "{" and i not in starts)
        for start in starts:
            try:
                parsed, _end = decoder.raw_decode(stripped[start:])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _first_hosted_action(response: Dict[str, Any]) -> Optional[str]:
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            action = item.get("action")
            if isinstance(action, dict):
                action_type = action.get("type")
                if isinstance(action_type, str) and action_type.strip():
                    return action_type.strip()
        return None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI Codex Web (OAuth)",
            "badge": "oauth",
            "tag": (
                "Hosted web_search/open_page via ChatGPT Codex OAuth. Requires "
                "`hermes auth add openai-codex`."
            ),
            "env_vars": [],
        }


def _text_blocks_from_item(item: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    if item.get("type") != "message":
        return texts
    content = item.get("content")
    if not isinstance(content, list):
        return texts
    for chunk in content:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("type") not in {"output_text", "text"}:
            continue
        text = chunk.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts
