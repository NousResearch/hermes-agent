"""Cloudflare Workers AI provider profile.

Workers AI's OpenAI-compatible surface is account-scoped::

    https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1

``CLOUDFLARE_BASE_URL`` wins when set; otherwise the URL is built from
``CLOUDFLARE_ACCOUNT_ID``. Auth is a Cloudflare API token with Workers AI
Read (``CLOUDFLARE_API_TOKEN``, then ``CLOUDFLARE_API_KEY``).

The live picker catalog comes from ``GET /accounts/{id}/ai/models/search``
(paginated), not the OpenAI-compat ``/v1/models`` path — that endpoint does
not return the full Workers AI roster.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from typing import Any
from urllib.parse import urlparse

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"
API_TOKEN_ENV = "CLOUDFLARE_API_TOKEN"
API_KEY_ENV = "CLOUDFLARE_API_KEY"
BASE_URL_ENV = "CLOUDFLARE_BASE_URL"
SIGNUP_URL = "https://developers.cloudflare.com/workers-ai/get-started/rest-api/"
_SEARCH_PER_PAGE = 50
_SEARCH_MAX_PAGES = 20
_NON_TEXT_TASK_MARKERS = (
    "embed", "image", "speech", "audio", "whisper", "translat", "classif",
    "rerank", "object detection", "caption", "to-image", "text-to-speech",
    "automatic speech", "summarization",
)
_NON_TEXT_ID_MARKERS = (
    "/bge-", "embed", "whisper", "flux", "stable-diffusion", "resnet",
    "melotts", "aura-1", "uform", "detr-", "resnet-",
)

# Offline floor for the picker when the live catalog is unreachable.
_FALLBACK_MODELS = (
    "@cf/moonshotai/kimi-k2.6",
    "@cf/moonshotai/kimi-k2.7-code",
    "@cf/zai-org/glm-5.3",
    "@cf/zai-org/glm-5.3-flash",
    "@cf/zai-org/glm-5.2",
    "@cf/zai-org/glm-4.7-flash",
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/deepseek-ai/deepseek-v4-flash-0731",
    "@cf/deepseek-ai/deepseek-v4-pro-0813",
    "@cf/ibm-granite/granite-4.0-h-micro",
)


def _env(name: str) -> str:
    """``.env``-preferred lookup; plain ``os.environ`` if the dotenv helper is unavailable."""
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv

        return str(get_env_value_prefer_dotenv(name) or "").strip()
    except Exception:
        return os.environ.get(name, "").strip()


def _account_id() -> str:
    return _env(ACCOUNT_ID_ENV).strip().strip("\"'")


def workers_ai_url_for_account(account_id: str) -> str:
    """OpenAI-compatible chat-completions root for a Cloudflare account."""
    account = (account_id or "").strip().strip("\"'")
    if not account:
        return ""
    return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"


def resolve_workers_ai_base_url(
    api_key: str = "", default_url: str = "", env_override: str = "",
) -> str:
    """Runtime base URL: explicit override, then account-id URL, then *default_url*.

    Signature matches ``_API_KEY_BASE_URL_RESOLVERS`` / ``ProviderProfile.resolve_base_url``.
    """
    override = (env_override or _env(BASE_URL_ENV)).strip().rstrip("/")
    if override:
        return override
    constructed = workers_ai_url_for_account(_account_id())
    if constructed:
        return constructed
    return (default_url or "").strip().rstrip("/")


def account_id_from_base_url(base_url: str) -> str:
    """Extract the account id from an OpenAI-compat Workers AI base URL."""
    parts = [p for p in urlparse(base_url or "").path.split("/") if p]
    if "accounts" in parts:
        idx = parts.index("accounts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def models_search_url(account_id: str, *, page: int = 1, per_page: int = _SEARCH_PER_PAGE) -> str:
    query = urllib.parse.urlencode({"per_page": per_page, "page": page})
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search?{query}"


def _catalog_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("data")
        if items is None:
            items = data.get("result")
        if isinstance(items, dict):
            items = items.get("data") or items.get("models") or []
        return items if isinstance(items, list) else []
    return []


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_workers_ai_model_id(mid: str) -> bool:
    """True for catalog slugs (``@cf/...``); false for Cloudflare's internal UUID ``id``."""
    value = (mid or "").strip()
    if not value or _UUID_RE.match(value):
        return False
    return value.startswith(("@cf/", "@hf/")) or "/" in value


def _item_id(item: Any) -> str:
    # models/search uses UUID ``id`` and the public slug in ``name``.
    if isinstance(item, dict):
        for key in ("name", "model", "id"):
            mid = item.get(key)
            if isinstance(mid, str) and is_workers_ai_model_id(mid):
                return mid.strip()
        return ""
    return item.strip() if isinstance(item, str) and is_workers_ai_model_id(item) else ""


def _task_name(item: dict) -> str:
    task = item.get("task")
    if isinstance(task, dict):
        return str(task.get("name") or task.get("id") or "").lower()
    if isinstance(task, str):
        return task.lower()
    return ""


def _is_text_generation_item(item: Any) -> bool:
    """Keep chat/text-generation rows; drop embeddings, image, ASR, TTS."""
    mid = _item_id(item).lower()
    if any(marker in mid for marker in _NON_TEXT_ID_MARKERS):
        return False
    if not isinstance(item, dict):
        return True
    task = _task_name(item)
    if task and any(marker in task for marker in _NON_TEXT_TASK_MARKERS):
        return False
    if task and not any(token in task for token in ("text", "generation", "convers", "chat", "llm")):
        return False
    return True


def _ids_from_catalog_payload(data: Any, *, text_generation_only: bool = False) -> list[str]:
    """Accept OpenAI ``{data: [...]}`` and Cloudflare ``{result: ...}`` envelopes."""
    ids: list[str] = []
    seen: set[str] = set()
    for item in _catalog_items(data):
        if text_generation_only and not _is_text_generation_item(item):
            continue
        mid = _item_id(item)
        if mid and mid not in seen:
            seen.add(mid)
            ids.append(mid)
    return ids


def _json_get(url: str, api_key: str | None, headers: dict[str, str], timeout: float) -> Any:
    from hermes_cli.urllib_security import open_credentialed_url

    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", _profile_user_agent())
    for key, value in headers.items():
        req.add_header(key, value)
    with open_credentialed_url(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_workers_ai_models(
    *,
    api_key: str | None,
    account_id: str = "",
    openai_base_url: str = "",
    timeout: float = 8.0,
    headers: dict[str, str] | None = None,
) -> list[str] | None:
    """Live catalog: paginated ``/ai/models/search``, then OpenAI-compat ``/models``."""
    account = (account_id or account_id_from_base_url(openai_base_url) or _account_id()).strip()
    hdrs = dict(headers or {})
    if account:
        ids: list[str] = []
        seen: set[str] = set()
        try:
            for page in range(1, _SEARCH_MAX_PAGES + 1):
                payload = _json_get(models_search_url(account, page=page), api_key, hdrs, timeout)
                page_ids = _ids_from_catalog_payload(payload, text_generation_only=True)
                if not page_ids:
                    break
                for mid in page_ids:
                    if mid not in seen:
                        seen.add(mid)
                        ids.append(mid)
                items = _catalog_items(payload)
                info = payload.get("result_info") if isinstance(payload, dict) else None
                total_pages = 0
                if isinstance(info, dict):
                    total_pages = int(info.get("total_pages") or 0)
                if total_pages and page >= total_pages:
                    break
                if len(items) < _SEARCH_PER_PAGE:
                    break
            if ids:
                return ids
        except Exception as exc:
            logger.debug("workers-ai models/search: %s", exc)

    openai_url = (openai_base_url or workers_ai_url_for_account(account)).rstrip("/")
    if not openai_url:
        return None
    try:
        payload = _json_get(openai_url + "/models", api_key, hdrs, timeout)
        ids = _ids_from_catalog_payload(payload, text_generation_only=True)
        return ids or None
    except Exception as exc:
        logger.debug("workers-ai /models: %s", exc)
        return None


class WorkersAIProfile(ProviderProfile):
    """Account-scoped OpenAI-compatible Workers AI endpoint."""

    def resolve_base_url(
        self, api_key: str = "", default_url: str = "", env_override: str = "",
    ) -> str:
        return resolve_workers_ai_base_url(api_key, default_url or self.base_url, env_override)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        resolved = (base_url or "").strip().rstrip("/") or self.resolve_base_url(
            api_key=api_key or "",
        )
        return fetch_workers_ai_models(
            api_key=api_key,
            account_id=account_id_from_base_url(resolved) or _account_id(),
            openai_base_url=resolved,
            timeout=timeout,
            headers=self.default_headers,
        )

    def get_hostname(self) -> str:
        if self.hostname:
            return self.hostname
        resolved = self.resolve_base_url()
        if resolved:
            return urlparse(resolved).hostname or ""
        return "api.cloudflare.com"


workers_ai = WorkersAIProfile(
    name="workers-ai",
    aliases=("cloudflare", "cloudflare-workers-ai", "cf-workers-ai", "workersai", "cloudflare-ai"),
    display_name="Cloudflare Workers AI",
    description="Cloudflare Workers AI — OpenAI-compatible edge inference (@cf/ model ids)",
    signup_url=SIGNUP_URL,
    env_vars=(API_TOKEN_ENV, API_KEY_ENV, BASE_URL_ENV),
    base_url="",  # account-scoped; resolve_base_url fills it at runtime
    hostname="api.cloudflare.com",
    auth_type="api_key",
    default_headers={"User-Agent": f"HermesAgent/{_HERMES_VERSION}"},
    default_aux_model="@cf/zai-org/glm-4.7-flash",
    fallback_models=_FALLBACK_MODELS,
)

register_provider(workers_ai)
