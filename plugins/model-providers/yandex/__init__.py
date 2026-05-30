"""Yandex Cloud AI Studio provider profile."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
_DEFAULT_MODEL = "deepseek-v4-flash/latest"


def _folder_id() -> str:
    return os.environ.get("YANDEX_FOLDER_ID", "").strip().strip('"').strip("'")


def normalize_yandex_model(model_input: str) -> str:
    """Resolve a config/user model id to a Yandex ``gpt://`` URI."""
    folder_id = _folder_id()
    name = (model_input or "").strip()
    if not name:
        return name
    if "${YANDEX_FOLDER_ID}" in name:
        if not folder_id:
            return name
        name = name.replace("${YANDEX_FOLDER_ID}", folder_id)
    if name.startswith("gpt://"):
        return name
    if not folder_id:
        return name
    return f"gpt://{folder_id}/{name.lstrip('/')}"


def _yandex_headers() -> dict[str, str]:
    headers = {"x-data-logging-enabled": "false"}
    folder_id = _folder_id()
    if folder_id:
        headers["x-folder-id"] = folder_id
    return headers


class YandexProfile(ProviderProfile):
    """Yandex AI Studio — folder headers read from env at access time."""

    @property
    def default_headers(self) -> dict[str, str]:
        return _yandex_headers()

    @default_headers.setter
    def default_headers(self, value: dict[str, str]) -> None:
        # dataclass compatibility; ignore static assignment
        return

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        url = (self.models_url or "").strip() or (self.base_url.rstrip("/") + "/models")
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Api-Key {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        for key, value in _yandex_headers().items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            return [m["id"] for m in items if isinstance(m, dict) and "id" in m]
        except Exception:
            return None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Yandex OpenAI-compat rejects extra_body.thinking for DeepSeek V4 Flash.
        return {}, {}


yandex = YandexProfile(
    name="yandex",
    aliases=("yandex-ai-studio", "yandex-aistudio"),
    env_vars=("YANDEX_API_KEY", "YANDEX_FOLDER_ID"),
    display_name="Yandex AI Studio",
    description="Yandex Cloud AI Studio — Model Gallery",
    signup_url="https://aistudio.yandex.ru/",
    base_url=_BASE_URL,
    hostname="llm.api.cloud.yandex.net",
    auth_type="api_key",
    fallback_models=(_DEFAULT_MODEL,),
    default_aux_model=_DEFAULT_MODEL,
)

register_provider(yandex)
