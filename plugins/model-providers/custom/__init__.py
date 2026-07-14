"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → extra_body.think = False
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
  - Ollama backends are additionally gated on the model's declared
    /api/show capabilities: non-thinking models (e.g. llama3.3:70b)
    400 with "does not support thinking" if think/reasoning_effort is
    sent at all, so we probe and skip both fields when unsupported.
"""

import json
import time as _time
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Cache of (model, base_url) -> (capabilities list, timestamp). Non-empty
# results are cached indefinitely (a model's declared capabilities don't
# change at runtime); empty/failed probes get a short TTL so a transient
# network hiccup doesn't wedge the model into "no thinking support" forever.
_OLLAMA_CAPS_CACHE: dict[tuple[str, str], tuple[list[str], float]] = {}
_OLLAMA_CAPS_TTL = 300  # seconds


def _is_ollama_base_url(base_url: str | None) -> bool:
    b = (base_url or "").lower()
    return "ollama" in b or ":11434" in b


def _ollama_model_capabilities(model: str | None, base_url: str | None, timeout: float = 3.0) -> list[str]:
    """Fetch Ollama's declared capabilities for `model` via /api/show.

    Ollama publishes per-model capabilities (e.g. ["completion","tools"] or
    ["completion","tools","thinking"]) at POST {root}/api/show. Only models
    that list "thinking" accept the think / reasoning_effort request fields —
    sending them to a non-thinking model 400s with
    '"<model>" does not support thinking'. Cached per (model, base_url).
    """
    if not base_url or not model:
        return []
    key = (model, base_url)
    cached = _OLLAMA_CAPS_CACHE.get(key)
    if cached is not None:
        caps, ts = cached
        if caps or (_time.monotonic() - ts) < _OLLAMA_CAPS_TTL:
            return caps

    caps: list[str] = []
    try:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        req = urllib.request.Request(
            root + "/api/show",
            data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        caps = list(data.get("capabilities") or [])
    except Exception:
        caps = []

    _OLLAMA_CAPS_CACHE[key] = (caps, _time.monotonic())
    return caps


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled  → extra_body.think = False (Ollama's thinking-off flag)
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # We deliberately do NOT emit think=True on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        #
        # For Ollama specifically, gate on the model's declared capabilities:
        # non-thinking models (e.g. llama3.3:70b — ["completion","tools"])
        # 400 with "does not support thinking" if think/reasoning_effort is
        # sent at all, so probe /api/show and skip emitting either field when
        # the model doesn't advertise "thinking". Non-Ollama custom backends
        # (vLLM/GLM/llama.cpp) keep the unconditional behavior since they
        # don't expose an equivalent capability probe.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            _base_url = ctx.get("base_url")
            _emit = True
            if _is_ollama_base_url(_base_url):
                _emit = "thinking" in _ollama_model_capabilities(ctx.get("model"), _base_url)
            if _emit:
                if _effort == "none" or _enabled is False:
                    extra_body["think"] = False
                elif _effort:
                    top_level["reasoning_effort"] = _effort

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
