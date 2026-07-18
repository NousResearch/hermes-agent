"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances and OpenAI-compatible reasoning endpoints (GLM-5.2 on
Volcengine ARK, vLLM, llama.cpp). Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled + local endpoint → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False for /api/chat and proxies
    Remote endpoints get neither (they reject "none" as invalid).
  - reasoning_config enabled + effort → top-level reasoning_effort
    (the native OpenAI-compatible format GLM/ARK expect; unset omits it
    so the endpoint's server default applies)
"""

import ipaddress
from typing import Any
from urllib.parse import urlparse

from providers import register_provider
from providers.base import ProviderProfile


def _is_local_endpoint(base_url: str) -> bool:
    """Return True if base_url points to a local/private/LAN endpoint.

    Checks for loopback (v4 + v6), private LAN ranges, and .local domains.
    Avoids false positives from remote URLs that happen to contain substrings
    like 'localhost' (e.g. https://localhost.example.com).
    """
    if not base_url:
        return False
    base_url = base_url.strip()
    if not base_url:
        return False

    # Parse URL properly instead of substring matching
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        base_url = f"http://{base_url}"
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
    except Exception:
        return False

    if not host:
        return False

    host_lower = host.lower()

    # Exact hostname matches for loopback/loopback aliases
    if host_lower in {"localhost", "ip6-localhost", "ip6-loopback", "0.0.0.0", "::", "::1"}:
        return True

    # .local mDNS domains (Bonjour/AVAHI — always local LAN)
    if host_lower.endswith(".local") or host_lower.endswith(".localhost"):
        return True

    # IP address check: loopback or private/LAN range
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private or ip.is_link_local:
            return True
    except ValueError:
        # Not an IP, hostname checked above
        pass

    return False


def _read_reasoning_disable_method(ctx: dict) -> str:
    """Return how to disable reasoning for the current custom provider.

    Reads ``custom_providers[<name>].reasoning_disable`` from config.
    Returns one of ``"auto"`` (default), ``"none"``, or ``"omit"``.
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        providers = cfg.get("custom_providers") or []
        # Find the matching provider by base_url
        for p in providers:
            if not isinstance(p, dict):
                continue
            p_url = (p.get("base_url") or "").strip()
            ctx_url = (ctx.get("base_url") or "").strip()
            if p_url and ctx_url and p_url.rstrip("/") == ctx_url.rstrip("/"):
                val = (p.get("reasoning_disable") or "").strip().lower()
                if val in {"none", "omit"}:
                    return val
                return "auto"
    except Exception:
        pass
    return "auto"


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

        # Resolve base URL for local detection
        _base_url = str(
            ctx.get("base_url")
            or getattr(self, "base_url", "")
            or ""
        ).strip()
        # If no base_url is provided (only happens in legacy tests, never production),
        # default to local behavior to preserve backward compatibility
        _is_local = True if not _base_url else _is_local_endpoint(_base_url)

        # Reasoning / thinking control for custom OpenAI-compatible endpoints
        # (GLM-5.2 on Volcengine ARK, vLLM, Ollama, llama.cpp, …).
        #
        #   - disabled + LOCAL endpoint → extra_body.think = False + reasoning_effort="none"
        #     (Ollama's thinking-off flag + top-level field that Ollama /v1/chat/completions
        #     actually respects; ollama#14820, #25758)
        #   - disabled + REMOTE endpoint → emit NEITHER. Remote OpenAI-compatible APIs
        #     (ofox, Volcengine ARK, etc.) reject reasoning_effort="none" as invalid, so we
        #     omit the field entirely and let the server default apply.
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     format GLM-5.2/ARK and other OpenAI-compatible reasoning APIs
        #     expect (GLM documents "high" and "max"; "max" is its default).
        #   - enabled + no effort  → omit both, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        #
        # We deliberately do NOT emit ``think=True`` on enable: it is an
        # Ollama-only flag and thinking is already server-default-on for these
        # backends, so forcing it risks a 400 on GLM/vLLM endpoints that don't
        # recognize it. Mirrors the DeepSeek/Zai profile precedent.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                # How to signal reasoning-off to the endpoint.
                #
                #   "auto" (default) → locality heuristic: send to local,
                #     omit for remote (Ollama needs it; ofox/ARK reject it)
                #   "none" → always send reasoning_effort="none" + think=False
                #     (for remote endpoints that accept it)
                #   "omit" → never send the disable fields
                #     (for endpoints that reject "none", let server default apply)
                #
                # Read from the per-provider config key ``reasoning_disable``
                # in ``custom_providers``. Falls back to "auto".
                _method = _read_reasoning_disable_method(ctx)
                if _method == "none":
                    top_level["reasoning_effort"] = "none"
                    extra_body["think"] = False
                elif _method == "auto":
                    if _is_local:
                        top_level["reasoning_effort"] = "none"
                        extra_body["think"] = False
                # "omit": send neither — server default applies
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
