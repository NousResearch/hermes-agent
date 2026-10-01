"""Pure local-catalog route classification.

Runtime/application layers supply configured endpoint facts and perform probes.
This module decides whether the catalog is definitely Ollama-native, definitely
OpenAI-style, or requires an Ollama-native reachability probe.
"""

from __future__ import annotations

from typing import Literal, Optional
from urllib.parse import urlparse, urlsplit

CatalogDecision = Literal["native", "openai", "probe"]

_OLLAMA_DEFAULT_PORT = 11434
_NEVER_OLLAMA_PROVIDERS = frozenset({
    "openrouter", "nous", "anthropic", "openai", "openai-codex",
    "gemini", "ollama-cloud",
})
_LOCAL_LIKE_PROVIDERS = frozenset({
    "", "custom", "local", "llamacpp", "llama.cpp", "llama-cpp", "vllm",
})


def ollama_native_root(base_url: Optional[str]) -> str:
    root = str(base_url or "").strip().rstrip("/")
    if root.startswith(":"):
        root = "http://127.0.0.1" + root
    elif root and "://" not in root:
        root = "http://" + root
    for suffix in ("/api/tags", "/v1/models", "/api", "/v1"):
        if root.endswith(suffix):
            return root[: -len(suffix)].rstrip("/")
    return root


def _url_origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    port = parsed.port
    defaults = {"http": 80, "https": 443}
    return (
        scheme,
        (parsed.hostname or "").lower().rstrip("."),
        port if port is not None else defaults.get(scheme),
    )


def same_ollama_native_root(left: str, right: str) -> bool:
    left_root = ollama_native_root(left).rstrip("/")
    right_root = ollama_native_root(right).rstrip("/")
    if not left_root or not right_root:
        return False
    try:
        left_parts = urlsplit(left_root)
        right_parts = urlsplit(right_root)
        return (
            _url_origin(left_root) == _url_origin(right_root)
            and left_parts.path.rstrip("/") == right_parts.path.rstrip("/")
        )
    except (AttributeError, ValueError):
        return False


def classify_ollama_catalog(
    provider: Optional[str],
    base_url: Optional[str],
    *,
    configured_base_url: Optional[str] = None,
) -> CatalogDecision:
    """Classify catalog routing without config access or network I/O."""
    requested = str(provider or "").strip().lower()
    root = ollama_native_root(base_url)

    if root:
        try:
            host = (urlparse(root).hostname or "").lower()
            if host == "ollama.com" or host.endswith(".ollama.com"):
                return "openai"
        except ValueError:
            pass

    if requested in _NEVER_OLLAMA_PROVIDERS:
        return "openai"

    configured = str(configured_base_url or "").strip()
    if requested == "ollama":
        if not root:
            return "openai"
        if configured and not same_ollama_native_root(root, configured):
            return "probe"
        return "native"

    if configured and same_ollama_native_root(root, configured):
        return "native"

    if not root:
        return "openai"

    if requested not in _LOCAL_LIKE_PROVIDERS and not requested.startswith("custom:"):
        return "openai"

    if requested == "custom:ollama" or requested.endswith("-ollama"):
        return "native"

    try:
        if urlparse(root).port != _OLLAMA_DEFAULT_PORT:
            return "openai"
    except ValueError:
        return "openai"

    return "probe"


__all__ = [
    "CatalogDecision",
    "classify_ollama_catalog",
    "ollama_native_root",
    "same_ollama_native_root",
]