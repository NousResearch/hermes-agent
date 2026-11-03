"""Endpoint-scoped context windows: models whose true window is confirmed for one
provider endpoint only (moved out of model_metadata to keep that module's
code-health line budget; see _endpoint_scoped_context_length)."""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


# (host, canonical paths, model ids, context) — see _endpoint_scoped_context_length.
_ENDPOINT_SCOPED_CONTEXT = (
    ("api.kimi.com", {"/coding", "/coding/v1"}, {"k3", "kimi-k3", "kimi-k3-cot"}, 1_048_576),
    ("integrate.api.nvidia.com", {"/v1"}, {"deepseek-ai/deepseek-v4-pro"}, 262_144),
)


def _endpoint_scoped_context_length(model: str, base_url: str) -> Optional[int]:
    """Context confirmed for one provider endpoint only (see _ENDPOINT_SCOPED_CONTEXT): Kimi Coding
    serves K3 at 1 Mi only on the canonical ``api.kimi.com/coding`` host (legacy Moonshot keys do
    not); NVIDIA NIM serves deepseek-v4-pro at 262,144 while DeepSeek's native endpoint is 1M."""
    try:
        parsed = urlparse(_normalize_base_url(base_url))
        port = parsed.port
    except ValueError:
        return None
    # Only canonical https://host[:443]/path with no credentials/query/fragment.
    if parsed.scheme.lower() != "https" or port not in (None, 443) or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        return None
    host, path, model_key = (parsed.hostname or "").lower(), parsed.path.rstrip("/"), model.strip().lower()
    return next((ctx for scoped_host, paths, models, ctx in _ENDPOINT_SCOPED_CONTEXT if host == scoped_host and path in paths and model_key in models), None)
