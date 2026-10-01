"""Provider-specific local-model reasoning capability probes."""

from __future__ import annotations


def _materialize_api_key(api_key: object) -> str:
    """Best-effort concrete bearer for capability probes."""
    try:
        token = api_key() if callable(api_key) else api_key
    except Exception:
        return ""
    return token.strip() if isinstance(token, str) else ""


def _server_root(base_url: str | None) -> str:
    root = str(base_url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root


def lmstudio_model_reasoning_options(
    model: str,
    base_url: str | None,
    api_key: object = None,
    timeout: float = 5.0,
) -> list[str]:
    """Return LM Studio's declared reasoning options for model."""
    root = _server_root(base_url)
    if not root or not str(model or "").strip():
        return []

    import httpx

    token = _materialize_api_key(api_key)
    headers = {"User-Agent": "HermesAgent/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
            response = client.get(root + "/api/v1/models")
            if response.status_code != 200:
                return []
            payload = response.json()
    except Exception:
        return []

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        return []
    target = str(model or "").strip()
    entry = next(
        (
            item
            for item in raw_models
            if isinstance(item, dict)
            and (item.get("key") == target or item.get("id") == target)
        ),
        None,
    )
    if entry is None:
        return []
    capabilities = entry.get("capabilities")
    reasoning = capabilities.get("reasoning") if isinstance(capabilities, dict) else None
    options = reasoning.get("allowed_options") if isinstance(reasoning, dict) else None
    if not isinstance(options, list):
        return []
    return [
        value
        for option in options
        if isinstance(option, str) and (value := option.strip().lower())
    ]


def _strip_ollama_cloud_suffix(model_id: str) -> str:
    for suffix in (":cloud", "-cloud"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def ollama_model_supports_thinking(
    model: str,
    base_url: str | None,
    api_key: object = None,
    timeout: float = 5.0,
) -> bool | None:
    """Tri-state Ollama thinking capability from the native /api/show API."""
    import httpx

    root = _server_root(base_url)
    bare_model = _strip_ollama_cloud_suffix(str(model or "").strip())
    if not root or not bare_model:
        return None

    token = _materialize_api_key(api_key)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
            response = client.post(root + "/api/show", json={"name": bare_model})
            if response.status_code != 200:
                return None
            capabilities = response.json().get("capabilities")
    except Exception:
        return None
    return "thinking" in capabilities if isinstance(capabilities, list) else None


__all__ = [
    "lmstudio_model_reasoning_options",
    "ollama_model_supports_thinking",
]
