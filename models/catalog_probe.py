"""Runtime model-catalog probes owned by the model domain."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_single_openai_model(base_url: str) -> str:
    """Return the sole model exposed by an OpenAI-style /v1/models endpoint."""
    if not base_url:
        return ""
    try:
        import requests

        url = base_url.rstrip("/")
        response = requests.get(
            (url if url.endswith("/v1") else url + "/v1") + "/models",
            timeout=(2, 3),
        )
        if response.ok:
            models = response.json().get("data", [])
            if len(models) == 1 and models[0].get("id", ""):
                return str(models[0]["id"])
    except Exception as exc:
        logger.debug("Auto-detect model from %s failed: %s", base_url, exc)
    return ""


__all__ = ["detect_single_openai_model"]
