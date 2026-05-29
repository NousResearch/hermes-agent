"""Auto-discovery module for local Ollama models in the provider gateway.

Queries the local Ollama instance for installed models and extracts their
optimal parameters (like native context length) dynamically.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class OllamaDiscovery:
    """Discovers installed Ollama models and their optimal parameters."""

    def __init__(self, host: str | None = None) -> None:
        if host is not None:
            self.host = host
        else:
            # Read Ollama standard env variable or fallback to default localhost port
            self.host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
        
        # Ensure scheme is present
        if not self.host.startswith("http://") and not self.host.startswith("https://"):
            self.host = "http://" + self.host

    def discover_local_models(self) -> list[dict[str, Any]]:
        """Query the Ollama daemon for active models.

        Returns a list of model info dictionaries:
        [
            {"model": "llama3:8b", "provider": "ollama", "num_ctx": 8192, "base_url": "..."},
            ...
        ]
        """
        api_url = f"{self.host}/api/tags"
        logger.debug("Ollama Discovery: querying %s", api_url)

        try:
            req = urllib.request.Request(api_url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status != 200:
                    return []
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.debug("Ollama Discovery: daemon not running or unreachable: %s", exc)
            return []

        raw_models = data.get("models", [])
        if not raw_models:
            return []

        discovered = []
        for item in raw_models:
            name = item.get("name")
            if not name:
                continue

            # Query context length for this model dynamically
            num_ctx = self._fetch_model_context_length(name)

            discovered.append({
                "model": name,
                "provider": "ollama",
                "num_ctx": num_ctx,
                "base_url": f"{self.host}/v1",
            })
            logger.info("Ollama Discovery: found model %s (context length: %d)", name, num_ctx)

        return discovered

    def _fetch_model_context_length(self, model_name: str) -> int:
        """Fetch custom properties from /api/show to resolve optimal context length."""
        api_url = f"{self.host}/api/show"
        payload = json.dumps({"name": model_name}).encode("utf-8")

        try:
            req = urllib.request.Request(
                api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status != 200:
                    return 8192  # Default conservative fallback
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return 8192

        # Extract context length from system parameter or model info
        # Ollama returns system parameter inside "parameters" string (e.g., "num_ctx        8192")
        parameters = data.get("parameters", "")
        if parameters:
            for line in parameters.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == "num_ctx":
                    try:
                        return int(parts[1])
                    except (TypeError, ValueError):
                        pass

        # Check model_info dictionary keys (e.g., "llama.context_length")
        model_info = data.get("model_info", {})
        if isinstance(model_info, dict):
            for key, val in model_info.items():
                if "context_length" in key:
                    try:
                        return int(val)
                    except (TypeError, ValueError):
                        pass

        return 8192  # Default fallback
