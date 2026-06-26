"""Camofox web extract — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses a
self-hosted Camofox browser server (https://github.com/jo-inc/camofox-browser)
to fetch and extract page content.

Config keys this provider responds to::

    web:
      extract_backend: "camofox"      # explicit per-capability
      backend: "camofox"              # shared fallback

Env vars::

    CAMOFOX_URL=http://localhost:9377
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import requests

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _camofox_url() -> str:
    """Return CAMOFOX_URL from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value("CAMOFOX_URL")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("CAMOFOX_URL", "")
    return (val or "http://localhost:9377").strip().rstrip("/")


class CamofoxWebExtractProvider(WebSearchProvider):
    """Extract page content via a self-hosted Camofox browser server."""

    @property
    def name(self) -> str:
        return "camofox"

    @property
    def display_name(self) -> str:
        return "Camofox (self-hosted)"
    def is_available(self) -> bool:
        """Return True when CAMOFOX_URL is set."""
        return bool(_camofox_url() and os.getenv("CAMOFOX_URL", "").strip())

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Camofox.

        Creates a tab per URL, navigates, snapshots the accessibility tree,
        and returns markdown-ish text.
        """
        base_url = _camofox_url()
        if not base_url:
            return [{"url": u, "title": "", "content": "", "error": "CAMOFOX_URL is not set"} for u in urls]

        results: List[Dict[str, Any]] = []
        for url in urls:
            try:
                # Create tab
                create_resp = requests.post(
                    f"{base_url}/tabs",
                    json={"userId": "hermes", "sessionKey": "web-extract", "url": url},
                    timeout=15,
                )
                create_resp.raise_for_status()
                tab = create_resp.json()
                tab_id = tab.get("tabId")
                if not tab_id:
                    results.append({"url": url, "title": "", "content": "", "error": "Camofox did not return a tabId"})
                    continue

                # Navigate
                nav_resp = requests.post(
                    f"{base_url}/tabs/{tab_id}/navigate",
                    json={"userId": "hermes", "url": url},
                    timeout=30,
                )
                nav_resp.raise_for_status()

                # Snapshot
                snap_resp = requests.get(
                    f"{base_url}/tabs/{tab_id}/snapshot?userId=hermes",
                    timeout=15,
                )
                snap_resp.raise_for_status()
                snap = snap_resp.json()

                title = snap.get("title", "")
                text = snap.get("text", "") or snap.get("snapshot", "")

                # Close tab
                try:
                    requests.delete(f"{base_url}/tabs/{tab_id}?userId=hermes", timeout=5)
                except Exception:  # noqa: BLE001
                    pass

                results.append({
                    "url": url,
                    "title": title,
                    "content": text,
                    "raw_content": text,
                    "metadata": {"source": "camofox"},
                })
            except requests.RequestException as exc:
                logger.warning("Camofox extract failed for %s: %s", url, exc)
                results.append({"url": url, "title": "", "content": "", "error": f"Camofox request failed: {exc}"})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Camofox extract error for %s: %s", url, exc)
                results.append({"url": url, "title": "", "content": "", "error": str(exc)})

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Camofox (self-hosted)",
            "badge": "free · self-hosted",
            "tag": "Local anti-detect browser extraction. Point CAMOFOX_URL at your instance.",
            "env_vars": [
                {
                    "key": "CAMOFOX_URL",
                    "prompt": "Camofox server URL (e.g. http://localhost:9377)",
                    "url": "https://github.com/jo-inc/camofox-browser",
                },
            ],
        }
