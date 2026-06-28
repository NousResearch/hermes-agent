"""Kernel cloud browser provider — plugin form.

Subclasses :class:`agent.browser_provider.BrowserProvider` (the plugin-facing
ABC introduced in PR #25214). Wraps Kernel (https://kernel.sh) sandboxed Chrome
browsers via the Kernel REST API.

Talks to the REST API with the core ``requests`` dependency rather than a
vendor SDK, matching the browserbase / firecrawl providers and the repo's
minimal-dependency policy — each lifecycle op is a single REST call.

Config keys this provider responds to::

    browser:
      cloud_provider: "kernel"

Auth env vars::

    KERNEL_API_KEY=...            # https://kernel.sh

Optional feature knobs::

    KERNEL_STEALTH=true           # default true  -> stealth mode (anti-bot)
    KERNEL_HEADLESS=true          # default true  -> headless (no live view)
    KERNEL_TIMEOUT_SECONDS=...    # inactivity timeout, seconds
    KERNEL_PROXY_ID=...           # attach a Kernel proxy by ID
    KERNEL_PROFILE=...            # profile name to load + persist across runs
    KERNEL_BASE_URL=...           # default https://api.onkernel.com
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

import requests

from agent.browser_provider import BrowserProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.onkernel.com"


class KernelBrowserProvider(BrowserProvider):
    """Kernel (https://kernel.sh) cloud browser backend.

    Stealth is on by default; headless is on by default (flip
    ``KERNEL_HEADLESS=false`` to surface a human-watchable ``live_view_url`` in
    the session metadata). Proxy and profile persistence are opt-in via env.
    """

    @property
    def name(self) -> str:
        return "kernel"

    @property
    def display_name(self) -> str:
        return "Kernel"

    def is_available(self) -> bool:
        return self._get_config_or_none() is not None

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def _get_config_or_none(self) -> Optional[Dict[str, Any]]:
        api_key = os.environ.get("KERNEL_API_KEY")
        if not api_key:
            return None
        return {
            "api_key": api_key,
            "base_url": os.environ.get("KERNEL_BASE_URL", _DEFAULT_BASE_URL).rstrip("/"),
        }

    def _get_config(self) -> Dict[str, Any]:
        config = self._get_config_or_none()
        if config is None:
            raise ValueError(
                "Kernel requires the KERNEL_API_KEY environment variable. "
                "Get your key at https://kernel.sh"
            )
        return config

    def _headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, task_id: str) -> Dict[str, object]:
        config = self._get_config()

        stealth = os.environ.get("KERNEL_STEALTH", "true").lower() != "false"
        headless = os.environ.get("KERNEL_HEADLESS", "true").lower() != "false"
        proxy_id = os.environ.get("KERNEL_PROXY_ID")
        profile_name = os.environ.get("KERNEL_PROFILE")

        session_name = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"

        body: Dict[str, object] = {
            "stealth": stealth,
            "headless": headless,
            "name": session_name,
        }

        timeout_seconds = os.environ.get("KERNEL_TIMEOUT_SECONDS")
        if timeout_seconds:
            try:
                body["timeout_seconds"] = int(timeout_seconds)
            except ValueError:
                logger.warning(
                    "Invalid KERNEL_TIMEOUT_SECONDS value: %s", timeout_seconds
                )

        if proxy_id:
            body["proxy_id"] = proxy_id

        if profile_name:
            body["profile"] = {"name": profile_name, "save_changes": True}

        try:
            response = requests.post(
                f"{config['base_url']}/browsers",
                headers=self._headers(config),
                json=body,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Kernel API connection failed: {exc}") from exc

        if not response.ok:
            raise RuntimeError(
                f"Failed to create Kernel session: "
                f"{response.status_code} {response.text}"
            )

        data = response.json()
        live_view_url = data.get("browser_live_view_url")

        features: Dict[str, object] = {
            "stealth": stealth,
            "headless": headless,
            "custom_timeout": "timeout_seconds" in body,
            "proxies": bool(proxy_id),
            "profile_persistence": profile_name is not None,
            "live_view": bool(live_view_url),
        }

        active = ", ".join(k for k, v in features.items() if v)
        logger.info("Created Kernel session %s with features: %s", session_name, active)

        result: Dict[str, object] = {
            "session_name": session_name,
            "bb_session_id": data["session_id"],
            "cdp_url": data["cdp_ws_url"],
            "features": features,
        }
        if live_view_url:
            result["live_view_url"] = live_view_url
        return result

    def close_session(self, session_id: str) -> bool:
        try:
            config = self._get_config()
        except ValueError:
            logger.warning(
                "Cannot close Kernel session %s — missing credentials", session_id
            )
            return False

        try:
            response = requests.delete(
                f"{config['base_url']}/browsers/{session_id}",
                headers=self._headers(config),
                timeout=10,
            )
            if response.status_code in {200, 201, 204}:
                logger.debug("Successfully closed Kernel session %s", session_id)
                return True
            logger.warning(
                "Failed to close Kernel session %s: HTTP %s - %s",
                session_id,
                response.status_code,
                response.text[:200],
            )
            return False
        except Exception as e:
            logger.error("Exception closing Kernel session %s: %s", session_id, e)
            return False

    def emergency_cleanup(self, session_id: str) -> None:
        config = self._get_config_or_none()
        if config is None:
            logger.warning(
                "Cannot emergency-cleanup Kernel session %s — missing credentials",
                session_id,
            )
            return
        try:
            requests.delete(
                f"{config['base_url']}/browsers/{session_id}",
                headers=self._headers(config),
                timeout=5,
            )
        except Exception as e:
            logger.debug(
                "Emergency cleanup failed for Kernel session %s: %s", session_id, e
            )

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Kernel",
            "badge": "paid",
            "tag": "Cloud browser with stealth, proxies, and live view",
            "env_vars": [
                {
                    "key": "KERNEL_API_KEY",
                    "prompt": "Kernel API key",
                    "url": "https://kernel.sh",
                },
            ],
            "post_setup": "agent_browser",
        }
