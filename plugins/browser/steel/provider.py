"""Steel cloud browser provider — plugin form.

Subclasses :class:`agent.browser_provider.BrowserProvider` (the plugin-facing
ABC introduced in PR #25214), mirroring ``plugins/browser/browserbase/`` and
``plugins/browser/firecrawl/``.

Steel exposes a standard CDP websocket per session, so the provider hands the
dispatcher a ``cdp_url`` and the provider-agnostic wrapper in
:mod:`tools.browser_tool` drives it via ``agent-browser --cdp`` — no Steel CLI
required for the browser path. (The standalone ``steel_scrape`` tool is a
separate HTTP-only tool and is unaffected.)

One Steel quirk is normalized here: the session's ``websocketUrl`` comes back
without a path segment (``wss://connect.steel.dev?sessionId=...``), which makes
some CDP clients build a malformed request-line and get ``400 Bad Request`` on
the upgrade. We insert the ``/`` before the query and append the API key so the
URL is ``wss://connect.steel.dev/?sessionId=...&apiKey=...``.

Config keys this provider responds to::

    browser:
      cloud_provider: "steel"

Auth env vars::

    STEEL_API_KEY=...            # https://steel.dev

Optional feature knobs::

    STEEL_BASE_URL=...           # default https://api.steel.dev (self-hosted)
    STEEL_USE_PROXY=true         # default false — residential proxy
    STEEL_SOLVE_CAPTCHA=true     # default false — CAPTCHA solving
    STEEL_SESSION_TIMEOUT=...     # milliseconds, integer
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from agent.browser_provider import BrowserProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.steel.dev"


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes")


def _normalize_cdp_url(ws_url: str, api_key: str) -> str:
    """Return a CDP-connectable websocket URL for *ws_url*.

    Ensures a ``/`` path segment (Steel's ``websocketUrl`` omits it, which
    trips up CDP clients during the WS upgrade) and appends ``apiKey``.
    """
    parts = urlsplit(ws_url)
    path = parts.path or "/"
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["apiKey"] = api_key
    return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), parts.fragment))


class SteelBrowserProvider(BrowserProvider):
    """Steel (https://steel.dev) cloud browser backend.

    Headless browser sessions with optional residential proxy and CAPTCHA
    solving. Explicit-only: activated via ``browser.cloud_provider: steel``;
    never auto-selected (not in the registry's legacy preference walk).
    """

    @property
    def name(self) -> str:
        return "steel"

    @property
    def display_name(self) -> str:
        return "Steel"

    def is_available(self) -> bool:
        return bool(os.environ.get("STEEL_API_KEY"))

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    def _api_key(self) -> str:
        api_key = os.environ.get("STEEL_API_KEY")
        if not api_key:
            raise ValueError(
                "Steel requires the STEEL_API_KEY environment variable. "
                "Get your key at https://steel.dev"
            )
        return api_key

    def _base_url(self) -> str:
        return os.environ.get("STEEL_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")

    def _headers(self, api_key: str) -> Dict[str, str]:
        return {"Content-Type": "application/json", "Steel-Api-Key": api_key}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, task_id: str) -> Dict[str, object]:
        api_key = self._api_key()
        base_url = self._base_url()

        enable_proxy = _truthy(os.environ.get("STEEL_USE_PROXY"))
        enable_captcha = _truthy(os.environ.get("STEEL_SOLVE_CAPTCHA"))

        session_config: Dict[str, object] = {}
        if enable_proxy:
            session_config["useProxy"] = True
        if enable_captcha:
            session_config["solveCaptcha"] = True

        custom_timeout_ms = os.environ.get("STEEL_SESSION_TIMEOUT")
        if custom_timeout_ms:
            try:
                timeout_val = int(custom_timeout_ms)
                if timeout_val > 0:
                    session_config["timeout"] = timeout_val
            except ValueError:
                logger.warning(
                    "Invalid STEEL_SESSION_TIMEOUT value: %s", custom_timeout_ms
                )

        try:
            response = requests.post(
                f"{base_url}/v1/sessions",
                headers=self._headers(api_key),
                json=session_config,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Steel API connection failed: {exc}") from exc

        if not response.ok:
            raise RuntimeError(
                f"Failed to create Steel session: "
                f"{response.status_code} {response.text}"
            )

        session_data = response.json()
        cloud_session_id = session_data.get("id")
        websocket_url = session_data.get("websocketUrl")
        if not cloud_session_id or not websocket_url:
            raise RuntimeError(
                f"Steel session response missing id/websocketUrl: {session_data!r}"
            )

        session_name = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"

        features_enabled = {
            "proxy": enable_proxy,
            "captcha_solving": enable_captcha,
            "custom_timeout": "timeout" in session_config,
        }

        viewer_url = session_data.get("sessionViewerUrl")
        if viewer_url:
            logger.info("Steel session %s — live viewer: %s", session_name, viewer_url)

        feature_str = ", ".join(k for k, v in features_enabled.items() if v) or "none"
        logger.info(
            "Created Steel session %s with features: %s", session_name, feature_str
        )

        result: Dict[str, object] = {
            "session_name": session_name,
            "bb_session_id": cloud_session_id,
            "cdp_url": _normalize_cdp_url(websocket_url, api_key),
            "features": features_enabled,
        }
        if viewer_url:
            result["session_viewer_url"] = viewer_url
        return result

    def close_session(self, session_id: str) -> bool:
        try:
            api_key = self._api_key()
        except ValueError:
            logger.warning(
                "Cannot close Steel session %s — missing STEEL_API_KEY", session_id
            )
            return False

        try:
            response = requests.post(
                f"{self._base_url()}/v1/sessions/{session_id}/release",
                headers=self._headers(api_key),
                timeout=10,
            )
            if response.status_code in {200, 201, 204}:
                logger.debug("Successfully closed Steel session %s", session_id)
                return True
            logger.warning(
                "Failed to close Steel session %s: HTTP %s - %s",
                session_id,
                response.status_code,
                response.text[:200],
            )
            return False
        except Exception as e:
            logger.error("Exception closing Steel session %s: %s", session_id, e)
            return False

    def emergency_cleanup(self, session_id: str) -> None:
        try:
            api_key = self._api_key()
        except ValueError:
            logger.warning(
                "Cannot emergency-cleanup Steel session %s — missing STEEL_API_KEY",
                session_id,
            )
            return
        try:
            requests.post(
                f"{self._base_url()}/v1/sessions/{session_id}/release",
                headers=self._headers(api_key),
                timeout=5,
            )
        except Exception as e:
            logger.debug(
                "Emergency cleanup failed for Steel session %s: %s", session_id, e
            )

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Steel",
            "badge": "paid",
            "tag": "Cloud browser with proxies and CAPTCHA solving",
            "env_vars": [
                {
                    "key": "STEEL_API_KEY",
                    "prompt": "Steel API key",
                    "url": "https://steel.dev",
                },
            ],
            "post_setup": "agent_browser",
        }
