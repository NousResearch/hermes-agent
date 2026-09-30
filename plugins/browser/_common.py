"""Shared base for the bundled cloud-browser plugins. Every vendor speaks the same REST shape
(POST to create a session, one request to release it); :class:`CloudBrowserProvider` owns that
lifecycle, subclasses supply attributes + hooks. Logs go to the subclass module's logger with the
vendor label so emitted text matches the pre-refactor per-vendor modules."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import requests

from agent.browser_provider import BrowserProvider

_CLOSE_OK = {200, 201, 204}


class CloudBrowserProvider(BrowserProvider):
    """Subclasses set ``provider_id``/``label``, ``release_method``/``release_path`` (``{session_id}``
    placeholder appended to ``config["base_url"]``), implement ``_get_config_or_none()`` and
    ``_headers(config)``, and build ``create_session`` on ``_post_create``/``_check_created``."""

    provider_id: str
    label: str
    release_method: str
    release_path: str
    missing_credentials_error: str = ""
    setup_tag: Optional[str] = None  # ``None`` hides the provider from the setup picker
    setup_env_vars: List[Dict[str, str]] = []
    create_label_suffix: str = ""  # "Failed to create <label><suffix> session"; Firecrawl: " browser"
    close_fail_fmt: Optional[str] = None  # Browserbase's close warning historically omits the vendor

    @property
    def name(self) -> str:
        return self.provider_id

    @property
    def display_name(self) -> str:
        return self.label

    @property
    def _log(self) -> logging.Logger:
        return logging.getLogger(type(self).__module__)

    def is_available(self) -> bool:
        return self._get_config_or_none() is not None

    def _get_config_or_none(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def _get_config(self) -> Dict[str, Any]:
        config = self._get_config_or_none()
        if config is None:
            raise ValueError(self.missing_credentials_error)
        return config

    def _headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        raise NotImplementedError

    def _release_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        return self._headers(config)

    def _release_body(self, config: Dict[str, Any]) -> Optional[Dict[str, object]]:
        return None

    def _release(self, config: Dict[str, Any], session_id: str, timeout: int) -> requests.Response:
        kwargs: Dict[str, Any] = {"headers": self._release_headers(config), "timeout": timeout}
        body = self._release_body(config)
        if body is not None:
            kwargs["json"] = body
        url = f"{config['base_url']}{self.release_path.format(session_id=session_id)}"
        return getattr(requests, self.release_method)(url, **kwargs)

    @staticmethod
    def _session_name(task_id: str) -> str:
        return f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"

    def _post_create(
        self, url: str, headers: Dict[str, str], payload: Dict[str, object], *, wrap_errors: bool = True
    ) -> requests.Response:
        """POST the create request; network failures → RuntimeError unless the managed gateway
        caller needs the raw exception to retry."""
        try:
            return requests.post(url, headers=headers, json=payload, timeout=30)
        except requests.RequestException as exc:
            if not wrap_errors:
                raise
            raise RuntimeError(f"{self.label} API connection failed: {exc}") from exc

    def _check_created(self, response: requests.Response) -> None:
        if not response.ok:
            raise RuntimeError(
                f"Failed to create {self.label}{self.create_label_suffix} session: "
                f"{response.status_code} {response.text}")

    def close_session(self, session_id: str) -> bool:
        try:
            config = self._get_config()
        except ValueError:
            self._log.warning("Cannot close %s session %s — missing credentials", self.label, session_id)
            return False
        try:
            response = self._release(config, session_id, timeout=10)
            if response.status_code in _CLOSE_OK:
                self._log.debug("Successfully closed %s session %s", self.label, session_id)
                return True
            self._log.warning(
                self.close_fail_fmt or f"Failed to close {self.label} session %s: HTTP %s - %s",
                session_id, response.status_code, response.text[:200])
            return False
        except Exception as e:
            self._log.error("Exception closing %s session %s: %s", self.label, session_id, e)
            return False

    def emergency_cleanup(self, session_id: str) -> None:
        config = self._get_config_or_none()
        if config is None:
            self._log.warning(
                "Cannot emergency-cleanup %s session %s — missing credentials", self.label, session_id
            )
            return
        try:
            self._release(config, session_id, timeout=5)
        except Exception as e:
            self._log.debug("Emergency cleanup failed for %s session %s: %s", self.label, session_id, e)

    def get_setup_schema(self) -> Optional[Dict[str, Any]]:
        if self.setup_tag is None:
            return None
        return {
            "name": self.label,
            "badge": "paid",
            "tag": self.setup_tag,
            "env_vars": [dict(v) for v in self.setup_env_vars],
            # Cloud-scoped hook: installs the agent-browser CLI only (the vendor hosts Chromium).
            "post_setup": "browserbase",
        }
