"""Local ``sxng-search`` command wrapper — bundled web provider plugin.

The provider is intentionally search-only. Configure the executable and timeout
in ``config.yaml``::

    web:
      search_backend: sxng
      sxng:
        command: sxng-search
        timeout: 45

No credential or behavioral setting is stored in ``.env``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

_DEFAULT_COMMAND = "sxng-search"
_DEFAULT_TIMEOUT = 45
_MAX_TIMEOUT = 300


def _sxng_config() -> Dict[str, Any]:
    """Return the ``web.sxng`` mapping from the merged Hermes config."""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:  # noqa: BLE001 — config remains optional for plugin import
        config = {}
    web = config.get("web", {}) if isinstance(config, dict) else {}
    if not isinstance(web, dict):
        return {}
    sxng = web.get("sxng", {})
    return sxng if isinstance(sxng, dict) else {}


def _command_name() -> str:
    value = _sxng_config().get("command", _DEFAULT_COMMAND)
    command = str(value or "").strip()
    return command or _DEFAULT_COMMAND


def _resolve_command() -> str:
    """Resolve the configured executable without invoking a shell."""
    return shutil.which(_command_name()) or ""


def _timeout_seconds() -> int:
    raw = _sxng_config().get("timeout", _DEFAULT_TIMEOUT)
    if raw is None:
        return _DEFAULT_TIMEOUT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_TIMEOUT
    return min(max(value, 1), _MAX_TIMEOUT)


def _candidate_results(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("results"), list):
        return payload["results"]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("web"), list):
        return data["web"]
    if isinstance(payload.get("web"), list):
        return payload["web"]
    return []


def _normalize_results(payload: Any, limit: int) -> List[Dict[str, Any]]:
    web_results: List[Dict[str, Any]] = []
    for item in _candidate_results(payload):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        if not title and not url:
            continue
        description = str(
            item.get("description")
            or item.get("content")
            or item.get("snippet")
            or item.get("summary")
            or ""
        ).strip()
        web_results.append(
            {
                "title": title,
                "url": url,
                "description": description,
                "position": len(web_results) + 1,
            }
        )
        if len(web_results) >= limit:
            break
    return web_results


class SxngWebSearchProvider(WebSearchProvider):
    """Search provider backed by a local ``sxng-search`` executable."""

    @property
    def name(self) -> str:
        return "sxng"

    @property
    def display_name(self) -> str:
        return "Local sxng-search wrapper"

    def is_available(self) -> bool:
        return bool(_resolve_command())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        command = _resolve_command()
        if not command:
            return {
                "success": False,
                "error": (
                    "sxng-search command not found. Install it on PATH or set "
                    "web.sxng.command in config.yaml."
                ),
            }

        try:
            result_limit = min(max(int(limit), 1), 100)
        except (TypeError, ValueError):
            result_limit = 5
        timeout = _timeout_seconds()
        # Put fixed options first and terminate option parsing before the query.
        # Without ``--``, a query such as ``--help`` is interpreted by argparse
        # as a command option rather than search text.
        args = [
            command,
            "--limit",
            str(result_limit),
            "--json",
            "--",
            str(query or ""),
        ]

        try:
            completed = subprocess.run(
                args,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"sxng-search timed out after {timeout} seconds",
            }
        except OSError:
            return {
                "success": False,
                "error": "sxng-search failed to start",
            }

        if completed.returncode != 0:
            return {
                "success": False,
                # Do not expose subprocess output to the model: it may contain
                # private paths, local endpoints, usernames, or credentials.
                "error": f"sxng-search failed with exit code {completed.returncode}",
            }

        try:
            payload = json.loads(completed.stdout or "{}")
        except (TypeError, json.JSONDecodeError):
            return {
                "success": False,
                "error": "Could not parse sxng-search output as JSON",
            }

        return {
            "success": True,
            "data": {"web": _normalize_results(payload, result_limit)},
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "free · local",
            "tag": (
                "Search only via an installed sxng-search command; configure "
                "advanced command/timeout settings under web.sxng."
            ),
            "env_vars": [],
        }
