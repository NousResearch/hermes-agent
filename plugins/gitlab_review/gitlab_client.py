"""GitLab REST API v4 client for the gitlab-review plugin.

Provides a thin, synchronous wrapper around GitLab's REST API using ``httpx``.
All tool handlers delegate to this client for HTTP communication.

Configuration via environment variables:
  GITLAB_TOKEN   — Personal access token (required). Needs api + read_api scope.
  GITLAB_URL     — Base URL for self-hosted GitLab (default: https://gitlab.com).

Self-hosted GitLab:
  Set GITLAB_URL to your instance root (e.g. ``https://gitlab.mycompany.com``).
  The client appends ``/api/v4`` automatically.

Project identifiers:
  GitLab API accepts either a numeric project ID or a URL-encoded path
  (``group%2Fproject``).  The ``encode_project`` helper handles encoding.
"""

from __future__ import annotations

import httpx
import json
import logging
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_GITLAB_URL = "https://gitlab.com"
_REQUEST_TIMEOUT = 30
_MAX_PAGES = 10
_PER_PAGE = 100


def get_config() -> Tuple[str, str]:
    """Return ``(gitlab_url, gitlab_token)`` from environment.

    ``gitlab_url`` has trailing slashes stripped and ``/api/v4`` is NOT
    included — callers build full URLs via :func:`api_url`.
    """
    gitlab_url = os.getenv("GITLAB_URL", _DEFAULT_GITLAB_URL).rstrip("/")
    gitlab_token = os.getenv("GITLAB_TOKEN", "").strip()
    return gitlab_url, gitlab_token


def is_available() -> bool:
    """Return True when GITLAB_TOKEN is set (plugin should be active)."""
    return bool(os.getenv("GITLAB_TOKEN", "").strip())


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def api_url(base_url: str, path: str) -> str:
    """Build a full GitLab API v4 URL.

    ``path`` should start with ``/`` and include the encoded project ID,
    e.g. ``/projects/group%2Frepo/merge_requests/1``.
    """
    return f"{base_url}/api/v4{path}"


def encode_project(project: str) -> str:
    """URL-encode a project path for the GitLab API.

    GitLab requires ``group/project`` paths to be percent-encoded as
    ``group%2Fproject`` in API URLs.  Numeric project IDs are returned
    unchanged.
    """
    # If it looks like a numeric ID, return as-is
    if project.isdigit():
        return project
    return urllib.parse.quote(project, safe="")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# Simple rate limiter: track the last 429 response and back off.
_last_rate_limit_reset: float = 0.0


def _check_rate_limit() -> None:
    """Sleep if we're inside a rate-limit cooldown window."""
    if _last_rate_limit_reset and time.monotonic() < _last_rate_limit_reset:
        remaining = _last_rate_limit_reset - time.monotonic()
        logger.debug("Rate limited — sleeping %.1fs", remaining)
        time.sleep(remaining)


def _handle_rate_limit_response(response) -> None:
    """Update rate limiter state from a 429 response."""
    global _last_rate_limit_reset
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", "60"))
        _last_rate_limit_reset = time.monotonic() + retry_after
        logger.warning("GitLab API rate limited — Retry-After: %ds", retry_after)


# ---------------------------------------------------------------------------
# HTTP request helpers
# ---------------------------------------------------------------------------

def _gitlab_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """Make a GitLab REST API v4 request and return parsed JSON.

    Raises :class:`GitLabAPIError` on non-2xx responses (except 429 which
    triggers a retry after cooldown).

    Args:
        method: HTTP method (GET, POST, PUT, DELETE).
        path: API path starting with ``/`` (after ``/api/v4``).
        params: Query parameters.
        json_body: JSON request body for POST/PUT.
        timeout: Request timeout in seconds.
    """
    import httpx

    base_url, token = get_config()
    if not token:
        raise GitLabAPIError("GITLAB_TOKEN not set")

    _check_rate_limit()

    url = api_url(base_url, path)
    headers = {
        "PRIVATE-TOKEN": token,
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_body,
            )

        _handle_rate_limit_response(response)

        if response.status_code == 429:
            # Retry once after cooldown
            _check_rate_limit()
            with httpx.Client(timeout=timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_body,
                )

        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"message": response.text}
            raise GitLabAPIError(
                f"GitLab API {response.status_code}: {error_body.get('message', response.text)}",
                status_code=response.status_code,
                body=error_body,
            )

        # Some endpoints return 204 No Content
        if response.status_code == 204:
            return {"status": "success"}

        return response.json()

    except httpx.TimeoutException:
        raise GitLabAPIError(f"Request timed out after {timeout}s")
    except httpx.ConnectError as e:
        raise GitLabAPIError(f"Connection failed: {e}")


def gitlab_get(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """Convenience wrapper for GET requests."""
    return _gitlab_request("GET", path, params=params, timeout=timeout)


def gitlab_post(
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """Convenience wrapper for POST requests."""
    return _gitlab_request("POST", path, params=params, json_body=json_body, timeout=timeout)


def gitlab_put(
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """Convenience wrapper for PUT requests."""
    return _gitlab_request("PUT", path, params=params, json_body=json_body, timeout=timeout)


def gitlab_delete(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = _REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """Convenience wrapper for DELETE requests."""
    return _gitlab_request("DELETE", path, params=params, timeout=timeout)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def gitlab_get_paginated(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    max_pages: int = _MAX_PAGES,
    per_page: int = _PER_PAGE,
    timeout: int = _REQUEST_TIMEOUT,
) -> List[Dict[str, Any]]:
    """Follow GitLab Link-header pagination and collect all pages.

    Returns a flat list of items from all pages.  Stops at ``max_pages``
    to prevent unbounded reads.
    """
    base_url, token = get_config()
    if not token:
        raise GitLabAPIError("GITLAB_TOKEN not set")

    _check_rate_limit()

    all_items: List[Dict[str, Any]] = []
    url = api_url(base_url, path)
    headers = {
        "PRIVATE-TOKEN": token,
        "Content-Type": "application/json",
    }
    query_params = dict(params or {})
    query_params["per_page"] = per_page

    with httpx.Client(timeout=timeout) as client:
        for page in range(1, max_pages + 1):
            query_params["page"] = page
            response = client.get(url, headers=headers, params=query_params)

            _handle_rate_limit_response(response)

            if response.status_code >= 400:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = {"message": response.text}
                raise GitLabAPIError(
                    f"GitLab API {response.status_code}: {error_body.get('message', response.text)}",
                    status_code=response.status_code,
                    body=error_body,
                )

            items = response.json()
            if not items:
                break
            all_items.extend(items)

            # Check Link header for next page
            link_header = response.headers.get("Link", "")
            if 'rel="next"' not in link_header:
                break

    return all_items


# ---------------------------------------------------------------------------
# Project path helper
# ---------------------------------------------------------------------------

def project_path(project: str) -> str:
    """Build the ``/projects/:id`` path segment with proper encoding."""
    return f"/projects/{encode_project(project)}"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GitLabAPIError(Exception):
    """Raised when a GitLab API call fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
