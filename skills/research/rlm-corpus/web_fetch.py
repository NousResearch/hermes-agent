"""URL → markdown fetchers.

Small abstraction over a couple of web-to-markdown backends so the RLM
skill can ingest websites alongside local documents. Two concrete impls:

* ``CloudflareFetcher`` — Cloudflare Browser Rendering REST API.
    - ``/markdown`` for single-page sync fetches
    - ``/crawl`` for async site-wide crawls (submit → poll → paginate results)

* ``JinaFetcher`` — ``https://r.jina.ai/<url>`` Reader endpoint. No auth,
    lower fidelity, no crawl. Useful as an unauthenticated fallback or for
    quick prototyping before provisioning CF credentials.

Both implement the same ``URLFetcher`` protocol so the ingestion pipeline
and kernel helpers are backend-agnostic.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("rlm_corpus.web_fetch")


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


@dataclass
class FetchedPage:
    url: str
    markdown: str
    title: str | None = None
    status: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class URLFetcher(Protocol):
    name: str

    def fetch_markdown(self, url: str, *, timeout: int = 60) -> FetchedPage:
        ...

    def crawl(
        self,
        start_url: str,
        *,
        max_depth: int = 2,
        limit: int = 50,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        timeout: int = 600,
    ) -> list[FetchedPage]:
        ...


# ---------------------------------------------------------------------------
# Cloudflare Browser Rendering
# ---------------------------------------------------------------------------


class CloudflareFetcher:
    """Cloudflare Browser Rendering REST API client.

    Requires a Cloudflare account ID and an API token with the
    "Browser Rendering - Edit" scope.

    Env fallbacks (used when args are None):
      * ``CLOUDFLARE_ACCOUNT_ID`` or ``RLM_CF_ACCOUNT_ID``
      * ``CLOUDFLARE_API_TOKEN``  or ``RLM_CF_API_TOKEN``
    """

    name = "cloudflare"
    BASE = "https://api.cloudflare.com/client/v4"

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        *,
        poll_interval: float = 2.0,
    ) -> None:
        self.account_id = (
            account_id
            or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            or os.environ.get("RLM_CF_ACCOUNT_ID")
        )
        self.api_token = (
            api_token
            or os.environ.get("CLOUDFLARE_API_TOKEN")
            or os.environ.get("RLM_CF_API_TOKEN")
        )
        if not self.account_id or not self.api_token:
            raise RuntimeError(
                "CloudflareFetcher requires CLOUDFLARE_ACCOUNT_ID and "
                "CLOUDFLARE_API_TOKEN (or their RLM_CF_* equivalents) to be set"
            )
        self.poll_interval = poll_interval

    # ---- internal helpers ---------------------------------------------------

    def _endpoint(self, path: str) -> str:
        return f"{self.BASE}/accounts/{self.account_id}/browser-rendering/{path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
        import requests

        resp = requests.post(
            self._endpoint(path),
            headers=self._headers(),
            json=body,
            timeout=timeout,
        )
        return self._unwrap(resp)

    def _get(self, path: str, timeout: int, params: dict[str, Any] | None = None) -> dict[str, Any]:
        import requests

        resp = requests.get(
            self._endpoint(path),
            headers=self._headers(),
            params=params or {},
            timeout=timeout,
        )
        return self._unwrap(resp)

    @staticmethod
    def _unwrap(resp: Any) -> dict[str, Any]:
        try:
            body = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Cloudflare API returned non-JSON (status {resp.status_code}): "
                f"{resp.text[:500]}"
            ) from exc
        if not body.get("success", False):
            errors = body.get("errors") or []
            raise RuntimeError(f"Cloudflare API error: {errors or body}")
        return body

    # ---- /markdown ----------------------------------------------------------

    def fetch_markdown(self, url: str, *, timeout: int = 60) -> FetchedPage:
        body = self._post("markdown", {"url": url}, timeout=timeout)
        result = body.get("result")
        # The API currently returns result as a markdown string, but older
        # changelogs suggest it may become an object. Accept both.
        if isinstance(result, dict):
            markdown = result.get("markdown") or result.get("content") or ""
            title = result.get("title")
        else:
            markdown = result or ""
            title = None
        return FetchedPage(url=url, markdown=markdown, title=title)

    # ---- /crawl -------------------------------------------------------------

    def crawl(
        self,
        start_url: str,
        *,
        max_depth: int = 2,
        limit: int = 50,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        timeout: int = 600,
    ) -> list[FetchedPage]:
        payload: dict[str, Any] = {
            "url": start_url,
            "maxDepth": max_depth,
            "limit": limit,
            "formats": ["markdown"],
            "render": True,
        }
        if include_patterns:
            payload["includePatterns"] = include_patterns
        if exclude_patterns:
            payload["excludePatterns"] = exclude_patterns

        submit = self._post("crawl", payload, timeout=60)
        result = submit.get("result") or {}
        job_id = result.get("id") or result.get("jobId")
        if not job_id:
            raise RuntimeError(f"crawl submit returned no job id: {submit}")

        log.info("cloudflare crawl job %s submitted for %s", job_id, start_url)
        return self._collect_crawl(job_id, deadline=time.time() + timeout)

    def _collect_crawl(self, job_id: str, *, deadline: float) -> list[FetchedPage]:
        pages: list[FetchedPage] = []
        cursor: str | None = None
        terminal = {"completed", "failed", "cancelled"}

        while True:
            if time.time() > deadline:
                raise TimeoutError(f"crawl job {job_id} did not complete before deadline")

            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            body = self._get(f"crawl/{job_id}", timeout=60, params=params)
            result = body.get("result") or {}
            status = (result.get("status") or "").lower()

            for rec in result.get("records") or []:
                pages.append(
                    FetchedPage(
                        url=rec.get("url", ""),
                        markdown=rec.get("markdown") or "",
                        title=rec.get("title"),
                        status=rec.get("status"),
                        extras={"job_id": job_id},
                    )
                )

            cursor = result.get("cursor")
            if cursor:
                continue  # drain pagination before deciding on terminality
            if status in terminal:
                if status != "completed":
                    log.warning("crawl job %s ended with status=%s", job_id, status)
                return pages

            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# Jina Reader (no-auth fallback)
# ---------------------------------------------------------------------------


class JinaFetcher:
    """Unauthenticated single-page fetch via https://r.jina.ai/<url>."""

    name = "jina"
    BASE = "https://r.jina.ai/"

    def fetch_markdown(self, url: str, *, timeout: int = 60) -> FetchedPage:
        import requests

        resp = requests.get(
            self.BASE + url,
            headers={"Accept": "text/markdown"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return FetchedPage(url=url, markdown=resp.text)

    def crawl(self, *args: Any, **kwargs: Any) -> list[FetchedPage]:
        raise NotImplementedError(
            "JinaFetcher does not support crawl; use CloudflareFetcher"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_fetcher(
    name: str | None = None,
    *,
    account_id: str | None = None,
    api_token: str | None = None,
) -> URLFetcher:
    """Build a fetcher from a name. Defaults to the ``RLM_WEB_FETCHER`` env
    var, falling back to ``cloudflare``."""
    name = (name or os.environ.get("RLM_WEB_FETCHER") or "cloudflare").lower().strip()
    if name == "cloudflare":
        return CloudflareFetcher(account_id=account_id, api_token=api_token)
    if name == "jina":
        return JinaFetcher()
    raise ValueError(f"unknown web fetcher: {name!r}")
