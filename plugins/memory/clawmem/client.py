"""ClawMem REST client — httpx wrapper for the ClawMem Git Server API.

Stateless async client.  All public methods are async; callers use
``run_sync()`` to bridge from synchronous MemoryProvider hooks.

Each ``run_sync()`` call spins up a fresh event loop so concurrent
threads (background prefetch, sync_turn, on_memory_write) never
collide on a shared loop.  This is safe because every ``_request``
creates its own ``httpx.AsyncClient`` — there is no connection pool
to preserve across calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date
from typing import Any
from urllib.parse import quote, urlencode

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sync-async bridging
# ---------------------------------------------------------------------------


def run_sync(coro):
    """Run an async coroutine from a synchronous context.

    Creates a throwaway event loop per call.  Thread-safe — multiple
    threads can call ``run_sync()`` concurrently without contention.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def label_color(label: str) -> str:
    if label.startswith("type:"):
        return "5319e7" if label == "type:memory" else "1d76db"
    if label.startswith("kind:"):
        return "5319e7"
    if label.startswith("topic:"):
        return "fbca04"
    if label.startswith("status:"):
        return "b60205"
    if label.startswith("agent:"):
        return "1d76db"
    return "0e8a16"


def label_description(label: str) -> str:
    prefixes = {
        "type:": "Issue type",
        "kind:": "Memory kind",
        "status:": "Conversation lifecycle status",
        "agent:": "Agent",
        "topic:": "Topic",
    }
    for pfx, desc in prefixes.items():
        if label.startswith(pfx):
            return f"{desc} label managed by clawmem."
    return "Label managed by clawmem."


# ---------------------------------------------------------------------------
# Issue body helpers
# ---------------------------------------------------------------------------

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def render_memory_body(detail: str, hash_val: str) -> str:
    return render_flat_yaml([
        ("memory_hash", hash_val),
        ("date", date.today().isoformat()),
        ("detail", detail.strip()),
    ])


def render_memory_title(detail: str, title: str | None = None) -> str:
    raw = title.strip() if title and title.strip() else detail.strip()
    prefix = "Memory: "
    if raw.startswith(prefix):
        return raw
    return f"{prefix}{raw[:120]}"


def render_flat_yaml(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in pairs)


def parse_flat_yaml(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key and value:
            result[key] = value
    return result


def extract_label_names(labels: list) -> list[str]:
    result: list[str] = []
    for item in labels:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = (item.get("name") or "").strip()
        else:
            continue
        if name:
            result.append(name)
    return result


def label_val(labels: list[str], prefix: str) -> str | None:
    for lbl in labels:
        if lbl.startswith(prefix):
            val = lbl[len(prefix):].strip()
            if val:
                return val
    return None


def parse_memory_issue(issue: dict) -> dict | None:
    """Parse a GitHub issue dict into a ClawMem memory record."""
    labels = extract_label_names(issue.get("labels", []))
    if "type:memory" not in labels:
        return None

    body = (issue.get("body") or "").strip()
    parsed = parse_flat_yaml(body)
    detail = (parsed.get("detail") or body).strip()
    if not detail:
        title_raw = (issue.get("title") or "").strip()
        detail = title_raw.removeprefix("Memory: ").strip() if title_raw else ""
    if not detail:
        return None

    kind = label_val(labels, "kind:")
    topics = [
        lbl[6:].strip()
        for lbl in labels
        if lbl.startswith("topic:") and lbl[6:].strip()
    ]
    status = "stale" if issue.get("state") == "closed" else "active"

    return {
        "issue_number": issue.get("number"),
        "title": (issue.get("title") or "").strip(),
        "memory_id": str(issue.get("number", "")),
        "memory_hash": parsed.get("memory_hash", ""),
        "date": parsed.get("date", ""),
        "detail": detail,
        "kind": kind,
        "topics": topics,
        "status": status,
    }


def format_memory_line(mem: dict) -> str:
    parts = [f"[{mem['memory_id']}]"]
    if mem.get("kind"):
        parts.append(f"({mem['kind']})")
    parts.append(mem["detail"][:120])
    if mem.get("topics"):
        parts.append(f"  topics: {', '.join(mem['topics'])}")
    return " ".join(parts)


def format_memory_block(mem: dict) -> str:
    lines = [
        f"ID: {mem['memory_id']}",
        f"Title: {mem['title']}",
        f"Status: {mem['status']}",
        f"Detail: {mem['detail']}",
    ]
    if mem.get("kind"):
        lines.append(f"Kind: {mem['kind']}")
    if mem.get("topics"):
        lines.append(f"Topics: {', '.join(mem['topics'])}")
    if mem.get("date"):
        lines.append(f"Date: {mem['date']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClawMemClient
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30.0


class ClawMemClient:
    """Async REST client for the ClawMem Git Server API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        default_repo: str,
        auth_scheme: str = "token",
    ):
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/api/v3"):
            base_url = f"{base_url}/api/v3"
        self.base_url = base_url
        self.token = token
        self.default_repo = default_repo
        self.auth_scheme = auth_scheme

    def _headers(self, *, omit_auth: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        }
        if not omit_auth and self.token:
            if self.auth_scheme == "bearer":
                headers["Authorization"] = f"Bearer {self.token}"
            else:
                headers["Authorization"] = f"token {self.token}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        allow_not_found: bool = False,
        allow_validation_error: bool = False,
        omit_auth: bool = False,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.request(
                method,
                url,
                json=body if body is not None else None,
                headers=self._headers(omit_auth=omit_auth),
            )

        if resp.status_code == 404 and allow_not_found:
            return None
        if resp.status_code == 422 and allow_validation_error:
            return None
        if resp.status_code == 204:
            return None

        if not resp.is_success:
            text = resp.text[:500] if resp.text else resp.reason_phrase
            raise RuntimeError(
                f"ClawMem API {method} {path}: HTTP {resp.status_code}: {text}"
            )

        raw = resp.text.strip()
        if not raw:
            return None
        try:
            return resp.json()
        except Exception:
            logger.warning("ClawMem: failed to parse response for %s %s", method, path)
            return None

    def _repo_path(self, suffix: str) -> str:
        if not self.default_repo:
            raise RuntimeError("ClawMem repository is not configured")
        owner, _, repo = self.default_repo.partition("/")
        return f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}/{suffix}"

    # -- Bootstrap ----------------------------------------------------------

    @staticmethod
    async def register_agent(
        base_url: str,
        prefix_login: str,
        default_repo_name: str,
    ) -> dict:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/api/v3"):
            base_url = f"{base_url}/api/v3"
        url = f"{base_url}/agents"
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                url,
                json={
                    "prefix_login": prefix_login,
                    "default_repo_name": default_repo_name,
                },
                headers={
                    "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json",
                },
            )
        if not resp.is_success:
            text = resp.text[:500] if resp.text else resp.reason_phrase
            raise RuntimeError(
                f"ClawMem agent registration failed: HTTP {resp.status_code}: {text}"
            )
        return resp.json()

    # -- Issues -------------------------------------------------------------

    async def create_issue(self, title: str, body: str, labels: list[str]) -> dict:
        return await self._request(
            "POST",
            self._repo_path("issues"),
            body={"title": title, "body": body, "labels": labels},
        )

    async def update_issue(
        self,
        issue_number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
    ) -> dict:
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if state is not None:
            payload["state"] = state
        if labels is not None:
            payload["labels"] = labels
        return await self._request(
            "PATCH",
            self._repo_path(f"issues/{issue_number}"),
            body=payload,
        )

    async def get_issue(self, issue_number: int) -> dict:
        return await self._request(
            "GET",
            self._repo_path(f"issues/{issue_number}"),
        )

    async def list_issues(
        self,
        *,
        labels: list[str] | None = None,
        state: str = "open",
        page: int = 1,
        per_page: int = 100,
    ) -> list[dict]:
        params = f"state={state}&page={page}&per_page={per_page}"
        if labels:
            params += f"&labels={','.join(labels)}"
        result = await self._request(
            "GET",
            self._repo_path(f"issues?{params}"),
        )
        return result if isinstance(result, list) else []

    # -- Comments -----------------------------------------------------------

    async def create_comment(self, issue_number: int, body: str) -> None:
        await self._request(
            "POST",
            self._repo_path(f"issues/{issue_number}/comments"),
            body={"body": body},
        )

    # -- Search -------------------------------------------------------------

    async def search_issues(
        self,
        query: str,
        *,
        page: int = 1,
        per_page: int = 100,
    ) -> list[dict]:
        params = urlencode({"q": query, "page": page, "per_page": per_page})
        result = await self._request("GET", f"search/issues?{params}")
        if isinstance(result, dict):
            return result.get("items", [])
        return []

    # -- Labels -------------------------------------------------------------

    async def ensure_labels(self, labels: list[str]) -> None:
        for lbl in labels:
            lbl = lbl.strip()
            if not lbl:
                continue
            await self._request(
                "POST",
                self._repo_path("labels"),
                body={
                    "name": lbl,
                    "color": label_color(lbl),
                    "description": label_description(lbl),
                },
                allow_validation_error=True,
            )

    async def sync_managed_labels(
        self,
        issue_number: int,
        desired: list[str],
    ) -> None:
        managed_prefixes = (
            "type:", "kind:", "session:", "date:", "topic:",
            "agent:", "status:",
        )

        issue = await self.get_issue(issue_number)
        current = extract_label_names(issue.get("labels", []))

        unmanaged = [
            lbl for lbl in current
            if not any(lbl.startswith(p) for p in managed_prefixes)
        ]
        merged = list(dict.fromkeys(unmanaged + desired))
        await self.update_issue(issue_number, labels=merged)
