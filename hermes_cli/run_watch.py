"""Interactive, fail-closed watcher for one authenticated ``/v1/runs`` run."""

from __future__ import annotations

import json
import ipaddress
from collections.abc import Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


_TERMINAL_EVENTS = {"run.completed": "completed", "run.failed": "failed", "run.cancelled": "cancelled"}


class _NoRedirect(HTTPRedirectHandler):
    """Never forward an API credential to a redirected endpoint."""

    def redirect_request(self, *_args, **_kwargs):
        return None


def _base_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("--url must be an absolute HTTP(S) URL without embedded credentials")
    host = (parsed.hostname or "").lower()
    # Do not resolve arbitrary host names to decide whether it is safe to send
    # a bearer token over HTTP: a DNS answer can change between this check and
    # urllib's connection. ``localhost`` and literal loopback addresses are
    # the only plaintext exception.
    loopback = host == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if parsed.scheme != "https" and not loopback:
        raise ValueError("HTTPS is required unless --url resolves to loopback")
    return value.rstrip("/")


def _request(url: str, api_key: str, *, data: bytes | None = None) -> Request:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "text/event-stream, application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")


def _events(response) -> Iterator[tuple[str, dict]]:
    """Parse the small SSE subset emitted by the runs endpoint."""
    event, data = "message", []
    for raw in response:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                try:
                    payload = json.loads("\n".join(data))
                except json.JSONDecodeError:
                    payload = {}
                if isinstance(payload, dict):
                    # ``api_server_runs`` uses a JSON event envelope rather
                    # than SSE's optional ``event:`` field. Honor either so
                    # this client follows the server's current wire contract.
                    yield (event if event != "message" else str(payload.get("event") or event)), payload
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


def _show_event(event: str, payload: dict, write: Callable[[str], None]) -> None:
    if event == "message.delta":
        write(str(payload.get("delta") or ""))
    elif event == "approval.request":
        write("\nApproval required")
        if payload.get("description"):
            write(f"  {payload['description']}")
        if payload.get("command"):
            write(f"  command: {payload['command']}")
    elif event.startswith("run."):
        write(f"\nRun {event.removeprefix('run.')}.")
    elif event in {"tool.start", "tool.complete", "subagent.start", "subagent.complete"}:
        preview = payload.get("preview") or payload.get("tool_name") or ""
        write(f"\n{event}: {preview}".rstrip())


def _choose(payload: dict, prompt: Callable[[str], str]) -> str:
    allowed = tuple(str(choice) for choice in payload.get("choices", []) if choice)
    if not allowed:
        allowed = ("once", "deny")
    while True:
        choice = prompt(f"Approve? ({'/'.join(allowed)}) [deny]: ").strip().lower() or "deny"
        if choice in allowed:
            return choice
        # Do not silently coerce an invalid input into consent.


def watch_run(
    run_id: str,
    base_url: str,
    api_key: str,
    *,
    opener=None,
    prompt: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    timeout: float = 30.0,
) -> str:
    """Stream one run and synchronously answer its correlated approval cards.

    The server remains the authorization authority: this client supplies the API
    credential and the exact pending ``request_id`` only. A closed stream or a
    failed approval request never grants a command.
    """
    run_id = run_id.strip()
    if not run_id or "/" in run_id:
        raise ValueError("run_id must be a non-empty run identifier")
    if not api_key.strip():
        raise ValueError("--api-key (or HERMES_API_KEY) is required")
    root = _base_url(base_url)
    opener = opener or build_opener(_NoRedirect()).open
    events_url = f"{root}/v1/runs/{run_id}/events"
    try:
        with opener(_request(events_url, api_key), timeout=timeout) as response:
            for event, payload in _events(response):
                _show_event(event, payload, write)
                if event == "approval.request":
                    request_id = str(payload.get("request_id") or "")
                    if not request_id:
                        raise RuntimeError("server sent an approval request without request_id")
                    choice = _choose(payload, prompt)
                    body = json.dumps({"choice": choice, "request_id": request_id}).encode("utf-8")
                    approval_url = f"{root}/v1/runs/{run_id}/approval"
                    with opener(_request(approval_url, api_key, data=body), timeout=timeout):
                        pass
                if event in _TERMINAL_EVENTS:
                    return _TERMINAL_EVENTS[event]
    except HTTPError as exc:
        raise RuntimeError(f"API request failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"API connection failed: {exc.reason}") from exc
    return "stream_closed"
