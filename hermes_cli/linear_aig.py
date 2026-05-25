"""Linear AIG webhook receiver primitives for Hermes.

This module keeps the Linear Agent Session wiring isolated from Hermes'
generic webhook adapter. Linear AIG has stricter timing and payload semantics:
webhook handlers must acknowledge quickly, then emit Agent Activities back to
the Agent Session.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

import httpx

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without aiohttp
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False


LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
DEFAULT_REPLAY_TOLERANCE_MS = 60_000

ActivitySender = Callable[[str, dict[str, Any]], Awaitable[None]]
TaskDispatcher = Callable[["AgentSessionEvent"], Awaitable[str | None]]


@dataclass(frozen=True)
class AgentSessionEvent:
    action: str
    agent_session_id: str
    prompt_context: str
    prompt: str
    issue_identifier: str | None = None
    issue_title: str | None = None
    comment_body: str | None = None


def _header(headers: Mapping[str, str], name: str) -> str:
    lower_name = name.lower()
    for key, value in headers.items():
        if key.lower() == lower_name:
            return value
    return ""


def verify_linear_signature(
    *,
    raw_body: bytes,
    headers: Mapping[str, str],
    signing_secret: str,
    payload: Mapping[str, Any] | None = None,
    now_ms: int | None = None,
    replay_tolerance_ms: int = DEFAULT_REPLAY_TOLERANCE_MS,
) -> bool:
    """Validate Linear's HMAC signature and webhook timestamp."""
    if not signing_secret:
        return False

    signature = _header(headers, "Linear-Signature").strip()
    if signature.startswith("sha256="):
        signature = signature.removeprefix("sha256=")
    if not signature:
        return False

    try:
        expected = hmac.new(
            signing_secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
    except Exception:
        return False

    if not hmac.compare_digest(expected, signature):
        return False

    if replay_tolerance_ms <= 0:
        return True

    body = payload
    if body is None:
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            return False

    timestamp = body.get("webhookTimestamp")
    if not isinstance(timestamp, (int, float)):
        return False

    current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    return abs(current_ms - int(timestamp)) <= replay_tolerance_ms


def parse_agent_session_event(payload: Mapping[str, Any]) -> AgentSessionEvent:
    """Extract the fields Hermes needs from a Linear AgentSessionEvent."""
    action = str(payload.get("action") or "").strip()
    agent_session = payload.get("agentSession")
    if not isinstance(agent_session, Mapping):
        data = payload.get("data")
        agent_session = data.get("agentSession") if isinstance(data, Mapping) else None
    if not isinstance(agent_session, Mapping):
        agent_session = {}

    agent_session_id = str(
        agent_session.get("id") or payload.get("agentSessionId") or ""
    ).strip()
    if not action:
        raise ValueError("Linear AgentSessionEvent is missing action")
    if not agent_session_id:
        raise ValueError("Linear AgentSessionEvent is missing agentSession.id")

    agent_activity = payload.get("agentActivity")
    if not isinstance(agent_activity, Mapping):
        agent_activity = {}

    issue = agent_session.get("issue")
    if not isinstance(issue, Mapping):
        issue = {}

    data = payload.get("data")
    if not isinstance(data, Mapping):
        data = {}

    prompt_context = str(
        payload.get("promptContext")
        or data.get("promptContext")
        or agent_session.get("promptContext")
        or ""
    )
    comment_body = str(agent_activity.get("body") or "")
    prompt = prompt_context or comment_body

    return AgentSessionEvent(
        action=action,
        agent_session_id=agent_session_id,
        prompt_context=prompt_context,
        prompt=prompt,
        issue_identifier=(str(issue.get("identifier")) if issue.get("identifier") else None),
        issue_title=(str(issue.get("title")) if issue.get("title") else None),
        comment_body=comment_body or None,
    )


def build_activity_content(kind: str, body: str, **extra: Any) -> dict[str, Any]:
    if kind not in {"thought", "elicitation", "action", "response", "error"}:
        raise ValueError(f"Unsupported Linear Agent Activity type: {kind}")
    content: dict[str, Any] = {"type": kind}
    if kind == "action":
        content["action"] = body
    else:
        content["body"] = body
    content.update({key: value for key, value in extra.items() if value is not None})
    return content


def build_agent_activity_input(
    agent_session_id: str,
    content: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "agentSessionId": agent_session_id,
        "content": dict(content),
    }


async def send_agent_activity(
    *,
    api_key: str,
    agent_session_id: str,
    content: Mapping[str, Any],
    graphql_url: str = LINEAR_GRAPHQL_URL,
    timeout_seconds: float = 10.0,
) -> None:
    """Send a Linear Agent Activity using GraphQL."""
    if not api_key:
        raise ValueError("Linear API key is required to send Agent Activities")

    mutation = """
    mutation AgentActivityCreate($input: AgentActivityCreateInput!) {
      agentActivityCreate(input: $input) {
        success
      }
    }
    """
    payload = {
        "query": mutation,
        "variables": {
            "input": build_agent_activity_input(agent_session_id, content),
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _post() -> None:
        response = httpx.post(
            graphql_url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        errors = data.get("errors")
        if errors:
            raise RuntimeError(f"Linear GraphQL error: {errors}")
        if not data.get("data", {}).get("agentActivityCreate", {}).get("success"):
            raise RuntimeError("Linear agentActivityCreate did not return success")

    await asyncio.to_thread(_post)


class LinearAIGReceiver:
    """Small aiohttp-compatible receiver for Linear AgentSessionEvent webhooks."""

    def __init__(
        self,
        *,
        signing_secret: str,
        activity_sender: ActivitySender,
        task_dispatcher: TaskDispatcher | None = None,
    ) -> None:
        self.signing_secret = signing_secret
        self.activity_sender = activity_sender
        self.task_dispatcher = task_dispatcher

    async def handle_request(self, request: Any) -> Any:
        raw_body = await request.read()
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        if not verify_linear_signature(
            raw_body=raw_body,
            headers=request.headers,
            signing_secret=self.signing_secret,
            payload=payload,
        ):
            return web.json_response({"error": "invalid_signature"}, status=401)

        try:
            event = parse_agent_session_event(payload)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        asyncio.create_task(self.process_event(event))
        return web.json_response({"ok": True})

    async def process_event(self, event: AgentSessionEvent) -> None:
        await self.activity_sender(
            event.agent_session_id,
            build_activity_content(
                "thought",
                f"Hermes received Linear Agent Session `{event.action}` and is starting.",
            ),
        )
        if not self.task_dispatcher:
            return
        try:
            result = await self.task_dispatcher(event)
        except Exception as exc:
            await self.activity_sender(
                event.agent_session_id,
                build_activity_content("error", f"Hermes failed: {exc}"),
            )
            return
        if result:
            await self.activity_sender(
                event.agent_session_id,
                build_activity_content("response", result),
            )


def create_app(receiver: LinearAIGReceiver) -> Any:
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required for the Linear AIG receiver")

    async def health(_request: Any) -> Any:
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/linear/aig", receiver.handle_request)
    app.router.add_get("/linear/aig/health", health)
    return app
