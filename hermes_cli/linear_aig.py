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
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8667

ACCESS_TOKEN_ENV = "HERMES_LINEAR_AIG_ACCESS_TOKEN"
WEBHOOK_SECRET_ENV = "HERMES_LINEAR_AIG_WEBHOOK_SECRET"
HOST_ENV = "HERMES_LINEAR_AIG_HOST"
PORT_ENV = "HERMES_LINEAR_AIG_PORT"
GRAPHQL_URL_ENV = "HERMES_LINEAR_AIG_GRAPHQL_URL"
TASK_MODE_ENV = "HERMES_LINEAR_AIG_TASK_MODE"
MODEL_ENV = "HERMES_LINEAR_AIG_MODEL"
PROVIDER_ENV = "HERMES_LINEAR_AIG_PROVIDER"
TOOLSETS_ENV = "HERMES_LINEAR_AIG_TOOLSETS"

TASK_MODE_BRIDGE = "bridge"
TASK_MODE_ONESHOT = "oneshot"
TASK_MODES = {TASK_MODE_BRIDGE, TASK_MODE_ONESHOT}

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


@dataclass(frozen=True)
class LinearAIGRuntimeConfig:
    access_token: str
    webhook_secret: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    graphql_url: str = LINEAR_GRAPHQL_URL
    ack_only: bool = False
    task_mode: str = TASK_MODE_BRIDGE
    model: str | None = None
    provider: str | None = None
    toolsets: str | None = None


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


def _redact(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def _env(env: Mapping[str, str], name: str) -> str:
    return str(env.get(name) or "").strip()


def _parse_port(value: str | int | None, *, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Linear AIG port must be an integer") from exc
    if port <= 0 or port > 65535:
        raise ValueError("Linear AIG port must be between 1 and 65535")
    return port


def load_runtime_config(args: Any, env: Mapping[str, str]) -> LinearAIGRuntimeConfig:
    access_token = (
        str(getattr(args, "access_token", "") or "").strip()
        or _env(env, ACCESS_TOKEN_ENV)
    )
    webhook_secret = (
        str(getattr(args, "webhook_secret", "") or "").strip()
        or _env(env, WEBHOOK_SECRET_ENV)
    )
    host = str(getattr(args, "host", "") or "").strip() or _env(env, HOST_ENV) or DEFAULT_HOST
    port = _parse_port(
        getattr(args, "port", None) or _env(env, PORT_ENV),
        default=DEFAULT_PORT,
    )
    graphql_url = (
        str(getattr(args, "graphql_url", "") or "").strip()
        or _env(env, GRAPHQL_URL_ENV)
        or LINEAR_GRAPHQL_URL
    )
    ack_only = bool(getattr(args, "ack_only", False))
    task_mode = (
        str(getattr(args, "task_mode", "") or "").strip()
        or _env(env, TASK_MODE_ENV)
        or TASK_MODE_BRIDGE
    ).lower()
    if task_mode not in TASK_MODES:
        raise ValueError(
            "Linear AIG task mode must be one of: "
            + ", ".join(sorted(TASK_MODES))
        )
    model = str(getattr(args, "model", "") or "").strip() or _env(env, MODEL_ENV) or None
    provider = (
        str(getattr(args, "provider", "") or "").strip()
        or _env(env, PROVIDER_ENV)
        or None
    )
    toolsets = (
        str(getattr(args, "toolsets", "") or "").strip()
        or _env(env, TOOLSETS_ENV)
        or None
    )

    missing = []
    if not access_token:
        missing.append(ACCESS_TOKEN_ENV)
    if not webhook_secret:
        missing.append(WEBHOOK_SECRET_ENV)
    if missing:
        raise ValueError(
            "Linear AIG is missing required environment variables: "
            + ", ".join(missing)
        )

    return LinearAIGRuntimeConfig(
        access_token=access_token,
        webhook_secret=webhook_secret,
        host=host,
        port=port,
        graphql_url=graphql_url,
        ack_only=ack_only,
        task_mode=task_mode,
        model=model,
        provider=provider,
        toolsets=toolsets,
    )


def describe_runtime_config(config: LinearAIGRuntimeConfig) -> str:
    return "\n".join(
        [
            "Linear AIG runtime config:",
            f"  {ACCESS_TOKEN_ENV}: {_redact(config.access_token)}",
            f"  {WEBHOOK_SECRET_ENV}: {_redact(config.webhook_secret)}",
            f"  host: {config.host}",
            f"  port: {config.port}",
            f"  graphql_url: {config.graphql_url}",
            f"  ack_only: {config.ack_only}",
            f"  task_mode: {config.task_mode}",
            f"  model: {config.model or '<configured default>'}",
            f"  provider: {config.provider or '<configured default>'}",
            f"  toolsets: {config.toolsets or '<configured default>'}",
        ]
    )


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


def build_activity_sender(
    *,
    access_token: str,
    graphql_url: str = LINEAR_GRAPHQL_URL,
) -> ActivitySender:
    async def _sender(agent_session_id: str, content: dict[str, Any]) -> None:
        await send_agent_activity(
            api_key=access_token,
            agent_session_id=agent_session_id,
            content=content,
            graphql_url=graphql_url,
        )

    return _sender


async def ack_only_dispatcher(event: AgentSessionEvent) -> str:
    target = f" for {event.issue_identifier}" if event.issue_identifier else ""
    return (
        f"Hermes Linear AIG bridge received the session{target}. "
        "Full task execution is not wired in this runtime yet."
    )


def build_oneshot_prompt(event: AgentSessionEvent) -> str:
    parts = [
        "You are Hermes responding to a Linear Agent Session.",
        f"Action: {event.action}",
    ]
    if event.issue_identifier:
        parts.append(f"Issue: {event.issue_identifier}")
    if event.issue_title:
        parts.append(f"Issue title: {event.issue_title}")
    if event.comment_body:
        parts.append(f"Comment: {event.comment_body}")
    if event.prompt:
        parts.append("")
        parts.append(event.prompt)
    return "\n".join(parts).strip()


def run_hermes_oneshot(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    toolsets: str | None = None,
) -> str:
    from hermes_cli.oneshot import _run_agent

    return _run_agent(
        prompt,
        model=model,
        provider=provider,
        toolsets=toolsets,
        use_config_toolsets=toolsets is None,
    )


def build_task_dispatcher(config: LinearAIGRuntimeConfig) -> TaskDispatcher | None:
    if config.ack_only:
        return None
    if config.task_mode == TASK_MODE_BRIDGE:
        return ack_only_dispatcher
    if config.task_mode == TASK_MODE_ONESHOT:
        async def _dispatch(event: AgentSessionEvent) -> str:
            prompt = build_oneshot_prompt(event)
            if not prompt:
                return "Hermes Linear AIG received an empty Agent Session prompt."
            response = await asyncio.to_thread(
                run_hermes_oneshot,
                prompt,
                model=config.model,
                provider=config.provider,
                toolsets=config.toolsets,
            )
            return response or "Hermes completed the Linear Agent Session without a final response."

        return _dispatch
    raise ValueError(f"Unsupported Linear AIG task mode: {config.task_mode}")


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


def run_server(config: LinearAIGRuntimeConfig) -> None:
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required for the Linear AIG receiver. Install with: "
            "pip install 'hermes-agent[messaging]' or `pip install aiohttp`."
        )

    receiver = LinearAIGReceiver(
        signing_secret=config.webhook_secret,
        activity_sender=build_activity_sender(
            access_token=config.access_token,
            graphql_url=config.graphql_url,
        ),
        task_dispatcher=build_task_dispatcher(config),
    )
    app = create_app(receiver)
    web.run_app(app, host=config.host, port=config.port)


def linear_aig_command(args: Any) -> int:
    import os

    subcommand = getattr(args, "linear_aig_action", None)
    if not subcommand:
        print("Usage: hermes linear-aig {serve|check-config}")
        return 2

    try:
        config = load_runtime_config(args, os.environ)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if subcommand == "check-config":
        print(describe_runtime_config(config))
        return 0

    if subcommand == "serve":
        print(describe_runtime_config(config))
        print(f"Listening on http://{config.host}:{config.port}/linear/aig")
        run_server(config)
        return 0

    print(f"Unknown linear-aig command: {subcommand}")
    return 2
