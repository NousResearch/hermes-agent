"""Tiny ACP router fallback for tests without agent-client-protocol."""

from __future__ import annotations

from typing import Any


def _snake_params(params: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "sessionId": "session_id",
        "modeId": "mode_id",
        "modelId": "model_id",
        "configId": "config_id",
        "mcpServers": "mcp_servers",
    }
    return {mapping.get(k, k): v for k, v in (params or {}).items()}


def _dump_response(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    return value


def build_agent_router(agent: Any, use_unstable_protocol: bool = False):
    async def router(method: str, params: dict[str, Any] | None = None, _notify: bool = False):
        kwargs = _snake_params(params or {})
        handlers = {
            "initialize": agent.initialize,
            "authenticate": agent.authenticate,
            "session/new": agent.new_session,
            "session/load": agent.load_session,
            "session/resume": agent.resume_session,
            "session/cancel": agent.cancel,
            "session/fork": agent.fork_session,
            "session/list": agent.list_sessions,
            "session/prompt": agent.prompt,
            "session/set_mode": agent.set_session_mode,
            "session/set_config_option": agent.set_config_option,
        }
        if use_unstable_protocol:
            handlers["session/set_model"] = agent.set_session_model
        handler = handlers.get(method)
        if handler is None:
            raise ValueError(f"unknown ACP method: {method}")
        return _dump_response(await handler(**kwargs))

    return router
