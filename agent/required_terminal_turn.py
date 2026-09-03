"""Host-owned contract for a required terminal tool on gateway turns.

The helper owns validation only. Provider calls, tool dispatch, transcript
persistence, and final delivery remain in Hermes' existing agent loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tools.registry import TerminalToolResult

DEFAULT_FAILURE_RESPONSE = (
    "I couldn't safely complete this turn. Please send your message again."
)


class RequiredTerminalTurnError(RuntimeError):
    """The configured terminal-turn contract cannot be satisfied safely."""


@dataclass(frozen=True)
class RequiredTerminalPolicy:
    name: str
    failure_response: str = DEFAULT_FAILURE_RESPONSE


_NATURAL_GATEWAY_TURN_SEAL = object()


@dataclass(frozen=True, slots=True)
class _NaturalGatewayTurn:
    platform_message_id: str
    seal: object


def _mint_natural_gateway_turn(
    platform_message_id: Any, *, internal: bool
) -> _NaturalGatewayTurn | None:
    """Mint the private marker used by the natural inbound gateway runner."""

    if internal:
        return None
    message_id = str(platform_message_id or "").strip()
    if not message_id:
        return None
    return _NaturalGatewayTurn(message_id, _NATURAL_GATEWAY_TURN_SEAL)


def _natural_gateway_message_id(gateway_turn: Any) -> str | None:
    if (
        isinstance(gateway_turn, _NaturalGatewayTurn)
        and gateway_turn.seal is _NATURAL_GATEWAY_TURN_SEAL
    ):
        return gateway_turn.platform_message_id
    return None


def load_required_terminal_policy(
    agent: Any, *, gateway_turn: Any, platform_message_id: Any
) -> RequiredTerminalPolicy | None:
    """Read the profile-owned policy only at the trusted gateway boundary."""

    capability_message_id = _natural_gateway_message_id(gateway_turn)
    if capability_message_id is None:
        return None
    durable_message_id = str(platform_message_id or "").strip()
    if not durable_message_id or durable_message_id != capability_message_id:
        raise RequiredTerminalTurnError(
            "natural gateway capability does not match durable inbound identity"
        )

    from hermes_cli.config import load_config_readonly

    config = load_config_readonly() or {}
    agent_config = config.get("agent") or {}
    if not isinstance(agent_config, dict):
        raise RequiredTerminalTurnError("agent config must be a mapping")
    raw = agent_config.get("required_terminal_tool")
    if raw in (None, False, ""):
        return None
    if not isinstance(raw, dict):
        raise RequiredTerminalTurnError(
            "agent.required_terminal_tool must be a mapping"
        )
    if str(raw.get("surface") or "gateway").strip().lower() != "gateway":
        raise RequiredTerminalTurnError(
            "required terminal tool surface must be 'gateway'"
        )

    name = str(raw.get("name") or "").strip()
    if not name:
        raise RequiredTerminalTurnError("required terminal tool name is missing")
    failure_response = str(
        raw.get("failure_response") or DEFAULT_FAILURE_RESPONSE
    ).strip()
    if not failure_response:
        failure_response = DEFAULT_FAILURE_RESPONSE
    return RequiredTerminalPolicy(name=name, failure_response=failure_response)


def validate_required_terminal_policy(
    agent: Any, policy: RequiredTerminalPolicy
) -> None:
    """Fail before provider access when the declared terminal tool is unusable."""

    from tools.registry import registry

    if getattr(agent, "api_mode", None) != "codex_responses":
        raise RequiredTerminalTurnError(
            "required terminal tools need codex_responses mode"
        )
    if getattr(agent, "provider", None) != "openai-codex":
        raise RequiredTerminalTurnError(
            "required terminal tools need the openai-codex provider"
        )

    entry = registry.get_entry(policy.name)
    if entry is None or not getattr(entry, "terminal", False):
        raise RequiredTerminalTurnError(
            f"required terminal tool {policy.name!r} is unavailable or non-terminal"
        )

    names = {
        str((tool.get("function") or {}).get("name") or "")
        for tool in (getattr(agent, "tools", None) or [])
        if isinstance(tool, dict)
    }
    if policy.name not in names:
        raise RequiredTerminalTurnError(
            f"required terminal tool {policy.name!r} is absent from this turn"
        )


def apply_required_terminal_request(
    request: dict[str, Any],
    *,
    policy: RequiredTerminalPolicy,
    provider: str,
) -> dict[str, Any]:
    """Apply the host-owned provider request constraint after all middleware."""

    if provider != "openai-codex":
        raise RequiredTerminalTurnError(
            f"required terminal tools are unsupported for provider {provider!r}"
        )
    names: set[str] = set()
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            names.add(str(function.get("name") or ""))
        names.add(str(tool.get("name") or ""))
    if policy.name not in names:
        raise RequiredTerminalTurnError(
            f"middleware removed required terminal tool {policy.name!r}"
        )
    updated = dict(request)
    updated["tool_choice"] = {"type": "function", "name": policy.name}
    updated["parallel_tool_calls"] = False
    return updated


def validate_required_terminal_call(
    assistant_message: Any, policy: RequiredTerminalPolicy
) -> Any:
    """Accept exactly one pure call to the configured tool, before execution."""

    content = str(getattr(assistant_message, "content", "") or "")
    calls = list(getattr(assistant_message, "tool_calls", None) or [])
    if content.strip():
        raise RequiredTerminalTurnError("provider mixed prose with terminal call")
    if len(calls) != 1:
        raise RequiredTerminalTurnError("provider must emit exactly one terminal call")

    call = calls[0]
    function = getattr(call, "function", None)
    if str(getattr(function, "name", "") or "") != policy.name:
        raise RequiredTerminalTurnError("provider emitted the wrong terminal tool")
    if not str(getattr(call, "id", "") or "").strip():
        raise RequiredTerminalTurnError("terminal tool call ID is missing")
    try:
        arguments = json.loads(str(getattr(function, "arguments", "") or "{}"))
    except (TypeError, ValueError) as exc:
        raise RequiredTerminalTurnError(
            "terminal tool arguments are invalid JSON"
        ) from exc
    if not isinstance(arguments, dict):
        raise RequiredTerminalTurnError("terminal tool arguments must be an object")
    return call


def validate_terminal_result(
    result: Any,
    *,
    turn_id: str,
    tool_call_id: str,
    request_id: str,
) -> TerminalToolResult:
    """Bind a trusted terminal receipt to this exact host turn and tool call."""

    if not isinstance(result, TerminalToolResult):
        raise RequiredTerminalTurnError("terminal tool returned no terminal receipt")
    if result.turn_id != turn_id:
        raise RequiredTerminalTurnError("terminal receipt turn ID mismatch")
    if result.tool_call_id != tool_call_id:
        raise RequiredTerminalTurnError("terminal receipt tool-call ID mismatch")
    if result.request_id != request_id:
        raise RequiredTerminalTurnError("terminal receipt request ID mismatch")
    if not isinstance(result.response_text, str) or not result.response_text.strip():
        raise RequiredTerminalTurnError("terminal receipt response is empty")
    if not isinstance(result.receipt, dict):
        raise RequiredTerminalTurnError("terminal receipt payload is invalid")
    try:
        json.dumps(result.receipt)
    except (TypeError, ValueError) as exc:
        raise RequiredTerminalTurnError(
            "terminal receipt payload is not JSON-serializable"
        ) from exc
    return result
