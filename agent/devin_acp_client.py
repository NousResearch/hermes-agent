"""OpenAI-compatible shim that forwards Hermes requests to ``devin acp``.

Mirrors :class:`agent.copilot_acp_client.CopilotACPClient` for Cognition's
Devin CLI ACP mode (JSON-RPC over stdio). Devin is launched as a short-lived
subprocess per request, the same lifecycle Copilot ACP uses.

Docs: https://docs.devin.ai/cli/acp/jetbrains (``devin acp`` subcommand).
"""

from __future__ import annotations

import os
import shlex
from typing import Any

from agent.copilot_acp_client import CopilotACPClient

ACP_MARKER_BASE_URL = "acp://devin"


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_DEVIN_ACP_COMMAND", "").strip()
        or os.getenv("DEVIN_CLI_PATH", "").strip()
        or "devin"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_DEVIN_ACP_ARGS", "").strip()
    if not raw:
        # Official JetBrains / Zed ACP config uses a single ``acp`` argument.
        return ["acp"]
    return shlex.split(raw)


class DevinACPClient(CopilotACPClient):
    """Minimal OpenAI-client-compatible facade for Devin CLI ACP."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key or "devin-acp",
            base_url=base_url or ACP_MARKER_BASE_URL,
            default_headers=default_headers,
            acp_command=acp_command or command or _resolve_command(),
            acp_args=list(acp_args if acp_args is not None else (args if args is not None else _resolve_args())),
            acp_cwd=acp_cwd,
            **kwargs,
        )

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        return super()._create_chat_completion(
            model=model or "devin-acp",
            messages=messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            **kwargs,
        )

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        try:
            return super()._run_prompt(prompt_text, timeout_seconds=timeout_seconds)
        except RuntimeError as exc:
            msg = str(exc)
            if "Could not start Copilot ACP command" in msg:
                raise RuntimeError(
                    f"Could not start Devin ACP command '{self._acp_command}'. "
                    "Install Devin CLI (https://docs.devin.ai/cli) and run "
                    "`devin auth login`, or set HERMES_DEVIN_ACP_COMMAND/DEVIN_CLI_PATH."
                ) from exc
            if "Copilot ACP process did not expose" in msg:
                raise RuntimeError(
                    "Devin ACP process did not expose stdin/stdout pipes."
                ) from exc
            raise
