"""Official Claude Agent SDK session and fail-closed Kanban options."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from agent.claude_agent_runtime import ClaudeProjection, project_claude_messages
from agent.claude_sdk_mcp import build_hermes_sdk_mcp_server
from agent.claude_process_scope import WorkerProcessBroker
from agent.claude_subscription_env import build_claude_subscription_env
from agent.claude_tool_guard import create_workspace_pre_tool_hook
from agent.claude_workspace_terminal import build_workspace_terminal_args
from agent.claude_workspace_files import WorkspaceFileBroker


_BUILTIN_TOOLS: tuple[str, ...] = ()


def load_claude_agent_sdk() -> Any:
    """Load the exact pinned optional SDK, installing it lazily if allowed."""

    from tools.lazy_deps import ensure

    ensure("runtime.claude_agent_sdk", prompt=False)
    import claude_agent_sdk

    return claude_agent_sdk


def _mcp_tool_names(
    definitions: Iterable[Mapping[str, Any]],
    *,
    capability_mode: str,
    auxiliary_tool_names: Iterable[str],
) -> list[str]:
    available: set[str] = set()
    names: list[str] = []
    for definition in definitions:
        function = definition.get("function") if isinstance(definition, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name") or "")
        if name:
            available.add(name)
        if capability_mode == "worker" and (name.startswith("kanban_") or name in {
            "terminal",
            "process",
            "read_file",
            "write_file",
        }):
            names.append(name)
    if capability_mode == "auxiliary":
        required = {str(name) for name in auxiliary_tool_names if str(name)}
        if not required:
            raise RuntimeError("Claude auxiliary runtime requires an explicit tool allowlist")
        missing = required - available
        if missing:
            raise RuntimeError(
                "Claude auxiliary runtime is missing required Hermes tools: "
                + ", ".join(sorted(missing))
            )
        names.extend(required)
    elif capability_mode != "worker":
        raise RuntimeError(f"Unknown Claude capability mode: {capability_mode}")
    return sorted(set(names))


def build_claude_agent_options(
    *,
    sdk: Any,
    model: str,
    system_prompt: str,
    workspace: str | Path,
    host_home: str | Path,
    profile_home: str | Path,
    inherited_env: Mapping[str, str] | None,
    tool_definitions: Iterable[Mapping[str, Any]],
    dispatch: Callable[..., Any],
    effective_task_id: str,
    kanban_task_id: str | None,
    max_turns: int | None = None,
    resume: str | None = None,
    effort: str | None = None,
    cli_path: str | Path | None = None,
    file_broker: WorkspaceFileBroker | None = None,
    capability_mode: str = "worker",
    auxiliary_tool_names: Iterable[str] = (),
) -> Any:
    """Create the sole supported Claude runtime policy: an isolated worker.

    Orchestrator use is intentionally rejected until Hermes can expose its
    stateful loop tools safely through the in-process MCP boundary.
    """

    if not kanban_task_id:
        raise RuntimeError(
            "Claude Agent SDK runtime currently supports Kanban workers only"
        )

    workspace_path = Path(workspace).expanduser().resolve()
    host_home_path = Path(host_home).expanduser().resolve()
    env = build_claude_subscription_env(
        inherited_env,
        host_home=host_home_path,
        profile_home=profile_home,
    )
    tool_names = _mcp_tool_names(
        tool_definitions,
        capability_mode=capability_mode,
        auxiliary_tool_names=auxiliary_tool_names,
    )
    file_broker = file_broker or WorkspaceFileBroker(workspace_path)
    process_broker = WorkerProcessBroker(effective_task_id) if "process" in tool_names else None

    def _transform(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "terminal":
            return arguments
        return build_workspace_terminal_args(
            arguments,
            workspace=workspace_path,
            host_home=host_home_path,
            exact_env=env,
        )

    mcp_server = build_hermes_sdk_mcp_server(
        tool_definitions,
        dispatch=dispatch,
        task_id=effective_task_id,
        sdk=sdk,
        allowed_names=set(tool_names),
        argument_transform=_transform,
        handler_overrides={
            **({"process": process_broker.handle} if process_broker is not None else {}),
            **(
                {"read_file": lambda args: file_broker.handle("read_file", args)}
                if "read_file" in tool_names
                else {}
            ),
            **(
                {"write_file": lambda args: file_broker.handle("write_file", args)}
                if "write_file" in tool_names
                else {}
            ),
        },
    )
    mcp_names = [f"mcp__hermes__{name}" for name in tool_names]
    exact_tools = [*_BUILTIN_TOOLS, *mcp_names]

    pre_tool_guard = create_workspace_pre_tool_hook(
        workspace_path,
        allowed_mcp_tools=mcp_names,
    )
    hook_matcher = sdk.HookMatcher(matcher=None, hooks=[pre_tool_guard])

    return sdk.ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        cwd=str(workspace_path),
        # The SDK overlays this mapping on os.environ. The public cli_path
        # points to an env-i launcher, which enforces the actual child boundary.
        env={},
        cli_path=str(cli_path) if cli_path is not None else None,
        tools=exact_tools,
        allowed_tools=exact_tools,
        disallowed_tools=["Task", "WebFetch", "WebSearch"],
        mcp_servers={"hermes": mcp_server},
        strict_mcp_config=True,
        setting_sources=[],
        skills=[],
        plugins=[],
        permission_mode="acceptEdits",
        sandbox=None,
        hooks={"PreToolUse": [hook_matcher]},
        fallback_model=None,
        max_budget_usd=None,
        max_turns=max_turns,
        resume=resume,
        effort=effort,
        include_partial_messages=True,
    )


class ClaudeAgentSdkSession:
    """Synchronous Hermes facade over one resumable SDK conversation."""

    def __init__(
        self,
        *,
        options_factory: Callable[[str | None], Any],
        sdk: Any | None = None,
        stream_delta_callback: Callable[[str | None], Any] | None = None,
        tool_progress_callback: Callable[..., Any] | None = None,
        resources: Iterable[Any] = (),
        initial_session_id: str | None = None,
    ) -> None:
        self._sdk = sdk
        self._options_factory = options_factory
        self.session_id: str | None = initial_session_id
        self._stream_delta_callback = stream_delta_callback
        self._tool_progress_callback = tool_progress_callback
        self._resources = list(resources)

    def close(self) -> None:
        for resource in self._resources:
            try:
                resource.close()
            except Exception:
                pass
        self._resources.clear()

    def run_turn(self, prompt: str) -> ClaudeProjection:
        projection = asyncio.run(self._run_turn(prompt))
        if projection.session_id:
            self.session_id = projection.session_id
        return projection

    async def _run_turn(self, prompt: str) -> ClaudeProjection:
        for resource in self._resources:
            begin_turn = getattr(resource, "begin_turn", None)
            if callable(begin_turn):
                begin_turn()
        sdk = self._sdk or load_claude_agent_sdk()
        options = self._options_factory(self.session_id)
        events: list[Any] = []
        started_tools: set[str] = set()
        async with sdk.ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for event in client.receive_response():
                events.append(event)
                if type(event).__name__ == "RateLimitEvent":
                    immediate = project_claude_messages([event]).failure
                    if immediate is not None and immediate.reason.value == "billing":
                        await client.interrupt()
                        break
                if type(event).__name__ == "StreamEvent":
                    raw = getattr(event, "event", None)
                    event_type = raw.get("type") if isinstance(raw, Mapping) else getattr(raw, "type", None)
                    delta = raw.get("delta") if isinstance(raw, Mapping) else getattr(raw, "delta", None)
                    delta_type = delta.get("type") if isinstance(delta, Mapping) else getattr(delta, "type", None)
                    text = delta.get("text") if isinstance(delta, Mapping) else getattr(delta, "text", None)
                    if (
                        event_type == "content_block_delta"
                        and delta_type == "text_delta"
                        and text
                        and self._stream_delta_callback is not None
                    ):
                        self._stream_delta_callback(str(text))
                if type(event).__name__ == "AssistantMessage" and self._tool_progress_callback:
                    for block in getattr(event, "content", None) or []:
                        if type(block).__name__ != "ToolUseBlock":
                            continue
                        call_id = str(getattr(block, "id", "") or "")
                        if call_id in started_tools:
                            continue
                        started_tools.add(call_id)
                        name = str(getattr(block, "name", "") or "")
                        if name.startswith("mcp__hermes__"):
                            name = name[len("mcp__hermes__") :]
                        args = getattr(block, "input", None)
                        if not isinstance(args, dict):
                            args = {"input": args}
                        self._tool_progress_callback(
                            "tool.started", name, str(args)[:500], args
                        )
        return project_claude_messages(events)


__all__ = [
    "ClaudeAgentSdkSession",
    "build_claude_agent_options",
    "load_claude_agent_sdk",
]
