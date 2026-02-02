"""
External ToolServer (Phase 4.5+).

This server executes tools that must NOT run inside the sandbox, typically
because they require credentials or access to external services.

Run (dev):
  uv run uvicorn atropos_agent.api.tool_server:app --host 0.0.0.0 --port 9002
"""

from __future__ import annotations

import asyncio
import os
import inspect
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from ..tools import ToolRegistry, build_tool_registry
from ..tools.base import ToolResultPayload, ToolServerExecuteRequest


class ToolServerConfig(BaseModel):
    token: Optional[str] = Field(
        default=None,
        description="Bearer token required for requests (optional in dev).",
    )
    max_concurrency: int = Field(default=16, ge=1, description="Max concurrent tool executions.")

    @classmethod
    def from_env(cls) -> "ToolServerConfig":
        # In dev, prefer loading secrets from the repo-local `.env` (not committed).
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:  # pragma: no cover
            load_dotenv = None  # type: ignore[assignment]
        if load_dotenv is not None:
            env_path = Path(__file__).resolve().parents[2] / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)

        token = os.getenv("TOOL_SERVER_TOKEN") or None
        max_concurrency = int(os.getenv("TOOL_SERVER_MAX_CONCURRENCY", "16"))
        return cls(token=token, max_concurrency=max_concurrency)


app = FastAPI(title="Atropos-Agent Tool Server")


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Atropos-Agent Tool Server"}


@app.on_event("startup")
async def _startup() -> None:
    cfg = ToolServerConfig.from_env()

    # External-only registry. It will only include tools that are enabled by toolsets and
    # whose Hermes requirements/keys are satisfied in this process.
    tools: ToolRegistry = build_tool_registry(
        enabled_toolsets=["all"],
        disabled_toolsets=["terminal", "sandbox", "filesystem", "terminal_stateful", "default"],
        tool_server_url="enabled",
    )

    app.state.cfg = cfg
    app.state.tools = tools
    app.state.semaphore = asyncio.Semaphore(cfg.max_concurrency)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/tools")
async def list_tools() -> Dict[str, Any]:
    tools: ToolRegistry = app.state.tools
    return {"tools": [s.to_dict() for s in tools.get_schemas()]}


def _check_auth(cfg: ToolServerConfig, authorization: Optional[str]) -> None:
    if not cfg.token:
        return
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if token != cfg.token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")


@app.post("/execute", response_model=ToolResultPayload)
async def execute_tool(
    req: ToolServerExecuteRequest,
    authorization: Optional[str] = Header(default=None),
) -> ToolResultPayload:
    cfg: ToolServerConfig = app.state.cfg
    _check_auth(cfg, authorization)

    tools: ToolRegistry = app.state.tools
    sem: asyncio.Semaphore = app.state.semaphore

    tool = tools.get(req.tool.name)
    if tool is None:
        return ToolResultPayload(
            success=False,
            error=f"Unknown tool: {req.tool.name}",
            uniq_id=req.tool.uniq_id,
        )

    async with sem:
        try:
            kwargs = dict(req.tool.arguments)
            sig = inspect.signature(tool.execute).parameters
            # Some tools can benefit from extra context.
            if req.trajectory_id and "trajectory_id" in sig:
                kwargs["trajectory_id"] = req.trajectory_id
            if req.slot_id and "slot_id" in sig:
                kwargs["slot_id"] = req.slot_id
            if req.container_addr and "container_addr" in sig:
                kwargs["container_addr"] = req.container_addr
            if "task_id" in sig:
                kwargs["task_id"] = req.trajectory_id
            result = await tool.execute(**kwargs)
        except Exception as e:
            return ToolResultPayload(
                success=False,
                error=f"Tool execution error: {e}",
                uniq_id=req.tool.uniq_id,
            )

    if result.uniq_id is None:
        result.uniq_id = req.tool.uniq_id
    return ToolResultPayload.from_tool_result(result)
