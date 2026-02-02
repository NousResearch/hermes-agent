"""
Tool Executor API (Phase 4)

This service provides a queued, batched execution layer on top of SlotPool.
It mirrors the stateful FastAPI + app.state pattern used in:
  atropos/atroposlib/api/server.py

Run (dev):
  uv run uvicorn atropos_agent.api.tool_executor_server:app --host 0.0.0.0 --port 9001
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from ..slots import SlotPool, SlotPoolConfig
from ..tools import BashTool, ImageGenerateTool, ReadFileTool, ToolRegistry, WriteFileTool
from ..tools.mixture_of_agents_tool import MixtureOfAgentsTool
from ..tools.terminal_tool import TerminalTool
from ..tools.vision_tools import VisionAnalyzeTool
from ..tools.web_tools import WebCrawlTool, WebExtractTool, WebSearchTool
from ..tools.base import (
    ArtifactArchiveRequestPayload,
    ArtifactArchiveResponsePayload,
    ArtifactListRequestPayload,
    ArtifactListResponsePayload,
    ArtifactReadRequestPayload,
    ArtifactReadResponsePayload,
    ToolExecutorExecuteRequest,
    ToolExecutorReleaseRequest,
    ToolResultPayload,
)
from ..tools.tool_executor import ToolExecutor, ToolExecutorConfig


class ToolExecutorServerConfig(BaseModel):
    nomad_address: str = Field(default="http://localhost:4646")
    job_id: str = Field(default="atropos-sandbox-tool-executor")
    image: str = Field(default="atropos-sandbox:local")
    slots_per_container: int = Field(default=10)
    min_containers: int = Field(default=1)
    max_containers: int = Field(default=10)
    privileged: bool = Field(default=False)
    acquire_timeout_s: float = Field(default=30.0)

    batch_window_ms: int = Field(default=20)
    max_batch_size: int = Field(default=200)
    allow_network: bool = Field(default=True)

    tool_server_url: Optional[str] = Field(default=None)
    tool_server_token: Optional[str] = Field(default=None)

    token: Optional[str] = Field(default=None, description="Bearer token required for requests (optional in dev).")

    purge_job_on_shutdown: bool = Field(default=True)

    @classmethod
    def from_env(cls) -> "ToolExecutorServerConfig":
        # In dev, prefer loading secrets/config from the repo-local `.env` (not committed).
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:  # pragma: no cover
            load_dotenv = None  # type: ignore[assignment]
        if load_dotenv is not None:
            env_path = Path(__file__).resolve().parents[2] / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)

        def _get_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

        return cls(
            nomad_address=os.getenv("TOOL_EXECUTOR_NOMAD_ADDRESS", "http://localhost:4646"),
            job_id=os.getenv("TOOL_EXECUTOR_JOB_ID", "atropos-sandbox-tool-executor"),
            image=os.getenv("TOOL_EXECUTOR_IMAGE", "atropos-sandbox:local"),
            slots_per_container=int(os.getenv("TOOL_EXECUTOR_SLOTS", "10")),
            min_containers=int(os.getenv("TOOL_EXECUTOR_MIN_CONTAINERS", "1")),
            max_containers=int(os.getenv("TOOL_EXECUTOR_MAX_CONTAINERS", "10")),
            privileged=_get_bool("TOOL_EXECUTOR_PRIVILEGED", False),
            acquire_timeout_s=float(os.getenv("TOOL_EXECUTOR_ACQUIRE_TIMEOUT_S", "30.0")),
            batch_window_ms=int(os.getenv("TOOL_EXECUTOR_BATCH_WINDOW_MS", "20")),
            max_batch_size=int(os.getenv("TOOL_EXECUTOR_MAX_BATCH_SIZE", "200")),
            allow_network=_get_bool("TOOL_EXECUTOR_ALLOW_NETWORK", True),
            tool_server_url=os.getenv("TOOL_EXECUTOR_TOOL_SERVER_URL") or None,
            tool_server_token=os.getenv("TOOL_EXECUTOR_TOOL_SERVER_TOKEN") or None,
            token=os.getenv("TOOL_EXECUTOR_TOKEN") or None,
            purge_job_on_shutdown=_get_bool("TOOL_EXECUTOR_PURGE_JOB_ON_SHUTDOWN", True),
        )


app = FastAPI(title="Atropos-Agent Tool Executor")


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Atropos-Agent Tool Executor"}


def _check_auth(cfg: ToolExecutorServerConfig, authorization: Optional[str]) -> None:
    if not cfg.token:
        return
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if token != cfg.token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")


@app.on_event("startup")
async def _startup() -> None:
    cfg = ToolExecutorServerConfig.from_env()

    tools = ToolRegistry()
    tools.register(BashTool())
    tools.register(TerminalTool())
    tools.register(ReadFileTool())
    tools.register(WriteFileTool())
    tools.register(ImageGenerateTool())
    tools.register(WebSearchTool())
    tools.register(WebExtractTool())
    tools.register(WebCrawlTool())
    tools.register(VisionAnalyzeTool())
    tools.register(MixtureOfAgentsTool())

    pool = SlotPool(
        SlotPoolConfig(
            nomad_address=cfg.nomad_address,
            job_id=cfg.job_id,
            image=cfg.image,
            slots_per_container=cfg.slots_per_container,
            min_containers=cfg.min_containers,
            max_containers=cfg.max_containers,
            privileged=cfg.privileged,
            acquire_timeout=cfg.acquire_timeout_s,
        )
    )
    await pool.start()

    executor = ToolExecutor(
        pool=pool,
        tools=tools,
        config=ToolExecutorConfig(
            batch_window_ms=cfg.batch_window_ms,
            max_batch_size=cfg.max_batch_size,
            allow_network=cfg.allow_network,
            tool_server_url=cfg.tool_server_url,
            tool_server_token=cfg.tool_server_token,
        ),
    )
    await executor.start()

    app.state.cfg = cfg
    app.state.pool = pool
    app.state.executor = executor


@app.on_event("shutdown")
async def _shutdown() -> None:
    executor: Optional[ToolExecutor] = getattr(app.state, "executor", None)
    pool: Optional[SlotPool] = getattr(app.state, "pool", None)
    cfg: Optional[ToolExecutorServerConfig] = getattr(app.state, "cfg", None)

    if executor is not None:
        await executor.close()

    if pool is not None:
        await pool.stop(purge_job=bool(cfg.purge_job_on_shutdown) if cfg else False)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/status")
async def status_endpoint() -> Dict[str, Any]:
    executor: ToolExecutor = app.state.executor
    pool: SlotPool = app.state.pool

    return {
        "queue_size": executor.queue_size(),
        "total_requests": executor.total_requests,
        "total_errors": executor.total_errors,
        "pool": pool.get_stats(),
    }


@app.post("/execute", response_model=ToolResultPayload)
async def execute_tool(
    req: ToolExecutorExecuteRequest,
    authorization: Optional[str] = Header(default=None),
    status_code: int = status.HTTP_200_OK,  # noqa: B008
) -> ToolResultPayload:
    cfg: ToolExecutorServerConfig = app.state.cfg
    _check_auth(cfg, authorization)

    executor: ToolExecutor = app.state.executor
    result = await executor.execute(
        trajectory_id=req.trajectory_id,
        call=req.tool.to_tool_call(),
        timeout_s=req.timeout_s,
    )
    return ToolResultPayload.from_tool_result(result)


@app.post("/release")
async def release_trajectory(
    req: ToolExecutorReleaseRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    cfg: ToolExecutorServerConfig = app.state.cfg
    _check_auth(cfg, authorization)

    executor: ToolExecutor = app.state.executor
    await executor.release_trajectory(req.trajectory_id, reset_workspace=req.reset_workspace)
    return {"status": "ok"}


@app.post("/artifacts/read", response_model=ArtifactReadResponsePayload)
async def artifacts_read(
    req: ArtifactReadRequestPayload,
    authorization: Optional[str] = Header(default=None),
) -> ArtifactReadResponsePayload:
    cfg: ToolExecutorServerConfig = app.state.cfg
    _check_auth(cfg, authorization)

    executor: ToolExecutor = app.state.executor
    return await executor.read_artifact(req)


@app.post("/artifacts/list", response_model=ArtifactListResponsePayload)
async def artifacts_list(
    req: ArtifactListRequestPayload,
    authorization: Optional[str] = Header(default=None),
) -> ArtifactListResponsePayload:
    cfg: ToolExecutorServerConfig = app.state.cfg
    _check_auth(cfg, authorization)

    executor: ToolExecutor = app.state.executor
    return await executor.list_artifacts(req)


@app.post("/artifacts/archive", response_model=ArtifactArchiveResponsePayload)
async def artifacts_archive(
    req: ArtifactArchiveRequestPayload,
    authorization: Optional[str] = Header(default=None),
) -> ArtifactArchiveResponsePayload:
    cfg: ToolExecutorServerConfig = app.state.cfg
    _check_auth(cfg, authorization)

    executor: ToolExecutor = app.state.executor
    return await executor.archive_artifacts(req)
