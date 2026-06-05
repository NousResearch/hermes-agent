"""Workflow workbench API for the Hermes desktop dashboard.

The workbench model is intentionally separate from the existing chat/session
surface.  Projects live in user-selected folders, keep readable JSON manifests
for review/diff, and use a small SQLite event log for recoverable execution
state.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from hermes_cli.config import get_hermes_home


WorkflowNodeStatus = Literal[
    "created",
    "ready",
    "queued",
    "running",
    "reviewing",
    "waiting_user_confirm",
    "completed",
    "revision_needed",
    "failed",
    "retrying",
    "skipped",
    "aborted",
]
RunStatus = Literal["idle", "running", "paused", "waiting_user_confirm", "completed", "failed", "stopped"]
ProjectStatus = Literal["draft", "clarifying", "generated", "failed"]
ExecutionMode = Literal["single_step", "semi_auto", "auto"]
SkillMode = Literal["auto", "manual"]
StreamEventType = Literal[
    "process_summary",
    "tool_call",
    "stage_result",
    "ai_reply",
    "node_status",
    "snapshot",
    "approval",
    "error",
]

WORKFLOW_DIR = ".agent-workflow"
REGISTRY_FILE = "workflow-projects.json"
DEFAULT_NODE_WIDTH = 260
DEFAULT_NODE_HEIGHT = 112


class WorkflowPosition(BaseModel):
    x: float
    y: float


class ReviewRules(BaseModel):
    required: bool = False
    checklist: List[str] = Field(default_factory=list)


class NodeFileChange(BaseModel):
    path: str
    status: str
    diff: str = ""
    truncated: bool = False
    isArtifact: bool = False
    isBinary: bool = False
    previewable: bool = True


class WorkflowNode(BaseModel):
    id: str
    type: str = "task"
    title: str
    description: str = ""
    position: WorkflowPosition = Field(default_factory=lambda: WorkflowPosition(x=0, y=0))
    status: WorkflowNodeStatus = "created"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    reviewRules: ReviewRules = Field(default_factory=ReviewRules)
    skills: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    promptOverride: Optional[str] = None
    skillMode: SkillMode = "auto"
    references: List[str] = Field(default_factory=list)
    modelOverride: Optional[str] = None
    fileChanges: List[NodeFileChange] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    optional: bool = False
    maxRetries: int = 1
    retryCount: int = 0
    agentSessionId: Optional[str] = None
    lastRunId: Optional[str] = None
    llmGenerated: bool = False


class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str = "dependency"
    label: str = ""
    optional: bool = False


class Workflow(BaseModel):
    id: str
    title: str
    nodes: List[WorkflowNode] = Field(default_factory=list)
    edges: List[WorkflowEdge] = Field(default_factory=list)
    updatedAt: float = Field(default_factory=time.time)


class ReferenceItem(BaseModel):
    id: str
    name: str
    path: str
    enabled: bool = True
    kind: str = "file"
    addedAt: float = Field(default_factory=time.time)


class SkillBinding(BaseModel):
    id: str
    name: str
    enabled: bool = True
    source: str = "hermes"


class Artifact(BaseModel):
    id: str
    nodeId: Optional[str] = None
    name: str
    path: str
    kind: str = "markdown"
    createdAt: float = Field(default_factory=time.time)


class VersionSnapshot(BaseModel):
    id: str
    label: str
    reason: str
    commit: Optional[str] = None
    createdAt: float = Field(default_factory=time.time)


class StreamEvent(BaseModel):
    id: str
    projectId: str
    runId: Optional[str] = None
    nodeId: Optional[str] = None
    type: StreamEventType
    label: str
    timestamp: float = Field(default_factory=time.time)
    summary: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    status: str = "info"
    durationMs: Optional[int] = None


class ExecutionRun(BaseModel):
    id: str
    projectId: str
    mode: ExecutionMode = "semi_auto"
    status: RunStatus = "idle"
    currentNodeId: Optional[str] = None
    maxConcurrency: int = 2
    startedAt: float = Field(default_factory=time.time)
    updatedAt: float = Field(default_factory=time.time)
    completedAt: Optional[float] = None


class WorkflowProject(BaseModel):
    id: str
    name: str
    root: str
    goal: str = ""
    status: ProjectStatus = "generated"
    createdAt: float = Field(default_factory=time.time)
    updatedAt: float = Field(default_factory=time.time)
    currentRunId: Optional[str] = None
    agentSessionId: Optional[str] = None
    lastOpenedAt: Optional[float] = None
    archived: bool = False


class ProjectBundle(BaseModel):
    project: WorkflowProject
    workflow: Workflow
    references: List[ReferenceItem] = Field(default_factory=list)
    skills: List[SkillBinding] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)
    snapshots: List[VersionSnapshot] = Field(default_factory=list)
    latestRun: Optional[ExecutionRun] = None
    error: Optional[str] = None


class ProjectCreateRequest(BaseModel):
    name: str
    goal: str = ""
    root: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class WorkflowIntakeStartRequest(ProjectCreateRequest):
    pass


class WorkflowIntakeMessageRequest(BaseModel):
    message: str


class WorkflowIntakeOption(BaseModel):
    id: str
    label: str
    description: str = ""
    priority: int = 1


class WorkflowIntakeQuestion(BaseModel):
    id: str
    question: str
    detail: str = ""
    options: List[WorkflowIntakeOption]


class WorkflowIntakeBatch(BaseModel):
    id: str
    questions: List[WorkflowIntakeQuestion]
    createdAt: float = Field(default_factory=time.time)


class WorkflowIntakeAnswer(BaseModel):
    questionId: str
    optionId: Optional[str] = None
    answer: str = ""
    custom: bool = False


class WorkflowIntakeAnswersRequest(BaseModel):
    answers: List[WorkflowIntakeAnswer] = Field(default_factory=list)


class WorkflowIntakeConfirmRequest(ProjectCreateRequest):
    intakeId: Optional[str] = None
    projectId: Optional[str] = None
    summary: str = ""


class ProjectOpenRequest(BaseModel):
    root: str


class ProjectPatchRequest(BaseModel):
    name: Optional[str] = None
    archived: Optional[bool] = None


class WorkflowUpdateRequest(BaseModel):
    workflow: Workflow
    snapshotLabel: str = "Workflow edited"


class RunCreateRequest(BaseModel):
    mode: ExecutionMode = "semi_auto"
    maxConcurrency: int = 2


class ChatRequest(BaseModel):
    text: str
    nodeId: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    skillIds: List[str] = Field(default_factory=list)


class ComposerSlashRequest(BaseModel):
    command: str
    nodeId: Optional[str] = None


class ComposerCompletionRequest(BaseModel):
    text: str = ""
    cursor: Optional[int] = None
    cwd: Optional[str] = None


class ComposerAttachmentRequest(BaseModel):
    paths: List[str] = Field(default_factory=list)


class ReferencesUpdateRequest(BaseModel):
    references: List[ReferenceItem]


class SkillsUpdateRequest(BaseModel):
    skills: List[SkillBinding]


class RestoreSnapshotRequest(BaseModel):
    commit: str


class WorkflowRuntime:
    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flags: dict[str, threading.Event] = {}

    def lock_for(self, project_id: str) -> threading.Lock:
        lock = self._locks.get(project_id)
        if lock is None:
            lock = threading.Lock()
            self._locks[project_id] = lock
        return lock

    def start(self, project: WorkflowProject, run: ExecutionRun) -> None:
        stop = threading.Event()
        self._stop_flags[run.id] = stop
        thread = threading.Thread(target=_run_engine, args=(project.id, run.id, stop), daemon=True)
        self._threads[run.id] = thread
        thread.start()

    def stop(self, run_id: str) -> None:
        flag = self._stop_flags.get(run_id)
        if flag:
            flag.set()


_runtime = WorkflowRuntime()
_TRANSIENT_EVENT_LIMIT = 800
_transient_events: dict[str, List["StreamEvent"]] = {}
_transient_events_lock = threading.Lock()
_intake_sessions: dict[str, Dict[str, Any]] = {}
_intake_sessions_lock = threading.Lock()
_WORKFLOW_DISABLED_TOOLSETS = {"memory", "session_search"}


@dataclass
class WorkflowAgentResult:
    text: str
    session_id: str
    status: str = "complete"
    usage: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, Any]]] = None


class WorkflowAgentRunner:
    """Thin adapter from workflow projects/nodes to Hermes AIAgent turns."""

    def run(
        self,
        *,
        project: WorkflowProject,
        prompt: str,
        session_id: str,
        node: Optional[WorkflowNode] = None,
        run: Optional[ExecutionRun] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 18,
        model_override: Optional[str] = None,
        message_label: str = "AI 回复",
        persist_final: bool = True,
    ) -> WorkflowAgentResult:
        started_at = time.time()
        deltas: List[str] = []
        last_delta_event = 0.0
        message_id = _new_id("msg")

        def append_stream_delta(delta: str) -> None:
            nonlocal last_delta_event
            if not delta:
                return
            deltas.append(str(delta))
            now = time.time()
            buffered = "".join(deltas)[-1200:]
            if len(buffered) < 240 and now - last_delta_event < 1.0:
                return
            last_delta_event = now
            _append_event(
                project.id,
                StreamEvent(
                    id=_new_id("evt"),
                    projectId=project.id,
                    runId=run.id if run else None,
                    nodeId=node.id if node else None,
                    type="ai_reply",
                    label=message_label,
                    summary=_strip_reasoning(buffered),
                    details={
                        "messageId": message_id,
                        "text": _strip_reasoning(buffered),
                        "streaming": True,
                        "rawReasoningExposed": False,
                    },
                    status="info",
                ),
                persist=False,
            )

        def tool_start(tool_call_id: str, name: str, args: Dict[str, Any]) -> None:
            _append_event(
                project.id,
                StreamEvent(
                    id=_new_id("evt"),
                    projectId=project.id,
                    runId=run.id if run else None,
                    nodeId=node.id if node else None,
                    type="tool_call",
                    label=f"工具开始：{name}",
                    summary=_tool_summary_from_args(name, args),
                    details={"toolCallId": tool_call_id, "name": name, "args": _safe_details(args)},
                    status="info",
                ),
            )

        def tool_complete(tool_call_id: str, name: str, args: Dict[str, Any], result: Any) -> None:
            _append_event(
                project.id,
                StreamEvent(
                    id=_new_id("evt"),
                    projectId=project.id,
                    runId=run.id if run else None,
                    nodeId=node.id if node else None,
                    type="stage_result",
                    label=f"工具完成：{name}",
                    summary=_truncate_text(str(result), 900),
                    details={"toolCallId": tool_call_id, "name": name, "args": _safe_details(args)},
                    status="success",
                    durationMs=int((time.time() - started_at) * 1000),
                ),
            )

        def tool_progress(event_type: str, name: Optional[str] = None, preview: Optional[str] = None, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            if not name and not preview:
                return
            _append_event(
                project.id,
                StreamEvent(
                    id=_new_id("evt"),
                    projectId=project.id,
                    runId=run.id if run else None,
                    nodeId=node.id if node else None,
                    type="tool_call",
                    label=str(name or event_type),
                    summary=_truncate_text(str(preview or event_type), 500),
                    details={"eventType": event_type, "args": _safe_details(args or {}), **_safe_details(kwargs)},
                    status="info",
                ),
            )

        agent = self._make_agent(
            project=project,
            session_id=session_id,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            model_override=model_override,
            tool_start_callback=tool_start,
            tool_complete_callback=tool_complete,
            tool_progress_callback=tool_progress,
        )
        history = _load_workflow_agent_history(project, session_id)
        _append_workflow_agent_message(
            project,
            session_id=session_id,
            role="user",
            content=prompt,
            node=node,
            run=run,
            metadata={"label": message_label},
        )
        tokens = _set_workflow_session_context(session_id, project.root)
        try:
            result = agent.run_conversation(
                prompt,
                conversation_history=history,
                stream_callback=append_stream_delta,
                task_id=session_id,
            )
        finally:
            _clear_workflow_session_context(tokens)

        if isinstance(result, dict):
            text = str(result.get("final_response") or "")
            status = "error" if result.get("error") else "complete"
            if not text and result.get("error"):
                text = f"Error: {result.get('error')}"
            parsed = WorkflowAgentResult(
                text=_strip_reasoning(text),
                session_id=getattr(agent, "session_id", session_id) or session_id,
                status=status,
                usage=result.get("usage") if isinstance(result.get("usage"), dict) else None,
                messages=result.get("messages") if isinstance(result.get("messages"), list) else None,
            )
        else:
            parsed = WorkflowAgentResult(text=_strip_reasoning(str(result)), session_id=session_id)

        if parsed.text:
            _append_workflow_agent_message(
                project,
                session_id=session_id,
                role="assistant",
                content=parsed.text,
                node=node,
                run=run,
                metadata={"label": message_label, "status": parsed.status, "usage": parsed.usage or {}},
            )

        if persist_final:
            _append_event(
                project.id,
                StreamEvent(
                    id=_new_id("evt"),
                    projectId=project.id,
                    runId=run.id if run else None,
                    nodeId=node.id if node else None,
                    type="ai_reply",
                    label=message_label,
                    summary=_truncate_text(parsed.text, 1800),
                    details={
                        "messageId": message_id,
                        "text": parsed.text,
                        "final": True,
                        "usage": parsed.usage or {},
                        "rawReasoningExposed": False,
                    },
                    status="error" if parsed.status == "error" else "success",
                    durationMs=int((time.time() - started_at) * 1000),
                ),
            )

        return parsed

    def _make_agent(
        self,
        *,
        project: WorkflowProject,
        session_id: str,
        system_prompt: Optional[str],
        max_iterations: int,
        model_override: Optional[str],
        tool_start_callback: Callable[..., None],
        tool_complete_callback: Callable[..., None],
        tool_progress_callback: Callable[..., None],
    ) -> Any:
        from run_agent import AIAgent
        from tui_gateway.server import (
            _load_enabled_toolsets,
            _load_reasoning_config,
            _load_service_tier,
            _resolve_startup_runtime,
        )
        from hermes_cli.runtime_provider import resolve_runtime_provider

        model, requested_provider = _resolve_startup_runtime()
        if model_override:
            model = model_override
        runtime = resolve_runtime_provider(requested=requested_provider, target_model=model or None)
        enabled_toolsets = [
            toolset for toolset in _load_enabled_toolsets()
            if str(toolset) not in _WORKFLOW_DISABLED_TOOLSETS
        ]
        return AIAgent(
            model=model,
            max_iterations=max_iterations,
            provider=runtime.get("provider"),
            base_url=runtime.get("base_url"),
            api_key=runtime.get("api_key"),
            api_mode=runtime.get("api_mode"),
            acp_command=runtime.get("command"),
            acp_args=runtime.get("args"),
            credential_pool=runtime.get("credential_pool"),
            quiet_mode=True,
            verbose_logging=False,
            reasoning_config=_load_reasoning_config(),
            service_tier=_load_service_tier(),
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=sorted(_WORKFLOW_DISABLED_TOOLSETS),
            platform="workflow",
            session_id=session_id,
            session_db=None,
            skip_memory=True,
            ephemeral_system_prompt=system_prompt,
            gateway_session_key=session_id,
            pass_session_id=True,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            tool_progress_callback=tool_progress_callback,
        )


_agent_runner: WorkflowAgentRunner = WorkflowAgentRunner()


def create_workflow_router(ws_auth_ok: Callable[[WebSocket], bool]) -> APIRouter:
    router = APIRouter(prefix="/api/workflows", tags=["workflows"])

    @router.get("/projects")
    async def list_projects(include_archived: bool = False) -> Dict[str, Any]:
        projects = [_load_project_by_path(path) for path in _read_registry().values()]
        projects = [project for project in projects if project is not None]
        if not include_archived:
            projects = [project for project in projects if not project.archived]
        projects.sort(key=lambda project: project.updatedAt, reverse=True)
        return {"projects": projects}

    @router.post("/intake/start")
    async def start_intake(body: WorkflowIntakeStartRequest) -> Dict[str, Any]:
        return _start_intake_session(body)

    @router.post("/intake/{intake_id}/message")
    async def message_intake(intake_id: str, body: WorkflowIntakeMessageRequest) -> Dict[str, Any]:
        return _message_intake_session(intake_id, body.message)

    @router.post("/intake/{intake_id}/answers")
    async def answer_intake(intake_id: str, body: WorkflowIntakeAnswersRequest) -> Dict[str, Any]:
        return _answer_intake_session(intake_id, body.answers)

    @router.post("/intake/{intake_id}/confirm")
    async def confirm_intake(intake_id: str, body: WorkflowIntakeConfirmRequest) -> ProjectBundle:
        try:
            project, state = _intake_project_and_state(intake_id, body.projectId)
            summary = body.summary.strip() or str(state.get("summary") or "")
            _finalize_intake_project(project, state, body, summary)
        except HTTPException as exc:
            if exc.status_code != 404 or body.projectId:
                raise
            summary = body.summary.strip() or _intake_summary_for(intake_id)
            project = _create_legacy_intake_project(body, intake_id, summary)
        error = _generate_workflow_for_project(project, reason="workflow_generate")
        return _project_bundle(project.id, error=error)

    @router.post("/projects")
    async def create_project(body: ProjectCreateRequest) -> ProjectBundle:
        project = _create_project(body)
        _register_project(project)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                type="process_summary",
                label="项目已创建",
                summary="已初始化项目结构、Workflow manifest、SQLite 事件库和 Git 仓库。",
                status="success",
            ),
        )
        _create_snapshot(project, "Project initialized", "project_init")
        error = _generate_workflow_for_project(project, reason="workflow_generate")
        return _project_bundle(project.id, error=error)

    @router.post("/projects/open")
    async def open_project(body: ProjectOpenRequest) -> ProjectBundle:
        project = _load_project_by_path(Path(body.root))
        if project is None:
            raise HTTPException(status_code=404, detail="Workflow project not found")
        project.lastOpenedAt = time.time()
        _save_project(project)
        _register_project(project)
        return _project_bundle(project.id)

    @router.get("/projects/{project_id}")
    async def get_project(project_id: str) -> ProjectBundle:
        return _project_bundle(project_id)

    @router.patch("/projects/{project_id}")
    async def patch_project(project_id: str, body: ProjectPatchRequest) -> ProjectBundle:
        project = _load_project(project_id)
        changed = False
        if body.name is not None:
            name = body.name.strip()
            if not name:
                raise HTTPException(status_code=422, detail="Project name is required")
            if name != project.name:
                project.name = name
                changed = True
        if body.archived is not None and bool(body.archived) != project.archived:
            project.archived = bool(body.archived)
            changed = True
        if changed:
            _touch_project(project)
            _append_event(
                project.id,
                StreamEvent(
                    id=_new_id("evt"),
                    projectId=project.id,
                    type="process_summary",
                    label="项目设置已更新",
                    summary="Workflow 项目名称或归档状态已更新。",
                    status="success",
                ),
            )
        return _project_bundle(project.id)

    @router.delete("/projects/{project_id}")
    async def remove_project_from_history(project_id: str) -> Dict[str, Any]:
        return _remove_project_from_history(project_id)

    @router.post("/projects/{project_id}/remove-from-history")
    async def remove_project_from_history_post(project_id: str) -> Dict[str, Any]:
        return _remove_project_from_history(project_id)

    @router.get("/projects/{project_id}/export")
    async def export_project(project_id: str) -> FileResponse:
        project = _load_project(project_id)
        archive_path = _create_project_export_zip(project)
        return FileResponse(
            archive_path,
            filename=f"{_slug(project.name) or 'workflow-project'}-{project.id[:8]}.zip",
            media_type="application/zip",
            background=BackgroundTask(_safe_unlink, archive_path),
        )

    @router.post("/projects/{project_id}/generate")
    async def generate_workflow(project_id: str) -> ProjectBundle:
        project = _load_project(project_id)
        error = _generate_workflow_for_project(project, reason="workflow_generate")
        if error:
            return _project_bundle(project.id, error=error)
        return _project_bundle(project.id)

    @router.put("/projects/{project_id}/workflow")
    async def update_workflow(project_id: str, body: WorkflowUpdateRequest) -> ProjectBundle:
        project = _load_project(project_id)
        workflow = body.workflow
        workflow.updatedAt = time.time()
        _validate_workflow(workflow)
        _save_workflow(project, workflow)
        _touch_project(project)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                type="stage_result",
                label="Workflow 已保存",
                summary="Canvas 节点、边和位置已经写入 workflow.flow.json。",
                status="success",
            ),
        )
        _create_snapshot(project, body.snapshotLabel, "workflow_edit")
        return _project_bundle(project.id)

    @router.post("/projects/{project_id}/runs")
    async def start_run(project_id: str, body: RunCreateRequest) -> Dict[str, Any]:
        project = _load_project(project_id)
        workflow = _load_workflow(project)
        if not workflow.nodes:
            error = _generate_workflow_for_project(project, reason="workflow_generate_before_run")
            if error:
                raise HTTPException(status_code=422, detail=f"Workflow is empty and generation failed: {error}")
            workflow = _load_workflow(project)
        if not workflow.nodes:
            raise HTTPException(status_code=422, detail="Workflow is empty. Generate a workflow before running.")
        _validate_workflow(workflow)

        run = ExecutionRun(
            id=_new_id("run"),
            projectId=project.id,
            mode=body.mode,
            maxConcurrency=max(1, min(8, int(body.maxConcurrency or 2))),
            status="running",
        )
        _save_run(run)
        project.currentRunId = run.id
        _touch_project(project)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                type="process_summary",
                label="执行已启动",
                summary=f"运行模式：{_mode_label(run.mode)}，并发上限：{run.maxConcurrency}。",
                status="success",
            ),
        )
        _runtime.start(project, run)
        return {"ok": True, "run": run, "project": project}

    @router.post("/runs/{run_id}/pause")
    async def pause_run(run_id: str) -> Dict[str, Any]:
        run = _load_run(run_id)
        project = _load_project(run.projectId)
        run.status = "paused"
        run.updatedAt = time.time()
        _save_run(run)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                type="process_summary",
                label="执行已暂停",
                summary="当前 run 已暂停，恢复后会从持久化状态继续调度 ready 节点。",
            ),
        )
        return {"ok": True, "run": run}

    @router.post("/runs/{run_id}/resume")
    async def resume_run(run_id: str) -> Dict[str, Any]:
        run = _load_run(run_id)
        project = _load_project(run.projectId)
        run.status = "running"
        run.updatedAt = time.time()
        _save_run(run)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                type="process_summary",
                label="执行已恢复",
                summary="调度器将继续检查依赖和 ready queue。",
                status="success",
            ),
        )
        _runtime.start(project, run)
        return {"ok": True, "run": run}

    @router.post("/runs/{run_id}/stop")
    async def stop_run(run_id: str) -> Dict[str, Any]:
        run = _load_run(run_id)
        project = _load_project(run.projectId)
        _runtime.stop(run_id)
        run.status = "stopped"
        run.completedAt = time.time()
        run.updatedAt = run.completedAt
        _save_run(run)
        workflow = _load_workflow(project)
        changed = False
        for node in workflow.nodes:
            if node.status in {"queued", "running", "reviewing"}:
                node.status = "aborted"
                changed = True
        if changed:
            _save_workflow(project, workflow)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                type="process_summary",
                label="执行已停止",
                summary="已请求停止当前 run，未完成节点被标记为 aborted。",
                status="warning",
            ),
        )
        return {"ok": True, "run": run}

    @router.post("/runs/{run_id}/nodes/{node_id}/confirm")
    async def confirm_node(run_id: str, node_id: str) -> Dict[str, Any]:
        run = _load_run(run_id)
        project = _load_project(run.projectId)
        workflow = _load_workflow(project)
        node = _node_by_id(workflow, node_id)
        if node.status not in {"waiting_user_confirm", "reviewing"}:
            raise HTTPException(status_code=409, detail="Node is not waiting for confirmation")
        node.status = "completed"
        node.outputs["confirmedAt"] = time.time()
        _promote_ready_nodes(workflow)
        _save_workflow(project, workflow)
        run.status = "running"
        run.updatedAt = time.time()
        _save_run(run)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                nodeId=node.id,
                type="approval",
                label="节点已确认",
                summary=f"已确认「{node.title}」，调度器继续推进下游节点。",
                status="success",
            ),
        )
        _create_snapshot(project, f"Confirmed {node.title}", "user_confirm")
        _runtime.start(project, run)
        return {"ok": True, "run": run, "workflow": workflow}

    @router.post("/runs/{run_id}/nodes/{node_id}/retry")
    async def retry_node(run_id: str, node_id: str) -> Dict[str, Any]:
        run = _load_run(run_id)
        project = _load_project(run.projectId)
        workflow = _load_workflow(project)
        node = _node_by_id(workflow, node_id)
        node.status = "retrying"
        node.retryCount += 1
        node.outputs.pop("completedAt", None)
        _save_workflow(project, workflow)
        run.status = "running"
        run.updatedAt = time.time()
        _save_run(run)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                nodeId=node.id,
                type="node_status",
                label="节点重试",
                summary=f"「{node.title}」已重新加入 ready queue。",
                status="warning",
            ),
        )
        node.status = "ready"
        _save_workflow(project, workflow)
        _runtime.start(project, run)
        return {"ok": True, "workflow": workflow}

    @router.post("/runs/{run_id}/nodes/{node_id}/skip")
    async def skip_node(run_id: str, node_id: str) -> Dict[str, Any]:
        run = _load_run(run_id)
        project = _load_project(run.projectId)
        workflow = _load_workflow(project)
        node = _node_by_id(workflow, node_id)
        node.status = "skipped"
        node.outputs["skippedAt"] = time.time()
        _promote_ready_nodes(workflow)
        _save_workflow(project, workflow)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                nodeId=node.id,
                type="node_status",
                label="节点已跳过",
                summary=f"「{node.title}」已跳过，下游可选路径会继续评估。",
                status="warning",
            ),
        )
        _runtime.start(project, run)
        return {"ok": True, "workflow": workflow}

    @router.post("/projects/{project_id}/chat")
    async def workflow_chat(project_id: str, body: ChatRequest) -> Dict[str, Any]:
        project = _load_project(project_id)
        workflow = _load_workflow(project)
        node = _node_by_id(workflow, body.nodeId) if body.nodeId else None
        target = f"节点「{node.title}」" if node else "当前 Workflow"
        patch: Dict[str, Any]
        reply: str
        try:
            result = _agent_runner.run(
                project=project,
                prompt=_workflow_chat_prompt(project, workflow, body, node),
                session_id=_ensure_project_agent_session(project),
                node=node,
                run=_load_run(project.currentRunId) if project.currentRunId else None,
                system_prompt=_workflow_chat_system_prompt(),
                max_iterations=14,
                message_label="工作台对话",
                persist_final=False,
            )
            project.agentSessionId = result.session_id
            _touch_project(project)
            reply, patch = _chat_result_from_agent_text(result.text, workflow, body)
        except Exception as exc:
            reply = f"Workflow chat failed: {exc}"
            patch = {"op": "error", "message": str(exc), "nodeId": body.nodeId}
            _append_event(
                project.id,
                StreamEvent(
                    id=_new_id("evt"),
                    projectId=project.id,
                    runId=project.currentRunId,
                    nodeId=body.nodeId,
                    type="error",
                    label="工作台对话失败",
                    summary=_truncate_text(str(exc), 1200),
                    details={"target": body.nodeId},
                    status="error",
                ),
            )
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=project.currentRunId,
                nodeId=body.nodeId,
                type="ai_reply",
                label="AI 工作台回复",
                summary=_truncate_text(reply or f"已收到面向{target}的调整请求，并生成结构化 patch proposal。", 1800),
                details={
                    "text": reply,
                    "userText": body.text,
                    "patch": patch,
                    "attachments": body.attachments,
                    "skillIds": body.skillIds,
                    "rawReasoningExposed": False,
                },
                status="error" if patch.get("op") == "error" else "success",
            ),
        )
        return {"ok": patch.get("op") != "error", "target": body.nodeId, "patch": patch, "reply": reply}

    @router.get("/projects/{project_id}/composer/slash")
    async def workflow_slash_catalog(project_id: str, q: str = "") -> Dict[str, Any]:
        _load_project(project_id)
        return {"items": _workflow_slash_catalog(q)}

    @router.post("/projects/{project_id}/composer/slash")
    async def workflow_slash_execute(project_id: str, body: ComposerSlashRequest) -> Dict[str, Any]:
        project = _load_project(project_id)
        workflow = _load_workflow(project)
        result = _execute_workflow_slash(project, workflow, body)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                nodeId=body.nodeId,
                type="process_summary",
                label="Slash 命令",
                summary=_truncate_text(result.get("output", ""), 1200),
                details={"command": body.command, "ok": result.get("ok", False)},
                status="success" if result.get("ok") else "warning",
            ),
        )
        return result

    @router.post("/projects/{project_id}/composer/complete")
    async def workflow_composer_complete(project_id: str, body: ComposerCompletionRequest) -> Dict[str, Any]:
        project = _load_project(project_id)
        return {"items": _workflow_completions(project, body)}

    @router.post("/projects/{project_id}/composer/attachments")
    async def workflow_composer_attachments(project_id: str, body: ComposerAttachmentRequest) -> Dict[str, Any]:
        project = _load_project(project_id)
        refs = _load_references(project)
        existing = {ref.path for ref in refs}
        added: List[ReferenceItem] = []
        for path in body.paths:
            if path and path not in existing:
                ref = _reference_from_path(path)
                refs.append(ref)
                added.append(ref)
                existing.add(ref.path)
        if added:
            _write_json(_meta_path(project, "references.manifest.json"), [ref.model_dump() for ref in refs])
            _touch_project(project)
        return {"ok": True, "attachments": [ref.model_dump() for ref in added], "references": [ref.model_dump() for ref in refs]}

    @router.get("/projects/{project_id}/events")
    async def list_events(project_id: str, since: Optional[float] = None, limit: int = 200) -> Dict[str, Any]:
        project = _load_project(project_id)
        return {"events": _read_events(project, since=since, limit=max(1, min(1000, limit)))}

    @router.websocket("/projects/{project_id}/events")
    async def workflow_events_ws(websocket: WebSocket, project_id: str) -> None:
        if not ws_auth_ok(websocket):
            await websocket.close(code=4401)
            return
        project = _load_project(project_id)
        await websocket.accept()
        since = _float_or_none(websocket.query_params.get("since"))
        try:
            while True:
                events = _read_events(project, since=since, limit=100)
                for event in events:
                    await websocket.send_json(event.model_dump())
                    since = max(since or 0, event.timestamp)
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    @router.get("/projects/{project_id}/snapshots")
    async def list_snapshots(project_id: str) -> Dict[str, Any]:
        project = _load_project(project_id)
        return {"snapshots": _list_snapshots(project)}

    @router.post("/projects/{project_id}/snapshots")
    async def create_snapshot(project_id: str) -> Dict[str, Any]:
        project = _load_project(project_id)
        snapshot = _create_snapshot(project, "Manual snapshot", "manual")
        return {"ok": True, "snapshot": snapshot}

    @router.post("/projects/{project_id}/snapshots/restore")
    async def restore_snapshot(project_id: str, body: RestoreSnapshotRequest) -> ProjectBundle:
        project = _load_project(project_id)
        _git(project.root, "checkout", body.commit, "--", WORKFLOW_DIR)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                type="snapshot",
                label="快照已恢复",
                summary=f"已从 {body.commit[:8]} 恢复 .agent-workflow 配置。",
                status="warning",
            ),
        )
        _create_snapshot(project, "Restore checkpoint", "restore_checkpoint")
        return _project_bundle(project.id)

    @router.get("/projects/{project_id}/references")
    async def get_references(project_id: str) -> Dict[str, Any]:
        project = _load_project(project_id)
        return {"references": _load_references(project)}

    @router.put("/projects/{project_id}/references")
    async def update_references(project_id: str, body: ReferencesUpdateRequest) -> ProjectBundle:
        project = _load_project(project_id)
        _write_json(_meta_path(project, "references.manifest.json"), [ref.model_dump() for ref in body.references])
        _touch_project(project)
        _create_snapshot(project, "References updated", "references_update")
        return _project_bundle(project.id)

    @router.get("/projects/{project_id}/skills")
    async def get_skills(project_id: str) -> Dict[str, Any]:
        project = _load_project(project_id)
        return {"skills": _load_skills(project)}

    @router.put("/projects/{project_id}/skills")
    async def update_skills(project_id: str, body: SkillsUpdateRequest) -> ProjectBundle:
        project = _load_project(project_id)
        _write_json(_meta_path(project, "skills.config.json"), [skill.model_dump() for skill in body.skills])
        _touch_project(project)
        _create_snapshot(project, "Skills updated", "skills_update")
        return _project_bundle(project.id)

    @router.get("/projects/{project_id}/files")
    async def get_project_files(project_id: str) -> Dict[str, Any]:
        project = _load_project(project_id)
        return {"tree": _file_tree(Path(project.root), max_depth=3)}

    return router


def _run_engine(project_id: str, run_id: str, stop: threading.Event) -> None:
    project = _load_project(project_id)
    lock = _runtime.lock_for(project.id)
    with lock:
        while not stop.is_set():
            run = _load_run(run_id)
            if run.status in {"paused", "waiting_user_confirm", "completed", "failed", "stopped"}:
                return
            workflow = _load_workflow(project)
            _promote_ready_nodes(workflow)
            ready = [node for node in workflow.nodes if node.status in {"ready", "retrying"}]
            if not ready:
                if _workflow_is_done(workflow):
                    run.status = "completed"
                    run.currentNodeId = None
                    run.completedAt = time.time()
                    run.updatedAt = run.completedAt
                    _save_run(run)
                    _append_event(
                        project.id,
                        StreamEvent(
                            id=_new_id("evt"),
                            projectId=project.id,
                            runId=run.id,
                            type="stage_result",
                            label="Workflow 已完成",
                            summary="所有必要节点已经完成、跳过或终止。",
                            status="success",
                        ),
                    )
                    _create_snapshot(project, "Workflow completed", "final_delivery")
                return

            batch = ready[: max(1, run.maxConcurrency)]
            for node in batch:
                if stop.is_set():
                    return
                run = _load_run(run_id)
                if run.status != "running":
                    return
                _execute_node(project, workflow, run, node)
                workflow = _load_workflow(project)
                node = _node_by_id(workflow, node.id)
                if node.status == "waiting_user_confirm":
                    run.status = "waiting_user_confirm"
                    run.currentNodeId = node.id
                    run.updatedAt = time.time()
                    _save_run(run)
                    return


def _execute_node(project: WorkflowProject, workflow: Workflow, run: ExecutionRun, node: WorkflowNode) -> None:
    started = time.time()
    before_change_paths = _git_change_paths(project)
    node.status = "queued"
    _save_workflow(project, workflow)
    _append_node_status(project, run, node, "节点已排队", "等待调度器执行。")

    node.status = "running"
    _save_workflow(project, workflow)
    run.currentNodeId = node.id
    run.updatedAt = time.time()
    _save_run(run)
    _append_node_status(project, run, node, "节点执行中", "已装载项目目标、上游输出、启用 references 和 skill bindings。")
    _append_event(
        project.id,
        StreamEvent(
            id=_new_id("evt"),
            projectId=project.id,
            runId=run.id,
            nodeId=node.id,
            type="tool_call",
            label="上下文装载",
            summary="读取节点输入、父节点产物和项目 manifest，形成 Hermes Agent runner prompt。",
            details={
                "skills": _effective_node_skills(project, node),
                "skillMode": node.skillMode,
                "model": node.modelOverride or node.model,
                "references": node.references,
                "referencePolicy": "enabled_only",
            },
        ),
    )
    node.lastRunId = run.id
    session_id = _ensure_node_agent_session(project, node)
    _save_workflow(project, workflow)
    prompt = _node_execution_prompt(project, workflow, run, node)
    try:
        result = _agent_runner.run(
            project=project,
            prompt=prompt,
            session_id=session_id,
            node=node,
            run=run,
            system_prompt=_node_execution_system_prompt(project, node),
            max_iterations=max(12, min(60, 12 + len(_effective_node_skills(project, node)) * 4)),
            model_override=node.modelOverride or node.model,
            message_label=f"节点输出：{node.title}",
        )
        node.agentSessionId = result.session_id
        final_text = _strip_reasoning(result.text)
    except Exception as exc:
        node.status = "failed"
        node.retryCount += 1
        node.outputs.update({"error": str(exc), "failedAt": time.time()})
        _save_workflow(project, workflow)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                nodeId=node.id,
                type="error",
                label="节点执行失败",
                summary=_truncate_text(str(exc), 1200),
                details={"agentSessionId": session_id, "rawReasoningExposed": False},
                status="error",
                durationMs=int((time.time() - started) * 1000),
            ),
        )
        return

    artifact = _write_node_artifact(project, run, node, final_text)
    node.artifacts = list(dict.fromkeys([*node.artifacts, artifact.path]))
    node.fileChanges = _collect_node_file_changes(project, before_change_paths)
    node.outputs.update(
        {
            "summary": _truncate_text(final_text, 1600),
            "artifact": artifact.path,
            "fileChanges": [change.model_dump() for change in node.fileChanges],
            "completedAt": time.time(),
            "agentSessionId": node.agentSessionId,
        }
    )
    _append_event(
        project.id,
        StreamEvent(
            id=_new_id("evt"),
            projectId=project.id,
            runId=run.id,
            nodeId=node.id,
            type="stage_result",
            label="阶段结果",
            summary=f"「{node.title}」已生成节点产物：{artifact.name}。",
            details={"artifact": artifact.model_dump()},
            status="success",
            durationMs=int((time.time() - started) * 1000),
        ),
    )

    if _node_needs_confirmation(run, node):
        node.status = "waiting_user_confirm"
        _save_workflow(project, workflow)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                runId=run.id,
                nodeId=node.id,
                type="approval",
                label="等待用户确认",
                summary=f"「{node.title}」需要确认后才能推进下游节点。",
                details={"reviewRules": node.reviewRules.model_dump(), "mode": run.mode},
                status="warning",
            ),
        )
        _create_snapshot(project, f"Review passed {node.title}", "review_pass")
        return

    node.status = "completed"
    _promote_ready_nodes(workflow)
    _save_workflow(project, workflow)
    _append_node_status(project, run, node, "节点已完成", "下游依赖已重新计算。", status="success")
    _create_snapshot(project, f"Completed {node.title}", "node_complete")


def _start_intake_session(body: WorkflowIntakeStartRequest) -> Dict[str, Any]:
    project = _create_project(ProjectCreateRequest(name=body.name, goal=body.goal, root=body.root, references=body.references))
    project.status = "clarifying"
    _touch_project(project)
    _append_event(
        project.id,
        StreamEvent(
            id=_new_id("evt"),
            projectId=project.id,
            type="process_summary",
            label="Intake started",
            summary="Workflow project created as a clarification draft.",
            details={"intake": True, "references": len(body.references), "rawReasoningExposed": False},
            status="success",
        ),
    )
    _create_snapshot(project, "Project initialized", "project_init")
    intake_id = _new_id("intake")
    state = {
        "id": intake_id,
        "projectId": project.id,
        "name": project.name,
        "goal": body.goal,
        "root": project.root,
        "references": list(body.references),
        "turns": 0,
        "summary": "",
        "ready": False,
        "currentBatch": None,
        "batches": [],
        "answeredCount": 0,
        "messages": [],
    }
    _remember_intake_state(project, state)
    return _advance_intake_with_agent(project, state, trigger="start")


def _message_intake_session(intake_id: str, message: str) -> Dict[str, Any]:
    text = _strip_reasoning(message)
    if not text:
        raise HTTPException(status_code=422, detail="Intake message is required")
    project, state = _intake_project_and_state(intake_id)
    _append_intake_message(state, "user", text)
    _remember_intake_state(project, state)
    return _advance_intake_with_agent(project, state, trigger="message", user_message=text)


def _answer_intake_session(intake_id: str, answers: List[WorkflowIntakeAnswer]) -> Dict[str, Any]:
    if not answers:
        raise HTTPException(status_code=422, detail="At least one intake answer is required")
    project, state = _intake_project_and_state(intake_id)
    current_batch = state.get("currentBatch") if isinstance(state.get("currentBatch"), dict) else None
    questions = current_batch.get("questions", []) if current_batch else []
    expected_ids = {str(question.get("id") or "") for question in questions if question.get("id")}
    answer_ids = {answer.questionId for answer in answers if answer.questionId}
    if expected_ids and expected_ids != answer_ids:
        raise HTTPException(status_code=422, detail="All questions in the current clarification batch must be answered")
    answer_payload = [answer.model_dump() for answer in answers]
    if current_batch:
        current_batch["answers"] = answer_payload
        state.setdefault("batches", []).append(current_batch)
    state["currentBatch"] = None
    state["answeredCount"] = int(state.get("answeredCount") or 0) + len(answers)
    answer_summary = _format_intake_answers_for_user(questions, answers)
    _append_intake_message(state, "user", answer_summary)
    _remember_intake_state(project, state)
    return _advance_intake_with_agent(project, state, trigger="answers", answers=answer_payload)


def _intake_response(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "intakeId": state["id"],
        "projectId": state.get("projectId"),
        "messages": state.get("messages", []),
        "ready": bool(state.get("ready")),
        "summary": state.get("summary") or _build_intake_summary(state),
        "currentBatch": state.get("currentBatch"),
        "answeredCount": int(state.get("answeredCount") or 0),
        "error": state.get("error") or None,
    }


def _workflow_intake_system_prompt() -> str:
    return (
        "You are Hermes Workflow Intake, a planning clarifier for an AI-agent workflow workbench. "
        "Analyze the user's objective, references, inputs, deliverables, acceptance criteria, risks, "
        "preferred models/skills/tools, and human confirmation needs. Ask only clarification questions "
        "that materially improve workflow planning. Return only JSON. Never expose hidden reasoning or chain-of-thought."
    )


def _workflow_intake_prompt(
    project: WorkflowProject,
    state: Dict[str, Any],
    *,
    trigger: str,
    user_message: Optional[str] = None,
    answers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    references = [ref.model_dump() for ref in _load_references(project) if ref.enabled]
    transcript = [
        {"role": item.get("role"), "content": _truncate_text(str(item.get("content") or ""), 1600)}
        for item in state.get("messages", [])[-12:]
        if item.get("content")
    ]
    schema = {
        "reply": "assistant message shown above the interactive questions",
        "ready": False,
        "summary": "planning summary when ready, otherwise best current summary",
        "questions": [
            {
                "id": "stable-question-id",
                "question": "one decision the user must clarify",
                "detail": "why this matters",
                "options": [
                    {"id": "a", "label": "recommended option", "description": "tradeoff", "priority": 1},
                    {"id": "b", "label": "second option", "description": "tradeoff", "priority": 2},
                    {"id": "c", "label": "third option", "description": "tradeoff", "priority": 3},
                ],
            }
        ],
    }
    return "\n".join(
        [
            "Create the next clarification batch for this Hermes Workflow project.",
            "Each question must include exactly three options ranked by priority. The user can still type a custom answer.",
            "If enough information is available, return ready=true, an empty questions array, and a concrete planning summary.",
            "If more information is needed, return ready=false and one or more questions.",
            "",
            f"Trigger: {trigger}",
            f"Project name: {project.name}",
            f"Project root: {project.root}",
            f"Initial goal:\n{project.goal or state.get('goal') or 'No goal provided.'}",
            f"Enabled references JSON:\n{json.dumps(references, ensure_ascii=False, indent=2)}",
            f"Prior transcript JSON:\n{json.dumps(transcript, ensure_ascii=False, indent=2)}",
            f"Latest freeform user message:\n{user_message or ''}",
            f"Latest structured answers JSON:\n{json.dumps(answers or [], ensure_ascii=False, indent=2)}",
            f"Current summary:\n{state.get('summary') or ''}",
            "",
            "Return JSON matching this exact shape:",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )


def _remember_intake_state(project: WorkflowProject, state: Dict[str, Any]) -> None:
    state["projectId"] = project.id
    state["updatedAt"] = time.time()
    with _intake_sessions_lock:
        _intake_sessions[str(state["id"])] = state
    _write_json(_intake_state_path(project), state)


def _intake_state_path(project: WorkflowProject) -> Path:
    return _meta_path(project, "intake.state.json")


def _read_intake_state(project: WorkflowProject) -> Optional[Dict[str, Any]]:
    path = _intake_state_path(project)
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _intake_project_and_state(intake_id: str, project_id: Optional[str] = None) -> tuple[WorkflowProject, Dict[str, Any]]:
    if project_id:
        project = _load_project(project_id)
        state = _read_intake_state(project)
        if state is None:
            raise HTTPException(status_code=404, detail="Workflow intake state not found")
        if str(state.get("id") or intake_id) != intake_id:
            raise HTTPException(status_code=404, detail="Workflow intake id mismatch")
        with _intake_sessions_lock:
            _intake_sessions[intake_id] = state
        return project, state

    with _intake_sessions_lock:
        cached = _intake_sessions.get(intake_id)
    if cached and cached.get("projectId"):
        project = _load_project(str(cached["projectId"]))
        state = _read_intake_state(project) or cached
        with _intake_sessions_lock:
            _intake_sessions[intake_id] = state
        return project, state

    for root in _read_registry().values():
        project = _load_project_by_path(Path(root))
        if project is None:
            continue
        state = _read_intake_state(project)
        if state and str(state.get("id") or "") == intake_id:
            with _intake_sessions_lock:
                _intake_sessions[intake_id] = state
            return project, state

    raise HTTPException(status_code=404, detail="Workflow intake state not found")


def _append_intake_message(state: Dict[str, Any], role: str, content: str) -> None:
    text = _strip_reasoning(content).strip()
    if role not in {"assistant", "user"} or not text:
        return
    state.setdefault("messages", []).append({"role": role, "content": text, "timestamp": time.time()})


def _advance_intake_with_agent(
    project: WorkflowProject,
    state: Dict[str, Any],
    *,
    trigger: str,
    user_message: Optional[str] = None,
    answers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    state["error"] = None
    prompt = _workflow_intake_prompt(project, state, trigger=trigger, user_message=user_message, answers=answers)
    try:
        result = _agent_runner.run(
            project=project,
            prompt=prompt,
            session_id=f"workflow-intake-{project.id}",
            system_prompt=_workflow_intake_system_prompt(),
            max_iterations=8,
            message_label="Workflow clarification",
            persist_final=False,
        )
        output = _intake_output_from_agent_text(project, state, result.text)
    except Exception as first_error:
        try:
            repair_prompt = _workflow_intake_repair_prompt(project, state, str(first_error))
            repaired = _agent_runner.run(
                project=project,
                prompt=repair_prompt,
                session_id=f"workflow-intake-{project.id}",
                system_prompt=_workflow_intake_system_prompt(),
                max_iterations=4,
                message_label="Workflow clarification repair",
                persist_final=False,
            )
            output = _intake_output_from_agent_text(project, state, repaired.text)
        except Exception as repair_error:
            message = _truncate_text(str(repair_error), 1000)
            state["error"] = message
            _append_intake_message(
                state,
                "assistant",
                "Workflow clarification could not parse the model response. Add more planning details or retry clarification.",
            )
            _append_event(
                project.id,
                StreamEvent(
                    id=_new_id("evt"),
                    projectId=project.id,
                    type="error",
                    label="Workflow clarification failed",
                    summary=message,
                    details={"stage": "intake", "rawReasoningExposed": False},
                    status="error",
                ),
            )
            _remember_intake_state(project, state)
            return _intake_response(state)

    reply = output.get("reply", "")
    if reply:
        _append_intake_message(state, "assistant", reply)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                type="ai_reply",
                label="Workflow clarification",
                summary=_truncate_text(reply, 1800),
                details={"text": reply, "final": True, "rawReasoningExposed": False},
                status="success",
            ),
        )

    summary = str(output.get("summary") or "").strip()
    if summary:
        state["summary"] = _strip_reasoning(summary)

    batch = output.get("batch")
    if batch:
        state["currentBatch"] = batch
        state["ready"] = False
    else:
        state["currentBatch"] = None
        state["ready"] = bool(output.get("ready"))

    state["turns"] = int(state.get("turns") or 0) + 1
    _remember_intake_state(project, state)
    return _intake_response(state)


def _workflow_intake_repair_prompt(project: WorkflowProject, state: Dict[str, Any], error: str) -> str:
    schema = {
        "reply": "assistant message shown above the interactive questions",
        "ready": False,
        "summary": "current or final planning summary",
        "questions": [
            {
                "id": "stable-question-id",
                "question": "one decision the user must clarify",
                "detail": "why this matters",
                "options": [
                    {"id": "a", "label": "recommended option", "description": "tradeoff", "priority": 1},
                    {"id": "b", "label": "second option", "description": "tradeoff", "priority": 2},
                    {"id": "c", "label": "third option", "description": "tradeoff", "priority": 3},
                ],
            }
        ],
    }
    return "\n".join(
        [
            "Repair the previous workflow intake response. Return valid JSON only.",
            f"Project: {project.name}",
            f"Goal:\n{project.goal or state.get('goal') or ''}",
            f"Parse error:\n{error}",
            "Use this exact schema and ensure every question has exactly three options:",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )


def _intake_output_from_agent_text(project: WorkflowProject, state: Dict[str, Any], text: str) -> Dict[str, Any]:
    data = _extract_json_object(text)
    reply = _strip_reasoning(str(data.get("reply") or "")).strip()
    summary = _strip_reasoning(str(data.get("summary") or "")).strip()
    questions_raw = data.get("questions")
    if questions_raw is None:
        questions_raw = []
    if not isinstance(questions_raw, list):
        raise ValueError("Workflow intake JSON field 'questions' must be a list")

    questions: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw_question in enumerate(questions_raw):
        if not isinstance(raw_question, dict):
            raise ValueError("Workflow intake question must be an object")
        question_text = _strip_reasoning(str(raw_question.get("question") or "")).strip()
        if not question_text:
            raise ValueError("Workflow intake question is missing text")
        question_id = _unique_id(_slug(str(raw_question.get("id") or question_text)) or f"question-{index + 1}", used_ids)
        used_ids.add(question_id)
        options_raw = raw_question.get("options")
        if not isinstance(options_raw, list) or len(options_raw) != 3:
            raise ValueError("Each workflow intake question must include exactly three options")
        options: List[Dict[str, Any]] = []
        used_option_ids: set[str] = set()
        for option_index, raw_option in enumerate(options_raw):
            if not isinstance(raw_option, dict):
                raise ValueError("Workflow intake option must be an object")
            label = _strip_reasoning(str(raw_option.get("label") or "")).strip()
            if not label:
                raise ValueError("Workflow intake option is missing label")
            priority_raw = raw_option.get("priority", option_index + 1)
            try:
                priority = int(priority_raw)
            except Exception:
                priority = option_index + 1
            option_id = _unique_id(
                _slug(str(raw_option.get("id") or label)) or chr(ord("a") + option_index),
                used_option_ids,
            )
            used_option_ids.add(option_id)
            options.append(
                WorkflowIntakeOption(
                    id=option_id,
                    label=label,
                    description=_strip_reasoning(str(raw_option.get("description") or "")).strip(),
                    priority=priority,
                ).model_dump()
            )
        options.sort(key=lambda item: int(item.get("priority") or 0))
        questions.append(
            WorkflowIntakeQuestion(
                id=question_id,
                question=question_text,
                detail=_strip_reasoning(str(raw_question.get("detail") or "")).strip(),
                options=options,
            ).model_dump()
        )

    batch = None
    if questions:
        batch = WorkflowIntakeBatch(id=_new_id("batch"), questions=questions).model_dump()

    ready = bool(data.get("ready")) and not questions
    if ready and not summary:
        summary = _build_intake_summary(state)
    return {"reply": reply, "ready": ready, "summary": summary, "batch": batch}


def _format_intake_answers_for_user(questions: List[Dict[str, Any]], answers: List[WorkflowIntakeAnswer]) -> str:
    question_lookup = {str(question.get("id") or ""): question for question in questions if isinstance(question, dict)}
    lines = ["Clarification answers:"]
    for answer in answers:
        question = question_lookup.get(answer.questionId, {})
        option_text = ""
        for option in question.get("options", []) if isinstance(question.get("options"), list) else []:
            if isinstance(option, dict) and str(option.get("id") or "") == str(answer.optionId or ""):
                option_text = str(option.get("label") or "")
                break
        response = answer.answer.strip() or option_text or "No answer"
        suffix = " (custom)" if answer.custom else ""
        lines.append(f"- {str(question.get('question') or answer.questionId)}: {response}{suffix}")
    return "\n".join(lines)


def _finalize_intake_project(
    project: WorkflowProject,
    state: Dict[str, Any],
    body: WorkflowIntakeConfirmRequest,
    summary: str,
) -> None:
    clean_summary = _strip_reasoning(summary).strip() or _build_intake_summary(state)
    goal_parts = [body.goal.strip() if body.goal else project.goal, f"Clarification summary:\n{clean_summary}"]
    project.goal = "\n\n".join(part for part in goal_parts if part)
    if body.name.strip() and body.name.strip() != project.name:
        project.name = body.name.strip()
    project.status = "draft"
    state["summary"] = clean_summary
    state["ready"] = True
    state["currentBatch"] = None
    _remember_intake_state(project, state)
    _touch_project(project)
    _append_event(
        project.id,
        StreamEvent(
            id=_new_id("evt"),
            projectId=project.id,
            type="process_summary",
            label="Workflow intake confirmed",
            summary="Clarification summary saved. Workflow generation is starting.",
            details={"intakeId": state.get("id"), "rawReasoningExposed": False},
            status="success",
        ),
    )


def _create_legacy_intake_project(
    body: WorkflowIntakeConfirmRequest,
    intake_id: str,
    summary: str,
) -> WorkflowProject:
    goal_parts = [body.goal.strip() if body.goal else "", f"Clarification summary:\n{summary}" if summary else ""]
    project = _create_project(
        ProjectCreateRequest(
            name=body.name,
            goal="\n\n".join(part for part in goal_parts if part),
            root=body.root,
            references=body.references,
        )
    )
    _register_project(project)
    _append_event(
        project.id,
        StreamEvent(
            id=_new_id("evt"),
            projectId=project.id,
            type="process_summary",
            label="项目已创建",
            summary="已初始化项目结构、Workflow manifest、SQLite 事件库和 Git 仓库。",
            details={"intakeId": intake_id, "legacyIntake": True},
            status="success",
        ),
    )
    _create_snapshot(project, "Project initialized", "project_init")
    return project


def _build_intake_summary(state: Dict[str, Any]) -> str:
    user_messages = [
        str(item.get("content") or "").strip()
        for item in state.get("messages", [])
        if item.get("role") == "user" and str(item.get("content") or "").strip()
    ]
    parts = [
        f"项目名称：{state.get('name') or '未命名 Workflow'}",
        f"初始目标：{state.get('goal') or '未填写'}",
        f"项目目录：{state.get('root') or '使用 Hermes 默认 workflows 目录'}",
        f"References：{len(state.get('references') or [])} 个初始条目",
    ]
    if user_messages:
        parts.append("澄清记录：")
        parts.extend(f"- {message}" for message in user_messages[-4:])
    else:
        parts.append("澄清记录：等待用户回复。")
    return "\n".join(parts)


def _intake_summary_for(intake_id: str) -> str:
    with _intake_sessions_lock:
        state = _intake_sessions.get(intake_id)
        if not state:
            return ""
        return str(state.get("summary") or _build_intake_summary(state))


def _create_project(body: ProjectCreateRequest) -> WorkflowProject:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Project name is required")
    root = Path(body.root).expanduser() if body.root else _default_project_root(name)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for dirname in ["references", "memory", "workflow", "artifacts", "outputs", "logs", WORKFLOW_DIR]:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    meta = root / WORKFLOW_DIR
    for dirname in ["runs", "stream-events"]:
        (meta / dirname).mkdir(parents=True, exist_ok=True)
    project = WorkflowProject(id=_new_id("proj"), name=name, root=str(root), goal=body.goal.strip(), status="draft")
    _init_db(project)
    _save_project(project)
    _save_workflow(project, _empty_workflow(project))
    _write_json(_meta_path(project, "settings.json"), {"executionMode": "semi_auto", "maxConcurrency": 2})
    refs = [_reference_from_path(path) for path in body.references]
    _write_json(_meta_path(project, "references.manifest.json"), [ref.model_dump() for ref in refs])
    _write_json(_meta_path(project, "skills.config.json"), [])
    _git_init(root)
    return project


def _default_project_root(name: str) -> Path:
    slug = _slug(name) or "workflow-project"
    return Path(get_hermes_home()) / "workflows" / slug


def _empty_workflow(project: WorkflowProject) -> Workflow:
    return Workflow(id=_new_id("wf"), title=project.name, nodes=[], edges=[])


def _ensure_project_agent_session(project: WorkflowProject) -> str:
    if project.agentSessionId:
        return project.agentSessionId
    project.agentSessionId = f"workflow-project-{project.id}"
    _save_project(project)
    _register_project(project)
    return project.agentSessionId


def _ensure_node_agent_session(project: WorkflowProject, node: WorkflowNode) -> str:
    if node.agentSessionId:
        return node.agentSessionId
    node.agentSessionId = f"workflow-node-{project.id}-{node.id}"
    return node.agentSessionId


def _generate_workflow_for_project(project: WorkflowProject, *, reason: str) -> Optional[str]:
    prompt = _workflow_generation_prompt(project)
    session_id = _ensure_project_agent_session(project)
    try:
        result = _agent_runner.run(
            project=project,
            prompt=prompt,
            session_id=session_id,
            system_prompt=_workflow_generation_system_prompt(),
            max_iterations=10,
            message_label="Workflow 规划生成",
        )
        project.agentSessionId = result.session_id
        workflow = _workflow_from_agent_text(project, result.text)
        _validate_workflow(workflow)
        _save_workflow(project, workflow)
        project.status = "generated"
        _touch_project(project)
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                type="stage_result",
                label="Workflow 已生成",
                summary="已通过 Hermes Agent/LLM 基于任务背景、references 和 skills 生成初始节点图。",
                details={"agentSessionId": result.session_id, "rawReasoningExposed": False},
                status="success",
            ),
        )
        _create_snapshot(project, "Workflow generated", reason)
        return None
    except Exception as exc:
        message = _truncate_text(str(exc), 1200)
        project.status = "failed"
        _append_event(
            project.id,
            StreamEvent(
                id=_new_id("evt"),
                projectId=project.id,
                type="error",
                label="Workflow 生成失败",
                summary=message,
                details={"stage": "generate", "rawReasoningExposed": False},
                status="error",
            ),
        )
        _touch_project(project)
        return message


def _workflow_generation_system_prompt() -> str:
    return (
        "You are the Hermes Workflow planner. Generate executable AI-agent workflow graphs for a desktop workbench. "
        "Return only JSON. Do not include markdown fences. Do not expose hidden reasoning or chain-of-thought."
    )


def _workflow_generation_prompt(project: WorkflowProject) -> str:
    references = [ref.model_dump() for ref in _load_references(project) if ref.enabled]
    skills = [skill.model_dump() for skill in _load_skills(project) if skill.enabled]
    schema = {
        "title": "string",
        "nodes": [
            {
                "id": "stable-lowercase-id",
                "type": "planning|reference|execution|review|delivery|task",
                "title": "short title",
                "description": "what the node must accomplish",
                "skills": ["optional skill names"],
                "optional": False,
                "reviewRules": {"required": False, "checklist": ["observable acceptance item"]},
                "position": {"x": 0, "y": 0},
            }
        ],
        "edges": [
            {
                "id": "edge-source-target",
                "source": "source-node-id",
                "target": "target-node-id",
                "type": "dependency|feedback",
                "label": "short label",
                "optional": False,
            }
        ],
    }
    return "\n".join(
        [
            "Create an initial workflow for this Hermes desktop project.",
            "The workflow must be executable by independent Hermes Agent node runs.",
            "Include planning, reference/context handling when useful, execution, review, and final delivery nodes.",
            "Use dependency edges for forward execution. Use type='feedback' only for bounded revision loops.",
            "At least one start node must have no dependencies. Include no ordinary dependency cycles.",
            "",
            f"Project name: {project.name}",
            f"Project root: {project.root}",
            f"Project goal:\n{project.goal or 'No goal was provided. Infer a useful software/AI workflow from project context.'}",
            "",
            f"Enabled references JSON:\n{json.dumps(references, ensure_ascii=False, indent=2)}",
            f"Enabled skills JSON:\n{json.dumps(skills, ensure_ascii=False, indent=2)}",
            "",
            "Return JSON matching this shape exactly:",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )


def _workflow_from_agent_text(project: WorkflowProject, text: str) -> Workflow:
    data = _extract_json_object(text)
    if "workflow" in data and isinstance(data["workflow"], dict):
        data = data["workflow"]
    nodes_raw = data.get("nodes")
    edges_raw = data.get("edges")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("LLM response did not include workflow.nodes")
    if not isinstance(edges_raw, list):
        edges_raw = []

    nodes: List[WorkflowNode] = []
    used_ids: set[str] = set()
    for idx, raw in enumerate(nodes_raw):
        if not isinstance(raw, dict):
            continue
        node = _workflow_node_from_raw(raw, idx, used_ids)
        used_ids.add(node.id)
        nodes.append(node)
    if not nodes:
        raise ValueError("LLM response did not include valid workflow nodes")

    edges: List[WorkflowEdge] = []
    node_ids = {node.id for node in nodes}
    for idx, raw in enumerate(edges_raw):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or "").strip()
        target = str(raw.get("target") or "").strip()
        if source not in node_ids or target not in node_ids or source == target:
            continue
        edge_type = str(raw.get("type") or "dependency").strip() or "dependency"
        if edge_type not in {"dependency", "feedback"}:
            edge_type = "dependency"
        edge_id = str(raw.get("id") or f"edge-{source}-{target}-{idx}").strip()
        edges.append(
            WorkflowEdge(
                id=_unique_id(_slug(edge_id) or f"edge-{source}-{target}-{idx}", {edge.id for edge in edges}),
                source=source,
                target=target,
                type=edge_type,
                label=str(raw.get("label") or ""),
                optional=bool(raw.get("optional", False)),
            )
        )

    workflow = Workflow(id=_new_id("wf"), title=str(data.get("title") or project.name), nodes=nodes, edges=edges)
    _promote_ready_nodes(workflow)
    return workflow


def _workflow_node_from_raw(raw: Dict[str, Any], index: int, used_ids: set[str]) -> WorkflowNode:
    title = str(raw.get("title") or raw.get("name") or f"节点 {index + 1}").strip()
    raw_id = str(raw.get("id") or title or f"node-{index + 1}")
    node_id = _unique_id(_slug(raw_id) or f"node-{index + 1}", used_ids)
    position_raw = raw.get("position") if isinstance(raw.get("position"), dict) else {}
    review_raw = raw.get("reviewRules") if isinstance(raw.get("reviewRules"), dict) else {}
    checklist = review_raw.get("checklist") if isinstance(review_raw, dict) else []
    skills = raw.get("skills") if isinstance(raw.get("skills"), list) else []
    return WorkflowNode(
        id=node_id,
        type=str(raw.get("type") or "task"),
        title=title,
        description=str(raw.get("description") or raw.get("summary") or ""),
        position=WorkflowPosition(
            x=float(position_raw.get("x", 60 + (index % 4) * 330)),
            y=float(position_raw.get("y", 80 + (index // 4) * 190)),
        ),
        status="created",
        inputs=raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {},
        outputs={},
        reviewRules=ReviewRules(
            required=bool(review_raw.get("required", raw.get("type") in {"review", "delivery"})),
            checklist=[str(item) for item in checklist if str(item).strip()],
        ),
        skills=[str(skill) for skill in skills if str(skill).strip()],
        model=str(raw.get("model")) if raw.get("model") else None,
        optional=bool(raw.get("optional", False)),
        maxRetries=max(0, int(raw.get("maxRetries", 1) or 1)),
        llmGenerated=True,
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_reasoning(text)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.IGNORECASE | re.DOTALL)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain a JSON object")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _sample_workflow(project: WorkflowProject) -> Workflow:
    goal = project.goal or "明确任务目标并完成可交付结果。"
    nodes = [
        WorkflowNode(
            id="intake",
            type="planning",
            title="任务澄清",
            description=f"整理任务背景、成功标准和约束：{goal}",
            position=WorkflowPosition(x=40, y=120),
            status="ready",
            reviewRules=ReviewRules(required=False, checklist=["目标明确", "范围边界明确"]),
            skills=["context-analysis"],
        ),
        WorkflowNode(
            id="references",
            type="reference",
            title="资料整理",
            description="读取并筛选项目 references，只把启用资料纳入节点上下文。",
            position=WorkflowPosition(x=360, y=40),
            reviewRules=ReviewRules(required=False, checklist=["禁用资料未进入上下文"]),
            skills=["file"],
        ),
        WorkflowNode(
            id="strategy",
            type="planning",
            title="执行策略",
            description="生成可执行 workflow、依赖关系和 review gate。",
            position=WorkflowPosition(x=360, y=220),
            reviewRules=ReviewRules(required=True, checklist=["包含分支和并行", "风险点可检查"]),
            skills=["planner"],
        ),
        WorkflowNode(
            id="implementation",
            type="execution",
            title="实现任务",
            description="调用 Hermes Agent runner 完成核心任务并沉淀 artifacts。",
            position=WorkflowPosition(x=700, y=110),
            reviewRules=ReviewRules(required=False, checklist=["产物路径已记录", "工具调用可追踪"]),
            skills=["terminal", "file"],
        ),
        WorkflowNode(
            id="review",
            type="review",
            title="半自动审查",
            description="检查输出质量、遗漏和需要用户确认的决策。",
            position=WorkflowPosition(x=1030, y=110),
            reviewRules=ReviewRules(required=True, checklist=["不暴露原始 CoT", "失败路径可重试", "输出可交付"]),
            skills=["reviewer"],
        ),
        WorkflowNode(
            id="delivery",
            type="delivery",
            title="最终交付",
            description="汇总 artifacts、stream 事件和最终说明。",
            position=WorkflowPosition(x=1360, y=110),
            reviewRules=ReviewRules(required=True, checklist=["快照已创建", "交付说明完整"]),
            skills=["writer"],
        ),
    ]
    edges = [
        WorkflowEdge(id="edge-intake-references", source="intake", target="references", label="资料"),
        WorkflowEdge(id="edge-intake-strategy", source="intake", target="strategy", label="策略"),
        WorkflowEdge(id="edge-references-implementation", source="references", target="implementation", label="上下文"),
        WorkflowEdge(id="edge-strategy-implementation", source="strategy", target="implementation", label="计划"),
        WorkflowEdge(id="edge-implementation-review", source="implementation", target="review", label="审查"),
        WorkflowEdge(id="edge-review-delivery", source="review", target="delivery", label="确认"),
        WorkflowEdge(id="edge-review-strategy", source="review", target="strategy", type="feedback", label="修订回路"),
    ]
    return Workflow(id=_new_id("wf"), title=project.name, nodes=nodes, edges=edges)


def _validate_workflow(workflow: Workflow) -> None:
    ids = {node.id for node in workflow.nodes}
    if len(ids) != len(workflow.nodes):
        raise HTTPException(status_code=422, detail="Workflow node ids must be unique")
    for edge in workflow.edges:
        if edge.source not in ids or edge.target not in ids:
            raise HTTPException(status_code=422, detail=f"Edge {edge.id} references an unknown node")
    _detect_dependency_cycles(workflow)


def _detect_dependency_cycles(workflow: Workflow) -> None:
    graph: dict[str, list[str]] = {node.id: [] for node in workflow.nodes}
    for edge in workflow.edges:
        if edge.type == "feedback":
            continue
        graph.setdefault(edge.source, []).append(edge.target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise HTTPException(
                status_code=422,
                detail="Dependency cycle detected. Use feedback edges for revision loops.",
            )
        if node_id in visited:
            return
        visiting.add(node_id)
        for target in graph.get(node_id, []):
            visit(target)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)


def _promote_ready_nodes(workflow: Workflow) -> None:
    terminal = {"completed", "skipped"}
    for node in workflow.nodes:
        if node.status not in {"created", "revision_needed", "retrying"}:
            continue
        deps = _incoming_dependencies(workflow, node.id)
        if all(_node_by_id(workflow, dep.source).status in terminal for dep in deps):
            node.status = "ready"


def _workflow_is_done(workflow: Workflow) -> bool:
    terminal = {"completed", "skipped", "aborted"}
    required = [node for node in workflow.nodes if not node.optional]
    return bool(required) and all(node.status in terminal for node in required)


def _node_needs_confirmation(run: ExecutionRun, node: WorkflowNode) -> bool:
    if run.mode == "auto":
        return False
    if run.mode == "single_step":
        return True
    return bool(node.reviewRules.required or node.type in {"review", "delivery"})


def _append_node_status(
    project: WorkflowProject,
    run: ExecutionRun,
    node: WorkflowNode,
    label: str,
    summary: str,
    *,
    status: str = "info",
) -> None:
    _append_event(
        project.id,
        StreamEvent(
            id=_new_id("evt"),
            projectId=project.id,
            runId=run.id,
            nodeId=node.id,
            type="node_status",
            label=label,
            summary=summary,
            status=status,
            details={"nodeStatus": node.status},
        ),
    )


def _safe_node_summary(node: WorkflowNode) -> str:
    return (
        f"已完成「{node.title}」的可展示过程摘要：输入已检查、启用资料已过滤、"
        "工具调用和阶段结果已记录；原始推理链未写入 Stream。"
    )


def _set_workflow_session_context(session_id: str, cwd: str) -> List[Any]:
    try:
        from tui_gateway.server import _set_session_context

        return _set_session_context(session_id, cwd=cwd)
    except Exception:
        return []


def _clear_workflow_session_context(tokens: List[Any]) -> None:
    if not tokens:
        return
    try:
        from tui_gateway.server import _clear_session_context

        _clear_session_context(tokens)
    except Exception:
        return


def _strip_reasoning(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"<think>.*?</think>\s*", "", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"```(?:reasoning|chain[-_ ]?of[-_ ]?thought).*?```", "", value, flags=re.DOTALL | re.IGNORECASE)
    return value.strip()


def _truncate_text(text: str, limit: int = 1200) -> str:
    value = _strip_reasoning(str(text or ""))
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _safe_details(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        dumped = json.dumps(data or {}, ensure_ascii=False, default=str)
        if len(dumped) > 4000:
            return {"preview": dumped[:4000] + "…"}
        loaded = json.loads(dumped)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {"preview": _truncate_text(str(data), 1000)}


def _tool_summary_from_args(name: str, args: Dict[str, Any]) -> str:
    preview = ""
    if isinstance(args, dict):
        for key in ("cmd", "command", "path", "file", "query", "text"):
            if args.get(key):
                preview = str(args[key])
                break
    return _truncate_text(f"{name}: {preview}" if preview else name, 500)


def _effective_node_skills(project: WorkflowProject, node: WorkflowNode) -> List[str]:
    if node.skillMode == "manual":
        return list(dict.fromkeys(skill for skill in node.skills if skill))
    project_skills = [skill.name for skill in _load_skills(project) if skill.enabled]
    return list(dict.fromkeys([*node.skills, *project_skills]))


def _node_reference_details(project: WorkflowProject, node: WorkflowNode) -> List[Dict[str, Any]]:
    root = Path(project.root).resolve()
    details: List[Dict[str, Any]] = []
    for value in node.references:
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / value
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(root)
            inside_project = True
        except Exception:
            resolved = path
            rel = Path(value)
            inside_project = False
        details.append(
            {
                "path": str(resolved),
                "relativePath": str(rel).replace("\\", "/"),
                "exists": resolved.exists(),
                "kind": "folder" if resolved.is_dir() else "file",
                "insideProject": inside_project,
            }
        )
    return details


def _node_execution_system_prompt(project: WorkflowProject, node: WorkflowNode) -> str:
    return (
        "You are a Hermes Agent executing one node inside a desktop workflow workbench. "
        "Work only on the assigned node objective. Use tools when useful, keep outputs auditable, "
        f"and write or reference artifacts under the project root: {project.root}. "
        "Do not expose hidden reasoning or chain-of-thought."
    )


def _node_execution_prompt(project: WorkflowProject, workflow: Workflow, run: ExecutionRun, node: WorkflowNode) -> str:
    parents = [_node_by_id(workflow, edge.source) for edge in _incoming_dependencies(workflow, node.id)]
    enabled_refs = [ref.model_dump() for ref in _load_references(project) if ref.enabled]
    selected_refs = _node_reference_details(project, node)
    enabled_skills = _effective_node_skills(project, node)
    parent_outputs = [
        {
            "id": parent.id,
            "title": parent.title,
            "outputs": parent.outputs,
            "artifacts": parent.artifacts,
        }
        for parent in parents
    ]
    return "\n".join(
        [
            "Execute this workflow node and return a concise final result suitable for the workflow Stream panel.",
            "If you create files, put them below artifacts/ or outputs/ and mention their paths.",
            "Do not reveal chain-of-thought; summarize process and decisions only.",
            "",
            f"Project: {project.name}",
            f"Project root: {project.root}",
            f"Project goal:\n{project.goal}",
            f"Run: {run.id} ({run.mode})",
            "",
            f"Current node JSON:\n{json.dumps(node.model_dump(), ensure_ascii=False, indent=2)}",
            f"Editable task prompt override:\n{node.promptOverride.strip() if node.promptOverride else 'None'}",
            f"Parent outputs JSON:\n{json.dumps(parent_outputs, ensure_ascii=False, indent=2)}",
            f"Enabled references JSON:\n{json.dumps(enabled_refs, ensure_ascii=False, indent=2)}",
            f"Node selected references JSON:\n{json.dumps(selected_refs, ensure_ascii=False, indent=2)}",
            f"Enabled skills JSON:\n{json.dumps(enabled_skills, ensure_ascii=False, indent=2)}",
            f"Effective model override: {node.modelOverride or node.model or 'global/default'}",
            "",
            "Final response requirements:",
            "- State what was done for this node.",
            "- List produced/updated artifacts with paths.",
            "- List any blockers or user decisions needed.",
            "- Keep it concise and directly actionable.",
        ]
    )


def _write_node_artifact(project: WorkflowProject, run: ExecutionRun, node: WorkflowNode, final_text: str) -> Artifact:
    artifacts_dir = Path(project.root) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{run.id}-{node.id}.md"
    path = artifacts_dir / file_name
    text = "\n".join(
        [
            f"# {node.title}",
            "",
            f"- Project: {project.name}",
            f"- Run: {run.id}",
            f"- Node: {node.id}",
            f"- Status: {node.status}",
            "",
            "## Summary",
            final_text or _safe_node_summary(node),
            "",
            "## Inputs",
            json.dumps(node.inputs, ensure_ascii=False, indent=2),
            "",
            "## Outputs",
            json.dumps(node.outputs, ensure_ascii=False, indent=2),
            "",
            "## Review Rules",
            json.dumps(node.reviewRules.model_dump(), ensure_ascii=False, indent=2),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    artifact = Artifact(id=_new_id("art"), nodeId=node.id, name=file_name, path=str(path), kind="markdown")
    _save_artifact(project, artifact)
    return artifact


def _chat_patch_from_text(workflow: Workflow, body: ChatRequest) -> Dict[str, Any]:
    text = body.text.strip()
    patch: Dict[str, Any] = {"op": "comment", "text": text, "nodeId": body.nodeId}
    lower = text.lower()
    if any(word in lower for word in ["add node", "新增节点", "增加节点"]):
        patch = {
            "op": "add_node",
            "after": body.nodeId or (workflow.nodes[-1].id if workflow.nodes else None),
            "title": "待确认的新节点",
            "description": text,
        }
    elif any(word in lower for word in ["retry", "重试", "重新执行"]):
        patch = {"op": "retry_node", "nodeId": body.nodeId, "reason": text}
    return patch


def _workflow_chat_system_prompt() -> str:
    return (
        "You are Hermes Workflow workbench chat. Help the user modify or operate a workflow. "
        "Return only JSON with keys: reply (string) and patch (object). "
        "Do not expose chain-of-thought."
    )


def _workflow_chat_prompt(project: WorkflowProject, workflow: Workflow, body: ChatRequest, node: Optional[WorkflowNode]) -> str:
    return "\n".join(
        [
            "Respond to the user's workflow workbench request.",
            "If a workflow change is needed, propose a patch object. Do not apply it yourself.",
            "Patch examples: {\"op\":\"comment\"}, {\"op\":\"add_node\"}, {\"op\":\"update_node\"}, {\"op\":\"add_edge\"}, {\"op\":\"retry_node\"}, {\"op\":\"regenerate_workflow\"}.",
            "",
            f"Project: {project.name}",
            f"Project root: {project.root}",
            f"Project goal:\n{project.goal}",
            f"Selected node JSON:\n{json.dumps(node.model_dump() if node else None, ensure_ascii=False, indent=2)}",
            f"Workflow JSON:\n{json.dumps(workflow.model_dump(), ensure_ascii=False, indent=2)}",
            f"Attachments:\n{json.dumps(body.attachments, ensure_ascii=False, indent=2)}",
            f"Skill ids:\n{json.dumps(body.skillIds, ensure_ascii=False, indent=2)}",
            "",
            f"User request:\n{body.text}",
            "",
            "Return only JSON: {\"reply\":\"visible response\", \"patch\":{...}}",
        ]
    )


def _chat_result_from_agent_text(text: str, workflow: Workflow, body: ChatRequest) -> tuple[str, Dict[str, Any]]:
    try:
        data = _extract_json_object(text)
        reply = _strip_reasoning(str(data.get("reply") or data.get("message") or ""))
        patch = data.get("patch") if isinstance(data.get("patch"), dict) else None
        if patch is None:
            patch = _chat_patch_from_text(workflow, body)
        return reply or _truncate_text(text, 1200), patch
    except Exception:
        return _truncate_text(text, 1200), _chat_patch_from_text(workflow, body)


def _workflow_slash_catalog(query: str = "") -> List[Dict[str, Any]]:
    from hermes_cli.commands import COMMAND_REGISTRY

    q = query.strip().lower().lstrip("/")
    items: List[Dict[str, Any]] = []
    for command in COMMAND_REGISTRY:
        if command.cli_only:
            continue
        name = f"/{command.name}"
        haystack = " ".join([command.name, command.description, command.category, *command.aliases]).lower()
        if q and q not in haystack:
            continue
        items.append(
            {
                "name": name,
                "description": command.description,
                "category": command.category,
                "argsHint": command.args_hint,
                "aliases": [f"/{alias}" for alias in command.aliases],
            }
        )
    return items[:80]


def _execute_workflow_slash(project: WorkflowProject, workflow: Workflow, body: ComposerSlashRequest) -> Dict[str, Any]:
    command = body.command.strip()
    if not command:
        return {"ok": False, "output": "empty slash command"}
    if not command.startswith("/"):
        command = f"/{command}"
    name, _, arg = command[1:].partition(" ")
    name = name.lower().strip()
    arg = arg.strip()

    if name in {"help", "commands"}:
        rows = _workflow_slash_catalog(arg)
        output = "\n".join(f"{row['name']:<18} {row['description']}" for row in rows[:40])
        return {"ok": True, "output": output or "No workflow commands matched."}
    if name in {"status"}:
        ready = sum(1 for node in workflow.nodes if node.status == "ready")
        completed = sum(1 for node in workflow.nodes if node.status == "completed")
        return {"ok": True, "output": f"Workflow: {project.name}\nNodes: {completed}/{len(workflow.nodes)} completed, {ready} ready\nRoot: {project.root}"}
    if name in {"generate", "regen", "regenerate"}:
        error = _generate_workflow_for_project(project, reason="workflow_slash_generate")
        return {"ok": error is None, "output": error or "Workflow generated."}
    if name in {"run"}:
        return {"ok": True, "output": "Use the toolbar Run button to start execution with the selected mode/concurrency."}
    if name in {"new", "reset"}:
        return {"ok": True, "output": "Use Workflow 工作台 to create a new workflow project."}

    return {
        "ok": False,
        "output": f"/{name} is listed for Hermes chat but is not directly executable in the workflow composer yet. Send a normal message to ask the workflow agent for the same change.",
    }


def _workflow_completions(project: WorkflowProject, body: ComposerCompletionRequest) -> List[Dict[str, Any]]:
    text = body.text or ""
    cursor = max(0, min(len(text), int(body.cursor if body.cursor is not None else len(text))))
    prefix = text[:cursor]
    slash_match = re.search(r"/([\w-]*)$", prefix)
    if slash_match:
        return [
            {"type": "slash", "text": item["name"], "label": item["name"], "detail": item["description"]}
            for item in _workflow_slash_catalog(slash_match.group(1))
        ][:40]

    at_match = re.search(r"@file:([^\s`]*)$", prefix)
    if at_match:
        query = at_match.group(1).lower()
        root = Path(body.cwd or project.root).expanduser()
        candidates: List[Dict[str, Any]] = []
        for item in _walk_completion_files(root):
            rel = item.relative_to(root)
            label = str(rel).replace("\\", "/")
            if query and query not in label.lower():
                continue
            candidates.append({"type": "file", "text": f"@file:`{label}`", "label": label, "path": str(item)})
            if len(candidates) >= 40:
                break
        return candidates
    return []


def _walk_completion_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    ignored = {".git", ".agent-workflow", "node_modules", "__pycache__", ".venv", "venv"}

    def iter_paths() -> Iterable[Path]:
        for base, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in ignored and not name.startswith(".cache")]
            for name in files:
                if name.startswith("."):
                    continue
                yield Path(base) / name

    return iter_paths()


def _project_bundle(project_id: str, error: Optional[str] = None) -> ProjectBundle:
    project = _load_project(project_id)
    return ProjectBundle(
        project=project,
        workflow=_load_workflow(project),
        references=_load_references(project),
        skills=_load_skills(project),
        artifacts=_load_artifacts(project),
        snapshots=_list_snapshots(project),
        latestRun=_load_run(project.currentRunId) if project.currentRunId else None,
        error=error,
    )


def _meta_path(project: WorkflowProject, name: str) -> Path:
    return Path(project.root) / WORKFLOW_DIR / name


def _db_path(project: WorkflowProject) -> Path:
    return _meta_path(project, "workflow.db")


def _registry_path() -> Path:
    return Path(get_hermes_home()) / REGISTRY_FILE


def _read_registry() -> Dict[str, str]:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _write_registry(registry: Dict[str, str]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, registry)


def _register_project(project: WorkflowProject) -> None:
    registry = _read_registry()
    registry[project.id] = project.root
    _write_registry(registry)


def _unregister_project(project_id: str) -> None:
    registry = _read_registry()
    if project_id in registry:
        registry.pop(project_id, None)
        _write_registry(registry)


def _remove_project_from_history(project_id: str) -> Dict[str, Any]:
    project = _load_project(project_id)
    _unregister_project(project_id)
    return {"ok": True, "projectId": project_id, "root": project.root, "rootPreserved": True}


def _load_project(project_id: str) -> WorkflowProject:
    root = _read_registry().get(project_id)
    if not root:
        raise HTTPException(status_code=404, detail="Workflow project not registered")
    project = _load_project_by_path(Path(root))
    if project is None:
        raise HTTPException(status_code=404, detail="Workflow project not found")
    return project


def _load_project_by_path(root: Path | str) -> Optional[WorkflowProject]:
    path = Path(root).expanduser().resolve() / WORKFLOW_DIR / "project.json"
    if not path.exists():
        return None
    try:
        return WorkflowProject(**_read_json(path))
    except Exception:
        return None


def _save_project(project: WorkflowProject) -> None:
    _write_json(_meta_path(project, "project.json"), project.model_dump())


def _touch_project(project: WorkflowProject) -> None:
    project.updatedAt = time.time()
    _save_project(project)
    _register_project(project)


def _load_workflow(project: WorkflowProject) -> Workflow:
    path = _meta_path(project, "workflow.flow.json")
    if not path.exists():
        return _empty_workflow(project)
    return Workflow(**_read_json(path))


def _save_workflow(project: WorkflowProject, workflow: Workflow) -> None:
    workflow.updatedAt = time.time()
    _write_json(_meta_path(project, "workflow.flow.json"), workflow.model_dump())


def _load_references(project: WorkflowProject) -> List[ReferenceItem]:
    return [ReferenceItem(**item) for item in _read_json_list(_meta_path(project, "references.manifest.json"))]


def _load_skills(project: WorkflowProject) -> List[SkillBinding]:
    return [SkillBinding(**item) for item in _read_json_list(_meta_path(project, "skills.config.json"))]


def _load_artifacts(project: WorkflowProject) -> List[Artifact]:
    return [Artifact(**item) for item in _read_json_list(_meta_path(project, "artifacts.manifest.json"))]


def _save_artifact(project: WorkflowProject, artifact: Artifact) -> None:
    artifacts = _load_artifacts(project)
    artifacts = [item for item in artifacts if item.id != artifact.id]
    artifacts.append(artifact)
    _write_json(_meta_path(project, "artifacts.manifest.json"), [item.model_dump() for item in artifacts])


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = _read_json(path)
    return data if isinstance(data, list) else []


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _init_db(project: WorkflowProject) -> None:
    with _connect(project) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              run_id TEXT,
              node_id TEXT,
              type TEXT NOT NULL,
              label TEXT NOT NULL,
              timestamp REAL NOT NULL,
              payload TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              status TEXT NOT NULL,
              payload TEXT NOT NULL,
              updated_at REAL NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
              id TEXT PRIMARY KEY,
              project_id TEXT NOT NULL,
              session_id TEXT NOT NULL,
              run_id TEXT,
              node_id TEXT,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              timestamp REAL NOT NULL,
              payload TEXT NOT NULL
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_session "
            "ON agent_messages(project_id, session_id, timestamp)"
        )
        db.commit()


def _connect(project: WorkflowProject) -> sqlite3.Connection:
    _db_path(project).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(_db_path(project)), timeout=30)
    db.row_factory = sqlite3.Row
    return db


def _append_event(project_id: str, event: StreamEvent, *, persist: bool = True) -> None:
    with _transient_events_lock:
        bucket = _transient_events.setdefault(project_id, [])
        bucket.append(event)
        if len(bucket) > _TRANSIENT_EVENT_LIMIT:
            del bucket[: len(bucket) - _TRANSIENT_EVENT_LIMIT]

    if not persist:
        return

    project = _load_project(project_id)
    _init_db(project)
    payload = event.model_dump()
    with _connect(project) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            """
            INSERT OR REPLACE INTO events (id, project_id, run_id, node_id, type, label, timestamp, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.projectId,
                event.runId,
                event.nodeId,
                event.type,
                event.label,
                event.timestamp,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        db.commit()
    stream_dir = Path(project.root) / WORKFLOW_DIR / "stream-events"
    stream_dir.mkdir(parents=True, exist_ok=True)
    with (stream_dir / f"{time.strftime('%Y-%m-%d')}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_workflow_agent_message(
    project: WorkflowProject,
    *,
    session_id: str,
    role: str,
    content: str,
    node: Optional[WorkflowNode] = None,
    run: Optional[ExecutionRun] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if role not in {"user", "assistant"} or not str(content or "").strip():
        return
    _init_db(project)
    timestamp = time.time()
    payload = {
        "id": _new_id("wam"),
        "projectId": project.id,
        "sessionId": session_id,
        "runId": run.id if run else None,
        "nodeId": node.id if node else None,
        "role": role,
        "content": _strip_reasoning(str(content)),
        "timestamp": timestamp,
        "metadata": metadata or {},
    }
    with _connect(project) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            """
            INSERT INTO agent_messages (id, project_id, session_id, run_id, node_id, role, content, timestamp, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                project.id,
                session_id,
                payload["runId"],
                payload["nodeId"],
                role,
                payload["content"],
                timestamp,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        db.commit()


def _load_workflow_agent_history(project: WorkflowProject, session_id: str, *, limit: int = 30) -> List[Dict[str, Any]]:
    _init_db(project)
    with _connect(project) as db:
        rows = db.execute(
            """
            SELECT role, content
            FROM agent_messages
            WHERE project_id = ? AND session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (project.id, session_id, max(1, min(100, int(limit)))),
        ).fetchall()
    history: List[Dict[str, Any]] = []
    for row in reversed(rows):
        role = str(row["role"] or "")
        content = str(row["content"] or "")
        if role in {"user", "assistant"} and content.strip():
            history.append({"role": role, "content": _truncate_text(content, 24000)})
    return history


def _read_events(project: WorkflowProject, *, since: Optional[float], limit: int) -> List[StreamEvent]:
    _init_db(project)
    with _connect(project) as db:
        if since is None:
            rows = db.execute(
                "SELECT payload FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = db.execute(
                "SELECT payload FROM events WHERE timestamp > ? ORDER BY timestamp ASC LIMIT ?",
                (since, limit),
            ).fetchall()
    persisted = [StreamEvent(**json.loads(row["payload"])) for row in rows]
    with _transient_events_lock:
        transient = [
            event
            for event in _transient_events.get(project.id, [])
            if since is None or event.timestamp > since
        ]

    by_id: Dict[str, StreamEvent] = {}
    for event in [*persisted, *transient]:
        by_id[event.id] = event
    return sorted(by_id.values(), key=lambda event: event.timestamp)[-limit:]


def _save_run(run: ExecutionRun) -> None:
    project = _load_project(run.projectId)
    _init_db(project)
    with _connect(project) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT OR REPLACE INTO runs (id, project_id, status, payload, updated_at) VALUES (?, ?, ?, ?, ?)",
            (run.id, run.projectId, run.status, json.dumps(run.model_dump(), ensure_ascii=False), run.updatedAt),
        )
        db.commit()


def _load_run(run_id: Optional[str]) -> ExecutionRun:
    if not run_id:
        raise HTTPException(status_code=404, detail="Run not found")
    for project_id in _read_registry():
        project = _load_project(project_id)
        _init_db(project)
        with _connect(project) as db:
            row = db.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row:
            return ExecutionRun(**json.loads(row["payload"]))
    raise HTTPException(status_code=404, detail="Run not found")


def _node_by_id(workflow: Workflow, node_id: Optional[str]) -> WorkflowNode:
    if not node_id:
        raise HTTPException(status_code=404, detail="Node not found")
    for node in workflow.nodes:
        if node.id == node_id:
            return node
    raise HTTPException(status_code=404, detail="Node not found")


def _incoming_dependencies(workflow: Workflow, node_id: str) -> List[WorkflowEdge]:
    return [edge for edge in workflow.edges if edge.target == node_id and edge.type != "feedback"]


def _reference_from_path(path: str) -> ReferenceItem:
    p = Path(path).expanduser()
    return ReferenceItem(id=_new_id("ref"), name=p.name or str(p), path=str(p), kind="folder" if p.is_dir() else "file")


def _git_init(root: Path) -> None:
    _git(str(root), "init")
    _git(str(root), "config", "user.name", "hermes-workflow")
    _git(str(root), "config", "user.email", "hermes-workflow@local")


def _create_snapshot(project: WorkflowProject, label: str, reason: str) -> VersionSnapshot:
    _git(project.root, "add", WORKFLOW_DIR, "references", "workflow", "artifacts", "outputs", "logs")
    message = f"workflow: {label}"
    commit = None
    try:
        _git(project.root, "commit", "--allow-empty", "-m", message)
        commit = _git(project.root, "rev-parse", "HEAD").strip()
    except HTTPException:
        commit = None
    snapshot = VersionSnapshot(id=_new_id("snap"), label=label, reason=reason, commit=commit)
    _append_event(
        project.id,
        StreamEvent(
            id=_new_id("evt"),
            projectId=project.id,
            runId=project.currentRunId,
            type="snapshot",
            label="版本快照",
            summary=f"{label} 已记录为本地版本快照。",
            details=snapshot.model_dump(),
            status="success" if commit else "warning",
        ),
    )
    return snapshot


def _list_snapshots(project: WorkflowProject) -> List[VersionSnapshot]:
    git_dir = Path(project.root) / ".git"
    if not git_dir.exists():
        return []
    try:
        raw = _git(project.root, "log", "--pretty=format:%H%x1f%ct%x1f%s", "--", WORKFLOW_DIR)
    except HTTPException:
        return []
    snapshots: List[VersionSnapshot] = []
    for line in raw.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 3:
            continue
        commit, ts, subject = parts
        label = subject.replace("workflow: ", "", 1)
        snapshots.append(
            VersionSnapshot(
                id=f"snap_{commit[:12]}",
                label=label,
                reason="git_log",
                commit=commit,
                createdAt=float(ts),
            )
        )
    return snapshots


def _git(root: str | Path, *args: str) -> str:
    if not shutil.which("git"):
        raise HTTPException(status_code=500, detail="Git executable not found")
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise HTTPException(status_code=500, detail=detail)
    return result.stdout.strip()


def _git_change_paths(project: WorkflowProject) -> set[str]:
    return {entry["path"] for entry in _git_status_entries(project)}


def _git_status_entries(project: WorkflowProject) -> List[Dict[str, str]]:
    try:
        raw = _git(project.root, "status", "--porcelain", "--untracked-files=all")
    except HTTPException:
        return []
    entries: List[Dict[str, str]] = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        path = path.strip('"')
        normalized = path.replace("\\", "/")
        if not normalized or _skip_file_change_path(normalized):
            continue
        entries.append({"code": code, "path": normalized})
    return entries


def _collect_node_file_changes(project: WorkflowProject, before_paths: set[str]) -> List[NodeFileChange]:
    changes: List[NodeFileChange] = []
    for entry in _git_status_entries(project):
        rel_path = entry["path"]
        if rel_path in before_paths and rel_path.startswith(WORKFLOW_DIR + "/"):
            continue
        if rel_path in before_paths and not _is_artifact_path(rel_path):
            continue
        status = _file_change_status(entry["code"])
        diff, truncated, is_binary, previewable = _file_change_diff(project, rel_path, status)
        changes.append(
            NodeFileChange(
                path=rel_path,
                status=status,
                diff=diff,
                truncated=truncated,
                isArtifact=_is_artifact_path(rel_path),
                isBinary=is_binary,
                previewable=previewable,
            )
        )
        if len(changes) >= 80:
            break
    return changes


def _file_change_status(code: str) -> str:
    if "D" in code:
        return "deleted"
    if "R" in code:
        return "renamed"
    if "?" in code or "A" in code:
        return "added"
    if "M" in code:
        return "modified"
    return "changed"


def _file_change_diff(project: WorkflowProject, rel_path: str, status: str) -> tuple[str, bool, bool, bool]:
    if status in {"modified", "deleted", "renamed", "changed"}:
        try:
            diff = _git(project.root, "diff", "--", rel_path)
            if diff:
                if _is_git_binary_diff(diff):
                    return "", False, True, False
                truncated_diff, truncated = _truncate_diff(diff)
                return truncated_diff, truncated, False, True
        except HTTPException:
            pass
    return _file_preview_as_diff(Path(project.root) / rel_path)


_BINARY_PREVIEW_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bin",
    ".bmp",
    ".db",
    ".dll",
    ".doc",
    ".docx",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".pyc",
    ".sqlite",
    ".so",
    ".tar",
    ".tif",
    ".tiff",
    ".wasm",
    ".wav",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}


def _file_preview_as_diff(path: Path, limit: int = 12000) -> tuple[str, bool, bool, bool]:
    if _is_likely_binary_path(path):
        return "", False, True, False

    try:
        with path.open("rb") as handle:
            payload = handle.read(limit + 1)
    except Exception:
        return "", False, True, False

    if _looks_binary(payload):
        return "", False, True, False

    try:
        text = payload[:limit].decode("utf-8")
    except UnicodeDecodeError:
        return "", False, True, False

    truncated = len(payload) > limit
    diff = "\n".join(f"+{line}" for line in text.splitlines())
    return diff + ("\n... diff truncated ..." if truncated else ""), truncated, False, True


def _is_likely_binary_path(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_PREVIEW_EXTENSIONS


def _looks_binary(payload: bytes) -> bool:
    return b"\x00" in payload


def _is_git_binary_diff(diff: str) -> bool:
    return "GIT binary patch" in diff or re.search(r"^Binary files .+ differ$", diff, re.MULTILINE) is not None


def _truncate_diff(diff: str, limit: int = 16000) -> tuple[str, bool]:
    if len(diff) <= limit:
        return diff, False
    return diff[:limit].rstrip() + "\n... diff truncated ...", True


def _skip_file_change_path(path: str) -> bool:
    if path.startswith(".git/") or path.startswith("node_modules/"):
        return True
    if path.startswith(".venv/") or path.startswith("venv/") or "/__pycache__/" in f"/{path}/":
        return True
    if path.startswith(WORKFLOW_DIR + "/"):
        return True
    return False


def _is_artifact_path(path: str) -> bool:
    return path.startswith("artifacts/") or path.startswith("outputs/")


def _file_tree(root: Path, *, max_depth: int, depth: int = 0) -> List[Dict[str, Any]]:
    if depth > max_depth or not root.exists():
        return []
    children: List[Dict[str, Any]] = []
    allowed_top = {"references", "memory", "workflow", "artifacts", "outputs", "logs", WORKFLOW_DIR}
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return children
    for entry in entries:
        if depth == 0 and entry.name not in allowed_top:
            continue
        if entry.name == ".git":
            continue
        item: Dict[str, Any] = {
            "name": entry.name,
            "path": str(entry),
            "kind": "folder" if entry.is_dir() else "file",
        }
        if entry.is_dir():
            item["children"] = _file_tree(entry, max_depth=max_depth, depth=depth + 1)
        children.append(item)
    return children


def _create_project_export_zip(project: WorkflowProject) -> str:
    root = Path(project.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail="Workflow project root not found")
    fd, archive_path = tempfile.mkstemp(prefix=f"{_slug(project.name) or 'workflow'}-", suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for base, dirs, files in os.walk(root):
                base_path = Path(base)
                dirs[:] = [name for name in dirs if not _skip_export_dir(name)]
                for name in files:
                    if _skip_export_file(name):
                        continue
                    path = base_path / name
                    try:
                        rel = path.resolve().relative_to(root)
                    except ValueError:
                        continue
                    archive.write(path, rel.as_posix())
    except Exception as exc:
        _safe_unlink(archive_path)
        raise HTTPException(status_code=500, detail=f"Could not export workflow project: {exc}") from exc
    return archive_path


def _skip_export_dir(name: str) -> bool:
    return name in {".git", "node_modules", ".venv", "venv", "__pycache__", ".cache"}


def _skip_export_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(("-wal", "-shm", ".db-wal", ".db-shm", ".sqlite-wal", ".sqlite-shm"))


def _safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def _mode_label(mode: ExecutionMode) -> str:
    return {"single_step": "单步", "semi_auto": "半自动", "auto": "自动"}[mode]


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return value[:80]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
