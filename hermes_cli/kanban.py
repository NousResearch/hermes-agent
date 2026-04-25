"""Native Hermes Kanban board backend.

The board is deliberately scoped to Hermes concepts: cards launch Hermes
sessions in a workspace, persist lightweight lifecycle state, and expose git
diffs for review.  No external board/runtime code is required.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from hermes_cli.config import get_compatible_custom_providers, get_hermes_home, load_config

ColumnId = Literal["backlog", "running", "review", "done", "trash"]
TaskStatus = Literal["idle", "running", "review", "done", "failed", "stopped"]

COLUMNS: list[dict[str, str]] = [
    {"id": "backlog", "title": "Backlog"},
    {"id": "running", "title": "Running"},
    {"id": "review", "title": "Review"},
    {"id": "done", "title": "Done"},
    {"id": "trash", "title": "Trash"},
]

_RUNNING: dict[str, subprocess.Popen[str]] = {}
_RUNNING_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _workspace_path(path: str) -> Path:
    if not path.strip():
        raise HTTPException(status_code=400, detail="workspace_path is required")
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise HTTPException(status_code=400, detail="Workspace path must be an existing directory")
    return candidate


def _run_git(workspace: Path, *args: str, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _is_git_repo(workspace: Path) -> bool:
    result = _run_git(workspace, "rev-parse", "--is-inside-work-tree", timeout=5)
    return result.returncode == 0 and result.stdout.strip() == "true"


def _tail_text(path: Path, *, lines: int = 80) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


class KanbanCard(BaseModel):
    id: str = Field(default_factory=_new_id)
    title: str
    prompt: str
    model: str | None = None
    column: ColumnId = "backlog"
    status: TaskStatus = "idle"
    workspace_path: str | None = None
    pid: int | None = None
    log_path: str | None = None
    last_activity: str | None = None
    error: str | None = None
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)
    started_at: float | None = None
    finished_at: float | None = None

    @field_validator("title", "prompt")
    @classmethod
    def _non_empty_trimmed(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed

    @field_validator("model")
    @classmethod
    def _empty_model_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class KanbanBoard(BaseModel):
    version: int = 1
    cards: list[KanbanCard] = Field(default_factory=list)
    updated_at: float = Field(default_factory=_now)


class BoardResponse(BaseModel):
    board: KanbanBoard
    columns: list[dict[str, str]]
    default_workspace_path: str
    active_model: str
    active_provider: str
    model_options: list[str]


class CardCreateRequest(BaseModel):
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    model: str | None = None
    workspace_path: str | None = None

    @field_validator("title", "prompt")
    @classmethod
    def _non_empty_trimmed(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed

    @field_validator("model")
    @classmethod
    def _empty_model_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class CardUpdateRequest(BaseModel):
    title: str | None = None
    prompt: str | None = None
    model: str | None = None
    column: ColumnId | None = None
    workspace_path: str | None = None

    @field_validator("title", "prompt")
    @classmethod
    def _non_empty_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be blank")
        return trimmed

    @field_validator("model")
    @classmethod
    def _empty_model_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class CardStartRequest(BaseModel):
    workspace_path: str | None = None
    model: str | None = None

    @field_validator("model")
    @classmethod
    def _empty_model_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class CardLogResponse(BaseModel):
    card_id: str
    log: str
    log_path: str | None


class CardDiffResponse(BaseModel):
    card_id: str
    workspace_path: str | None
    is_git_repo: bool
    summary: str
    diff: str


class KanbanStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_hermes_home() / "kanban" / "board.json"
        self._lock = threading.RLock()

    def load(self) -> KanbanBoard:
        with self._lock:
            if not self.path.exists():
                return KanbanBoard()
            try:
                return KanbanBoard.model_validate_json(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Unable to read Kanban board at {self.path}: {exc}") from exc

    def save(self, board: KanbanBoard) -> KanbanBoard:
        with self._lock:
            board.updated_at = _now()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(board.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(self.path)
            return board

    def get_card(self, card_id: str) -> tuple[KanbanBoard, KanbanCard]:
        board = self.load()
        for card in board.cards:
            if card.id == card_id:
                return board, card
        raise HTTPException(status_code=404, detail="Kanban card not found")

    def upsert_card(self, card: KanbanCard) -> KanbanCard:
        board = self.load()
        for idx, existing in enumerate(board.cards):
            if existing.id == card.id:
                card.updated_at = _now()
                board.cards[idx] = card
                self.save(board)
                return card
        card.updated_at = _now()
        board.cards.append(card)
        self.save(board)
        return card


store = KanbanStore()
router = APIRouter()


def _default_workspace() -> str:
    return str(Path.cwd().resolve())


def _logs_dir() -> Path:
    path = get_hermes_home() / "kanban" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configured_model_info() -> tuple[str, str]:
    config = load_config()
    model_config = config.get("model", "")
    if isinstance(model_config, dict):
        model = str(model_config.get("default") or model_config.get("model") or model_config.get("name") or "")
        provider = str(model_config.get("provider") or "")
    else:
        model = str(model_config or "")
        provider = ""
    provider = provider or os.getenv("HERMES_INFERENCE_PROVIDER", "auto")
    return model, provider


def _model_options(active_model: str, active_provider: str) -> list[str]:
    options: list[str] = []

    def add(model: Any) -> None:
        value = str(model or "").strip()
        if value and value not in options:
            options.append(value)

    add(active_model)
    try:
        from hermes_cli.models import _PROVIDER_MODELS
        for model in _PROVIDER_MODELS.get(active_provider, [])[:40]:
            add(model)
    except Exception:
        pass

    try:
        config = load_config()
        for entry in get_compatible_custom_providers(config):
            add(entry.get("model"))
    except Exception:
        pass

    return options


def _finish_card(card_id: str, return_code: int) -> None:
    with _RUNNING_LOCK:
        _RUNNING.pop(card_id, None)

    try:
        board, card = store.get_card(card_id)
    except HTTPException:
        return

    card.pid = None
    card.finished_at = _now()
    card.updated_at = _now()
    if card.status == "stopped":
        if not card.last_activity:
            card.last_activity = "Hermes task stopped."
        store.save(board)
        return
    if return_code == 0:
        card.status = "review"
        card.column = "review"
        card.error = None
        card.last_activity = "Hermes task finished; ready for review."
    else:
        card.status = "failed"
        card.column = "review"
        card.error = f"Hermes task exited with code {return_code}"
        tail = _tail_text(Path(card.log_path)) if card.log_path else ""
        card.last_activity = tail.splitlines()[-1] if tail else card.error
    store.save(board)


def _watch_process(card_id: str, process: subprocess.Popen[str]) -> None:
    return_code = process.wait()
    _finish_card(card_id, return_code)


def _terminate_process(card: KanbanCard) -> None:
    process: subprocess.Popen[str] | None
    with _RUNNING_LOCK:
        process = _RUNNING.pop(card.id, None)

    if process and process.poll() is None:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        return

    if card.pid:
        try:
            if os.name == "nt":
                os.kill(card.pid, signal.SIGTERM)
            else:
                os.killpg(card.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


@router.get("/board", response_model=BoardResponse)
async def get_board() -> BoardResponse:
    active_model, active_provider = _configured_model_info()
    return BoardResponse(
        board=store.load(),
        columns=COLUMNS,
        default_workspace_path=_default_workspace(),
        active_model=active_model,
        active_provider=active_provider,
        model_options=_model_options(active_model, active_provider),
    )


@router.post("/cards", response_model=KanbanCard)
async def create_card(body: CardCreateRequest) -> KanbanCard:
    workspace = body.workspace_path.strip() if body.workspace_path else None
    if workspace:
        _workspace_path(workspace)
    card = KanbanCard(
        title=body.title,
        prompt=body.prompt,
        model=body.model,
        workspace_path=workspace,
    )
    return store.upsert_card(card)


@router.put("/cards/{card_id}", response_model=KanbanCard)
async def update_card(card_id: str, body: CardUpdateRequest) -> KanbanCard:
    _, card = store.get_card(card_id)
    if card.status == "running":
        raise HTTPException(status_code=409, detail="Cannot edit a running card")
    if body.title is not None:
        card.title = body.title
    if body.prompt is not None:
        card.prompt = body.prompt
    if "model" in body.model_fields_set:
        card.model = body.model
    if body.column is not None:
        card.column = body.column
        if body.column == "done":
            card.status = "done"
        elif body.column == "backlog" and card.status in {"review", "failed", "stopped"}:
            card.status = "idle"
    if body.workspace_path is not None:
        workspace = body.workspace_path.strip()
        card.workspace_path = str(_workspace_path(workspace)) if workspace else None
    return store.upsert_card(card)


@router.delete("/cards/{card_id}")
async def delete_card(card_id: str) -> dict[str, bool]:
    board, card = store.get_card(card_id)
    if card.status == "running":
        raise HTTPException(status_code=409, detail="Stop the card before deleting it")
    board.cards = [c for c in board.cards if c.id != card_id]
    store.save(board)
    return {"ok": True}


@router.post("/cards/{card_id}/start", response_model=KanbanCard)
async def start_card(card_id: str, body: CardStartRequest) -> KanbanCard:
    _, card = store.get_card(card_id)
    if card.status == "running":
        raise HTTPException(status_code=409, detail="Card is already running")

    workspace_raw = body.workspace_path or card.workspace_path or _default_workspace()
    workspace = _workspace_path(workspace_raw)
    model = body.model or card.model
    log_path = _logs_dir() / f"{card.id}-{int(_now())}.log"
    command = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "chat",
        "-Q",
        "--source",
        "kanban",
    ]
    if model:
        command.extend(["--model", model])
    command.extend([
        "-q",
        card.prompt,
    ])
    env = os.environ.copy()
    env.setdefault("HERMES_ACCEPT_HOOKS", "1")

    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
    except Exception as exc:
        log_handle.close()
        card.status = "failed"
        card.column = "review"
        card.error = str(exc)
        card.last_activity = f"Failed to start Hermes task: {exc}"
        return store.upsert_card(card)
    finally:
        # The child owns the descriptor; the parent should not keep it open.
        log_handle.close()

    card.status = "running"
    card.column = "running"
    card.workspace_path = str(workspace)
    card.model = model
    card.pid = process.pid
    card.log_path = str(log_path)
    card.started_at = _now()
    card.finished_at = None
    card.error = None
    card.last_activity = "Hermes task started."
    updated = store.upsert_card(card)

    with _RUNNING_LOCK:
        _RUNNING[card.id] = process
    watcher = threading.Thread(target=_watch_process, args=(card.id, process), daemon=True)
    watcher.start()
    return updated


@router.post("/cards/{card_id}/stop", response_model=KanbanCard)
async def stop_card(card_id: str) -> KanbanCard:
    _, card = store.get_card(card_id)
    if card.status != "running":
        return card
    _terminate_process(card)
    card.status = "stopped"
    card.column = "review"
    card.pid = None
    card.finished_at = _now()
    card.last_activity = "Hermes task stopped."
    return store.upsert_card(card)


@router.get("/cards/{card_id}/log", response_model=CardLogResponse)
async def get_card_log(card_id: str, lines: int = 120) -> CardLogResponse:
    _, card = store.get_card(card_id)
    log_path = Path(card.log_path) if card.log_path else None
    return CardLogResponse(
        card_id=card.id,
        log=_tail_text(log_path, lines=max(1, min(lines, 500))) if log_path else "",
        log_path=card.log_path,
    )


@router.get("/cards/{card_id}/diff", response_model=CardDiffResponse)
async def get_card_diff(card_id: str) -> CardDiffResponse:
    _, card = store.get_card(card_id)
    if not card.workspace_path:
        return CardDiffResponse(
            card_id=card.id,
            workspace_path=None,
            is_git_repo=False,
            summary="No workspace path is attached to this card.",
            diff="",
        )

    workspace = _workspace_path(card.workspace_path)
    if not _is_git_repo(workspace):
        return CardDiffResponse(
            card_id=card.id,
            workspace_path=str(workspace),
            is_git_repo=False,
            summary="Workspace is not a git repository.",
            diff="",
        )

    summary = _run_git(workspace, "status", "--short", timeout=10)
    diff = _run_git(workspace, "diff", "--", timeout=20)
    return CardDiffResponse(
        card_id=card.id,
        workspace_path=str(workspace),
        is_git_repo=True,
        summary=summary.stdout.strip() or "Working tree is clean.",
        diff=diff.stdout,
    )


def include_kanban_routes(app: Any) -> None:
    app.include_router(router, prefix="/api/kanban", tags=["kanban"])
