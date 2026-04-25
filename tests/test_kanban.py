import asyncio
import subprocess
from pathlib import Path

import pytest
from fastapi import HTTPException

from hermes_cli import kanban
from hermes_cli.kanban import (
    CardCreateRequest,
    CardDiffResponse,
    CardStartRequest,
    CardUpdateRequest,
    KanbanCard,
    KanbanStore,
)


def test_kanban_store_round_trips_board(tmp_path: Path):
    store = KanbanStore(tmp_path / "board.json")
    card = KanbanCard(title="Build thing", prompt="Implement the thing")

    store.upsert_card(card)

    loaded = store.load()
    assert loaded.cards[0].id == card.id
    assert loaded.cards[0].title == "Build thing"
    assert loaded.cards[0].status == "idle"


def test_create_card_validates_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    card = asyncio.run(kanban.create_card(
        CardCreateRequest(
            title="Native task",
            prompt="Run inside this workspace",
            workspace_path=str(workspace),
        )
    ))

    assert card.workspace_path == str(workspace)
    assert kanban.store.load().cards[0].id == card.id


def test_create_card_stores_trimmed_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    card = asyncio.run(kanban.create_card(
        CardCreateRequest(
            title="Model task",
            prompt="Run on a specific model",
            model=" anthropic/claude-sonnet-4.6 ",
            workspace_path=str(workspace),
        )
    ))

    assert card.model == "anthropic/claude-sonnet-4.6"
    assert kanban.store.load().cards[0].model == "anthropic/claude-sonnet-4.6"


def test_update_card_clears_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))
    card = kanban.store.upsert_card(
        KanbanCard(
            title="Switch back",
            prompt="Use the dashboard default again",
            model="anthropic/claude-sonnet-4.6",
        )
    )

    updated = asyncio.run(kanban.update_card(card.id, CardUpdateRequest(model="")))

    assert updated.model is None
    assert kanban.store.load().cards[0].model is None


def test_create_card_rejects_missing_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(kanban.create_card(
            CardCreateRequest(
                title="Bad workspace",
                prompt="This should not be saved",
                workspace_path=str(tmp_path / "missing"),
            )
        ))

    assert exc.value.status_code == 400
    assert kanban.store.load().cards == []


def test_update_card_rejects_blank_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))
    card = kanban.store.upsert_card(KanbanCard(title="Keep title", prompt="Keep prompt"))

    with pytest.raises(ValueError):
        CardUpdateRequest(prompt="   ")

    loaded = kanban.store.load().cards[0]
    assert loaded.id == card.id
    assert loaded.prompt == "Keep prompt"


def test_card_diff_reports_git_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    (workspace / "notes.txt").write_text("hello\n", encoding="utf-8")

    card = kanban.store.upsert_card(
        KanbanCard(
            title="Review changes",
            prompt="Inspect the diff",
            workspace_path=str(workspace),
        )
    )

    response = asyncio.run(kanban.get_card_diff(card.id))

    assert isinstance(response, CardDiffResponse)
    assert response.is_git_repo is True
    assert "notes.txt" in response.summary


def test_start_card_records_start_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    card = kanban.store.upsert_card(
        KanbanCard(
            title="Launch failure",
            prompt="This launch is patched to fail",
            workspace_path=str(workspace),
        )
    )

    def fail_popen(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(kanban.subprocess, "Popen", fail_popen)

    updated = asyncio.run(kanban.start_card(card.id, CardStartRequest()))

    assert updated.status == "failed"
    assert updated.column == "review"
    assert updated.error == "boom"


def test_start_card_passes_model_to_chat_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))
    monkeypatch.setattr(kanban, "get_hermes_home", lambda: tmp_path / "hermes-home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    card = kanban.store.upsert_card(
        KanbanCard(
            title="Launch with model",
            prompt="Use the selected model",
            model="anthropic/claude-sonnet-4.6",
            workspace_path=str(workspace),
        )
    )
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

        def wait(self):
            return 0

    class NoopThread:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return FakeProcess()

    monkeypatch.setattr(kanban.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kanban.threading, "Thread", NoopThread)

    updated = asyncio.run(kanban.start_card(card.id, CardStartRequest()))

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--model") + 1] == "anthropic/claude-sonnet-4.6"
    assert command[command.index("-q") + 1] == "Use the selected model"
    assert captured["cwd"] == str(workspace)
    assert updated.status == "running"
    assert updated.model == "anthropic/claude-sonnet-4.6"


def test_board_exposes_configured_model_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))

    def fake_load_config():
        return {
            "model": {"provider": "nous", "default": "nous/deephermes-3-mistral-24b-preview"},
            "providers": {
                "local": {
                    "name": "Local",
                    "base_url": "http://localhost:1234/v1",
                    "model": "local/hermes",
                }
            },
        }

    monkeypatch.setattr(kanban, "load_config", fake_load_config)

    response = asyncio.run(kanban.get_board())

    assert response.active_provider == "nous"
    assert response.active_model == "nous/deephermes-3-mistral-24b-preview"
    assert "nous/deephermes-3-mistral-24b-preview" in response.model_options
    assert "local/hermes" in response.model_options


def test_finish_does_not_overwrite_stopped_card(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(kanban, "store", KanbanStore(tmp_path / "board.json"))
    card = kanban.store.upsert_card(
        KanbanCard(
            title="Stopped",
            prompt="Do not resurrect as failed",
            status="stopped",
            column="review",
            pid=123,
            last_activity="Hermes task stopped.",
        )
    )

    kanban._finish_card(card.id, -15)

    loaded = kanban.store.load().cards[0]
    assert loaded.status == "stopped"
    assert loaded.column == "review"
    assert loaded.pid is None
    assert loaded.last_activity == "Hermes task stopped."
