"""Tests for configurable Kanban worker model/reasoning dispatch."""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


MATRIX = {
    "default_marker": "medium",
    "none": {
        "first_attempt": {"model": "luna", "reasoning_effort": "none"},
        "second_attempt": {"model": "terra", "reasoning_effort": "none"},
        "third_plus": {"model": "terra", "reasoning_effort": "none"},
    },
    "low": {
        "first_attempt": {"model": "luna", "reasoning_effort": "medium"},
        "second_attempt": {"model": "terra", "reasoning_effort": "low"},
        "third_plus": {"model": "terra", "reasoning_effort": "medium"},
    },
    "medium": {
        "first_attempt": {"model": "terra", "reasoning_effort": "medium"},
        "second_attempt": {"model": "sol", "reasoning_effort": "low"},
        "third_plus": {"model": "sol", "reasoning_effort": "medium"},
    },
    "high": {
        "first_attempt": {"model": "sol", "reasoning_effort": "medium"},
        "second_attempt": {"model": "sol", "reasoning_effort": "medium"},
        "third_plus": {"model": "sol", "reasoning_effort": "high"},
    },
}


def _task(body="", failures=0, model_override=None):
    return kb.Task(
        id="t_matrix",
        title="matrix",
        body=body,
        assignee="dev",
        status="ready",
        priority=0,
        created_by="test",
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        consecutive_failures=failures,
        model_override=model_override,
    )


@pytest.mark.parametrize(
    ("marker", "failures", "expected"),
    [
        ("low", 0, ("luna", "medium")),
        ("low", 1, ("terra", "low")),
        ("low", 2, ("terra", "medium")),
        ("low", 3, ("terra", "medium")),
        ("medium", 0, ("terra", "medium")),
        ("medium", 1, ("sol", "low")),
        ("medium", 2, ("sol", "medium")),
        (None, 0, ("terra", "medium")),
        (None, 1, ("sol", "low")),
        (None, 2, ("sol", "medium")),
        ("high", 0, ("sol", "medium")),
        ("high", 1, ("sol", "medium")),
        ("high", 2, ("sol", "high")),
        ("none", 0, ("luna", "none")),
        ("none", 2, ("terra", "none")),
    ],
)
def test_dispatch_matrix(monkeypatch, marker, failures, expected):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config", lambda: {"kanban": {"worker_dispatch_matrix": MATRIX}})
    body = "unmarked" if marker is None else f"thinking_budget: {marker}"
    assert kb._resolve_kanban_worker_dispatch(_task(body, failures)) == expected


@pytest.mark.parametrize("failures", [0, 1, 2, 99])
def test_explicit_model_override_is_strongest(monkeypatch, failures):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config", lambda: {"kanban": {"worker_dispatch_matrix": MATRIX}})
    assert kb._resolve_kanban_worker_dispatch(
        _task("reasoning_effort: high", failures, "  custom-model  ")
    ) == ("custom-model", "medium" if failures < 2 else "high")


def test_unconfigured_matrix_preserves_existing_behavior(monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config", lambda: {"kanban": {"worker_dispatch_matrix": None}})
    assert kb._resolve_kanban_worker_dispatch(_task(model_override="pinned")) == ("pinned", None)
    assert kb._resolve_kanban_worker_dispatch(_task()) == (None, None)


def test_malformed_cell_fails_closed_to_existing_behavior(monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {"worker_dispatch_matrix": {"default_marker": "medium", "medium": []}}},
    )
    assert kb._resolve_kanban_worker_dispatch(_task(model_override="pinned")) == ("pinned", None)


def test_spawn_exports_atomic_model_and_reasoning_cell(tmp_path, monkeypatch):
    from hermes_cli import config, profiles

    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs["env"]
            self.pid = 4242

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(config, "load_config", lambda: {"kanban": {"worker_dispatch_matrix": MATRIX}})
    monkeypatch.setattr(profiles, "resolve_profile_env", lambda _profile: str(tmp_path))
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: tmp_path / "kanban.db")
    monkeypatch.setattr(kb, "workspaces_root", lambda board=None: tmp_path / "workspaces")
    monkeypatch.setattr("subprocess.Popen", _FakePopen)

    kb._default_spawn(_task("reasoning_effort: medium", failures=1), str(workspace))

    assert captured["env"]["HERMES_KANBAN_WORKER_MODEL"] == "sol"
    assert captured["env"]["HERMES_KANBAN_REASONING_EFFORT"] == "low"
    model_index = captured["cmd"].index("-m")
    assert captured["cmd"][model_index + 1] == "sol"
