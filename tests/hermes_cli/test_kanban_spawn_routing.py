"""Regression tests for deterministic local-first Kanban worker routing."""

import subprocess

from hermes_cli.kanban_db import (
    _resolve_explicit_local_task_route,
    _resolve_local_first_route,
)


def test_local_first_route_uses_effective_local_fallback_before_remote() -> None:
    """A remote primary must not cost a doomed egress attempt first."""
    profile = {
        "model": {"provider": "nous", "default": "poolside/laguna-xs-2.1:free"},
        "fallback_model": [
            {"provider": "openai-codex", "model": "gpt-5.6-luna"},
            {"provider": "ollama-launch", "model": "hermes-review-fast:latest"},
        ],
    }

    assert _resolve_local_first_route(profile) == (
        "ollama-launch",
        "hermes-review-fast:latest",
    )


def test_local_first_route_prefers_local_primary() -> None:
    profile = {
        "model": {"provider": "ollama-launch", "default": "hermes-cron-fast:latest"},
        "fallback_model": [
            {"provider": "nous", "model": "poolside/laguna-xs-2.1:free"},
        ],
    }

    assert _resolve_local_first_route(profile) == (
        "ollama-launch",
        "hermes-cron-fast:latest",
    )


def test_local_first_route_fails_closed_when_no_local_model_is_configured() -> None:
    profile = {
        "model": {"provider": "nous", "default": "poolside/laguna-xs-2.1:free"},
        "fallback_model": [
            {"provider": "openai-codex", "model": "gpt-5.6-luna"},
        ],
    }

    assert _resolve_local_first_route(profile) is None


def test_explicit_local_task_pin_beats_profile_local_first_route() -> None:
    from hermes_cli.kanban_db import Task

    task = Task(
        id="t_explicit_local",
        title="explicit local route",
        body=None,
        assignee="researcher-a",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        model_override="qwen3.6:35b-a3b",
        provider_override="ollama-launch",
    )

    assert _resolve_explicit_local_task_route(task) == (
        "ollama-launch",
        "qwen3.6:35b-a3b",
    )


def test_default_spawn_honors_explicit_local_pin_when_local_first_is_enabled(
    monkeypatch, tmp_path
) -> None:
    """A per-card local recovery pin must reach the child argv unchanged."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "researcher-a"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text(
        "kanban:\n  local_first: true\n", encoding="utf-8"
    )
    profile.joinpath("config.yaml").write_text(
        """
model:
  provider: nous
  default: poolside/laguna-xs-2.1:free
fallback_model:
  - provider: ollama-launch
    model: gemma4:e2b-it-qat
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    task = kb.Task(
        id="t_explicit_local_spawn",
        title="explicit local route",
        body=None,
        assignee="researcher-a",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        model_override="qwen3.6:35b-a3b",
        provider_override="ollama-launch",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert kb._default_spawn(task, str(workspace)) == 4245
    model_index = captured["cmd"].index("-m")
    provider_index = captured["cmd"].index("--provider")
    assert captured["cmd"][model_index + 1] == "qwen3.6:35b-a3b"
    assert captured["cmd"][provider_index + 1] == "ollama-launch"
    reasoning_index = captured["cmd"].index("--reasoning")
    assert captured["cmd"][reasoning_index + 1] == "none"
    assert "gemma4:e2b-it-qat" not in captured["cmd"]


def test_default_spawn_bridges_process_local_provider_into_profile_worker(
    monkeypatch, tmp_path
) -> None:
    """A profile worker must not turn a root-only local provider into remote fallback."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "researcher-a"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text(
        "providers:\n  ollama-launch:\n    api: http://127.0.0.1:11434/v1\n",
        encoding="utf-8",
    )
    profile.joinpath("config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-5.6-luna\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4247

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    task = kb.Task(
        id="t_profile_local_provider",
        title="profile local provider",
        body=None,
        assignee="researcher-a",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        model_override="qwen3.5:4b",
        provider_override="ollama-launch",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert kb._default_spawn(task, str(workspace)) == 4247
    provider_index = captured["cmd"].index("--provider")
    assert captured["cmd"][provider_index + 1] == "ollama"
    assert captured["env"]["CUSTOM_BASE_URL"] == "http://127.0.0.1:11434/v1"
    assert captured["env"]["HERMES_KANBAN_LOCAL_ONLY"] == "1"
    toolsets = captured["cmd"][captured["cmd"].index("--toolsets") + 1].split(",")
    assert "terminal" in toolsets
    assert "code_execution" not in toolsets


def test_default_spawn_honors_explicit_remote_pin_when_local_first_is_enabled(
    monkeypatch, tmp_path
) -> None:
    """A PR card's explicit Codex route must not be replaced by local-first."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "pr-repair-steward"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text(
        "kanban:\n  local_first: true\n", encoding="utf-8"
    )
    profile.joinpath("config.yaml").write_text(
        """
model:
  provider: nous
  default: poolside/laguna-xs-2.1:free
fallback_model:
  - provider: ollama-launch
    model: gemma4:e2b-it-qat
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4246

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    task = kb.Task(
        id="t_explicit_remote_spawn",
        title="explicit remote route",
        body=None,
        assignee="pr-repair-steward",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        model_override="gpt-5.6-terra",
        provider_override="openai-codex",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert kb._default_spawn(task, str(workspace)) == 4246
    model_index = captured["cmd"].index("-m")
    provider_index = captured["cmd"].index("--provider")
    assert captured["cmd"][model_index + 1] == "gpt-5.6-terra"
    assert captured["cmd"][provider_index + 1] == "openai-codex"
    assert "gemma4:e2b-it-qat" not in captured["cmd"]
