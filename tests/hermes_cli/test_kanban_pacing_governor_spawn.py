from __future__ import annotations

import json
import subprocess


def _make_task(kb, *, assignee: str = "elias", provider_override=None, model_override=None):
    return kb.Task(
        id="t_pace_spawn",
        title="pace spawn",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
        provider_override=provider_override,
        model_override=model_override,
    )


def _write_state_file(path, *, chain, providers):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"chain": chain, "providers": providers}))


def _over_pace_provider(used_pct=90.0, allowed_pct=20.0):
    return {"five_hour_used_pct": used_pct, "five_hour_allowed_pct": allowed_pct, "error": None}


def _within_pace_provider(used_pct=5.0, allowed_pct=20.0):
    return {"five_hour_used_pct": used_pct, "five_hour_allowed_pct": allowed_pct, "error": None}


# --- Unit tests of the integration point itself (kanban_db_dispatch_pacing) -------------


def test_resolve_worker_provider_explicit_override_skips_governor(monkeypatch, tmp_path):
    """task.provider_override always wins and never even reaches the governor."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    called = {}
    monkeypatch.setattr(
        kbdp, "_resolve_profile_active_provider",
        lambda hermes_home: called.setdefault("called", True) or "should-not-be-used",
    )

    task = _make_task(kb, provider_override="pinned-provider")
    provider = kbdp.resolve_worker_provider(task, str(tmp_path))

    assert provider == "pinned-provider"
    assert "called" not in called  # governor resolution never invoked


def test_resolve_worker_provider_routes_to_fallback_when_primary_over_pace(monkeypatch, tmp_path):
    """ACCEPTANCE: a kanban worker spawn is routed to the fallback provider when the
    primary is over pace."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    state_path = tmp_path / "state" / "provider_pace.json"
    decision_log = tmp_path / "state" / "decisions.jsonl"
    _write_state_file(
        state_path,
        chain=["claude-subscription", "openai-codex"],
        providers={
            "claude-subscription": _over_pace_provider(),
            "openai-codex": _within_pace_provider(),
        },
    )
    monkeypatch.setattr(kbdp, "DEFAULT_STATE_PATH", state_path)
    monkeypatch.setattr(kbdp, "DEFAULT_DECISION_LOG_PATH", decision_log)
    monkeypatch.setattr(kbdp, "_aos_resolve_provider_for_new_spawn", None)
    monkeypatch.setattr(kbdp, "_resolve_profile_active_provider", lambda hermes_home: "claude-subscription")

    task = _make_task(kb)
    provider = kbdp.resolve_worker_provider(task, str(tmp_path))

    assert provider == "openai-codex"
    logged = [json.loads(line) for line in decision_log.read_text().splitlines()]
    assert logged and logged[-1]["provider"] == "openai-codex"
    assert logged[-1]["context"] == "kanban:t_pace_spawn"


def test_resolve_worker_provider_sticky_session_never_touched(monkeypatch, tmp_path):
    """ACCEPTANCE: a resumed/already-running task's provider is never touched -- sticky_provider
    short-circuits the governor unconditionally, even when the state file says the sticky
    provider is over pace."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    state_path = tmp_path / "state" / "provider_pace.json"
    _write_state_file(
        state_path,
        chain=["claude-subscription", "openai-codex"],
        providers={
            "claude-subscription": _over_pace_provider(),
            "openai-codex": _within_pace_provider(),
        },
    )
    monkeypatch.setattr(kbdp, "DEFAULT_STATE_PATH", state_path)
    monkeypatch.setattr(kbdp, "_aos_resolve_provider_for_new_spawn", None)

    task = _make_task(kb)
    provider = kbdp.resolve_worker_provider(
        task, str(tmp_path), sticky_provider="claude-subscription",
    )

    # Unchanged, despite claude-subscription being over pace in the state file above.
    assert provider == "claude-subscription"


def test_resolve_worker_provider_fails_open_on_missing_state_file(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    monkeypatch.setattr(kbdp, "DEFAULT_STATE_PATH", tmp_path / "no" / "such" / "file.json")
    monkeypatch.setattr(kbdp, "_aos_resolve_provider_for_new_spawn", None)
    monkeypatch.setattr(kbdp, "_resolve_profile_active_provider", lambda hermes_home: "claude-subscription")

    task = _make_task(kb)
    provider = kbdp.resolve_worker_provider(task, str(tmp_path))

    assert provider == "claude-subscription"  # default_provider, unchanged


def test_resolve_worker_provider_fails_open_on_governor_exception(monkeypatch, tmp_path):
    """ANY exception from governor resolution must fall back to None (no --provider flag),
    never block or crash the spawn."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    def _boom(hermes_home):
        raise RuntimeError("governor state unreachable")

    monkeypatch.setattr(kbdp, "_resolve_profile_active_provider", _boom)

    task = _make_task(kb)
    provider = kbdp.resolve_worker_provider(task, str(tmp_path))

    assert provider is None


def test_resolve_worker_provider_no_profile_provider_resolved(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    monkeypatch.setattr(kbdp, "_resolve_profile_active_provider", lambda hermes_home: None)

    task = _make_task(kb)
    assert kbdp.resolve_worker_provider(task, str(tmp_path)) is None


def test_resolve_worker_provider_none_hermes_home_is_noop(monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    task = _make_task(kb)
    assert kbdp.resolve_worker_provider(task, None) is None


# --- E2E: the real _worker_argv/_default_spawn integration ------------------------------


def test_default_spawn_pins_governed_provider_when_no_override(monkeypatch, tmp_path):
    """The dispatcher's actual spawn path (_worker_argv via _default_spawn) must carry the
    pacing governor's decision into --provider when the task has no explicit override."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    profile.joinpath("auth.json").write_text(
        json.dumps({"active_provider": "claude-subscription", "providers": {"claude-subscription": {}}}),
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    state_path = tmp_path / "state" / "provider_pace.json"
    _write_state_file(
        state_path,
        chain=["claude-subscription", "openai-codex"],
        providers={
            "claude-subscription": _over_pace_provider(),
            "openai-codex": _within_pace_provider(),
        },
    )

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kbdp, "DEFAULT_STATE_PATH", state_path)
    monkeypatch.setattr(kbdp, "_aos_resolve_provider_for_new_spawn", None)

    captured = {}

    class FakeProc:
        pid = 5150

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kbd._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 5150
    assert "--provider" in captured["cmd"]
    provider_arg = captured["cmd"][captured["cmd"].index("--provider") + 1]
    assert provider_arg == "openai-codex"  # routed away from the over-pace primary


def test_default_spawn_explicit_provider_override_wins_over_governor(monkeypatch, tmp_path):
    """An explicit task.provider_override must reach argv unchanged even when the governor's
    state file would have routed elsewhere."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    profile.joinpath("auth.json").write_text(
        json.dumps({"active_provider": "claude-subscription", "providers": {"claude-subscription": {}}}),
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    state_path = tmp_path / "state" / "provider_pace.json"
    _write_state_file(
        state_path,
        chain=["claude-subscription", "openai-codex"],
        providers={
            "claude-subscription": _over_pace_provider(),
            "openai-codex": _within_pace_provider(),
        },
    )

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_db_dispatch_pacing as kbdp

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kbdp, "DEFAULT_STATE_PATH", state_path)
    monkeypatch.setattr(kbdp, "_aos_resolve_provider_for_new_spawn", None)

    captured = {}

    class FakeProc:
        pid = 5151

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(
        kb, assignee="elias", model_override="claude-sonnet-4.6", provider_override="claude-subscription",
    )
    kbd._default_spawn(task, str(workspace))

    assert "--provider" in captured["cmd"]
    provider_arg = captured["cmd"][captured["cmd"].index("--provider") + 1]
    assert provider_arg == "claude-subscription"  # explicit override, not the governed openai-codex


def test_default_spawn_omits_provider_flag_when_profile_provider_unresolved(monkeypatch, tmp_path):
    """When the profile has no active_provider at all (fresh/never-logged-in profile), no
    --provider flag is added -- identical to pre-governor dispatch."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 5152

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kbd._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert "--provider" not in captured["cmd"]
