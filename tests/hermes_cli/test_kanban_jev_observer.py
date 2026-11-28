from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_decompose as decompose
from hermes_cli import kanban_jev_observer as observer


def _roster():
    return [
        {"name": "default", "description": "Local implementation worker"},
        {"name": "cloud", "description": "Complex reasoning specialist"},
    ]


def test_build_request_creates_bounded_per_candidate_questions():
    candidates = [
        {"title": "Implement parser", "body": "Add tests first", "assignee": "default"},
        {"title": "Review architecture", "body": "Inspect tradeoffs", "assignee": "cloud"},
    ]

    state, questions = observer._build_request(candidates, _roster(), max_candidates=1)

    assert len(state["candidate_tasks"]) == 1
    assert set(questions) == {"route_0", "risk_0", "approval_0", "complexity_0"}
    assert set(questions["route_0"]["criteria"]) == {"default", "cloud", "ask_user"}


def test_summarize_compares_jev_route_without_changing_actual_route():
    candidates = [{"title": "Implement parser", "body": "", "assignee": "default"}]
    result = {
        "model": "jev-test",
        "latency_ms": 12.5,
        "usage": {"input_tokens": 10, "output_tokens": 4},
        "answers": {
            "route_0": {"type": "choice", "choice": "cloud", "confidence": 0.8},
            "risk_0": {"type": "choice", "choice": "low", "confidence": 0.9},
            "approval_0": {"type": "noul", "noul": 0.1},
            "complexity_0": {"type": "choice", "choice": "moderate", "confidence": 0.7},
        },
    }

    event = observer._summarize_result("t_123", candidates, result)

    assert event["mode"] == "observe"
    assert event["decisions"][0]["actual_assignee"] == "default"
    assert event["decisions"][0]["jev_assignee"] == "cloud"
    assert event["decisions"][0]["route_agrees"] is False
    assert event["decisions"][0]["requires_human_approval_probability"] == 0.1


def test_observe_routing_is_fail_open_and_audits_safe_error(monkeypatch, tmp_path: Path):
    audit = tmp_path / "jev.jsonl"
    cfg = {
        "kanban": {
            "jev_observer": {
                "enabled": True,
                "mode": "observe",
                "profile": "milly",
                "audit_log": str(audit),
            }
        }
    }
    monkeypatch.setattr(observer, "_load_config", lambda: cfg)
    monkeypatch.setattr(observer, "_evaluate", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("secret text")))

    result = observer.observe_routing(
        task_id="t_123",
        candidates=[{"title": "Read file", "body": "No writes", "assignee": "default"}],
        roster=_roster(),
    )

    assert result is None
    event = json.loads(audit.read_text().strip())
    assert event["status"] == "error"
    assert event["error_type"] == "RuntimeError"
    assert "secret text" not in audit.read_text()


def test_decomposer_bridge_passes_actual_candidates_to_observer(monkeypatch):
    seen = {}

    def fake_observe_routing(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(observer, "observe_routing", fake_observe_routing)
    routing = decompose._Routing(
        orchestrator="milly",
        default_assignee="default",
        auto_promote=True,
        roster=_roster(),
        valid_names={"default", "cloud"},
    )
    candidates = [{"title": "Inspect", "body": "Read only", "assignee": "default"}]

    decompose._observe_jev("t_bridge", candidates, routing)

    assert seen == {"task_id": "t_bridge", "candidates": candidates, "roster": _roster()}


def test_audit_write_failure_is_fail_open(monkeypatch, tmp_path: Path):
    cfg = {
        "kanban": {
            "jev_observer": {
                "enabled": True,
                "mode": "observe",
                "profile": "milly",
                "audit_log": str(tmp_path / "jev.jsonl"),
            }
        }
    }
    monkeypatch.setattr(observer, "_load_config", lambda: cfg)
    monkeypatch.setattr(
        observer,
        "_evaluate",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("network failed")),
    )
    monkeypatch.setattr(
        observer,
        "_append_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = observer.observe_routing(
        task_id="t_audit_failure",
        candidates=[{"title": "Read file", "body": "No writes", "assignee": "default"}],
        roster=_roster(),
    )

    assert result is None


def test_single_mutation_failure_is_not_observed(monkeypatch):
    seen = []
    routing = decompose._Routing("milly", "default", True, _roster(), {"default", "cloud"})
    task = SimpleNamespace(id="t_race", title="Original", body="Body", assignee=None)

    @contextmanager
    def fake_connect():
        yield object()

    monkeypatch.setattr(decompose.kbc, "connect_closing", fake_connect)
    monkeypatch.setattr(decompose.kb, "specify_triage_task", lambda *args, **kwargs: False)
    monkeypatch.setattr(decompose, "_observe_jev", lambda *args, **kwargs: seen.append(args))

    outcome = decompose._apply_single(
        task,
        {"title": "Proposed", "body": "New body", "assignee": "default"},
        routing,
        "tester",
    )

    assert outcome.ok is False
    assert seen == []


def test_single_success_observes_committed_concurrent_values(monkeypatch):
    seen = []
    routing = decompose._Routing("milly", "default", True, _roster(), {"default", "cloud"})
    stale = SimpleNamespace(id="t_race", title="Stale", body="Old body", assignee=None)
    committed = SimpleNamespace(
        id="t_race", title="Proposed", body="Concurrent body", assignee="cloud",
    )

    @contextmanager
    def fake_connect():
        yield object()

    def fake_specify(*args, **kwargs):
        assert kwargs["return_snapshot"] is True
        return committed

    monkeypatch.setattr(decompose.kbc, "connect_closing", fake_connect)
    monkeypatch.setattr(decompose.kb, "specify_triage_task", fake_specify)
    monkeypatch.setattr(decompose, "_observe_jev", lambda *args: seen.append(args))

    outcome = decompose._apply_single(
        stale,
        {"title": "Proposed", "body": None, "assignee": "default"},
        routing,
        "tester",
    )

    assert outcome.ok is True
    assert seen[0][1] == [{
        "title": "Proposed", "body": "Concurrent body", "assignee": "cloud",
    }]


def test_fanout_rejection_is_not_observed(monkeypatch):
    seen = []
    routing = decompose._Routing("milly", "default", True, _roster(), {"default", "cloud"})

    @contextmanager
    def fake_connect():
        yield object()

    monkeypatch.setattr(decompose.kbc, "connect_closing", fake_connect)
    monkeypatch.setattr(decompose, "decompose_triage_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(decompose, "_observe_jev", lambda *args, **kwargs: seen.append(args))

    outcome = decompose._apply_fanout(
        "t_race",
        {"tasks": [{"title": "Child", "body": "Work", "assignee": "default", "parents": []}]},
        routing,
        "tester",
    )

    assert outcome.ok is False
    assert seen == []


def test_success_audit_excludes_body_and_credentials(monkeypatch, tmp_path: Path):
    audit = tmp_path / "jev.jsonl"
    cfg = {
        "kanban": {
            "jev_observer": {
                "enabled": True,
                "mode": "observe",
                "profile": "milly",
                "audit_log": str(audit),
            }
        }
    }
    secret = "credential-must-not-be-logged"
    body = "private task body must not be logged"
    monkeypatch.setattr(observer, "_load_config", lambda: cfg)
    monkeypatch.setattr(
        observer,
        "_evaluate",
        lambda **kwargs: {
            "model": "jev-test",
            "backend": "typesafe",
            "latency_ms": 1,
            "usage": {},
            "answers": {},
            "debug_token": secret,
        },
    )

    observer.observe_routing(
        task_id="t_private",
        candidates=[{"title": "Safe title", "body": body, "assignee": "default"}],
        roster=_roster(),
    )

    logged = audit.read_text(encoding="utf-8")
    assert secret not in logged
    assert body not in logged


def test_evaluate_uses_only_explicit_profile_credentials(monkeypatch, tmp_path: Path):
    profile_home = tmp_path / "profiles" / "milly"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "plugins:\n  entries:\n    jev:\n      settings:\n        connection:\n          backend: typesafe\n",
        encoding="utf-8",
    )
    (profile_home / ".env").write_text("TYPESAFE_API_KEY=profile-only-token\n", encoding="utf-8")
    seen = {}
    client = SimpleNamespace(evaluate=lambda **kwargs: seen.update(kwargs) or {"answers": {}})
    monkeypatch.setattr(observer, "_profile_home", lambda profile: profile_home)
    monkeypatch.setattr(observer, "_load_plugin_client", lambda path: client)
    monkeypatch.setenv("TYPESAFE_API_KEY", "process-token-must-not-be-used")

    observer._evaluate(profile="milly", state={}, questions={}, timeout=1)

    assert seen["token"] == "profile-only-token"


def test_evaluate_does_not_fall_back_when_profile_credential_missing(monkeypatch, tmp_path: Path):
    profile_home = tmp_path / "profiles" / "milly"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "plugins:\n  entries:\n    jev:\n      settings:\n        connection:\n          backend: typesafe\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(observer, "_profile_home", lambda profile: profile_home)
    monkeypatch.setenv("TYPESAFE_API_KEY", "process-token-must-not-be-used")

    with pytest.raises(RuntimeError, match="credential is unavailable"):
        observer._evaluate(profile="milly", state={}, questions={}, timeout=1)


def test_evaluate_disables_dotenv_interpolation(monkeypatch, tmp_path: Path):
    profile_home = tmp_path / "profiles" / "milly"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "plugins:\n  entries:\n    jev:\n      settings:\n        connection:\n          backend: typesafe\n",
        encoding="utf-8",
    )
    (profile_home / ".env").write_text(
        "TYPESAFE_API_KEY=${OUTSIDE_PROFILE_TEST}\n", encoding="utf-8"
    )
    seen = {}
    client = SimpleNamespace(evaluate=lambda **kwargs: seen.update(kwargs) or {"answers": {}})
    monkeypatch.setattr(observer, "_profile_home", lambda profile: profile_home)
    monkeypatch.setattr(observer, "_load_plugin_client", lambda path: client)
    monkeypatch.setenv("OUTSIDE_PROFILE_TEST", "process-secret-must-not-be-used")

    observer._evaluate(profile="milly", state={}, questions={}, timeout=1)

    assert seen["token"] == "${OUTSIDE_PROFILE_TEST}"


def test_profile_home_rejects_path_escape(monkeypatch, tmp_path: Path):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="invalid Hermes profile name"):
        observer._profile_home("..")


@pytest.mark.parametrize("profile", ["default", "cloud", "other"])
def test_profile_home_accepts_only_milly(monkeypatch, tmp_path: Path, profile: str):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="must be milly"):
        observer._profile_home(profile)


def test_profile_home_rejects_symlink_escape(monkeypatch, tmp_path: Path):
    import hermes_constants

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.yaml").write_text("{}\n", encoding="utf-8")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "milly").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="unavailable"):
        observer._profile_home("milly")


@pytest.mark.parametrize("resource", ["config.yaml", ".env"])
def test_evaluate_rejects_profile_file_symlink_escape(monkeypatch, tmp_path: Path, resource: str):
    profile_home = tmp_path / "profiles" / "milly"
    profile_home.mkdir(parents=True)
    outside = tmp_path / f"outside-{resource.replace('.', '')}"
    outside.write_text(
        "TYPESAFE_API_KEY=outside-token\n" if resource == ".env" else "{}\n",
        encoding="utf-8",
    )
    (profile_home / resource).symlink_to(outside)
    if resource != "config.yaml":
        (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(observer, "_profile_home", lambda profile: profile_home)

    with pytest.raises(ValueError, match="escapes configured profile"):
        observer._evaluate(profile="milly", state={}, questions={}, timeout=1)


def test_load_plugin_client_rejects_nested_symlink_escape(tmp_path: Path):
    profile_home = tmp_path / "profiles" / "milly"
    profile_home.mkdir(parents=True)
    outside = tmp_path / "outside-plugin"
    outside.mkdir()
    (outside / "client.py").write_text("def evaluate(**kwargs): return {}\n", encoding="utf-8")
    plugins = profile_home / "plugins"
    plugins.mkdir()
    (plugins / "jev").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes configured profile"):
        observer._load_plugin_client(profile_home)


def test_append_event_repairs_existing_audit_permissions(tmp_path: Path):
    audit = tmp_path / "jev.jsonl"
    audit.write_text("", encoding="utf-8")
    audit.chmod(0o644)

    observer._append_event(audit, {"status": "ok"})

    assert audit.stat().st_mode & 0o777 == 0o600


def test_append_event_rejects_audit_symlink(tmp_path: Path):
    target = tmp_path / "outside.jsonl"
    target.write_text("untouched\n", encoding="utf-8")
    audit = tmp_path / "jev.jsonl"
    audit.symlink_to(target)

    with pytest.raises(OSError):
        observer._append_event(audit, {"status": "ok"})

    assert target.read_text(encoding="utf-8") == "untouched\n"
