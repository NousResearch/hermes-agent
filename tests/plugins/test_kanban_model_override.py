"""Per-task model/provider override — DB layer, worker spawn, dashboard API.

Covers the model-dropdown feature: kanban_db.set_model_override(),
create_task(model_override=..., provider_override=...), the dispatcher
passing task identity to the owner for trusted policy resolution, and the dashboard
PATCH/bulk/model-options surfaces.

A pin is judged by PROVENANCE (``kanban_db._validate_model_override``): it is accepted
only when it equals the assignee profile's own configured ``model.default`` or an
``approved_pins`` entry of that profile. Plumbing tests therefore declare the profile's
model first via :func:`configure_profile` — pinning a model a profile does not run is the
defect this guard exists to refuse, not a flow to exercise.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def configure_profile(home, name, model, provider=None, approved_pins=None):
    """Declare what profile *name* runs — the provenance a model pin is judged against."""
    lines = ["model:", f"  default: {model}"]
    if provider:
        lines.append(f"  provider: {provider}")
    if approved_pins:
        lines.append("  approved_pins:")
        for pin in approved_pins:
            if isinstance(pin, dict):
                lines.append(f"    - model: {pin['model']}")
                lines.append(f"      provider: {pin['provider']}")
            else:
                lines.append(f"    - {pin}")
    profile = home / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    profile.joinpath("config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profile


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    yield c
    c.close()


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_model_override_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


# ---------------------------------------------------------------------------
# DB layer — set_model_override
# ---------------------------------------------------------------------------


def test_set_and_clear_model_override(conn, kanban_home):
    configure_profile(kanban_home, "worker", "gpt-5.6-sol", "openai")
    tid = kb.create_task(conn, title="t", assignee="worker")
    assert kb.set_model_override(conn, tid, "gpt-5.6-sol", provider="openai")
    t = kb.get_task(conn, tid)
    assert t.model_override == "gpt-5.6-sol"
    assert t.provider_override == "openai"

    # Clearing the model clears the provider too.
    assert kb.set_model_override(conn, tid, None)
    t = kb.get_task(conn, tid)
    assert t.model_override is None
    assert t.provider_override is None


def test_provider_without_model_rejected(conn):
    tid = kb.create_task(conn, title="t", assignee="worker")
    with pytest.raises(ValueError):
        kb.set_model_override(conn, tid, None, provider="openrouter")
    with pytest.raises(ValueError):
        kb.create_task(
            conn, title="t2", assignee="worker", provider_override="openrouter",
        )


def test_create_task_with_model_and_provider(conn, kanban_home):
    configure_profile(kanban_home, "worker", "qwen-max", "openrouter")
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="qwen-max", provider_override="openrouter",
    )
    t = kb.get_task(conn, tid)
    assert t.model_override == "qwen-max"
    assert t.provider_override == "openrouter"
    # Creation event carries the override for auditability.
    ev = next(e for e in kb.list_events(conn, tid) if e.kind == "created")
    assert ev.payload["model_override"] == "qwen-max"
    assert ev.payload["provider_override"] == "openrouter"




# ---------------------------------------------------------------------------
# Worker spawn — task identity resolves trusted owner policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, expected_model, expected_provider, expected_reasoning",
    [
        ({"model_override": "glm-5", "provider_override": "openrouter",
          "reasoning_effort": "high"}, "glm-5", "openrouter", "high"),
        ({"reasoning_effort": "high"}, "profile-model", "custom", "high"),
        ({"reasoning_effort": "none"}, "profile-model", "custom", None),
        ({}, "profile-model", "custom", "low"),
    ],
)
def test_spawn_resolves_claimed_task_policy(
    monkeypatch, tmp_path, conn, kanban_home, overrides,
    expected_model, expected_provider, expected_reasoning,
):
    """The worker passes identity, while the owner reads overrides from SQLite."""
    from gateway.session_contract import SessionActor
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli import kanban_worker_client
    from persistence.core.errors import AuthorityRefusal

    profile = kanban_home / "profiles" / "elias"
    profile.mkdir(parents=True)
    # The provenance gate judges a pin against this file, so elias declares the pair case 1
    # pins (glm-5/openrouter); the in-memory `config` below stays the owner-side view this
    # test exercises.
    (profile / "config.yaml").write_text(
        "model:\n  default: glm-5\n  provider: openrouter\n", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tid = kb.create_task(conn, title="t", assignee="elias",
                         workspace_kind="dir", workspace_path=str(workspace), **overrides)
    task = kb.claim_task(conn, tid)
    assert task is not None
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(cmd=list(cmd), env=kwargs["env"])
        kwargs["stdout"].close()
        return SimpleNamespace(pid=4245)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kbd._default_spawn(task, str(workspace))
    assert captured["cmd"] == [sys.executable, "-m", "hermes_cli.kanban_worker_client"]
    assert captured["env"]["HERMES_HOME"] == str(profile)
    actor = SessionActor("owner", str(profile), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(
        authority=SimpleNamespace(profile_id=str(profile)), actor=actor, native_owner=True)
    config = {"model": {"default": "profile-model", "provider": "custom"},
              "agent": {"reasoning_effort": "medium",
                        "reasoning_overrides": {"profile-model": "low", "glm-5": "low"}},
              "platform_toolsets": {"cli": ["terminal"]}}
    original_config = json.dumps(config, sort_keys=True)

    async def resolve_at_owner(params, board_db):
        params = dict(params, db=board_db)
        policy, _ = build_kanban_policy(connection, params, config)
        assert policy.model == expected_model
        assert policy.provider == expected_provider
        expected = ({"enabled": False} if expected_reasoning is None else
                    {"enabled": True, "effort": expected_reasoning})
        assert policy.reasoning_config == expected
        assert policy.cwd == str(workspace) and "kanban" in policy.toolsets
        context = json.loads(policy.kanban_json)
        assert (context["task_id"], context["run_id"], context["claim_lock"]) == (
            task.id, task.current_run_id, task.claim_lock)
        # A worker cannot replace trusted settings or reuse another claim.
        with pytest.raises(AuthorityRefusal, match="invalid_params"):
            build_kanban_policy(connection, params | {"model": "forged"}, config)
        with pytest.raises(AuthorityRefusal, match="invalid_kanban_claim"):
            build_kanban_policy(connection, params | {"claim_lock": "forged"}, config)
        assert json.dumps(config, sort_keys=True) == original_config
        captured["resolved"] = True
        return 0

    monkeypatch.setattr(kanban_worker_client, "run", resolve_at_owner)
    with monkeypatch.context() as child:
        for key, value in captured["env"].items():
            child.setenv(key, value)
        assert kanban_worker_client.main() == 0
    assert captured["resolved"]


# ---------------------------------------------------------------------------
# Dashboard API — PATCH / bulk / create / model-options
# ---------------------------------------------------------------------------


def _create(client, **kwargs):
    body = {"title": "task", "assignee": "worker"}
    body.update(kwargs)
    r = client.post("/api/plugins/kanban/tasks", json=body)
    assert r.status_code == 200, r.text
    return r.json()["task"]


def test_patch_sets_model_override(client, kanban_home):
    configure_profile(kanban_home, "worker", "gpt-5.6-sol", "openai")
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"model_override": "gpt-5.6-sol", "provider_override": "openai"},
    )
    assert r.status_code == 200, r.text
    updated = r.json()["task"]
    assert updated["model_override"] == "gpt-5.6-sol"
    assert updated["provider_override"] == "openai"


def test_bulk_model_override(client, kanban_home):
    configure_profile(kanban_home, "worker", "fallback-model", "nous")
    t1 = _create(client)
    t2 = _create(client)
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={
            "ids": [t1["id"], t2["id"]],
            "model_override": "fallback-model",
            "provider_override": "nous",
        },
    )
    assert r.status_code == 200, r.text
    assert all(entry["ok"] for entry in r.json()["results"])
    for tid in (t1["id"], t2["id"]):
        got = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert got["model_override"] == "fallback-model"
        assert got["provider_override"] == "nous"


def test_model_options_endpoint_shape(client, monkeypatch):
    """The endpoint returns {providers: [{slug,label,models}]} and degrades
    to an empty catalog when the inventory substrate raises."""
    r = client.get("/api/plugins/kanban/model-options")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    assert isinstance(data["providers"], list)
    for row in data["providers"]:
        assert "slug" in row and "label" in row and "models" in row
        assert isinstance(row["models"], list)
        assert len(row["models"]) >= 1  # empty-model rows are filtered out


# ---------------------------------------------------------------------------
# Per-task reasoning effort — the depth half of the board's model picker
# ---------------------------------------------------------------------------


def test_reasoning_effort_normalizes_and_rejects(conn):
    tid = kb.create_task(conn, title="t", assignee="worker", reasoning_effort="  HIGH ")
    assert kb.get_task(conn, tid).reasoning_effort == "high"

    # "none" is a VALUE (thinking off), not a clear.
    assert kb.set_reasoning_effort(conn, tid, "none")
    assert kb.get_task(conn, tid).reasoning_effort == "none"

    # Empty clears back to "inherit the profile".
    assert kb.set_reasoning_effort(conn, tid, "")
    assert kb.get_task(conn, tid).reasoning_effort is None

    with pytest.raises(ValueError):
        kb.set_reasoning_effort(conn, tid, "extremely-hard")


def test_reasoning_effort_survives_clearing_the_model(conn, kanban_home):
    """Depth and model are independent knobs: dropping a model override must
    not silently reset the thinking depth the operator chose."""
    configure_profile(kanban_home, "worker", "glm-5", "openrouter")
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="glm-5", provider_override="openrouter",
        reasoning_effort="ultra",
    )
    assert kb.set_model_override(conn, tid, None)
    t = kb.get_task(conn, tid)
    assert t.model_override is None
    assert t.provider_override is None
    assert t.reasoning_effort == "ultra"




def test_patch_sets_and_clears_reasoning_effort(client):
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"reasoning_effort": "xhigh"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["task"]["reasoning_effort"] == "xhigh"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"clear_reasoning_effort": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["task"]["reasoning_effort"] is None


def test_patch_rejects_an_unknown_level(client):
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"reasoning_effort": "bogus"},
    )
    assert r.status_code == 400


def test_create_accepts_reasoning_effort(client):
    task = _create(client, reasoning_effort="minimal")
    assert task["reasoning_effort"] == "minimal"


def test_bulk_reasoning_effort(client):
    t1 = _create(client)
    t2 = _create(client)
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [t1["id"], t2["id"]], "reasoning_effort": "max"},
    )
    assert r.status_code == 200, r.text
    assert all(entry["ok"] for entry in r.json()["results"])
    for tid in (t1["id"], t2["id"]):
        got = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert got["reasoning_effort"] == "max"


# ---------------------------------------------------------------------------
# Provenance guard — a pin may only be what the assignee profile actually runs
#
# An observed defect: a minted card carried an INVENTED model string — a retired dated
# snapshot, a duplicated provider prefix, a truncated id, a foreign model on the wrong
# provider — and the worker ran on it, or died at startup. The gate has to be
# provenance (what the target profile's own config declares), never membership (a
# retired dated snapshot is a real catalog entry).
# ---------------------------------------------------------------------------


def test_duplicate_provider_prefix_normalized_and_conforming_pair_kept(kanban_home):
    """RED→GREEN control pair, at the validator."""
    configure_profile(kanban_home, "pinned-worker", "deepseek/deepseek-v4.1-flash", "nous")
    # RED: the exact recorded argv shape (`-m nous/deepseek/... --provider nous`).
    assert kb._validate_model_override(
        "nous/deepseek/deepseek-v4.1-flash", "nous", target_profile="pinned-worker",
    ) == ("deepseek/deepseek-v4.1-flash", "nous")
    # GREEN: the conforming pair passes unchanged.
    assert kb._validate_model_override(
        "deepseek/deepseek-v4.1-flash", "nous", target_profile="pinned-worker",
    ) == ("deepseek/deepseek-v4.1-flash", "nous")


def test_retired_dated_snapshot_refused_naming_the_allowed_value(kanban_home):
    configure_profile(kanban_home, "pinned-worker", "deepseek/deepseek-v4.1-flash", "nous")
    with pytest.raises(ValueError) as exc:
        kb._validate_model_override(
            "deepseek/deepseek-v4-flash-0731", "nous", target_profile="pinned-worker",
        )
    message = str(exc.value)
    assert "deepseek/deepseek-v4.1-flash" in message  # names the allowed value
    assert "model.approved_pins" in message           # names the deliberate-pin route


@pytest.mark.parametrize("model", [
    "deepseek/deepseek-v4-flash-0731",  # retired dated snapshot (a REAL catalog entry)
    "deepseek/deepseek-v4.1-fla",       # truncated id
    "anthropic/claude-sonnet-4.5",      # foreign model on the profile's provider
    "grok-4.7",                         # another profile's model
])
def test_models_the_profile_does_not_run_are_refused(kanban_home, model):
    configure_profile(kanban_home, "pinned-worker", "deepseek/deepseek-v4.1-flash", "nous")
    with pytest.raises(ValueError):
        kb._validate_model_override(model, "nous", target_profile="pinned-worker")


def test_profile_default_on_a_foreign_provider_is_refused(kanban_home):
    """`model X with provider Y` is the classic board-stall — a matching model is not
    enough when the provider would route it to a backend that profile does not use."""
    configure_profile(kanban_home, "pinned-worker", "deepseek/deepseek-v4.1-flash", "nous")
    with pytest.raises(ValueError) as exc:
        kb._validate_model_override(
            "deepseek/deepseek-v4.1-flash", "deepseek", target_profile="pinned-worker",
        )
    assert "provider" in str(exc.value)


def test_approved_pin_allows_a_declared_model(kanban_home):
    configure_profile(
        kanban_home, "pinned-worker", "deepseek/deepseek-v4.1-flash", "nous",
        approved_pins=["deepseek/deepseek-v4-pro",
                       {"model": "grok-4.7", "provider": "xai-oauth"}],
    )
    # A bare-id pin inherits the profile's provider; a mapping pin names its own.
    assert kb._validate_model_override(
        "deepseek/deepseek-v4-pro", None, target_profile="pinned-worker",
    ) == ("deepseek/deepseek-v4-pro", None)
    assert kb._validate_model_override(
        "grok-4.7", "xai-oauth", target_profile="pinned-worker",
    ) == ("grok-4.7", "xai-oauth")
    # Approved model, unapproved provider -> still refused.
    with pytest.raises(ValueError):
        kb._validate_model_override("grok-4.7", "nous", target_profile="pinned-worker")


def test_profile_with_no_configured_model_refuses_every_pin(conn, kanban_home):
    profile = kanban_home / "profiles" / "bare"
    profile.mkdir(parents=True, exist_ok=True)
    profile.joinpath("config.yaml").write_text("toolsets:\n  - terminal\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        kb.create_task(conn, title="t", assignee="bare", model_override="glm-5")
    assert "model.default" in str(exc.value)


def test_card_with_no_assignee_refuses_a_pin(conn, kanban_home):
    """Nothing can justify a pin when there is no profile to declare it."""
    with pytest.raises(ValueError) as exc:
        kb.create_task(
            conn, title="t", model_override="deepseek/deepseek-v4.1-flash",
            provider_override="nous",
        )
    assert "no assignee profile" in str(exc.value)


def test_refusals_fire_from_create_task_and_set_model_override(conn, kanban_home):
    configure_profile(kanban_home, "pinned-worker", "deepseek/deepseek-v4.1-flash", "nous")
    with pytest.raises(ValueError):
        kb.create_task(
            conn, title="t", assignee="pinned-worker",
            model_override="deepseek/deepseek-v4-flash-0731", provider_override="nous",
        )
    tid = kb.create_task(conn, title="ok", assignee="pinned-worker")
    with pytest.raises(ValueError):
        kb.set_model_override(conn, tid, "deepseek/deepseek-v4-flash-0731", provider="nous")
    assert kb.get_task(conn, tid).model_override is None


def test_clearing_a_pin_and_a_missing_task_never_need_a_policy(conn):
    """Clearing drops both columns and must work with no profile config at all; an
    unknown task id still reports False rather than raising a validation error."""
    tid = kb.create_task(conn, title="t", assignee="worker")
    assert kb.set_model_override(conn, tid, None)
    assert kb.get_task(conn, tid).model_override is None
    assert kb.set_model_override(conn, "t_missing", "deepseek/deepseek-v4.1-flash") is False


def test_dashboard_patch_refuses_a_pin_the_profile_does_not_run(client, kanban_home):
    configure_profile(kanban_home, "worker", "deepseek/deepseek-v4.1-flash", "nous")
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"model_override": "deepseek/deepseek-v4-flash-0731", "provider_override": "nous"},
    )
    assert r.status_code == 400, r.text
    assert "deepseek/deepseek-v4.1-flash" in r.text
    got = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]
    assert got["model_override"] is None
