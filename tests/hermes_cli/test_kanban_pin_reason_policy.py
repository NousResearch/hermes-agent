"""``kanban.pin_reason_required_providers``: an opt-in policy that makes
pinning a worker to listed providers need a stated ``--pin-reason``.

Both knob states are covered: unset (the default) must leave every route
write exactly as before, and set must refuse matching routes without a
reason on every write surface (create, set-model single + batch, lane-model).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_lanes as kbl
from hermes_cli.kanban_pin_policy import check_route_pin, matching_glob, required_provider_globs


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_policy(home: Path, globs) -> None:
    """Write the knob into the temp home's real config.yaml (E2E, no mocks)."""
    (home / "config.yaml").write_text(yaml.safe_dump({"kanban": {"pin_reason_required_providers": globs}}))


def _events(conn, task_id, kind):
    return [e.payload for e in kb.list_events(conn, task_id) if e.kind == kind]


# --- pure predicate -------------------------------------------------------

def test_globs_match_case_insensitively_and_fail_open_on_unknown():
    assert matching_glob("Pool-3", ["pool-*"]) == "pool-*"
    assert matching_glob("openrouter", ["pool-*"]) is None
    assert matching_glob(None, ["*"]) is None


def test_config_shapes():
    assert required_provider_globs({}) == ()
    assert required_provider_globs({"kanban": {}}) == ()
    assert required_provider_globs({"kanban": {"pin_reason_required_providers": "pool-*"}}) == ("pool-*",)
    assert required_provider_globs({"kanban": {"pin_reason_required_providers": [" a ", "", 3]}}) == ("a",)
    assert required_provider_globs({"kanban": {"pin_reason_required_providers": 7}}) == ()


def test_check_route_pin_rules():
    assert check_route_pin("pool-1", None, globs=()) is None
    assert check_route_pin("pool-1", "  why ", globs=["pool-*"]) == "why"
    with pytest.raises(ValueError, match="pin_reason_required_providers"):
        check_route_pin("pool-1", "  ", globs=["pool-*"])
    with pytest.raises(ValueError, match="needs a --provider"):
        check_route_pin(None, "why", globs=["pool-*"])


# --- knob unset: behaviour unchanged ---------------------------------------

def test_unset_knob_leaves_every_surface_unchanged(kanban_home):
    with kbc.connect_closing() as conn:
        t = kb.create_task(conn, title="t", assignee="alpha", model_override="m", provider_override="pool-1")
        assert kb.set_model_override(conn, t, "m2", provider="pool-2")
        kb.apply_batch_route_writes(conn, [kb.BatchRouteWrite(task_id=t, model="m3", provider="pool-3")])
        kbl.set_lane_model_override(conn, provider="pool-4", model="m", expires_at=int(time.time()) + 60)
        assert kb.get_task(conn, t).provider_override == "pool-3"
        created = _events(conn, t, "created")[0]
        assert "pin_reason" not in created
        assert all(set(p) == {"model", "provider"} for p in _events(conn, t, "model_override_set"))


# --- knob set: refusal + override on every write surface -------------------

def test_create_refused_without_reason_and_recorded_with_one(kanban_home):
    _set_policy(kanban_home, ["pool-*"])
    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError, match="--pin-reason"):
            kb.create_task(conn, title="t", assignee="alpha", model_override="m", provider_override="Pool-1")
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
        other = kb.create_task(conn, title="o", assignee="alpha", model_override="m", provider_override="openrouter")
        assert kb.get_task(conn, other).provider_override == "openrouter"
        t = kb.create_task(conn, title="t", assignee="alpha", model_override="m",
                           provider_override="pool-1", pin_reason="long context job")
        assert _events(conn, t, "created")[0]["pin_reason"] == "long context job"


def test_set_model_refused_without_reason_and_clearing_is_never_gated(kanban_home):
    _set_policy(kanban_home, ["pool-*"])
    with kbc.connect_closing() as conn:
        t = kb.create_task(conn, title="t", assignee="alpha")
        with pytest.raises(ValueError, match="pool-\\*"):
            kb.set_model_override(conn, t, "m", provider="pool-1")
        assert kb.get_task(conn, t).provider_override is None
        assert kb.set_model_override(conn, t, "m", provider="pool-1", pin_reason="why")
        assert _events(conn, t, "model_override_set")[-1]["pin_reason"] == "why"
        assert kb.set_model_override(conn, t, None)
        assert kb.get_task(conn, t).provider_override is None


def test_batch_refuses_whole_batch_before_writing(kanban_home):
    _set_policy(kanban_home, ["pool-*"])
    with kbc.connect_closing() as conn:
        a = kb.create_task(conn, title="a", assignee="alpha")
        b = kb.create_task(conn, title="b", assignee="alpha")
        writes = [kb.BatchRouteWrite(task_id=a, model="m", provider="openrouter"),
                  kb.BatchRouteWrite(task_id=b, model="m", provider="pool-1")]
        with pytest.raises(ValueError):
            kb.apply_batch_route_writes(conn, writes)
        assert kb.get_task(conn, a).provider_override is None
        writes[1].pin_reason = "why"
        assert kb.apply_batch_route_writes(conn, writes) == [a, b]


def test_lane_override_refused_without_reason(kanban_home):
    _set_policy(kanban_home, ["pool-*"])
    with kbc.connect_closing() as conn:
        exp = int(time.time()) + 60
        with pytest.raises(ValueError, match="--pin-reason"):
            kbl.set_lane_model_override(conn, provider="pool-1", model="m", expires_at=exp)
        assert kbl.list_lane_model_overrides(conn) == []
        kbl.set_lane_model_override(conn, provider="pool-1", model="m", expires_at=exp, pin_reason="why")
        assert kbl.get_lane_model_override(conn).provider == "pool-1"


def test_cli_flags_reach_the_policy(kanban_home):
    _set_policy(kanban_home, ["pool-*"])
    with kbc.connect_closing() as conn:
        t = kb.create_task(conn, title="t", assignee="alpha")
    assert "--pin-reason" in kc.run_slash(f"set-model {t} m --provider pool-1")
    kc.run_slash(f"set-model {t} m --provider pool-1 --pin-reason 'long job'")
    assert "--pin-reason" in kc.run_slash("create c --assignee alpha --model m --provider pool-2")
    kc.run_slash("create c2 --assignee alpha --model m --provider pool-2 --pin-reason why")
    assert "--pin-reason" in kc.run_slash("lane-model set pool-3/m --ttl 1h --reason r")
    kc.run_slash("lane-model set pool-3/m --ttl 1h --reason r --pin-reason why")
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, t).provider_override == "pool-1"
        assert _events(conn, t, "model_override_set")[-1]["pin_reason"] == "long job"
        titles = {r["title"]: r["provider_override"] for r in conn.execute("SELECT title, provider_override FROM tasks")}
        assert titles == {"t": "pool-1", "c2": "pool-2"}
        assert kbl.get_lane_model_override(conn).provider == "pool-3"
