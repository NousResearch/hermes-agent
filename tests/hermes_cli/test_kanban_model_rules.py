"""Dispatch-time model routing — mechanical card classes to a cheap model.

Covered here:
  * A card titled like a mechanical job with NO explicit override dispatches
    with the config rule's model injected into the spawn args.
  * A card with an explicit model_override keeps its override — no rule fires.
  * A card matching no rule spawns exactly as today (no override injected).
  * A malformed regex in a rule is skipped with a logged warning, never a crash.
  * The fired rule is recorded as a ``model_rule_applied`` audit event and the
    card itself is never written back (the worker's spawn is what changes).
"""
from __future__ import annotations

import json
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_model_rules_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if (mod.startswith("hermes_cli") or mod.startswith("hermes_state")
                or mod == "hermes_constants"):
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db, test_home


def _default_rules():
    return [
        {"match": "(audit|triage|routing|inventory|classif)",
         "model": "deepseek/deepseek-v4-flash-0731"},
        {"match": "^Deploy:", "model": "deepseek/deepseek-v4-flash-0731"},
    ]


def _fake_spawn(*args, **kwargs):
    return 12345


# ---------------------------------------------------------------------------
# Unit: resolver + apply on an in-memory Task
# ---------------------------------------------------------------------------


def test_resolver_matches_title_case_insensitively(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    rule = kb._resolve_model_rule(
        "BackupBrain: route audit for region ANZ", rules=_default_rules(),
    )
    assert rule is not None
    assert rule["model"] == "deepseek/deepseek-v4-flash-0731"


def test_resolver_matches_deploy_anchor(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    assert kb._resolve_model_rule(
        "Deploy: merge wt/x", rules=_default_rules(),
    ) is not None
    # Anchor is title-beginning only: a mid-title "Deploy:" does not fire.
    assert kb._resolve_model_rule(
        "Something Deploy: later", rules=_default_rules(),
    ) is None


def test_resolver_returns_none_when_no_rule_fires(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    assert kb._resolve_model_rule("Build feature XYZ", rules=_default_rules()) is None


def test_apply_mutates_only_in_memory_task(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Inventory scan", assignee="worker")
        task = kb.get_task(conn, tid)
    fired = kb.apply_model_rule(task, rules=_default_rules())
    assert fired is not None
    assert task.model_override == "deepseek/deepseek-v4-flash-0731"
    # The DB row is untouched — routing is spawn-scoped only.
    with kb.connect_closing() as conn:
        row = kb.get_task(conn, tid)
    assert row.model_override is None
    assert row.provider_override is None


def test_apply_respects_explicit_override(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="route audit", assignee="worker",
            model_override="glm-5", provider_override="openrouter",
        )
        task = kb.get_task(conn, tid)
    # Rule matches the title, but the explicit override always wins.
    fired = kb.apply_model_rule(task, rules=_default_rules())
    assert fired is None
    assert task.model_override == "glm-5"
    assert task.provider_override == "openrouter"


def test_apply_sets_provider_from_rule(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="classify threats", assignee="worker")
        task = kb.get_task(conn, tid)
    rules = [{"match": "classif", "model": "m2", "provider": "nous"}]
    fired = kb.apply_model_rule(task, rules=rules)
    assert fired is not None
    assert task.model_override == "m2"
    assert task.provider_override == "nous"


def test_malformed_regex_is_skipped_not_a_crash(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    rules = [
        {"match": "[unclosed", "model": "bad-model"},
        {"match": "routing", "model": "deepseek/deepseek-v4-flash-0731"},
    ]
    # First rule's regex is malformed -> skipped; the valid later rule fires.
    resolved = kb._resolve_model_rule("audit routing pass", rules=rules)
    assert resolved is not None
    assert resolved["model"] == "deepseek/deepseek-v4-flash-0731"


def test_all_malformed_rules_yield_none_not_a_crash(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    rules = [{"match": "[unclosed", "model": "bad-model"}]
    assert kb._resolve_model_rule("route audit", rules=rules) is None


# ---------------------------------------------------------------------------
# Integration: the dispatcher applies the rule and emits an audit event
# ---------------------------------------------------------------------------


def test_dispatch_applies_rule_and_emits_event(isolated_kanban_home):
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        tid = kb.create_task(
            conn, title="BackupBrain: route audit ...", assignee="default",
        )
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            model_rules=_default_rules(),
        )
    assert any(s[0] == tid for s in res.spawned)
    with kb.connect_closing() as conn:
        evs = list(conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id=? AND kind='model_rule_applied'",
            (tid,),
        ))
    assert len(evs) == 1
    payload = json.loads(evs[0][1])
    assert payload["source"] == "kanban.model_rules"
    assert payload["model"] == "deepseek/deepseek-v4-flash-0731"