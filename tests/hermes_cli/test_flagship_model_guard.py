"""Policy gates for explicit flagship model overrides."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.model_policy import firepower_guard_error, route_kind


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _created_id(output: str) -> str:
    match = re.search(r"(t_[a-f0-9]+)", output)
    assert match, output
    return match.group(1)


@pytest.mark.parametrize("model", [
    "gpt-6-astra-900k",
    "openai-codex/gpt-6-astra-900k",
    "claude-fable-5-1",
])
def test_guard_requires_nonempty_reason_for_flagship(model):
    assert "--firepower" in firepower_guard_error(model, None)
    assert "--firepower" in firepower_guard_error(model, "  ")
    assert firepower_guard_error(model, "hard architecture adjudication") is None


def test_guard_does_not_restrict_sub_flagship():
    assert firepower_guard_error("gpt-5.6-sol-900k", None) is None
    assert firepower_guard_error("claude-opus-5", None) is None
    assert route_kind("openai-codex/gpt-5.6-sol-900k") == "standard"
    assert route_kind("openai-codex/gpt-6-astra-900k") == "firepower"


def _stub_alias(monkeypatch, mapping):
    """Resolve only *mapping*; tests must not depend on ambient model config.

    ``tests/conftest.py`` sandboxes ``HERMES_HOME`` away from the real root, so
    asserting against live aliases would make these tests environment-dependent.
    """
    import hermes_cli.model_switch as model_switch

    def fake_resolve_alias(raw_input, current_provider):
        hit = mapping.get(str(raw_input).strip().lower())
        return (hit[0], hit[1], raw_input) if hit else None

    monkeypatch.setattr(model_switch, "resolve_alias", fake_resolve_alias)


def test_route_kind_resolves_aliases_before_classifying(monkeypatch):
    """An alias must not walk past the guard on spelling alone."""
    _stub_alias(monkeypatch, {
        "astra": ("openai-codex", "gpt-6-astra-900k"),
        "fable": ("claude-apr", "claude-fable-5-1"),
        "sol": ("openai-codex", "gpt-5.6-sol-900k"),
    })
    assert route_kind("openai-codex/astra") == "firepower"
    assert route_kind("astra") == "firepower"
    assert route_kind("fable") == "firepower"
    assert route_kind("sol") == "standard"


def test_guard_refuses_alias_spelling_without_a_reason(monkeypatch):
    _stub_alias(monkeypatch, {
        "astra": ("openai-codex", "gpt-6-astra-900k"),
        "sol": ("openai-codex", "gpt-5.6-sol-900k"),
    })
    assert "--firepower" in (firepower_guard_error("astra", None) or "")
    assert firepower_guard_error("astra", "hard adjudication") is None
    assert firepower_guard_error("sol", None) is None


def test_unresolvable_model_falls_back_to_the_literal_input(monkeypatch):
    """An uncatalogued name must still be classified on its literal spelling."""
    _stub_alias(monkeypatch, {})
    assert route_kind("openai-codex/gpt-6-astra-900k") == "firepower"
    assert route_kind("openai-codex/gpt-5.6-sol-900k") == "standard"


def test_create_refuses_flagship_without_firepower_reason(kanban_home):
    output = kc.run_slash(
        "create 'expensive task' --assignee worker "
        "--model gpt-6-astra-900k --provider openai-codex"
    )
    assert "--firepower" in output
    with kb.connect() as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_create_accepts_flagship_and_appends_audit_comment(kanban_home):
    reason = "cross-module concurrency diagnosis"
    output = kc.run_slash(
        "create 'expensive task' --assignee worker "
        "--model gpt-6-astra-900k --provider openai-codex "
        f"--firepower '{reason}'"
    )
    task_id = _created_id(output)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)
    assert task.model_override == "gpt-6-astra-900k"
    assert len(comments) == 1
    assert reason in comments[0].body
    assert "openai-codex/gpt-6-astra-900k" in comments[0].body


def test_create_rolls_back_if_firepower_audit_comment_fails(kanban_home, monkeypatch):
    def fail_comment(*_args, **_kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(kb, "add_comment", fail_comment)
    output = kc.run_slash(
        "create 'must stay audited' --assignee worker "
        "--model gpt-6-astra-900k --provider openai-codex "
        "--firepower 'hard recovery'"
    )
    assert "audit unavailable" in output
    with kb.connect() as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_set_model_refuses_before_mutating_existing_pin(kanban_home):
    output = kc.run_slash("create 'plain task' --assignee worker --model gpt-5.6-sol-900k")
    task_id = _created_id(output)
    rejected = kc.run_slash(
        f"set-model {task_id} gpt-6-astra-900k --provider openai-codex"
    )
    assert "--firepower" in rejected
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)
    assert task.model_override == "gpt-5.6-sol-900k"
    assert comments == []


def test_set_model_accepts_flagship_and_appends_audit_comment(kanban_home):
    task_id = _created_id(kc.run_slash("create 'plain task' --assignee worker"))
    reason = "ambiguous multi-file recovery"
    changed = kc.run_slash(
        f"set-model {task_id} gpt-6-astra-900k --provider openai-codex "
        f"--firepower '{reason}'"
    )
    assert "Set model override" in changed
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)
    assert task.model_override == "gpt-6-astra-900k"
    assert len(comments) == 1
    assert reason in comments[0].body


def test_set_model_rolls_back_if_firepower_audit_comment_fails(kanban_home, monkeypatch):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="atomic override", assignee="worker",
            model_override="gpt-5.6-sol-900k",
        )

    def fail_comment(*_args, **_kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(kb, "add_comment", fail_comment)
    with kb.connect() as conn:
        with pytest.raises(RuntimeError, match="audit unavailable"):
            kb.set_model_override(
                conn, task_id, "gpt-6-astra-900k", provider="openai-codex",
                audit_comment_author="operator",
                audit_comment_body="firepower override: reason=hard recovery",
            )
        task = kb.get_task(conn, task_id)
    assert task.model_override == "gpt-5.6-sol-900k"


def test_effective_worker_route_prefers_card_override(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="pinned", assignee="worker",
            model_override="gpt-5.6-sol-900k", provider_override="openai-codex",
        )
        task = kb.get_task(conn, task_id)
    assert kbd.effective_worker_route(task) == "openai-codex/gpt-5.6-sol-900k"
