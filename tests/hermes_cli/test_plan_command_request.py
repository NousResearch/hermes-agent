"""Regression: `/plan [request]` must preserve the documented skill dispatch.

Registering the enforcement-mode `plan` CommandDef shadowed the bundled `plan`
skill's `/plan [request]` workflow (website/docs/user-guide/features/skills.md):
`/plan implement X` entered plan mode but DISCARDED the request text.

Fix (chosen behavior): `/plan` with no args toggles enforcement mode (our
feature); `/plan <request>` enters enforcement mode AND seeds the documented
plan skill with the request text — the request is never dropped. The plan
skill is resolved by NAME because its `/plan` slash command is intentionally
skipped from the registry (it collides with the core `/plan` command).
"""

from __future__ import annotations

import queue
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Seed a minimal `plan` skill so the documented `/plan [request]` dispatch
    # resolves it by name (skills live under HERMES_HOME/skills).
    skill_dir = home / "skills" / "plan"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: plan\n"
        "description: Write a markdown implementation plan instead of executing.\n"
        "---\n"
        "# Plan skill\n\n"
        "Inspect context, write a markdown plan under `.hermes/plans/`, and do "
        "not execute the task.\n"
    )

    from hermes_cli import plan_mode

    plan_mode._DB_CACHE.clear()
    yield home
    plan_mode._DB_CACHE.clear()


def _make_fake_cli(sid: str):
    from hermes_cli.plan_mode import PlanManager

    fake = SimpleNamespace(
        session_id=sid,
        agent=object(),
        _pending_input=queue.Queue(),
    )
    fake._get_plan_manager = lambda: PlanManager(session_id=sid)
    fake._seed_plan_skill_request = types.MethodType(
        cli.HermesCLI._seed_plan_skill_request, fake
    )
    fake._handle_plan_command = types.MethodType(
        cli.HermesCLI._handle_plan_command, fake
    )
    return fake


def test_bare_plan_enters_enforcement_without_seeding(hermes_home):
    from hermes_cli.plan_mode import PlanManager

    fake = _make_fake_cli("cli-plan-bare")
    fake._handle_plan_command("/plan")

    assert PlanManager("cli-plan-bare").is_active() is True
    # No request text → nothing queued as a turn.
    assert fake._pending_input.empty()
    # Cached agent dropped so the restricted toolset policy rebuilds.
    assert fake.agent is None


def test_plan_request_enters_mode_and_seeds_skill(hermes_home):
    from hermes_cli.plan_mode import PlanManager

    fake = _make_fake_cli("cli-plan-req")
    fake._handle_plan_command("/plan implement the login flow")

    # Enforcement mode entered (our feature) …
    assert PlanManager("cli-plan-req").is_active() is True
    assert fake.agent is None
    # … AND the request was seeded through the plan skill (args NOT discarded).
    assert not fake._pending_input.empty()
    queued = fake._pending_input.get_nowait()
    assert "implement the login flow" in queued
    # The bundled plan skill's instructions rode along.
    assert "plan" in queued.lower()


def test_seed_helper_queues_plan_skill_with_request(hermes_home):
    fake = _make_fake_cli("cli-plan-seed")
    ok = fake._seed_plan_skill_request("draft a migration plan")
    assert ok is True
    queued = fake._pending_input.get_nowait()
    assert "draft a migration plan" in queued


def test_plan_status_subcommand_still_dispatches(hermes_home):
    """A recognized subcommand is NOT treated as a request."""
    from hermes_cli.plan_mode import PlanManager

    fake = _make_fake_cli("cli-plan-status")
    PlanManager("cli-plan-status").enter()
    fake._handle_plan_command("/plan status")
    # status is read-only: no turn queued, mode unchanged.
    assert fake._pending_input.empty()
    assert PlanManager("cli-plan-status").is_active() is True
