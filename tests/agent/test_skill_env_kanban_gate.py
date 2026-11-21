"""Kanban relevance gate for ``environments: [kanban]`` skills.

The gate opens for dispatcher-spawned workers (``HERMES_KANBAN_TASK`` pinned by
the dispatcher at spawn) or profiles opted into the kanban toolset. It must NOT
open for a plain interactive chat: ``cmd_chat`` pins ``HERMES_KANBAN_BOARD`` at
every chat boot (fix #20074) for board consistency, and the board alone is
scoping, not worker identity — otherwise every CLI/TUI session offers
kanban-gated skills in its index even with the toolset disabled.
"""

import pytest

from agent.skill_utils import skill_matches_environment

KANBAN_FRONTMATTER = {"environments": ["kanban"]}


@pytest.fixture(autouse=True)
def _no_kanban_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    # Keep the opt-in branch deterministic: these tests exercise the env gate.
    import tools.kanban_tools as kt

    monkeypatch.setattr(kt, "_profile_has_kanban_toolset", lambda: False)


def test_board_pin_at_chat_boot_does_not_open_gate(monkeypatch):
    # Regression: cmd_chat's board pin must not be mistaken for worker context.
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "trading")
    assert skill_matches_environment(KANBAN_FRONTMATTER) is False


def test_dispatcher_worker_env_opens_gate(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "trading")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "42")
    assert skill_matches_environment(KANBAN_FRONTMATTER) is True


def test_closed_without_env_or_opt_in():
    assert skill_matches_environment(KANBAN_FRONTMATTER) is False


def test_toolset_opt_in_opens_gate_without_env(monkeypatch):
    import tools.kanban_tools as kt

    monkeypatch.setattr(kt, "_profile_has_kanban_toolset", lambda: True)
    assert skill_matches_environment(KANBAN_FRONTMATTER) is True
