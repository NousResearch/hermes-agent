"""Regression tests for docs in skills/autonomous-ai-agents/hermes-agent/SKILL.md.

The bundled hermes-agent skill is the source every Hermes session reads for
slash-command reference. If the doc lists a command that does not exist in
``hermes_cli/commands.py``, agents propagate the phantom into the system
prompt and users hit dead ends. These tests guard the docs against two
specific drift cases reported in issue #50605:

1. ``/skill <name>`` — listed as a slash command, but no
   ``CommandDef("skill", ...)`` exists in the registry.
2. ``/q`` — listed as an alias for ``/quit`` in the Exit section, but
   the registry declares ``q`` as the alias for ``/queue`` and only
   ``exit`` as an alias for ``/quit``.

Fixes #50605
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.commands import COMMAND_REGISTRY


# Repo-relative path to the bundled hermes-agent skill docs.
HERMES_AGENT_SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "hermes-agent"
    / "SKILL.md"
)


def _read_skill_md() -> str:
    assert HERMES_AGENT_SKILL_MD.is_file(), (
        f"hermes-agent SKILL.md missing at {HERMES_AGENT_SKILL_MD}"
    )
    return HERMES_AGENT_SKILL_MD.read_text(encoding="utf-8")


def _registered_names() -> set[str]:
    return {cmd.name for cmd in COMMAND_REGISTRY}


def _registered_aliases() -> set[str]:
    aliases: set[str] = set()
    for cmd in COMMAND_REGISTRY:
        aliases.update(cmd.aliases)
    return aliases


@pytest.fixture(scope="module")
def skill_md_text() -> str:
    return _read_skill_md()


class TestSlashCommandsMatchRegistry:
    """Every slash command named in the SKILL.md doc must exist in the registry."""

    def test_no_phantom_skill_slash_command(self, skill_md_text: str) -> None:
        """``/skill <name>`` is not a registered command — only ``/skills`` is.

        The doc previously listed ``/skill <name>`` as "Load a skill into
        session", which misleads agents and users into typing a dead command.
        """
        registered = _registered_names()
        # ``/skills`` (the catalog browser) is the real skills-related slash
        # command and is allowed to appear; the singular ``/skill`` is not.
        assert "skill" not in registered, (
            "Test setup invariant violated: a CommandDef('skill', ...) appeared "
            "in the registry — update this test to reflect the new command."
        )

        # The phantom appears in two known places: the slash-command table and
        # the "Skills not showing" troubleshooting section. Match the command
        # line (with its argument placeholder) and the standalone invocation.
        assert "/skill <name>" not in skill_md_text, (
            "Phantom slash command ``/skill <name>`` is documented in the "
            "hermes-agent SKILL.md but no CommandDef('skill', ...) exists in "
            "the registry. Issue #50605."
        )
        assert "`/skill name`" not in skill_md_text, (
            "Phantom slash command ``/skill name`` is documented in the "
            "hermes-agent SKILL.md troubleshooting section but no such command "
            "exists in the registry. Issue #50605."
        )

    def test_q_alias_for_quit_is_documented_correctly(self, skill_md_text: str) -> None:
        """``/quit`` is only aliased as ``/exit``; ``/q`` is the alias for ``/queue``.

        The previous Exit section claimed ``/quit (/exit, /q)``, but the
        registry binds ``q`` to ``queue`` (which queues a prompt for the next
        turn) — typing ``/q`` to exit actually queues, silently. This test
        catches the regression at the doc level.
        """
        registered = _registered_names()
        assert "quit" in registered, "Registry invariant violated: /quit missing"

        # Find which command the alias ``q`` actually maps to.
        q_owner: str | None = None
        for cmd in COMMAND_REGISTRY:
            if "q" in cmd.aliases:
                q_owner = cmd.name
                break
        assert q_owner is not None, "Registry invariant violated: no command has alias 'q'"
        assert q_owner != "quit", (
            f"Test setup invariant violated: alias 'q' now belongs to /{q_owner}; "
            "update this test to reflect the new binding."
        )

        # The Exit section must not advertise ``/q`` as an alias for /quit.
        # We look for the exact Exit-line pattern from the issue.
        assert "/quit (/exit, /q)" not in skill_md_text, (
            "The Exit section of hermes-agent SKILL.md incorrectly documents "
            "``/q`` as an alias for ``/quit``. In the registry, ``/q`` is the "
            "alias for ``/queue``. Issue #50605."
        )
        # And no standalone ``/q`` should be advertised in the Exit block.
        # Match a trailing-alias pattern only inside the Exit section to keep
        # the assertion scoped; the bullet line is ``/quit (/exit, /q)``.
        exit_section_marker = "### Exit"
        idx = skill_md_text.find(exit_section_marker)
        assert idx != -1, "Exit section header missing from SKILL.md"
        # Take a window after the Exit header up to the next blank-line section.
        exit_block = skill_md_text[idx : idx + 400]
        assert "(/exit, /q)" not in exit_block, (
            "The Exit section still lists ``/q`` as an alias for ``/quit``. "
            "In the registry, ``/q`` is the alias for ``/queue``. Issue #50605."
        )