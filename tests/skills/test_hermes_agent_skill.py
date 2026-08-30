"""The `hermes-agent` skill is what a running Hermes knows about itself.

`website/` is never packaged, so an installed Hermes has no local copy of the
user guide; skills ARE synced into `$HERMES_HOME/skills/`. The skill therefore
does not try to restate the product — it routes to the published `llms.txt`,
which is generated from the docs tree on every build and so can never be behind
the feature set. These tests keep that routing honest: the index has to be where
the skill says it is, and every reference has to be reachable, otherwise a
shipped feature is invisible and the agent answers "Hermes can't do that."
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "autonomous-ai-agents" / "hermes-agent"
SKILL_MD = SKILL_DIR / "SKILL.md"
GENERATOR = REPO / "website" / "scripts" / "generate-llms-txt.py"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_every_referenced_file_exists(skill_text):
    """Routing a question to a file that isn't there is a dead end."""
    targets = set(re.findall(r"`((?:references|templates)/[^`]+)`", skill_text))

    assert targets, "the skill's routing table no longer references any files"
    for target in sorted(targets):
        assert (SKILL_DIR / target).exists(), f"SKILL.md routes to missing {target}"


def test_every_reference_is_reachable_from_the_skill(skill_text):
    """An unrouted reference is one the agent will never think to open.

    This is the failure that produced the original complaint: content can exist
    and still be invisible because nothing points at it.
    """
    on_disk = {f"references/{path.name}" for path in (SKILL_DIR / "references").glob("*.md")}
    routed = set(re.findall(r"`(references/[^`]+)`", skill_text))

    assert not (on_disk - routed), (
        f"reference files no reader will ever reach: {sorted(on_disk - routed)} — "
        "add a routing-table row in SKILL.md"
    )


def test_unknown_features_route_to_the_published_index(skill_text):
    """The catch-all is what makes coverage of the whole product possible."""
    assert "/docs/llms.txt" in skill_text
    # web_extract can be disabled; terminal never is.
    assert "curl" in skill_text, "no way to reach the index without web tools"


def test_the_index_is_published_where_the_skill_says_it_is(skill_text):
    """A skill pointing at a URL nobody generates is worse than no routing."""
    spec = importlib.util.spec_from_file_location("generate_llms_txt", GENERATOR)
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    assert f"{gen.SITE_BASE}/llms.txt" in skill_text


def test_busy_reference_lists_status_subcommand():
    from hermes_cli.commands import COMMAND_REGISTRY

    busy_command = next(command for command in COMMAND_REGISTRY if command.name == "busy")
    slash_commands = (SKILL_DIR / "references" / "slash-commands.md").read_text(encoding="utf-8")
    assert "status" in busy_command.subcommands
    assert f"/busy {busy_command.args_hint}" in slash_commands
