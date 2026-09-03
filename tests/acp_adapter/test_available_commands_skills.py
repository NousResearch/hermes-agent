"""ACP available_commands includes skills/bundles; slash expands skill bodies.

Dogfood gate for Buzz composer palette (#2528 / #3537): hermes-acp must
advertise real skills and expand /skill into the agent turn (TUI parity).

Fixtures are synthetic under the autouse HERMES_HOME tempdir — do not rely
on a developer-installed skill library (tests/conftest.py isolates skills/).
"""

from __future__ import annotations

from pathlib import Path
import pytest

import agent.skill_bundles as skill_bundles
import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionState

# Long enough that expand assertions (len > 500) stay meaningful.
_SKILL_BODY = (
    "Synthetic skill body for ACP advertise/expand unit tests. "
    "This is not a real skill — it only exists so the test suite can "
    "exercise advertisement and slash expansion without a live install. "
    "Repeat block for length: " + ("scaffold-line\n" * 40)
)


def _reset_skill_and_bundle_caches() -> None:
    skill_commands._skill_commands = {}
    skill_commands._skill_commands_platform = None
    skill_bundles._bundles_cache = {}
    skill_bundles._bundles_cache_mtime = None


@pytest.fixture()
def synthetic_skill_library(tmp_path, monkeypatch):
    """Install one skill + one bundle under the hermetic HERMES_HOME.

    conftest already redirects HERMES_HOME to ``tmp_path/hermes_test`` and
    creates an empty ``skills/`` dir. We write into that tree (or the live
    env path) and clear caches so discovery picks up the fixtures.
    """
    import os

    hermes_home = Path(os.environ["HERMES_HOME"])
    skills_dir = hermes_home / "skills"
    skill_dir = skills_dir / "acp-fixture-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: acp-fixture-skill\n"
        "description: Synthetic skill for ACP available_commands tests\n"
        "---\n\n"
        f"# acp-fixture-skill\n\n{_SKILL_BODY}\n",
        encoding="utf-8",
    )

    bundles_dir = hermes_home / "skill-bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    (bundles_dir / "acp-fixture-bundle.yaml").write_text(
        "name: acp-fixture-bundle\n"
        "description: Synthetic bundle for ACP available_commands tests\n"
        "skills:\n"
        "  - acp-fixture-skill\n",
        encoding="utf-8",
    )

    # Match test_session_skill_previews: pin SKILLS_DIR + clear caches so
    # import-time SKILLS_DIR does not stick to a real ~/.hermes tree.
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    _reset_skill_and_bundle_caches()
    skill_commands.scan_skill_commands()
    skill_bundles.scan_bundles()
    yield {
        "skill": "acp-fixture-skill",
        "bundle": "acp-fixture-bundle",
        "skills_dir": skills_dir,
        "bundles_dir": bundles_dir,
    }
    _reset_skill_and_bundle_caches()


@pytest.fixture()
def help_named_bundle(synthetic_skill_library, monkeypatch):
    """User bundle that would collide with built-in /help if unguarded."""
    bundles_dir = synthetic_skill_library["bundles_dir"]
    (bundles_dir / "help.yaml").write_text(
        "name: help\n"
        "description: Malicious-looking user bundle shadowing /help\n"
        "skills:\n"
        "  - acp-fixture-skill\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    _reset_skill_and_bundle_caches()
    skill_bundles.scan_bundles()
    yield
    _reset_skill_and_bundle_caches()


@pytest.fixture()
def status_named_bundle(synthetic_skill_library, monkeypatch):
    """User bundle that collides with a non-ACP Hermes CLI command."""
    bundles_dir = synthetic_skill_library["bundles_dir"]
    (bundles_dir / "status.yaml").write_text(
        "name: status\n"
        "description: User bundle shadowing the core /status command\n"
        "skills:\n"
        "  - acp-fixture-skill\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    _reset_skill_and_bundle_caches()
    skill_bundles.scan_bundles()
    yield
    _reset_skill_and_bundle_caches()


def test_available_commands_static_invariants():
    """Required ACP static commands exist with unique names (order free)."""
    cmds = HermesACPAgent._available_commands()
    names = [c.name for c in cmds]
    assert len(names) == len({n.lower() for n in names})

    required = {
        "help",
        "model",
        "tools",
        "context",
        "reset",
        "compress",
        "steer",
        "queue",
        "version",
    }
    present = {n.lower() for n in names}
    missing = required - present
    assert not missing, f"missing required static commands: {sorted(missing)}"

    by_name = {c.name.lower(): c for c in cmds}
    assert by_name["help"].description
    # input-bearing commands keep a hint for ACP clients
    assert by_name["model"].input is not None
    assert by_name["steer"].input is not None
    assert by_name["queue"].input is not None


def test_available_commands_includes_skills_and_unique(synthetic_skill_library):
    cmds = HermesACPAgent._available_commands()
    names = [c.name for c in cmds]
    assert len(names) == len({n.lower() for n in names})
    assert synthetic_skill_library["skill"] in names
    assert synthetic_skill_library["bundle"] in names


def test_available_commands_does_not_advertise_bundle_shadowing_cli_command(
    status_named_bundle,
):
    names = [command.name for command in HermesACPAgent._available_commands()]

    assert skill_bundles.resolve_bundle_command_key("status") == "/status"
    assert "status" not in names


def test_available_commands_reserves_cap_for_skills(monkeypatch):
    bundles = {
        f"/bundle-{index}": {"description": f"Bundle {index}"}
        for index in range(HermesACPAgent._MAX_ADVERTISED_SKILL_COMMANDS)
    }
    skills = {"/priority-skill": {"description": "A directly invokable skill"}}
    monkeypatch.setattr(skill_bundles, "get_skill_bundles", lambda: bundles)
    monkeypatch.setattr(skill_commands, "get_skill_commands", lambda: skills)

    names = [command.name for command in HermesACPAgent._available_commands()]

    assert "priority-skill" in names
    assert len(names) == len(HermesACPAgent._ADVERTISED_COMMANDS) + 250


def test_expand_unknown_returns_none():
    assert HermesACPAgent._expand_skill_or_bundle_slash("/not-a-skill-zzzz") is None
    assert HermesACPAgent._expand_skill_or_bundle_slash("nope") is None
    assert HermesACPAgent._expand_skill_or_bundle_slash("") is None


def test_expand_skill_includes_body_and_instruction(synthetic_skill_library):
    slug = synthetic_skill_library["skill"]
    msg = HermesACPAgent._expand_skill_or_bundle_slash(
        f"/{slug} smoke-instruction-token"
    )
    assert isinstance(msg, str)
    assert "IMPORTANT" in msg
    assert "smoke-instruction-token" in msg
    assert len(msg) > 500


def test_expand_skill_uses_acp_session_identity(
    synthetic_skill_library, monkeypatch, tmp_path
):
    """Usage/provenance writes stay scoped to the ACP session and cwd."""
    slug = synthetic_skill_library["skill"]
    original = skill_commands.build_skill_invocation_message
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(skill_commands, "build_skill_invocation_message", capture)
    state = SessionState(
        session_id="acp-session-123",
        agent=object(),
        cwd=str(tmp_path),
    )

    msg = HermesACPAgent._expand_skill_or_bundle_slash(f"/{slug} verify", state)

    assert isinstance(msg, str)
    assert captured["task_id"] == "acp-session-123"


def test_expand_bundle_returns_string_message(synthetic_skill_library):
    slug = synthetic_skill_library["bundle"]
    msg = HermesACPAgent._expand_skill_or_bundle_slash(f"/{slug} orient-token")
    assert isinstance(msg, str)
    assert "IMPORTANT" in msg or "bundle" in msg.lower()
    assert "orient-token" in msg
    assert len(msg) > 500


def test_expand_skill_underscore_alias(synthetic_skill_library):
    msg = HermesACPAgent._expand_skill_or_bundle_slash("/acp_fixture_skill x")
    assert isinstance(msg, str)
    assert "IMPORTANT" in msg


def test_builtin_help_not_shadowed_by_user_bundle(help_named_bundle):
    """TUI parity: resolve_command wins before bundle expansion."""
    # Bundle is registered...
    assert skill_bundles.resolve_bundle_command_key("help") == "/help"
    # ...but ACP expand must not treat /help as a skill-bundle scaffold.
    expanded = HermesACPAgent._expand_skill_or_bundle_slash("/help")
    assert expanded is None
