"""Tests for the `hermes skills verify` CLI surface — the user-facing opt-in
for a skill's declared outcome verifier.

Exercises ``do_verify`` (the handler), the router wiring in
``skills_command``, the ``/skills verify`` slash path, and the verify column
in ``do_list``. Uses a real isolated HERMES_HOME so the sidecar flag actually
persists.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from hermes_cli.skills_hub import do_list, do_verify, skills_command, handle_skills_slash


def _verify_block(run: str = "scripts/verify.py") -> str:
    return "{run: " + json.dumps(run) + ", timeout_seconds: 30}"


def _write_skill(skills_dir: Path, name: str, *, with_verify: bool) -> Path:
    """Create a skill under HERMES_HOME/skills/ — the agent-created shape."""
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    scripts = d / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "verify.py").write_text("print('{}')\n", encoding="utf-8")
    verify_yaml = f"    verify: {_verify_block()}\n" if with_verify else ""
    (d / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: test skill\n"
        "version: 1.0.0\n"
        "metadata:\n"
        "  hermes:\n"
        f"{verify_yaml}"
        "---\n"
        f"# {name}\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with agent-created skills."""
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.skill_usage as skill_usage
    return {"home": home, "skills": skills, "skill_usage": skill_usage}


def _capture(fn) -> str:
    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    fn(console)
    return sink.getvalue()


def test_verify_enable_optin_records_sidecar_flag(cli_env):
    """do_verify(name) flips verify_enabled on the sidecar for an eligible skill."""
    _write_skill(cli_env["skills"], "my-skill", with_verify=True)
    cli_env["skill_usage"].mark_agent_created("my-skill")

    rc = _capture(lambda c: do_verify("my-skill", enable=True, console=c))

    assert "verify: enabled for 'my-skill'" in rc
    assert cli_env["skill_usage"].is_verify_enabled("my-skill") is True


def test_verify_disable_clears_flag(cli_env):
    """do_verify(name, enable=False) turns the verifier off again."""
    _write_skill(cli_env["skills"], "my-skill", with_verify=True)
    cli_env["skill_usage"].mark_agent_created("my-skill")
    cli_env["skill_usage"].set_verify_enabled("my-skill", True)

    rc = _capture(lambda c: do_verify("my-skill", enable=False, console=c))

    assert "verify: disabled for 'my-skill'" in rc
    assert cli_env["skill_usage"].is_verify_enabled("my-skill") is False


def test_verify_refuses_skill_without_verify_block(cli_env):
    """No declared verify block ⇒ clear refusal, not a silent no-op."""
    _write_skill(cli_env["skills"], "no-verify", with_verify=False)
    cli_env["skill_usage"].mark_agent_created("no-verify")

    rc = _capture(lambda c: do_verify("no-verify", enable=True, console=c))

    assert "declares no verify block" in rc
    assert "verify: enabled" not in rc
    assert cli_env["skill_usage"].is_verify_enabled("no-verify") is False


def test_verify_refuses_unknown_skill(cli_env):
    rc = _capture(lambda c: do_verify("ghost", enable=True, console=c))
    assert "not found" in rc
    assert rc.startswith("Error")


def test_verify_refuses_hub_installed_skill(cli_env, monkeypatch):
    """Hub-installed skills are never curator-managed, so the opt-in is refused."""
    import tools.skill_usage as su

    _write_skill(cli_env["skills"], "hub-skill", with_verify=True)
    monkeypatch.setattr(su, "is_hub_installed", lambda name: name == "hub-skill")
    monkeypatch.setattr(su, "is_bundled", lambda name: False)
    monkeypatch.setattr(su, "is_protected_builtin", lambda name: False)

    rc = _capture(lambda c: do_verify("hub-skill", enable=True, console=c))

    assert "Cannot verify" in rc
    assert cli_env["skill_usage"].is_verify_enabled("hub-skill") is False


def test_verify_disable_allowed_even_when_ineligible(cli_env, monkeypatch):
    """Disabling is a consent-revocation safety valve, never gated on eligibility.

    A skill may drift into ineligibility after being enabled (hub-replaced,
    provenance cleared); revoking consent for a subprocess runner must still
    work — otherwise a verifier could become unstoppable.
    """
    import tools.skill_usage as su

    _write_skill(cli_env["skills"], "drifted", with_verify=True)
    su.set_verify_enabled("drifted", True)
    assert su.is_verify_enabled("drifted") is True
    # Now the skill becomes ineligible (e.g. the same name gets hub-installed).
    monkeypatch.setattr(su, "is_hub_installed", lambda name: name == "drifted")
    monkeypatch.setattr(su, "is_bundled", lambda name: False)
    monkeypatch.setattr(su, "is_protected_builtin", lambda name: False)

    rc = _capture(lambda c: do_verify("drifted", enable=False, console=c))

    assert "verify: disabled for 'drifted'" in rc
    assert "Cannot verify" not in rc
    assert su.is_verify_enabled("drifted") is False


def test_verify_refuses_plain_local_unmanaged_skill(cli_env):
    """A plain local skill (no curator-management marker) is refused: its
    outcomes would never surface in curator review, so the opt-in is offered
    only where the verifier has something to feed."""
    _write_skill(cli_env["skills"], "plain", with_verify=True)

    rc = _capture(lambda c: do_verify("plain", enable=True, console=c))

    assert "Cannot verify" in rc
    assert "never surface" in rc
    assert cli_env["skill_usage"].is_verify_enabled("plain") is False


def test_verify_refuses_bundled_builtin_without_prune(cli_env, monkeypatch):
    """Bundled built-ins are only verifiable when curator.prune_builtins is on."""
    import tools.skill_usage as su

    _write_skill(cli_env["skills"], "builtin-skill", with_verify=True)
    monkeypatch.setattr(su, "is_hub_installed", lambda name: False)
    monkeypatch.setattr(su, "is_bundled", lambda name: name == "builtin-skill")
    monkeypatch.setattr(su, "is_protected_builtin", lambda name: False)
    monkeypatch.setattr(su, "_prune_builtins_enabled", lambda: False)

    rc = _capture(lambda c: do_verify("builtin-skill", enable=True, console=c))

    assert "Cannot verify" in rc
    assert cli_env["skill_usage"].is_verify_enabled("builtin-skill") is False


def test_verify_enables_bundled_builtin_with_prune(cli_env, monkeypatch):
    """With curator.prune_builtins on, a bundled built-in's verifier is usable."""
    import tools.skill_usage as su

    _write_skill(cli_env["skills"], "builtin-skill", with_verify=True)
    monkeypatch.setattr(su, "is_hub_installed", lambda name: False)
    monkeypatch.setattr(su, "is_bundled", lambda name: name == "builtin-skill")
    monkeypatch.setattr(su, "is_protected_builtin", lambda name: False)
    monkeypatch.setattr(su, "_prune_builtins_enabled", lambda: True)

    rc = _capture(lambda c: do_verify("builtin-skill", enable=True, console=c))

    assert "verify: enabled for 'builtin-skill'" in rc
    assert cli_env["skill_usage"].is_verify_enabled("builtin-skill") is True


def test_do_list_hides_verify_cell_for_hub_skill(cli_env, monkeypatch):
    """Hub-installed skills get no verify cell even when they declare one —
    do_verify refuses them, so the listing must not invite the opt-in."""
    import tools.skills_hub as hub_mod
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    _write_skill(cli_env["skills"], "hub-with-verify", with_verify=True)

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **kwargs: [
        {"name": "hub-with-verify", "category": "x", "description": "h",
         "dir": str(cli_env["skills"] / "hub-with-verify"), "external": False,
         "verify_declared": True},
    ])
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

    class _Lock:
        def list_installed(self):
            return [{"name": "hub-with-verify", "source": "github", "trust_level": "community"}]

    monkeypatch.setattr(hub_mod.HubLockFile, "list_installed", _Lock().list_installed)

    out = _capture(lambda c: do_list(console=c))

    row = next(l for l in out.splitlines() if "hub-with-verify" in l)
    cells = [c.strip() for c in row.split("│")]
    assert cells[5] == ""  # Verify column is empty — the opt-in is not invited


def test_do_list_hides_verify_cell_for_builtin_without_prune(cli_env, monkeypatch):
    """A bundled built-in with curator.prune_builtins off shows no verify cell."""
    import tools.skill_usage as su
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    _write_skill(cli_env["skills"], "builtin-with-verify", with_verify=True)

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **kwargs: [
        {"name": "builtin-with-verify", "category": "x", "description": "b",
         "dir": str(cli_env["skills"] / "builtin-with-verify"), "external": False,
         "verify_declared": True},
    ])
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {"builtin-with-verify"})
    monkeypatch.setattr(su, "prune_builtins_enabled", lambda: False)

    out = _capture(lambda c: do_list(console=c))

    row = next(l for l in out.splitlines() if "builtin-with-verify" in l)
    cells = [c.strip() for c in row.split("│")]
    assert cells[5] == ""  # Verify column is empty — prune_builtins is off


def test_do_list_shows_verify_cell_for_builtin_with_prune(cli_env, monkeypatch):
    """With curator.prune_builtins on, a bundled built-in that declares a
    verifier gets an on/off cell — the branch most likely to drift."""
    import tools.skill_usage as su
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    _write_skill(cli_env["skills"], "builtin-with-verify", with_verify=True)
    su.set_verify_enabled("builtin-with-verify", True)

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **kwargs: [
        {"name": "builtin-with-verify", "category": "x", "description": "b",
         "dir": str(cli_env["skills"] / "builtin-with-verify"), "external": False,
         "verify_declared": True},
    ])
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {"builtin-with-verify"})
    monkeypatch.setattr(su, "prune_builtins_enabled", lambda: True)

    out = _capture(lambda c: do_list(console=c))

    row = next(l for l in out.splitlines() if "builtin-with-verify" in l)
    cells = [c.strip() for c in row.split("│")]
    assert cells[5] == "on"  # verify_enabled True → eligible built-in shows on


def test_skills_command_routes_verify(cli_env):
    """skills_command wires the parser flag through to the handler."""
    _write_skill(cli_env["skills"], "routed", with_verify=True)
    cli_env["skill_usage"].mark_agent_created("routed")

    class _Args:
        skills_action = "verify"
        name = "routed"
        disable = False

    skills_command(_Args())

    assert cli_env["skill_usage"].is_verify_enabled("routed") is True


def test_skills_command_verify_disable(cli_env):
    _write_skill(cli_env["skills"], "routed", with_verify=True)
    cli_env["skill_usage"].mark_agent_created("routed")
    cli_env["skill_usage"].set_verify_enabled("routed", True)

    class _Args:
        skills_action = "verify"
        name = "routed"
        disable = True

    skills_command(_Args())

    assert cli_env["skill_usage"].is_verify_enabled("routed") is False


def test_slash_verify_enable_and_disable(cli_env):
    _write_skill(cli_env["skills"], "slash-skill", with_verify=True)
    cli_env["skill_usage"].mark_agent_created("slash-skill")

    handle_skills_slash("/skills verify slash-skill", console=Console(file=StringIO(), force_terminal=False, color_system=None))
    assert cli_env["skill_usage"].is_verify_enabled("slash-skill") is True

    handle_skills_slash("/skills verify slash-skill --disable", console=Console(file=StringIO(), force_terminal=False, color_system=None))
    assert cli_env["skill_usage"].is_verify_enabled("slash-skill") is False


def test_slash_verify_requires_name(cli_env):
    sink = StringIO()
    c = Console(file=sink, force_terminal=False, color_system=None)
    handle_skills_slash("/skills verify", console=c)
    assert "Usage: /skills verify <name>" in sink.getvalue()


def test_do_list_shows_verify_column_only_for_declared_skills(cli_env, monkeypatch):
    """Skills that declare a verify block get an on/off cell; others stay quiet."""
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    _write_skill(cli_env["skills"], "with-verify", with_verify=True)
    _write_skill(cli_env["skills"], "plain", with_verify=False)
    cli_env["skill_usage"].mark_agent_created("with-verify")
    cli_env["skill_usage"].mark_agent_created("plain")
    cli_env["skill_usage"].set_verify_enabled("with-verify", True)

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **kwargs: [
        {"name": "with-verify", "category": "x", "description": "v",
         "dir": str(cli_env["skills"] / "with-verify"), "external": False,
         "verify_declared": True},
        {"name": "plain", "category": "x", "description": "p",
         "dir": str(cli_env["skills"] / "plain"), "external": False,
         "verify_declared": False},
    ])
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

    out = _capture(lambda c: do_list(console=c))

    assert "on" in out  # with-verify enabled shows verify:on
    # The plain skill row must not contain a verify cell.
    plain_row = out.splitlines()[next(i for i, l in enumerate(out.splitlines()) if "plain" in l)]
    assert "verify" not in plain_row.lower()