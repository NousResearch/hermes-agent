"""QA scenario for skill discoverability after create / update / install / delete.

Origin
------
Audit follow-up from the OpenClaw compatibility review (Spearhead task
``t_13bc8f81``, parent ``t_00ca2300``).  The audit recommended grafting
"OpenClaw skill reload/status QA ideas" — i.e. proving end-to-end that a
skill written, updated, or removed on disk becomes (or stops being)
visible through every Hermes surface a user can hit it from.

What this file covers
---------------------
The other ``tests/agent/`` and ``tests/tools/`` modules cover each surface
in isolation:

* ``test_skill_commands.py`` — ``scan_skill_commands`` against a fixed disk.
* ``test_skill_commands_reload.py`` — ``reload_skills`` diff output.
* ``test_skill_manager_tool.py`` — ``skill_manage`` CRUD against a temp dir.
* ``test_plugin_skills.py`` — plugin namespacing + ``skill_view`` dispatch.

This module is a stitched-together QA scenario: it runs the real
``skill_manage`` create / edit / delete dispatcher, then asserts the
results show up consistently across **all three** runtime discoverability
surfaces a user can hit a skill from:

1. ``agent.skill_commands.reload_skills()`` — the diff the CLI / gateway
   ``/reload-skills`` slash command exposes.
2. ``agent.skill_commands.get_skill_commands()`` — the ``/skill-name``
   slash-command map the gateway dispatchers and CLI route off.
3. ``tools.skills_tool.skills_list()`` — the tier-1 catalog the model
   reads via the ``skills_list`` tool.

It also pins three contracts the OpenClaw audit specifically called out:

* **External dirs (the "install" path proxy):** skills landing under a
  configured ``skills.external_dirs`` entry — the same on-disk location
  hub-installed and externally-managed skills end up in — must surface
  through all three runtime paths after ``reload_skills``.
* **Plugin-provided skill contract:** skills registered via
  ``PluginContext.register_skill`` are intentionally **not** in the
  slash-command map or ``skills_list`` catalog; they are explicit-load
  only via ``skill_view("plugin:skill")``.  Tested as a positive
  assertion so a future change that surfaces them in slash commands has
  to update this contract explicitly.
* **Profile isolation:** ``HERMES_HOME`` is the profile root.  Two
  profiles must not see each other's skills through any runtime
  discoverability surface.

The scenario is reproducible from one ``pytest`` invocation; see Kanban
task ``t_13bc8f81`` and review gate ``t_ecd707a1`` for the acceptance
evidence and the exact reproduction command.
"""

import json
import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────


VALID_SKILL_CONTENT = textwrap.dedent(
    """\
    ---
    name: qa-demo
    description: QA discoverability demo skill.
    ---

    # QA Demo

    Run the QA demo procedure.
    """
)

UPDATED_SKILL_CONTENT = textwrap.dedent(
    """\
    ---
    name: qa-demo
    description: QA discoverability demo skill (revised description).
    ---

    # QA Demo (revised)

    Run the revised QA demo procedure.
    """
)


def _write_external_skill(dir_: Path, name: str, description: str) -> Path:
    """Write a SKILL.md under *dir_* the same way an installer / external
    vault would.  Used to model the disk layout that hub-installed skills
    and ``skills.external_dirs`` entries produce."""
    skill_dir = dir_ / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: {description}
            ---

            # {name}

            External body.
            """
        )
    )
    return skill_dir


def _skills_list_names(category: str | None = None) -> set[str]:
    """Decode ``skills_list`` JSON and return the names it advertises."""
    from tools.skills_tool import skills_list

    payload = json.loads(skills_list(category=category))
    assert payload.get("success") is True, payload
    return {s["name"] for s in payload.get("skills", [])}


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_profile(monkeypatch):
    """Build an isolated HERMES_HOME profile root and redirect every
    discoverability surface at it.

    Mirrors the ``hermes_home`` fixture in ``test_skill_commands_reload.py``
    but also redirects ``tools.skill_manager_tool`` (the CRUD surface) and
    ``agent.skill_utils.get_all_skills_dirs`` (the manager's search path)
    so the full create-to-discover round-trip runs against the same temp
    dir.

    Yields the profile root path.  External dirs are not configured by
    this fixture — the caller patches ``get_external_skills_dirs`` /
    ``get_all_skills_dirs`` when they want to add one (see
    ``TestExternalDirInstallDiscoverability``).
    """
    td = tempfile.mkdtemp(prefix="hermes-skill-qa-")
    monkeypatch.setenv("HERMES_HOME", td)
    home = Path(td)
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    import agent.skill_commands as _sc
    import tools.skill_manager_tool as _smt
    import tools.skills_tool as _st

    monkeypatch.setattr(_st, "HERMES_HOME", home, raising=False)
    monkeypatch.setattr(_st, "SKILLS_DIR", skills_dir, raising=False)
    monkeypatch.setattr(_smt, "HERMES_HOME", home, raising=False)
    monkeypatch.setattr(_smt, "SKILLS_DIR", skills_dir, raising=False)

    # The slash-command map is process-global; clear it so each scenario
    # starts from an empty state.
    monkeypatch.setattr(_sc, "_skill_commands", {}, raising=False)
    monkeypatch.setattr(_sc, "_skill_commands_platform", None, raising=False)

    # Confine ``_find_skill`` (used by skill_manage) to the profile root.
    monkeypatch.setattr(
        "agent.skill_utils.get_all_skills_dirs",
        lambda: [skills_dir],
    )

    yield home

    shutil.rmtree(td, ignore_errors=True)


# ── Lifecycle: create → reload → discoverable ────────────────────────────


class TestCreateUpdateDeleteLifecycle:
    """End-to-end QA: an agent creates a skill via ``skill_manage``; the
    skill must then be visible on every surface a user can discover it
    from, and must disappear from every surface when deleted.

    Each test is a single contiguous scenario rather than per-surface
    asserts — the point of this file is to prove the surfaces stay in
    sync, so the assertions are intentionally bundled."""

    def test_create_then_reload_makes_skill_discoverable_everywhere(
        self, hermes_profile
    ):
        from agent.skill_commands import get_skill_commands, reload_skills
        from tools.skill_manager_tool import skill_manage

        # Prime the slash-command cache against the empty dir so the
        # subsequent reload sees the new skill in its diff.
        get_skill_commands()
        assert get_skill_commands() == {}
        assert _skills_list_names() == set()

        raw = skill_manage(action="create", name="qa-demo", content=VALID_SKILL_CONTENT)
        assert json.loads(raw)["success"] is True

        diff = reload_skills()

        # 1. reload diff names the new skill.
        assert {"name": "qa-demo", "description": "QA discoverability demo skill."} in diff["added"]
        assert diff["removed"] == []
        assert diff["total"] == 1
        assert diff["commands"] == 1

        # 2. /reload-skills' slash-command map exposes the same skill.
        commands = get_skill_commands()
        assert "/qa-demo" in commands
        assert commands["/qa-demo"]["description"] == (
            "QA discoverability demo skill."
        )

        # 3. skills_list (tier-1 catalog used by the model) sees it too.
        assert "qa-demo" in _skills_list_names()

    def test_edit_description_propagates_through_reload(self, hermes_profile):
        from agent.skill_commands import get_skill_commands, reload_skills
        from tools.skill_manager_tool import skill_manage
        from tools.skills_tool import skills_list

        # Prime the cache so the first reload sees the create, the second
        # sees only the description change.
        get_skill_commands()
        skill_manage(action="create", name="qa-demo", content=VALID_SKILL_CONTENT)
        first = reload_skills()
        assert any(a["name"] == "qa-demo" for a in first["added"])

        # Edit replaces SKILL.md whole-cloth; the description in
        # frontmatter is the only thing the discoverability surfaces
        # care about.
        raw = skill_manage(action="edit", name="qa-demo", content=UPDATED_SKILL_CONTENT)
        assert json.loads(raw)["success"] is True

        second = reload_skills()

        # The edit is not a structural diff — name stays the same — so
        # ``added`` / ``removed`` are empty and the skill shows in
        # ``unchanged``.  This is the intended behaviour: the reload
        # diff signals catalog churn, not metadata churn.
        assert second["added"] == []
        assert second["removed"] == []
        assert "qa-demo" in second["unchanged"]

        # The slash-command map and skills_list, however, do reflect the
        # updated description after the rescan — that is what makes a
        # description edit observable to the user.
        new_desc = "QA discoverability demo skill (revised description)."
        assert get_skill_commands()["/qa-demo"]["description"] == new_desc

        payload = json.loads(skills_list())
        rows = {s["name"]: s["description"] for s in payload["skills"]}
        assert rows["qa-demo"] == new_desc

    def test_delete_removes_skill_from_every_surface(self, hermes_profile):
        from agent.skill_commands import get_skill_commands, reload_skills
        from tools.skill_manager_tool import skill_manage

        skill_manage(action="create", name="qa-demo", content=VALID_SKILL_CONTENT)
        reload_skills()
        assert "/qa-demo" in get_skill_commands()
        assert "qa-demo" in _skills_list_names()

        raw = skill_manage(action="delete", name="qa-demo")
        assert json.loads(raw)["success"] is True

        diff = reload_skills()

        # 1. reload diff names the removed skill, preserving its
        #    pre-delete description (so the ``Removed Skills:`` block
        #    in the user-facing render survives the on-disk delete).
        assert {"name": "qa-demo", "description": "QA discoverability demo skill."} in diff["removed"]
        assert diff["added"] == []
        assert diff["total"] == 0
        assert diff["commands"] == 0

        # 2. Slash-command map and skills_list both drop the skill.
        assert "/qa-demo" not in get_skill_commands()
        assert "qa-demo" not in _skills_list_names()


# ── External dirs (the "install" path proxy) ─────────────────────────────


class TestExternalDirInstallDiscoverability:
    """A skill that lands under a configured ``skills.external_dirs``
    entry — the on-disk layout shared by hub-installed skills and
    externally managed vaults — must surface through every runtime
    discoverability path after ``reload_skills``.

    This is the OpenClaw "skill reload/status after install" QA item:
    proving that an "installed" skill (modelled here by the same on-disk
    layout) is reachable from the slash-command map and ``skills_list``
    once the user runs ``/reload-skills``.
    """

    @pytest.fixture
    def hermes_with_external(self, hermes_profile, monkeypatch, tmp_path):
        """Add an external skills dir alongside the isolated profile."""
        from pathlib import Path as _Path

        external = tmp_path / "external-vault"
        external.mkdir()
        local_skills = hermes_profile / "skills"

        monkeypatch.setattr(
            "agent.skill_utils.get_external_skills_dirs",
            lambda: [external],
        )
        # `_find_all_skills` and `scan_skill_commands` both import
        # `get_external_skills_dirs` from agent.skill_utils — patching the
        # attribute on the module covers both call sites.

        # The manager search path also includes the external dir so the
        # rest of the lifecycle (edit / delete) can reach the install.
        monkeypatch.setattr(
            "agent.skill_utils.get_all_skills_dirs",
            lambda: [local_skills, external],
        )

        return _Path(external)

    def test_external_install_surfaces_after_reload(self, hermes_with_external):
        from agent.skill_commands import get_skill_commands, reload_skills

        # Prime against the empty state so the install shows in the diff.
        get_skill_commands()
        assert "/installed-skill" not in get_skill_commands()
        assert "installed-skill" not in _skills_list_names()

        _write_external_skill(
            hermes_with_external,
            "installed-skill",
            "Pretend hub-installed skill.",
        )

        diff = reload_skills()

        assert {"name": "installed-skill", "description": "Pretend hub-installed skill."} in diff["added"]
        assert "/installed-skill" in get_skill_commands()
        assert "installed-skill" in _skills_list_names()

    def test_local_skill_shadows_external_with_same_name(
        self, hermes_profile, hermes_with_external
    ):
        """When the same skill name exists under both the local profile and
        an external dir, ``scan_skill_commands`` scans local first and
        records the first hit, so the local copy wins the slash-command
        slot.  Pinning this contract protects users who fork an installed
        skill into their local profile (the local override must be the one
        that runs).

        Direct-write at the file level rather than via ``skill_manage`` —
        ``skill_manage(action='create')`` refuses to create a skill whose
        name is already discoverable in any configured skill root, so the
        "fork to override" flow is necessarily a raw-disk operation
        (``git pull``, ``cp``, hub install into the local profile).
        """
        from agent.skill_commands import get_skill_commands, reload_skills

        _write_external_skill(
            hermes_with_external,
            "qa-demo",
            "External copy — should be shadowed.",
        )
        _write_external_skill(
            hermes_profile / "skills",
            "qa-demo",
            "Local override — should win.",
        )
        reload_skills()

        assert get_skill_commands()["/qa-demo"]["description"] == (
            "Local override — should win."
        )


# ── Plugin-provided skills (contract assertion) ──────────────────────────


class TestPluginProvidedSkillContract:
    """Plugin skills registered via ``PluginContext.register_skill`` are
    intentionally **not** members of the slash-command map or the
    ``skills_list`` catalog (see ``hermes_cli/plugins.py`` near the
    ``register_skill`` docstring: *"plugin skills are opt-in explicit
    loads only"*).  They are reachable only via the qualified-name
    ``skill_view("plugin:bare-name")`` form.

    Pinning this as a positive assertion protects the contract — if a
    future change starts surfacing plugin skills in slash commands or
    ``skills_list``, this test forces an explicit update."""

    def test_plugin_skill_is_skill_view_only(
        self, hermes_profile, monkeypatch, tmp_path
    ):
        from agent.skill_commands import get_skill_commands, reload_skills
        from hermes_cli import plugins as plugins_mod
        from hermes_cli.plugins import PluginManager
        from tools.skills_tool import skill_view

        pm = PluginManager()
        monkeypatch.setattr(plugins_mod, "_plugin_manager", pm)

        plugin_root = tmp_path / "plugin"
        skill_dir = plugin_root / "skills" / "plugin-only"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: plugin-only\ndescription: explicit-load only\n---\nBody.\n"
        )
        pm._plugin_skills["demoplugin:plugin-only"] = {
            "path": skill_dir / "SKILL.md",
            "plugin": "demoplugin",
            "bare_name": "plugin-only",
            "description": "explicit-load only",
        }

        # Sanity: skill_view reaches the plugin skill by qualified name.
        result = json.loads(skill_view("demoplugin:plugin-only"))
        assert result["success"] is True
        assert "Body." in result["content"]

        # Contract: reload_skills + slash-command map do NOT register
        # the plugin skill.  ``/plugin-only`` is the bare-name slash key
        # that a user would type, so we assert by that key as well as the
        # qualified key — neither should appear.
        reload_skills()
        commands = get_skill_commands()
        assert "/plugin-only" not in commands
        assert "/demoplugin:plugin-only" not in commands

        # Contract: skills_list catalog also omits plugin skills.
        assert "plugin-only" not in _skills_list_names()


# ── Profile isolation ────────────────────────────────────────────────────


class TestProfileIsolation:
    """Two distinct ``HERMES_HOME`` values are two distinct profiles.
    No discoverability surface may leak between them.

    Switching profile is modelled by re-binding ``HERMES_HOME`` /
    ``SKILLS_DIR`` and resetting the slash-command cache — the same
    sequence ``profile_use`` triggers when the user swaps profiles
    interactively.
    """

    def test_skills_do_not_leak_across_profiles(self, monkeypatch):
        import agent.skill_commands as _sc
        import tools.skill_manager_tool as _smt
        import tools.skills_tool as _st
        from agent.skill_commands import get_skill_commands, reload_skills
        from tools.skill_manager_tool import skill_manage

        profile_a = Path(tempfile.mkdtemp(prefix="hermes-qa-a-"))
        profile_b = Path(tempfile.mkdtemp(prefix="hermes-qa-b-"))
        (profile_a / "skills").mkdir(parents=True, exist_ok=True)
        (profile_b / "skills").mkdir(parents=True, exist_ok=True)

        try:
            # ── Bind profile A and create a skill there ──
            monkeypatch.setenv("HERMES_HOME", str(profile_a))
            monkeypatch.setattr(_st, "HERMES_HOME", profile_a, raising=False)
            monkeypatch.setattr(_st, "SKILLS_DIR", profile_a / "skills", raising=False)
            monkeypatch.setattr(_smt, "HERMES_HOME", profile_a, raising=False)
            monkeypatch.setattr(_smt, "SKILLS_DIR", profile_a / "skills", raising=False)
            monkeypatch.setattr(_sc, "_skill_commands", {}, raising=False)
            monkeypatch.setattr(_sc, "_skill_commands_platform", None, raising=False)
            monkeypatch.setattr(
                "agent.skill_utils.get_all_skills_dirs",
                lambda: [profile_a / "skills"],
            )

            skill_manage(
                action="create",
                name="profile-a-skill",
                content=VALID_SKILL_CONTENT.replace("qa-demo", "profile-a-skill"),
            )
            reload_skills()
            assert "/profile-a-skill" in get_skill_commands()
            assert "profile-a-skill" in _skills_list_names()

            # ── Switch to profile B; A's skill must not be visible ──
            monkeypatch.setenv("HERMES_HOME", str(profile_b))
            monkeypatch.setattr(_st, "HERMES_HOME", profile_b, raising=False)
            monkeypatch.setattr(_st, "SKILLS_DIR", profile_b / "skills", raising=False)
            monkeypatch.setattr(_smt, "HERMES_HOME", profile_b, raising=False)
            monkeypatch.setattr(_smt, "SKILLS_DIR", profile_b / "skills", raising=False)
            monkeypatch.setattr(_sc, "_skill_commands", {}, raising=False)
            monkeypatch.setattr(_sc, "_skill_commands_platform", None, raising=False)
            monkeypatch.setattr(
                "agent.skill_utils.get_all_skills_dirs",
                lambda: [profile_b / "skills"],
            )

            reload_skills()
            assert "/profile-a-skill" not in get_skill_commands()
            assert "profile-a-skill" not in _skills_list_names()

            # ── Create a profile-B-local skill; A is still hidden ──
            skill_manage(
                action="create",
                name="profile-b-skill",
                content=VALID_SKILL_CONTENT.replace("qa-demo", "profile-b-skill"),
            )
            reload_skills()
            visible_b = get_skill_commands()
            assert "/profile-b-skill" in visible_b
            assert "/profile-a-skill" not in visible_b

            list_b = _skills_list_names()
            assert "profile-b-skill" in list_b
            assert "profile-a-skill" not in list_b
        finally:
            shutil.rmtree(profile_a, ignore_errors=True)
            shutil.rmtree(profile_b, ignore_errors=True)
