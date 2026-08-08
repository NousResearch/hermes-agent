from pathlib import Path

from prompt_toolkit.document import Document

def _make_skill(skills_dir: Path, name: str, body: str = "Do the thing.") -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""\
---
name: {name}
description: Description for {name}.
---

# {name}

{body}
""",
        encoding="utf-8",
    )


def _make_bundle(bundles_dir: Path, slug: str, skills: list[str]) -> None:
    bundles_dir.mkdir(parents=True, exist_ok=True)
    (bundles_dir / f"{slug}.yaml").write_text(
        "name: {slug}\nskills:\n{skills}\n".format(
            slug=slug,
            skills="\n".join(f"  - {skill}" for skill in skills),
        ),
        encoding="utf-8",
    )


def _configure_protected_governance(home: Path) -> None:
    (home / "governance").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """\
skills:
  governance:
    registry_path: governance/skills-registry.yaml
    task_class: ardyn_engineering
    protected_task_classes:
      - ardyn_engineering
""",
        encoding="utf-8",
    )
    (home / "governance" / "skills-registry.yaml").write_text(
        """\
version: 1
skills:
  - name: ToolTrust
    classification: COMPATIBILITY_ONLY
  - name: SafeSkill
    classification: CURRENT
""",
        encoding="utf-8",
    )


def _make_cli():
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj.config = {}
    return obj


def _reset_skill_discovery_caches(monkeypatch) -> None:
    import agent.skill_commands as skill_commands_mod

    # Clear scan-time discovery state so each test re-resolves the patched
    # skills dir instead of inheriting a prior governance test's command map.
    monkeypatch.setattr(skill_commands_mod, "_skill_commands", {})
    monkeypatch.setattr(skill_commands_mod, "_skill_commands_platform", None)


def test_show_help_hides_governance_blocked_skills(monkeypatch, tmp_path):
    import agent.skill_commands as skill_commands_mod
    import cli as cli_mod

    home = tmp_path / "home"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _configure_protected_governance(home)
    _make_skill(skills_dir, "ToolTrust", body="blocked")
    _make_skill(skills_dir, "SafeSkill", body="allowed")

    printed: list[str] = []

    class _FakeChatConsole:
        def print(self, *args, **kwargs):
            printed.append(" ".join(str(arg) for arg in args))

    cli_obj = _make_cli()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr(cli_mod, "_skill_commands", None)
    monkeypatch.setattr(skill_commands_mod, "_skill_commands", {})
    monkeypatch.setattr(skill_commands_mod, "_skill_commands_platform", None)
    monkeypatch.setattr(cli_mod, "ChatConsole", _FakeChatConsole)
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: printed.append(str(text)))

    cli_obj.show_help()

    output = "\n".join(printed)
    assert "/safeskill" in output
    assert "/tooltrust" not in output


def test_cli_completer_hides_governance_blocked_skills(monkeypatch, tmp_path):
    import agent.skill_commands as skill_commands_mod
    import cli as cli_mod
    from hermes_cli.commands import SlashCommandCompleter

    home = tmp_path / "home"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _configure_protected_governance(home)
    _make_skill(skills_dir, "ToolTrust", body="blocked")
    _make_skill(skills_dir, "SafeSkill", body="allowed")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr(cli_mod, "_skill_commands", None)
    monkeypatch.setattr(skill_commands_mod, "_skill_commands", {})
    monkeypatch.setattr(skill_commands_mod, "_skill_commands_platform", None)

    completer = SlashCommandCompleter(
        skill_commands_provider=lambda: cli_mod.get_skill_commands(),
    )
    completions = list(completer.get_completions(Document("/s"), None))
    texts = {item.text for item in completions}

    assert "safeskill" in texts
    assert "tooltrust" not in texts


def test_show_help_hides_bundles_when_governance_config_is_malformed(monkeypatch, tmp_path):
    import agent.skill_bundles as skill_bundles_mod
    import cli as cli_mod

    home = tmp_path / "home"
    skills_dir = tmp_path / "skills"
    bundles_dir = tmp_path / "bundles"
    home.mkdir()
    skills_dir.mkdir()
    (home / "config.yaml").write_text("skills:\n  governance: []\n", encoding="utf-8")
    _make_skill(skills_dir, "SafeSkill", body="allowed")
    _make_bundle(bundles_dir, "safe-pack", ["SafeSkill"])

    printed: list[str] = []

    class _FakeChatConsole:
        def print(self, *args, **kwargs):
            printed.append(" ".join(str(arg) for arg in args))

    cli_obj = _make_cli()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr(cli_mod, "_skill_bundles", None)
    monkeypatch.setattr(skill_bundles_mod, "_bundles_cache", {})
    monkeypatch.setattr(skill_bundles_mod, "_bundles_cache_mtime", None)
    monkeypatch.setattr(cli_mod, "ChatConsole", _FakeChatConsole)
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: printed.append(str(text)))

    cli_obj.show_help()

    output = "\n".join(printed)
    assert "/safe-pack" not in output


def test_show_help_keeps_bundles_visible_when_unprotected(monkeypatch, tmp_path):
    import agent.skill_bundles as skill_bundles_mod
    import cli as cli_mod

    home = tmp_path / "home"
    skills_dir = tmp_path / "skills"
    bundles_dir = tmp_path / "bundles"
    skills_dir.mkdir()
    _make_skill(skills_dir, "SafeSkill", body="allowed")
    _make_bundle(bundles_dir, "safe-pack", ["SafeSkill"])

    printed: list[str] = []

    class _FakeChatConsole:
        def print(self, *args, **kwargs):
            printed.append(" ".join(str(arg) for arg in args))

    cli_obj = _make_cli()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr(cli_mod, "_skill_bundles", None)
    monkeypatch.setattr(skill_bundles_mod, "_bundles_cache", {})
    monkeypatch.setattr(skill_bundles_mod, "_bundles_cache_mtime", None)
    monkeypatch.setattr(cli_mod, "ChatConsole", _FakeChatConsole)
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: printed.append(str(text)))

    cli_obj.show_help()

    output = "\n".join(printed)
    assert "/safe-pack" in output


def test_slash_exec_help_and_commands_hide_governance_blocked_skills(monkeypatch, tmp_path):
    import agent.skill_commands as skill_commands_mod
    from hermes_cli.slash_exec import CommandContext, execute_command

    home = tmp_path / "home"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _configure_protected_governance(home)
    _make_skill(skills_dir, "ToolTrust", body="blocked")
    _make_skill(skills_dir, "SafeSkill", body="allowed")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    _reset_skill_discovery_caches(monkeypatch)
    assert skill_commands_mod._skill_commands == {}

    help_reply = execute_command("help", CommandContext(surface="gateway"))
    commands_reply = execute_command(
        "commands",
        CommandContext(surface="gateway", options={"page_size": 200}),
    )

    assert "/safeskill" in help_reply.text
    assert "/tooltrust" not in help_reply.text
    assert "/safeskill" in commands_reply.text
    assert "/tooltrust" not in commands_reply.text


def test_slash_exec_bundles_hide_bundles_when_governance_config_is_malformed(monkeypatch, tmp_path):
    from hermes_cli.slash_exec import CommandContext, execute_command

    home = tmp_path / "home"
    skills_dir = tmp_path / "skills"
    bundles_dir = tmp_path / "bundles"
    home.mkdir()
    skills_dir.mkdir()
    (home / "config.yaml").write_text("skills:\n  governance: []\n", encoding="utf-8")
    _make_skill(skills_dir, "SafeSkill", body="allowed")
    _make_bundle(bundles_dir, "safe-pack", ["SafeSkill"])

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)

    reply = execute_command("bundles", CommandContext(surface="gateway"))

    assert "/safe-pack" not in reply.text
    assert reply.data["bundles"] == []


def test_slash_exec_bundles_preserve_unprotected_behavior(monkeypatch, tmp_path):
    from hermes_cli.slash_exec import CommandContext, execute_command

    home = tmp_path / "home"
    skills_dir = tmp_path / "skills"
    bundles_dir = tmp_path / "bundles"
    skills_dir.mkdir()
    _make_skill(skills_dir, "SafeSkill", body="allowed")
    _make_bundle(bundles_dir, "safe-pack", ["SafeSkill"])

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)

    reply = execute_command("bundles", CommandContext(surface="gateway"))

    assert "/safe-pack" in reply.text
    assert [info["slug"] for info in reply.data["bundles"]] == ["safe-pack"]
