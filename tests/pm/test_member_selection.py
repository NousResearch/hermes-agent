"""The workspace and admission paths agree on scoped keys and disabled plugins."""
from pm.workspace import enabled_member_dirs


def test_scoped_members_respect_disabled_and_keep_other_profiles(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    current = home / "plugins" / "group" / "plugin"
    other_home = home / "profiles" / "other"
    other = other_home / "plugins" / "same"
    for member in (current, other):
        member.mkdir(parents=True)
        (member / "pyproject.toml").write_text('[project]\nname="test"\nversion="1"\n')
    (home / "config.yaml").write_text('plugins:\n  enabled: [group/plugin]\n  disabled: []\n')
    (other_home / "config.yaml").write_text('plugins:\n  enabled: [same]\n')
    assert set(enabled_member_dirs()) == {current, other}
    (home / "config.yaml").write_text('plugins:\n  enabled: [group/plugin]\n  disabled: [group/plugin]\n')
    assert enabled_member_dirs() == [other]


def test_manifest_only_member_is_named_after_its_plugin_dir_and_stays_unique(tmp_path):
    """uv's conflict text names the workspace member (``hermes-plugin-<key> depends on …``);
    a bare path hash left the user with nothing to disable. Two same-named plugins from
    different homes must still be distinct members."""
    import tomllib
    from pm.workspace import _workspace_member

    members = []
    for home in ("home-a", "home-b"):
        plugin = tmp_path / home / "plugins" / "My Plugin!"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: my-plugin\npip_dependencies: [left-pad-py]\n", encoding="utf-8")
        root = tmp_path / f"gen-{home}"
        root.mkdir()
        members.append(_workspace_member(plugin, root, identity=plugin))
    names = [tomllib.loads((m / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"] for m in members]
    assert all(name.startswith("hermes-plugin-my-plugin-") for name in names), names
    assert names[0] != names[1]
    assert members[0].name != members[1].name


def test_pyproject_member_is_renamed_by_its_key_and_stays_unique(tmp_path):
    """A plugin that ships its own pyproject declares its real [project].name, and
    uv identifies workspace members by that name — enabling the plugin in two
    profiles then fails ``uv lock`` with "Two workspace members are both named
    …". Rename metadata-only members by their unique key, like manifest-only
    ones; a buildable member keeps the name its package metadata reports."""
    import tomllib
    from pm.workspace import _workspace_member

    members = []
    for home in ("home-a", "home-b"):
        plugin = tmp_path / home / "plugins" / "hindsight"
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            '[project]\nname = "hermes-plugin-hindsight"\nversion = "1.0.0"\n'
            'dependencies = ["hindsight-client>=0.10.1"]\n',
            encoding="utf-8",
        )
        root = tmp_path / f"gen-{home}"
        root.mkdir()
        members.append(_workspace_member(plugin, root, identity=plugin))
    names = [tomllib.loads((m / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"] for m in members]
    assert all(name.startswith("hermes-plugin-hindsight-") for name in names), names
    assert names[0] != names[1]
    document = tomllib.loads((members[0] / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["version"] == "1.0.0"
    assert document["project"]["dependencies"] == ["hindsight-client>=0.10.1"]


def test_metadata_only_member_is_always_lockable(tmp_path):
    """uv refuses a workspace member whose [project] table has neither ``version``
    nor a dynamic entry for it (PEP 621), and it refuses that while *parsing* the
    workspace — so one plugin with a sloppy pyproject broke every later ``uv lock``.
    Staging adds what uv requires, and must not stamp a static version onto a member
    that already declared it dynamic."""
    import tomllib
    from pm.workspace import _workspace_member

    variants = {
        "tooling-only": '[tool.ruff]\ntarget-version = "py311"\n',
        "versionless": '[project]\nname = "hermes-lcm"\n',
        "declared": '[project]\nname = "hermes-pkg"\nversion = "4.5.6"\n',
        "dynamic": '[project]\nname = "hermes-scm"\ndynamic = ["version"]\n',
    }
    staged = {}
    for kind, pyproject in variants.items():
        plugin = tmp_path / "home" / "plugins" / kind
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        root = tmp_path / f"gen-{kind}"
        root.mkdir()
        member = _workspace_member(plugin, root, identity=plugin)
        project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        staged[kind] = project
        # The contract uv enforces, asserted directly: exactly one version source.
        assert ("version" in project) != ("version" in project.get("dynamic", ())), kind
    # A staged version is the inert placeholder manifest-only members already use;
    # a declared or dynamic one is left exactly as the plugin shipped it.
    assert staged["versionless"]["version"] == "0.0.0"
    assert staged["declared"]["version"] == "4.5.6"
    assert "version" not in staged["dynamic"]
    assert staged["dynamic"]["dynamic"] == ["version"]


def test_tooling_only_pyproject_leaves_the_manifest_in_charge(tmp_path):
    """A pyproject that carries only tooling config is not a packaging surface, so
    it must not become a workspace member or shadow the manifest's dependencies —
    otherwise a plugin's own ``pip_dependencies`` are silently dropped."""
    from pm.plugin_declarations import read_python_declaration
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "hermes-lcm-x"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[tool.ruff]\ntarget-version = "py311"\nline-length = 100\n', encoding="utf-8")
    (plugin / "plugin.yaml").write_text(
        'name: hermes-lcm-x\npip_dependencies: ["lcm-x>=0.24,<1"]\n', encoding="utf-8")

    declaration = read_python_declaration(plugin)

    assert declaration.pyproject is None
    assert declaration.requirements == ("lcm-x>=0.24,<1",)
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    assert "plugin-deps" in member.parts, member


def test_buildable_pyproject_member_keeps_its_declared_name(tmp_path):
    """uv verifies a buildable member's [project].name against the package metadata
    its backend produces, so renaming it breaks the build ("Package metadata name
    … does not match given name"); only metadata-only members may be renamed."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "replay"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "replay-plugin"\nversion = "1.0"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    assert (member / "pyproject.toml").read_text(encoding="utf-8") == (
        plugin / "pyproject.toml").read_text(encoding="utf-8")
