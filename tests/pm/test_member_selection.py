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


def test_versionless_pyproject_member_stages_a_version(tmp_path):
    """uv rejects any [project] table whose `version` is neither set nor listed in
    `project.dynamic` at parse time — before resolution or build — so a version-less
    plugin pyproject failed every workspace `uv lock` (#125583). Staging fills one in."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "lcm"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "hermes-lcm"\ndependencies = ["lcm>=1.0"]\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["version"] == "0.0.0"
    assert document["project"]["name"].startswith("hermes-plugin-lcm-")
    assert document["project"]["dependencies"] == ["lcm>=1.0"]


def test_versionless_buildable_pyproject_member_stages_a_version(tmp_path):
    """Buildable members keep their declared name (uv verifies it against the built
    metadata) but still need a staged version when the plugin omitted one."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "replay"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "replay-plugin"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["version"] == "0.0.0"
    assert document["project"]["name"] == "replay-plugin"


def test_declared_or_dynamic_version_is_left_alone(tmp_path):
    """A plugin that declares its own version — statically or via `dynamic` —
    must not be rewritten."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "mine"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "mine"\nversion = "2.5"\n[build-system]\n'
        'requires = []\nbuild-backend = "backend"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["version"] == "2.5"

    dynamic = tmp_path / "home" / "plugins" / "dyn"
    dynamic.mkdir(parents=True)
    (dynamic / "pyproject.toml").write_text(
        '[project]\nname = "dyn"\ndynamic = ["version", "dependencies"]\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
        encoding="utf-8",
    )
    root2 = tmp_path / "gen2"
    root2.mkdir()
    member2 = _workspace_member(dynamic, root2, identity=dynamic)
    document2 = tomllib.loads((member2 / "pyproject.toml").read_text(encoding="utf-8"))
    assert document2["project"]["dynamic"] == ["version", "dependencies"]
    assert "version" not in document2["project"]
