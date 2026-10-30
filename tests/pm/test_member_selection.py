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
        (member / "pyproject.toml").write_text('[project]\nname="test"\nversion="1"\ndependencies=["requests>=2"]\n')
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


def test_manifest_member_extras_are_marked_off_android(tmp_path):
    """Extras drag wheel-less transitive chains into the Android fork of every PM
    lock (onnxruntime ships no Android wheels at all), and uv >= 0.13 fails the
    whole workspace there — during hermes update that failure disables the plugin
    and blanks its settings instead of refusing it (#143305). Core's own extras
    carry the same marker since 07ee800dc7; plain dependencies stay untouched."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "mnemosyne"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: mnemosyne\npip_dependencies:\n"
        "  - mnemosyne-memory[embeddings]>=4.0.0b3\n"
        "  - aiohttp>=3.9,<4\n",
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["dependencies"] == [
        "mnemosyne-memory[embeddings]>=4.0.0b3; sys_platform != 'android'",
        "aiohttp>=3.9,<4",
    ]


def test_pyproject_member_extras_get_the_marker_and_plain_specs_stay_verbatim(tmp_path):
    """A plugin-shipped pyproject is rewritten only when an extra actually gains
    the marker; everything else the author declared is copied verbatim."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "memory"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "memory-plugin"\nversion = "1.0"\n'
        'dependencies = [\n'
        '  "mnemosyne-memory[embeddings]>=4.0.0b3",\n'
        '  "left-pad-py>=1",\n'
        ']\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["dependencies"] == [
        "mnemosyne-memory[embeddings]>=4.0.0b3; sys_platform != 'android'",
        "left-pad-py>=1",
    ]


def test_member_extras_with_own_sys_platform_marker_are_left_alone(tmp_path):
    """An extra that already scopes itself to platforms (positively or negatively)
    has expressed its author's intent; appending another clause could contradict it."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "scoped"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "scoped-plugin"\nversion = "1.0"\n'
        'dependencies = [\n'
        '  "webrtcvad-wheels[all]>=1; sys_platform == \'linux\'",\n'
        '  "plain-extra[tip]>=2",\n'
        ']\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["dependencies"] == [
        "webrtcvad-wheels[all]>=1; sys_platform == 'linux'",
        "plain-extra[tip]>=2; sys_platform != 'android'",
    ]


def test_buildable_member_without_dependencies_stays_byte_identical(tmp_path):
    """The extras rewrite must not materialize an empty [project].dependencies
    list into a buildable member that never declared one."""
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "bare"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "bare-plugin"\nversion = "1.0"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    assert (member / "pyproject.toml").read_text(encoding="utf-8") == (
        plugin / "pyproject.toml").read_text(encoding="utf-8")


def test_virtual_member_without_project_metadata_gets_a_version(tmp_path):
    """A metadata-only member whose pyproject declares no [project] (lint config
    only) is emitted as ``[project] name = …``; uv rejects a [project] table
    without ``version`` (or ``dynamic``) at ``uv lock`` time, aborting the whole
    update. Generated virtual members carry the same inert version plugin-deps
    use."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "hermes-lcm"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text('[tool.ruff]\ntarget-version = "py311"\n', encoding="utf-8")
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.0.0"


def test_virtual_member_declaring_dynamic_version_is_left_alone(tmp_path):
    """PEP 621 forbids supplying ``project.version`` while it is listed under
    ``project.dynamic`` ("You cannot provide a value for project.version and
    list it under project.dynamic at the same time"). A metadata-only member
    may legitimately declare ``dynamic = ["version"]``; the generated member
    must not gain a static version in that case, or ``uv lock`` rejects the
    conflicted manifest."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "dynver"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "dynver"\ndynamic = ["version"]\n', encoding="utf-8"
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "version" not in project
    assert project["dynamic"] == ["version"]


def test_buildable_member_without_a_version_gets_one_and_keeps_its_name(tmp_path):
    """uv rejects a [project] table with no version at parse time, buildable or not.
    A buildable member still keeps its declared name (uv checks it against its backend)."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "noversion"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "noversion-plugin"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project == {"name": "noversion-plugin", "version": "0.0.0"}
