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


def test_tool_only_pyproject_member_locks_cleanly(tmp_path):
    """A plugin whose pyproject is tool-only (ruff config, no [project] and no
    [build-system] table) is a virtual member: its synthesized [project] table
    must carry the version and requires-python uv demands. Without them
    ``uv lock`` exits 2 ("required project.version ... not set") and the
    updater silently disables the plugin while config still lists it enabled."""
    import os
    import shutil
    import subprocess
    import sys
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "hermes-lcm"
    plugin.mkdir(parents=True)
    shipped = '[tool.ruff]\ntarget-version = "py311"\n'
    (plugin / "pyproject.toml").write_text(shipped, encoding="utf-8")
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)

    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    project = document["project"]
    assert project["name"].startswith("hermes-plugin-hermes-lcm-"), project["name"]
    assert project["version"] == "0.0.0"
    assert project["requires-python"] == ">=3.11"
    assert document["tool"]["ruff"]["target-version"] == "py311", \
        "the plugin's tool config must survive the synthesis verbatim"
    # Synthesis copies to the member; the plugin checkout itself is untouched.
    assert (plugin / "pyproject.toml").read_text(encoding="utf-8") == shipped

    # The point of the fix: uv accepts the synthesized member instead of
    # rejecting it and silently disabling the plugin.
    uv = shutil.which("uv")
    assert uv is not None, "real uv is required to reproduce the lock rejection"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "PYTHON")) and key != "VIRTUAL_ENV"}
    result = subprocess.run(
        [uv, "lock", "--offline", "--python", sys.executable],
        cwd=member, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_tool_only_pyproject_member_keeps_fields_it_declared_itself(tmp_path):
    """``setdefault`` in the virtual branch: a tool-only member that already carries
    a [project] name/version/requires-python of its own must not be overwritten."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "self-declared"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "self-declared"\nversion = "2.5"\n'
        'requires-python = ">=3.12"\n'
        "[tool.ruff]\n"
        'target-version = "py312"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["version"] == "2.5"
    assert document["project"]["requires-python"] == ">=3.12"
    # Per-identity renaming still applies to virtual members even when the plugin
    # declares its own name (uv identifies members by [project].name; two profiles
    # enabling the same plugin must not collide).
    assert document["project"]["name"].startswith("hermes-plugin-self-declared-")
