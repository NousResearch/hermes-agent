"""Workspace folding must preserve distinct resolved dependency inputs (PR #125272)."""

import tomllib

from pm.workspace import _generate_pyproject


def test_identical_plugin_text_with_distinct_relative_dependencies_is_not_folded(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="hermes-agent"\nversion="1"\n', encoding="utf-8")
    plugins = []
    dependency_paths = set()
    for profile, version in (("first", "1.0"), ("second", "2.0")):
        parent = tmp_path / profile / "plugins"
        plugin = parent / "example"
        dependency = parent / "local-dep"
        plugin.mkdir(parents=True)
        dependency.mkdir()
        (dependency / "pyproject.toml").write_text(
            f'[project]\nname="local-dep"\nversion="{version}"\n', encoding="utf-8")
        (plugin / "pyproject.toml").write_text(
            '[project]\nname="example-plugin"\nversion="1"\n'
            'dependencies=["local-dep>=1,<3"]\n'
            '[build-system]\nrequires=["hatchling>=1.26,<2"]\n'
            'build-backend="hatchling.build"\n'
            '[tool.uv.sources]\nlocal-dep={path="../local-dep"}\n', encoding="utf-8")
        plugins.append(plugin)
        dependency_paths.add(dependency.resolve().as_posix())

    root = tmp_path / "generated"
    _generate_pyproject(plugins, root, source=core)
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    retained_paths = set()
    for member in document["tool"]["uv"]["workspace"]["members"]:
        metadata = tomllib.loads((root / member / "pyproject.toml").read_text(encoding="utf-8"))
        retained_paths.add(metadata["tool"]["uv"]["sources"]["local-dep"]["path"])
    assert retained_paths == dependency_paths
