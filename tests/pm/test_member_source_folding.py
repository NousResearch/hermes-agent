"""Fold only members with the same effective installed dependency inputs."""

import tomllib

import pytest

from pm.workspace import _generate_pyproject


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("kind", ["distinct", "shared", "internal", "absolute"])
def test_folding_respects_installed_path_sources(tmp_path, staged, kind):
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="hermes-agent"\nversion="1"\n', encoding="utf-8")
    shared = tmp_path / "shared"
    shared.mkdir()
    inputs = {}
    expected_paths = set()
    for profile in ("first", "second"):
        identity = tmp_path / profile / "plugins" / "example"
        identity.mkdir(parents=True)
        source = tmp_path / "staging" / profile / "example" if staged else identity
        source.mkdir(parents=True, exist_ok=True)
        if kind == "distinct":
            relative = "../local-dep"
        elif kind == "shared":
            relative = "../../../shared"
        elif kind == "internal":
            relative = "local-dep"
            (source / relative).mkdir()
            (source / relative / "data.txt").write_text("same", encoding="utf-8")
        else:
            relative = shared.as_posix()
        # Array-shaped sources must obey the same boundary as single tables.
        (source / "pyproject.toml").write_text(
            '[project]\nname="example-plugin"\nversion="1"\n'
            'dependencies=["local-dep"]\n'
            '[build-system]\nrequires=["hatchling"]\n'
            'build-backend="hatchling.build"\n'
            f'[tool.uv.sources]\nlocal-dep=[{{path="{relative}"}}]\n',
            encoding="utf-8")
        inputs[identity] = source
        expected_paths.add(relative if kind == "internal" else (identity / relative).resolve().as_posix())

    root = tmp_path / "workspace"
    _generate_pyproject(inputs if staged else list(inputs), root, source=core)
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    members = document["tool"]["uv"]["workspace"]["members"]
    assert len(members) == (2 if kind == "distinct" else 1)
    actual_paths = set()
    for member in members:
        metadata = tomllib.loads((root / member / "pyproject.toml").read_text(encoding="utf-8"))
        actual_paths.add(metadata["tool"]["uv"]["sources"]["local-dep"][0]["path"])
        if kind == "internal":
            assert (root / member / "local-dep" / "data.txt").read_text(encoding="utf-8") == "same"
    assert actual_paths == expected_paths
