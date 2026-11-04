"""Install-local provenance and tool caches do not distinguish buildable twins."""

import tomllib

import pytest

from pm.workspace import _generate_pyproject, _member_key, members_stamp


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("noise", [
    ".hermes-catalog.json", ".hermes-install.json",
    ".pytest_cache/state", ".mypy_cache/state", ".ruff_cache/state",
])
@pytest.mark.parametrize("both", [False, True])
def test_install_metadata_only_affects_snapshot_not_folding(tmp_path, staged, noise, both):
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text('[project]\nname="core"\nversion="1"\n')
    inputs = {}
    for profile in ("first", "second"):
        identity = tmp_path / profile / "plugins" / "example"
        source = tmp_path / "staging" / profile if staged else identity
        source.mkdir(parents=True)
        (source / "pyproject.toml").write_text(
            '[project]\nname="example-plugin"\nversion="1"\n'
            '[build-system]\nrequires=[]\nbuild-backend="backend"\n')
        if both or profile == "first":
            local = source / noise
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text('{"installed_at": "' + profile + '"}\n')
        inputs[identity] = source

    first, second = inputs
    assert members_stamp({first: inputs[first]}) != members_stamp({first: inputs[second]})
    root = tmp_path / "workspace"
    _generate_pyproject(inputs if staged else list(inputs), root, source=core)
    members = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]
    assert members == [f"plugin-sources/{_member_key(first)}"]
    # Equality ignores install-local data, but copying and generation stamps do not.
    assert (root / members[0] / noise).read_bytes() == (inputs[first] / noise).read_bytes()

    (inputs[second] / "plugin.py").write_text("CHANGED = True\n")
    diverged = tmp_path / "diverged"
    _generate_pyproject(inputs if staged else list(inputs), diverged, source=core)
    assert len(tomllib.loads((diverged / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]) == 2
