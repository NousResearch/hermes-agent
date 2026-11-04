"""Retain the seed's exact editable identity when equal buildable twins fold."""

import os
import shutil
import subprocess
import sys
import tomllib

import pytest

from pm.workspace import _generate_pyproject, _member_key, lock_and_sync


@pytest.fixture
def twins(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="core"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\n')
    inputs = {}
    for profile in ("first", "second", "third"):
        identity = tmp_path / profile / "plugins" / "example"
        source = tmp_path / "staging" / profile
        source.mkdir(parents=True)
        (source / "pyproject.toml").write_text(
            '[project]\nname="example-plugin"\ndynamic=["version", "dependencies"]\nrequires-python=">=3.11"\n'
            '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n')
        # Dependency-free, in-tree PEP 517 backend: metadata is actually built offline.
        (source / "backend.py").write_text(
            'import os\nfrom pathlib import Path\n'
            'def get_requires_for_build_wheel(config_settings=None): return []\n'
            'get_requires_for_build_editable = get_requires_for_build_wheel\n'
            'def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):\n'
            '    name = "example_plugin-1.0.dist-info"\n'
            '    target = Path(metadata_directory) / name\n'
            '    target.mkdir()\n'
            '    (target / "METADATA").write_text("Metadata-Version: 2.3\\nName: example-plugin\\nVersion: 1.0\\n")\n'
            '    if "MEMBER_METADATA_LOG" in os.environ:\n'
            '        with open(os.environ["MEMBER_METADATA_LOG"], "a") as log: log.write("metadata built\\n")\n'
            '    return name\n'
            'prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel\n')
        inputs[identity] = source
    return core, inputs


def _members(root):
    return tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]


def _seed(identity):
    return f'[[package]]\nname="example-plugin"\nversion="1.0"\nsource={{editable="plugin-sources/{_member_key(identity)}"}}\n'


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("reference", ["exact", "prefix", "comment", "metadata", "directory", "invalid", "none", "both"])
def test_only_exact_editable_package_sources_prefer_a_twin(tmp_path, twins, staged, reference):
    core, inputs = twins
    first, second, third = inputs
    if not staged:
        for identity, source in inputs.items():
            shutil.copytree(source, identity)
    text = _seed(second)
    if reference == "prefix":
        text = text.replace(_member_key(second), _member_key(second) + "-other")
    elif reference == "comment":
        text = "# " + text.replace("\n", "\n# ")
    elif reference == "metadata":
        text = f'[metadata]\npath="plugin-sources/{_member_key(second)}"\n'
    elif reference == "directory":
        text = text.replace("editable=", "directory=")
    elif reference == "invalid":
        text = '[not valid TOML ' + text
    elif reference == "none":
        text = ""
    elif reference == "both":
        text += _seed(first)
    root = tmp_path / "generated"
    _generate_pyproject(inputs if staged else list(inputs), root, source=core, seed_text=text)
    chosen = second if reference == "exact" else first
    assert _members(root) == [f"plugin-sources/{_member_key(chosen)}"]
    assert not (root / "plugin-sources" / _member_key(third)).exists()


@pytest.mark.parametrize("difference", ["bytes", "external"])
def test_seed_preference_never_shadows_different_inputs(tmp_path, twins, difference):
    core, inputs = twins
    first, second, _ = inputs
    inputs = {first: inputs[first], second: inputs[second]}
    if difference == "bytes":
        (inputs[second] / "plugin.py").write_text("DIFFERENT = True\n")
    else:
        for source in inputs.values():
            with (source / "pyproject.toml").open("a") as f:
                f.write('[tool.uv.sources]\nlocal-dep={path="../local-dep"}\n')
    root = tmp_path / "generated"
    _generate_pyproject(inputs, root, source=core, seed_text=_seed(second))
    assert set(_members(root)) == {f"plugin-sources/{_member_key(identity)}" for identity in inputs}


@pytest.mark.parametrize("staged", [False, True])
def test_real_offline_lock_keeps_seeded_editable_through_profile_changes(tmp_path, twins, staged):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    core, inputs = twins
    first, second, third = inputs
    if not staged:
        for identity, source in inputs.items():
            shutil.copytree(source, identity)
    env = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    env.update(UV_NO_CACHE="1", UV_PYTHON_DOWNLOADS="never", UV_OFFLINE="1",
               MEMBER_METADATA_LOG=str(tmp_path / "metadata.log"))

    class LockOnlyEnvironment:
        def sync(self, root, *, extras, frozen):
            # Exercise generation -> seed copying -> real uv metadata resolution,
            # without installing anything or creating a managed environment.
            result = subprocess.run(
                [uv, "lock", "--offline", "--no-config", "--python", sys.executable],
                cwd=root, env=env, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr

    seed = None
    for number, selected in enumerate(([second], [first, second], [first, second, third], [second, third])):
        root = tmp_path / f"generation-{number}"
        selection = {identity: inputs[identity] for identity in selected} if staged else selected
        lock_and_sync(selection, [], root=root, source=core, seed_lock=seed,
                      environment=LockOnlyEnvironment())
        expected = f"plugin-sources/{_member_key(second)}"
        assert _members(root) == [expected]
        packages = tomllib.loads((root / "uv.lock").read_text())["package"]
        plugin = next(package for package in packages if package["name"] == "example-plugin")
        assert plugin["source"] == {"editable": expected}
        assert not (root / ".venv").exists()
        seed = root / "uv.lock"
    assert "metadata built" in (tmp_path / "metadata.log").read_text()
