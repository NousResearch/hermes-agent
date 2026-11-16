"""The sealed package carries the remote store's wire conformance resources."""

from pathlib import Path
import subprocess
import sys


def test_remote_vendor_fixtures_ship_with_the_plugin(tmp_path):
    from setuptools import Distribution
    from setuptools.command.build_py import build_py
    from setuptools.config.pyprojecttoml import read_configuration

    root = Path(__file__).resolve().parents[3]
    config = read_configuration(str(root / "pyproject.toml"), expand=False)
    plugin_data = config["tool"]["setuptools"]["package-data"]["plugins"]
    dist = Distribution({"packages": ["plugins"], "package_dir": {"": str(root)},
                         "package_data": {"plugins": plugin_data}})
    dist.script_name = str(root / "setup.py")
    command = build_py(dist)
    command.build_lib = str(tmp_path / "lib")
    command.ensure_finalized()
    outputs = {Path(path) for path in command.get_outputs()}
    resources = list((root / "plugins/secret_store/remote/vendor").rglob("*.json"))
    assert resources
    for resource in resources:
        assert tmp_path / "lib" / resource.relative_to(root) in outputs
