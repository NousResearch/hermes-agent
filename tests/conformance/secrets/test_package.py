"""The secret conformance suite ships in the installed package, importable outside a checkout."""

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _packages():
    from setuptools import find_packages
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return find_packages(str(ROOT), include=config["tool"]["setuptools"]["packages"]["find"]["include"])


def test_package_discovery_ships_the_suite_and_no_tests():
    packages = _packages()
    assert {"hermes_conformance", "hermes_conformance.secrets"} <= set(packages)
    assert not [p for p in packages if p == "tests" or p.startswith("tests.")]


def test_a_plugin_imports_the_suite_from_outside_the_repo(tmp_path):
    """Stage what a wheel installs (the discovered packages and the root modules, never ``tests``)
    and import the suite from a directory that is not the checkout, as a plugin's tests would."""
    site = tmp_path / "site-packages"
    site.mkdir()
    for package in _packages():
        top = package.split(".")[0]
        if not (site / top).exists():
            shutil.copytree(ROOT / top, site / top, ignore=shutil.ignore_patterns("__pycache__", "tests"))
    for module in ROOT.glob("*.py"):
        if module.name != "setup.py":
            shutil.copyfile(module, site / module.name)
    elsewhere = tmp_path / "plugin"
    elsewhere.mkdir()
    (elsewhere / "test_backend.py").write_text(
        "from hermes_conformance.secrets import Backend, SecretsConformance\n"
        "def test_imported():\n"
        "    assert SecretsConformance.test_an_owner_wide_secret_resolves_on_every_agent_the_owner_owns\n"
        "    assert 'bind_member' in Backend.__dataclass_fields__\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(site)}
    env.pop("PYTHONSAFEPATH", None)
    code = (f"import sys, importlib.util, hermes_conformance.secrets as s; "
            f"assert not any(p and p.rstrip('/') == {str(ROOT)!r} for p in sys.path); "
            f"assert s.__file__.startswith({str(site)!r}), s.__file__; "
            f"spec = importlib.util.find_spec('tests'); "
            f"assert spec is None or not str(spec.origin).startswith({str(ROOT)!r})")
    result = subprocess.run([sys.executable, "-c", code], cwd=elsewhere, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir",
                          str(elsewhere), "-c", os.devnull, "test_backend.py"],
                         cwd=elsewhere, env=env, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
