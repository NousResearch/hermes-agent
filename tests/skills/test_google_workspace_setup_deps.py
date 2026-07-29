"""Regression test: google-workspace setup.py REQUIRED_PACKAGES must pin httplib2.

GHSA-j5g9-f88f-gfj3 (HIGH) — Decompression Bomb DoS via unbounded gzip/deflate
response handling.  Fixed in httplib2 0.32.0.

There are three install paths for google-workspace dependencies:
  1. pyproject.toml [project.optional-dependencies].google
  2. tools/lazy_deps.py LAZY_DEPS['skill.google_workspace']
  3. skills/productivity/google-workspace/scripts/setup.py REQUIRED_PACKAGES

This test ensures path 3 stays pinned and consistent with the other two.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SETUP_PY = REPO_ROOT / "skills/productivity/google-workspace/scripts/setup.py"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
LAZY_DEPS_PY = REPO_ROOT / "tools/lazy_deps.py"

# ---------------------------------------------------------------------------
# Static parsers
# ---------------------------------------------------------------------------

_GOOGLE_EXTRA_KEY = "google"
_LAZY_DEPS_KEY = "skill.google_workspace"


def _parse_setup_py_required_packages() -> list[str]:
    """Parse setup.py and return the REQUIRED_PACKAGES list."""
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REQUIRED_PACKAGES":
                    if isinstance(node.value, ast.List):
                        return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
    raise AssertionError("REQUIRED_PACKAGES not found in setup.py")


def _parse_pyproject_google_extra() -> list[str]:
    """Parse pyproject.toml and return the google extra dependency list."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    data = tomllib.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))
    optional_deps = data["project"]["optional-dependencies"]
    return list(optional_deps[_GOOGLE_EXTRA_KEY])


def _parse_lazy_deps_google_workspace() -> list[str]:
    """Parse tools/lazy_deps.py and return the LAZY_DEPS for skill.google_workspace."""
    tree = ast.parse(LAZY_DEPS_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # LAZY_DEPS is declared as AnnAssign: `LAZY_DEPS: dict[str, tuple[str, ...]] = {...}`
        target_name = None
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target_name = node.target.id
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    target_name = t.id
                    break
        if target_name != "LAZY_DEPS":
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Dict):
            dict_val = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            dict_val = node.value
        else:
            continue
        for k, v in zip(dict_val.keys, dict_val.values):
            if isinstance(k, ast.Constant) and k.value == _LAZY_DEPS_KEY:
                if isinstance(v, (ast.Tuple, ast.List)):
                    return [elt.value for elt in v.elts if isinstance(elt, ast.Constant)]  # pyright: ignore[reportReturnType]
    raise AssertionError(f"LAZY_DEPS[{_LAZY_DEPS_KEY!r}] not found")


def _extract_pins(packages: list[str]) -> dict[str, str]:
    """Extract pinned versions: {package_name: version} for entries with == pin."""
    pins: dict[str, str] = {}
    for pkg in packages:
        if "==" in pkg:
            name, version = pkg.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGoogleWorkspaceSetupDepsPins:
    """Security pin consistency across all three google-workspace install paths."""

    def test_setup_py_pins_httplib2(self):
        """setup.py REQUIRED_PACKAGES must include httplib2==0.32.0."""
        packages = _parse_setup_py_required_packages()
        pins = _extract_pins(packages)
        assert "httplib2" in pins, (
            f"httplib2 not found in setup.py REQUIRED_PACKAGES.\n"
            f"  Current entries: {packages}"
        )
        assert pins["httplib2"] == "0.32.0", (
            f"httplib2 pin mismatch in setup.py: expected 0.32.0, got {pins['httplib2']}.\n"
            f"  Full REQUIRED_PACKAGES: {packages}"
        )

    def test_setup_py_pins_match_pyproject_toml(self):
        """httplib2 pin in setup.py must match pyproject.toml google extra."""
        required_packages = _parse_setup_py_required_packages()
        pyproject_packages = _parse_pyproject_google_extra()

        required_pins = _extract_pins(required_packages)
        pyproject_pins = _extract_pins(pyproject_packages)

        for pkg in ("httplib2", "google-api-python-client", "google-auth-oauthlib", "google-auth-httplib2"):
            setup_ver = required_pins.get(pkg)
            toml_ver = pyproject_pins.get(pkg)
            if setup_ver is None and toml_ver is None:
                continue  # neither path pins it, skip
            assert toml_ver is not None, (
                f"{pkg} is pinned in setup.py ({setup_ver}) but NOT in pyproject.toml google extra.\n"
                f"  setup.py: {required_pins}\n"
                f"  pyproject.toml google: {pyproject_pins}"
            )
            assert setup_ver is not None, (
                f"{pkg} is pinned in pyproject.toml ({toml_ver}) but NOT in setup.py.\n"
                f"  pyproject.toml google: {pyproject_pins}\n"
                f"  setup.py: {required_pins}"
            )
            assert setup_ver == toml_ver, (
                f"{pkg} pin mismatch: setup.py has {setup_ver}, pyproject.toml has {toml_ver}.\n"
                f"  setup.py: {required_pins}\n"
                f"  pyproject.toml google: {pyproject_pins}"
            )

    def test_setup_py_pins_match_lazy_deps(self):
        """httplib2 pin in setup.py must match tools/lazy_deps.py skill.google_workspace."""
        required_packages = _parse_setup_py_required_packages()
        lazy_packages = _parse_lazy_deps_google_workspace()

        required_pins = _extract_pins(required_packages)
        lazy_pins = _extract_pins(lazy_packages)

        for pkg in ("httplib2", "google-api-python-client", "google-auth-oauthlib", "google-auth-httplib2"):
            setup_ver = required_pins.get(pkg)
            lazy_ver = lazy_pins.get(pkg)
            if setup_ver is None and lazy_ver is None:
                continue
            assert lazy_ver is not None, (
                f"{pkg} is pinned in setup.py ({setup_ver}) but NOT in lazy_deps.py.\n"
                f"  setup.py: {required_pins}\n"
                f"  lazy_deps.py: {lazy_pins}"
            )
            assert setup_ver is not None, (
                f"{pkg} is pinned in lazy_deps.py ({lazy_ver}) but NOT in setup.py.\n"
                f"  lazy_deps.py: {lazy_pins}\n"
                f"  setup.py: {required_pins}"
            )
            assert setup_ver == lazy_ver, (
                f"{pkg} pin mismatch: setup.py has {setup_ver}, lazy_deps.py has {lazy_ver}.\n"
                f"  setup.py: {required_pins}\n"
                f"  lazy_deps.py: {lazy_pins}"
            )

    def test_all_google_packages_are_pinned_in_all_paths(self):
        """Every google workspace package that is version-pinned in any path must appear in all three."""
        pyproject_packages = _parse_pyproject_google_extra()
        lazy_packages = _parse_lazy_deps_google_workspace()
        setup_packages = _parse_setup_py_required_packages()

        all_pins: dict[str, set[str]] = {}
        for label, pkgs in [
            ("pyproject.toml", pyproject_packages),
            ("lazy_deps.py", lazy_packages),
            ("setup.py", setup_packages),
        ]:
            for pkg in pkgs:
                if "==" in pkg:
                    name, ver = pkg.split("==", 1)
                    all_pins.setdefault(name.strip(), set()).add(f"{label}={ver.strip()}")

        for pkg, entries in sorted(all_pins.items()):
            versions = {e.split("=", 1)[1] for e in entries}
            assert len(versions) == 1, (
                f"{pkg} has inconsistent pins across install paths:\n"
                + "\n".join(f"  {e}" for e in sorted(entries))
            )
            assert len(entries) == 3, (
                f"{pkg} is not pinned in all three install paths.  Found {len(entries)}/3:\n"
                + "\n".join(f"  {e}" for e in sorted(entries))
            )
