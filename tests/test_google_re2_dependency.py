"""The regex engine hermes_re runs on is the one pyproject.toml declares.

``google-re2`` is the first core dependency some targets build from source (musl) or in the
project's own wheelhouse (Termux). An environment that kept an older build, or whose extension
module does not load, would otherwise first show up as a failing regex somewhere else.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]


def _declared() -> Requirement:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))["project"]
    declared = [requirement for spec in project["dependencies"]
                if canonicalize_name((requirement := Requirement(spec)).name) == "google-re2"]
    assert len(declared) == 1, "google-re2 must be one core dependency"
    return declared[0]


def test_core_dependency_is_one_exact_pin_for_this_interpreter():
    requirement = _declared()
    assert [spec.operator for spec in requirement.specifier] == ["=="], str(requirement)
    assert requirement.marker is None or requirement.marker.evaluate()


def test_installed_engine_is_the_declared_version_and_its_extension_loads():
    (pin,) = _declared().specifier
    assert Version(importlib.metadata.version("google-re2")) == Version(pin.version)
    # hermes_re loads this module and nothing else of the distribution.
    assert importlib.import_module("re2._re2").RE2 is not None
