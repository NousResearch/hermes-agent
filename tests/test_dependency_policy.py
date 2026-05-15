"""Regression tests for dependency-level supply-chain guardrails.

These tests intentionally inspect project metadata instead of importing Hermes.
They protect optional extras from resolving a known-bad upstream package version
and keep the lockfile from reintroducing the same exact release later.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_BAD_MISTRALAI = Version("2.4.6")


def _mistralai_specs(dependencies: list[str]) -> list[str]:
    """Return Mistral SDK requirements using Python's case-insensitive names."""

    return [
        dep
        for dep in dependencies
        if canonicalize_name(Requirement(dep).name) == "mistralai"
    ]


def test_mistralai_dependency_detection_is_case_insensitive() -> None:
    """Package names are case-insensitive, so mixed-case restores must be caught."""

    assert _mistralai_specs(["MistralAI==2.4.6"]) == ["MistralAI==2.4.6"]


def test_mistral_extra_absent_or_excludes_known_bad_mistralai_version() -> None:
    """The Mistral extra may stay quarantined, but must not allow 2.4.6."""

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    optional_deps = pyproject["project"]["optional-dependencies"]
    mistral_deps = optional_deps.get("mistral", [])

    # Upstream may remove the extra entirely while the SDK is quarantined.
    # If it is restored later, this regression test forces the restored
    # specifier to keep excluding the compromised exact release.
    mistralai_specs = _mistralai_specs(mistral_deps)
    if not mistralai_specs:
        assert "mistral: extra REMOVED" in (PROJECT_ROOT / "pyproject.toml").read_text()
        return

    assert len(mistralai_specs) == 1
    spec = Requirement(mistralai_specs[0]).specifier
    assert isinstance(spec, SpecifierSet)
    assert KNOWN_BAD_MISTRALAI not in spec


def test_uv_lock_does_not_pin_known_bad_mistralai_version() -> None:
    """The committed lockfile must not contain the known-bad exact release."""

    lock_text = (PROJECT_ROOT / "uv.lock").read_text()

    assert not re.search(
        r'(?m)^name = "mistralai"\nversion = "2\.4\.6"$',
        lock_text,
    )
    assert "mistralai-2.4.6" not in lock_text


def test_supply_chain_audit_blocks_known_bad_mistralai_version() -> None:
    """CI should fail fast if a future PR reintroduces the bad Mistral SDK."""

    workflow = (PROJECT_ROOT / ".github/workflows/supply-chain-audit.yml").read_text()

    assert "pyproject.toml" in workflow
    assert "uv.lock" in workflow
    assert "mistralai" in workflow
    assert "2.4.6" in workflow
    assert ".lower()" in workflow
    assert 'name = "mistralai"' in workflow
    assert 'version = "2.4.6"' in workflow
