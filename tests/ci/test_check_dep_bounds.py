"""Tests for scripts/ci/check_dep_bounds.py (supply-chain-audit ``dep-bounds`` job).

The job used to grep for ``"name>=X.Y"`` with the closing quote right after the
version, so every core dependency shape in pyproject.toml (environment marker,
spaces around the operator, dotted name) slipped through unbounded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_dep_bounds.py"
_spec = importlib.util.spec_from_file_location("check_dep_bounds", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load check_dep_bounds.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_unbounded_specs_are_reported_in_every_pyproject_shape():
    added = [
        '+  "plainpkg>=1.2.3",',
        "+  \"markerpkg>=1.2.3; python_version >= '3.14'\",",
        '+  "spacepkg >= 1.2.3",',
        '+  "ruamel.yaml>=0.18",',
        '+  "extras[socks]>0.28",',
    ]
    assert _mod.unbounded_requirements(added) == [
        '"plainpkg>=1.2.3"',
        "\"markerpkg>=1.2.3; python_version >= '3.14'\"",
        '"spacepkg >= 1.2.3"',
        '"ruamel.yaml>=0.18"',
        '"extras[socks]>0.28"',
    ]


def test_bounded_pinned_and_non_requirement_strings_pass():
    added = [
        "+  \"boundedpkg>=1.2,<2; python_version >= '3.14'\",",
        '+  "reordered<2,>=1",',
        '+  "exact==2.24.0",',
        '+  "compatible~=1.4",',
        '+  "pinned @ git+https://github.com/o/r@0123456789abcdef0123456789abcdef01234567",',
        '+requires-python = ">=3.11,<3.15"',
        '+readme = "README.md"',
    ]
    assert _mod.unbounded_requirements(added) == []
