"""Every entry on the published plugin API (``hermes_cli/data/plugin_api.json``) exists on this tree.

``hermes plugins validate`` and ``scripts/boundary_lint.py`` tell plugin authors these are the core
imports they may use, and ``public-api.md`` is generated from the same file. Both checks are static:
a listed module that was moved, or a ``module:Name`` whose name was renamed away, still passes them
while every plugin following the list fails to load. This imports each entry for real.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from hermes_agent.compat.paths import current_module, current_path

# The list keeps the names it was published with while packages move; each resolves through MOVES.
_PLUGIN_API = Path(__file__).resolve().parents[3] / current_path("hermes_cli/data/plugin_api.json")


def _entries() -> list[str]:
    data = json.loads(_PLUGIN_API.read_text(encoding="utf-8-sig"))
    return sorted({entry for group in data["groups"] for entry in group["entries"]})


@pytest.mark.parametrize("entry", _entries())
def test_published_entry_imports(entry):
    module_name, _, name = entry.partition(":")
    module = importlib.import_module(current_module(module_name))
    if name:
        assert hasattr(module, name), f"{entry}: {module_name} has no attribute {name!r}"
