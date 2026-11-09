"""The ``session_rules`` modules import no application package.

Each module may import the standard library, ``psutil``, the persistence contracts and the other ``session_rules`` modules, and nothing else; a database driver (``sqlite3``, ``_sqlite3``) and the SQLite backend (``persistence.sqlite``) are refused although they sit under those roots. Each is imported in a fresh interpreter whose import system refuses every other top-level name and records the attempt, so a module that reaches any other package fails to load, and a refused import its own ``try`` swallows still fails the case.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import session_rules

ROOT = Path(__file__).resolve().parents[2]

NEUTRAL_MODULES = ("session_rules", *sorted(f"session_rules.{info.name}" for info in pkgutil.iter_modules(session_rules.__path__)))

# Runs in the fresh interpreter: refuse and record every import outside the neutral roots, and a database driver
# when argv[1] is a session_rules module, then import argv[1].
_IMPORT_BLOCKED = """
import importlib, sys

ALLOWED = {"persistence", "psutil", "session_rules", *sys.stdlib_module_names}
refused = []


class RefuseApplicationPackages:
    drivers = ("sqlite3", "_sqlite3", "persistence.sqlite") if sys.argv[1].partition(".")[0] == "session_rules" else ()

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] not in ALLOWED or any(name == d or name.startswith(d + ".") for d in self.drivers):
            refused.append(name)
            raise ImportError(f"{name} is not a neutral import")
        return None


sys.meta_path.insert(0, RefuseApplicationPackages())
importlib.import_module(sys.argv[1])
if refused:
    sys.exit(f"refused imports: {sorted(set(refused))}")
"""


def _import_in_fresh_process(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", _IMPORT_BLOCKED, module], cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)


@pytest.mark.parametrize("name", NEUTRAL_MODULES)
def test_a_neutral_module_imports_only_what_it_is_allowed(name):
    result = _import_in_fresh_process(name)
    assert result.returncode == 0, result.stderr
