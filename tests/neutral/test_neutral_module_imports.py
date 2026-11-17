"""The neutral helper modules import no application package at load.

Each module may import the standard library, ``psutil``, the persistence contracts, ``session_rules``, the other neutral modules and the few non-application modules in ``ALLOWED_AT_LOAD`` (directly or through another neutral module); a ``session_rules`` module may not import a database driver (``sqlite3``, ``_sqlite3``) or the SQLite backend (``persistence.sqlite``), which the top-level SQLite helpers here do by design. Each is imported in a fresh interpreter whose import system refuses every other top-level name and records the attempt, so a module that reaches an application package at load fails, and a refused import its own ``try`` swallows still fails the case. The services a module reads at run time (configuration, the plugin hook dispatcher, the gateway's media release) are imported inside the function that uses them.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import session_rules

ROOT = Path(__file__).resolve().parents[2]

TOP_LEVEL_NEUTRAL = (
    "lineage_adoption",
    "process_identity",
    "session_keys",
    "session_rescue",
    "session_retention",
    "sqlite_support",
    "storage_errors",
    "storage_limits",
    "storage_user_copy",
)

NEUTRAL_MODULES = (
    *TOP_LEVEL_NEUTRAL,
    "session_rules",
    *sorted(f"session_rules.{info.name}" for info in pkgutil.iter_modules(session_rules.__path__)),
)

# Non-application modules the neutral modules import at load (``session_rescue``, ``session_retention``,
# ``sqlite_support`` and every module importing one of them).
ALLOWED_AT_LOAD = ("hermes_constants", "hermes_dir_acl", "hermes_startup_watchdog")

# Runs in the fresh interpreter: refuse and record every import outside the allowed roots, and a database driver
# when argv[1] is a session_rules module, then import argv[1].
_IMPORT_BLOCKED = """
import importlib, os, sys, sysconfig

ALLOWED = {"persistence", "psutil", "session_rules", *sys.stdlib_module_names, *sys.argv[2].split(",")}
# An import made by standard-library code is the interpreter's own: Python 3.11's ``copy`` probes Jython's
# ``org.python.core`` under ``except ImportError``. Only imports a neutral module makes itself are judged.
STDLIB_DIRS = [os.path.realpath(sysconfig.get_paths()[key]) + os.sep for key in ("stdlib", "platstdlib")]
refused = []


def _importer_is_stdlib():
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.startswith("<frozen"):
        frame = frame.f_back
    return frame is not None and os.path.realpath(frame.f_code.co_filename).startswith(tuple(STDLIB_DIRS))


class RefuseApplicationPackages:
    drivers = ("sqlite3", "_sqlite3", "persistence.sqlite") if sys.argv[1].partition(".")[0] == "session_rules" else ()

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] not in ALLOWED or any(name == d or name.startswith(d + ".") for d in self.drivers):
            if name.partition(".")[0] not in ALLOWED and _importer_is_stdlib():
                return None
            refused.append(name)
            raise ImportError(f"{name} is not a neutral import")
        return None


sys.meta_path.insert(0, RefuseApplicationPackages())
importlib.import_module(sys.argv[1])
if refused:
    sys.exit(f"refused imports: {sorted(set(refused))}")
"""


def _import_in_fresh_process(module: str) -> subprocess.CompletedProcess[str]:
    allowed = ",".join((*TOP_LEVEL_NEUTRAL, *ALLOWED_AT_LOAD))
    return subprocess.run([sys.executable, "-c", _IMPORT_BLOCKED, module, allowed], cwd=ROOT, capture_output=True,
                          text=True, timeout=60, check=False)


@pytest.mark.parametrize("name", NEUTRAL_MODULES)
def test_a_neutral_module_imports_only_what_it_is_allowed(name):
    result = _import_in_fresh_process(name)
    assert result.returncode == 0, result.stderr


def test_a_standard_library_probe_for_an_optional_module_is_not_refused(tmp_path):
    # Python 3.11's ``copy`` runs ``from org.python.core import PyStringMap`` under ``except ImportError`` (a Jython probe);
    # an import made from inside the standard library is the interpreter's own, not the neutral module's.
    probe = tmp_path / "stdlib_probe.py"
    probe.write_text("try:\n    from org.python.core import PyStringMap\nexcept ImportError:\n    PyStringMap = None\n")
    script = _IMPORT_BLOCKED.replace("importlib.import_module(sys.argv[1])", f"STDLIB_DIRS.append({str(tmp_path)!r}); sys.path.insert(0, {str(tmp_path)!r}); importlib.import_module('stdlib_probe'); importlib.import_module(sys.argv[1])")
    result = subprocess.run([sys.executable, "-c", script, "session_rules", ""], cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr


def test_a_neutral_module_reaching_an_application_package_is_still_refused(tmp_path):
    # Control: an import made by a neutral module itself is judged, even when its own ``try`` swallows the refusal.
    package = tmp_path / "session_rules"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "probe_swallows.py").write_text("try:\n    import gateway\nexcept ImportError:\n    pass\n")
    script = _IMPORT_BLOCKED.replace("importlib.import_module(sys.argv[1])", f"import session_rules; session_rules.__path__.insert(0, {str(package)!r}); importlib.import_module(sys.argv[1])")
    result = subprocess.run([sys.executable, "-c", script, "session_rules.probe_swallows", ""], cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode != 0
    assert "refused imports: ['gateway']" in result.stderr
