"""The neutral helper modules import no application package.

Each module may import the standard library, ``psutil``, the persistence contracts, ``session_rules`` and the other neutral modules. ``ALLOWED`` lists, per module, every other import it makes, each the service the code reads at run time (configuration, the plugin hook dispatcher, the gateway's media release); an application package is imported only inside the function that uses it, never at module load.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

NEUTRAL_MODULES = (
    "lineage_adoption.py",
    "process_identity.py",
    "session_keys.py",
    "session_rescue.py",
    "session_retention.py",
    "sqlite_support.py",
    "storage_errors.py",
    "storage_limits.py",
    "storage_user_copy.py",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "session_rules").glob("*.py")),
)

# The root modules any neutral module may import at load: the contracts and the neutral modules themselves.
NEUTRAL_ROOTS = {"persistence", "session_rules", *(Path(name).stem for name in NEUTRAL_MODULES if "/" not in name)}

# Imports outside the standard library and the neutral roots, per module: (module imported, names, at module level).
ALLOWED = {
    "session_rescue.py": {("hermes_dir_acl", ("make_user_dir",), True)},
    "session_retention.py": {
        ("hermes_startup_watchdog", ("report_startup_progress",), True),
        ("hermes_cli.lifecycle", ("invoke_hook",), False),
        ("hermes_constants", ("get_hermes_home", "profile_name_for_home", "reset_hermes_home_override", "set_hermes_home_override"), False),
        ("tools.tool_result_storage", ("referenced_spillover_names",), False),
        ("gateway.session_ingress_media", ("release_unheld_media",), False),
    },
    "sqlite_support.py": {
        ("hermes_constants", ("get_hermes_home",), True),
        ("hermes_constants", ("get_hermes_home_override",), False),
        ("hermes_cli.config", ("load_config_readonly",), False),
        ("hermes_cli.config", ("detect_install_method", "get_project_root", "recommended_update_command_for_method"), False),
        ("hermes_cli.config", ("cfg_get", "load_config_readonly"), False),
    },
    "storage_limits.py": {("hermes_cli.config", ("load_config_readonly",), False)},
    "storage_user_copy.py": {("hermes_constants", ("profile_cli_selector",), False)},
}


def _imports(tree: ast.Module):
    """``(module, names, at_module_level)`` for every import statement, ``import x`` giving ``(x, ())``."""
    top = {id(node) for node in tree.body}
    top |= {id(n) for node in tree.body if isinstance(node, ast.Try) for n in node.body + [h for handler in node.handlers for h in handler.body]}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, (), id(node) in top
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module, tuple(alias.name for alias in node.names), id(node) in top


def _neutral(module: str) -> bool:
    root = module.partition(".")[0]
    return root in sys.stdlib_module_names or root in NEUTRAL_ROOTS or root == "psutil"


@pytest.mark.parametrize("name", NEUTRAL_MODULES)
def test_a_neutral_module_imports_only_what_it_is_allowed(name):
    tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
    found = {entry for entry in _imports(tree) if not _neutral(entry[0])}
    assert found == ALLOWED.get(name, set())


def test_an_allowed_application_import_runs_inside_its_function():
    application = {"agent", "cron", "gateway", "hermes_cli", "plugins", "tools", "tui_gateway"}
    loaded = [(name, module) for name, entries in ALLOWED.items() for module, _names, top in entries
              if top and (module.partition(".")[0] in application or module.startswith("hermes_state"))]
    assert loaded == []
