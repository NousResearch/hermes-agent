"""Test that setup_wizard.py has shutil available for Matrix E2EE auto-install."""
import ast

import pytest


def _parse_setup_imports():
    """Parse setup_wizard.py and return top-level import names."""
    with open("hermes_cli/setup_wizard.py") as f:
        tree = ast.parse(f.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


class TestSetupShutilImport:
    def test_shutil_imported_at_module_level(self):
        """shutil must be imported at module level so setup_gateway can use it
        for the mautrix auto-install path."""
        names = _parse_setup_imports()
        assert "shutil" in names, (
            "shutil is not imported at the top of hermes_cli/setup_wizard.py. "
            "This causes a NameError when the Matrix E2EE auto-install "
            "tries to call shutil.which('uv')."
        )
