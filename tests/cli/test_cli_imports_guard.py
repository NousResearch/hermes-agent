"""Unit tests for defensive prompt_toolkit import guarding in cli.py (#96075)."""

import sys
from unittest.mock import patch


def test_cli_import_tolerates_missing_prompt_toolkit():
    with patch.dict(sys.modules, {"prompt_toolkit.patch_stdout": None}):
        import cli
        assert hasattr(cli, "patch_stdout")
        with cli.patch_stdout():
            pass
