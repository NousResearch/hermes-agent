"""Tests for hermes_cli/project_utils.py — project utility helpers."""


def test_project_root_detection():
    from hermes_cli.project_utils import _find_project_root
    result = _find_project_root()
    assert result is None or isinstance(result, str)
