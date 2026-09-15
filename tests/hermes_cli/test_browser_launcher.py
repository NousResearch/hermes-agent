"""Tests for hermes_cli/browser_launcher.py — browser launcher path resolution."""


def test_browser_path_resolution_exists():
    from hermes_cli.browser_launcher import _resolve_browser_executable
    path = _resolve_browser_executable()
    assert path is None or isinstance(path, str)
