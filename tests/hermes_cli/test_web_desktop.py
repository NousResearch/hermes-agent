"""Tests for hermes_cli/web_desktop.py — web desktop helpers."""


def test_env_detector_defaults():
    from hermes_cli.web_desktop import _env_detector
    result = _env_detector()
    assert isinstance(result, dict)
