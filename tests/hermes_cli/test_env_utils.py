"""Tests for hermes_cli/env_utils.py — environment utility helpers."""


def test_get_env_bool_true():
    from hermes_cli.env_utils import _env_bool
    assert _env_bool("HERMES_TEST_BOOL_DEFAULT") is False
