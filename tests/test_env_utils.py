"""Tests for shared env-var parsing helpers in utils.py."""

from utils import env_var_is_not_false, env_var_is_truthy


def test_env_var_is_truthy(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_FLAG", "yes")
    assert env_var_is_truthy("HERMES_TEST_FLAG") is True


def test_env_var_is_truthy_uses_default(monkeypatch):
    monkeypatch.delenv("HERMES_TEST_FLAG", raising=False)
    assert env_var_is_truthy("HERMES_TEST_FLAG", "true") is True


def test_env_var_is_not_false(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_FLAG", "maybe")
    assert env_var_is_not_false("HERMES_TEST_FLAG") is True


def test_env_var_is_not_false_respects_false_values(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_FLAG", "off")
    assert env_var_is_not_false("HERMES_TEST_FLAG", "true") is False
