"""Kanban/Telegram global env membership is exact-name, never prefix."""
import pytest

from agent import secret_scope


@pytest.fixture
def multiplexed(monkeypatch):
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    token = secret_scope.set_secret_scope({})
    yield
    secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("name", ["HERMES_KANBAN_CUSTOM_TOKEN", "HERMES_TELEGRAM_CUSTOM_KEY"])
def test_unlisted_prefix_sibling_never_reads_foreign_environ(multiplexed, monkeypatch, name):
    monkeypatch.setenv(name, "foreign")
    assert secret_scope.get_secret(name) is None
    token = secret_scope.set_secret_scope({name: "owned"})
    try:
        assert secret_scope.get_secret(name) == "owned"
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("prefix", ["HERMES_KANBAN_", "HERMES_TELEGRAM_"])
def test_listed_deployment_global_still_reads_environ(multiplexed, monkeypatch, prefix):
    # Relationship, not snapshot: every listed name in the family keeps reading os.environ.
    listed = sorted(n for n in secret_scope._GLOBAL_ENV_EXACT if n.startswith(prefix))
    assert listed
    for name in listed:
        monkeypatch.setenv(name, "deployment-value")
        assert secret_scope.get_secret(name) == "deployment-value", name
