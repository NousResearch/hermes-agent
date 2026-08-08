"""Tests for the oneshot faulthandler/SIGUSR1 diagnostics hook."""

import faulthandler

import pytest

from hermes_cli import oneshot


@pytest.fixture(autouse=True)
def _reset_faulthandler():
    yield
    if faulthandler.is_enabled():
        faulthandler.disable()


def test_diagnostics_enabled_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_ONESHOT_FAULTHANDLER", raising=False)

    oneshot._install_oneshot_timeout_diagnostics()

    assert faulthandler.is_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_diagnostics_opt_out(monkeypatch, value):
    monkeypatch.setenv("HERMES_ONESHOT_FAULTHANDLER", value)
    if faulthandler.is_enabled():
        faulthandler.disable()

    oneshot._install_oneshot_timeout_diagnostics()

    assert not faulthandler.is_enabled()
