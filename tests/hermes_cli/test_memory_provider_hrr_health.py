"""Health surfaces for the holographic provider's numpy/HRR degradation (#136707).

The plugin's ``is_available()`` returns True even without numpy (SQLite alone
works), so the generic installed/available rows cannot reveal that semantic
HRR recall silently fell back to FTS-only. ``hermes doctor`` and
``hermes memory status`` must report the HRR kernel's own state instead.
"""

from __future__ import annotations

import sys
import types

import pytest

from hermes_cli import memory_provider_health as health


class _Kernel:
    """Stand-in for the plugin's ``holographic`` HRR kernel module."""


class _Provider:
    """Stand-in whose class module points at a fake provider package."""

    def is_available(self) -> bool:
        return True


_PROVIDER_PKG = "fake_hermes_holographic_pkg"


@pytest.fixture
def fake_provider_package(monkeypatch):
    """Register a fake provider package + kernel in sys.modules, return the kernel."""
    pkg = types.ModuleType(_PROVIDER_PKG)
    _Provider.__module__ = _PROVIDER_PKG
    monkeypatch.setitem(sys.modules, _PROVIDER_PKG, pkg)
    kernel = _Kernel()
    monkeypatch.setitem(sys.modules, f"{_PROVIDER_PKG}.holographic", kernel)
    yield kernel
    monkeypatch.delitem(sys.modules, _PROVIDER_PKG, raising=False)
    monkeypatch.delitem(sys.modules, f"{_PROVIDER_PKG}.holographic", raising=False)


def test_state_reports_numpy_absent(fake_provider_package):
    fake_provider_package._HAS_NUMPY = False
    assert health.holographic_hrr_state(_Provider()) == {"numpy": False}


def test_state_reports_numpy_present_with_version(fake_provider_package, monkeypatch):
    fake_provider_package._HAS_NUMPY = True
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.__version__ = "2.4.1"
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    assert health.holographic_hrr_state(_Provider()) == {"numpy": True, "numpy_version": "2.4.1"}


def test_state_none_when_kernel_flag_missing(fake_provider_package):
    # An older/different plugin build without the kernel flag: caller stays silent.
    assert health.holographic_hrr_state(_Provider()) is None


def test_state_none_when_package_absent(monkeypatch):
    _Provider.__module__ = "fake_hermes_missing_pkg"
    assert health.holographic_hrr_state(_Provider()) is None


# --- doctor integration -------------------------------------------------------

def _run_holographic_check(monkeypatch, provider, capsys):
    from hermes_cli import doctor_state
    from hermes_cli.doctor_report import Finding

    monkeypatch.setattr(
        "plugins.memory.load_memory_provider", lambda name, **kw: provider, raising=False)
    f = Finding()
    doctor_state._check_holographic_provider(f)
    return capsys.readouterr().out, f


def test_doctor_warns_and_issues_when_numpy_missing(monkeypatch, fake_provider_package, capsys):
    from hermes_cli import doctor_state  # ensure registration exists

    fake_provider_package._HAS_NUMPY = False
    assert "holographic" in doctor_state._MEMORY_PROVIDER_CHECKS

    out, f = _run_holographic_check(monkeypatch, _Provider(), capsys)
    assert "holographic provider active" in out
    assert "numpy missing" in out
    assert "FTS-only" in out
    assert f.manual_issues  # surfaced in the doctor summary as a manual fix


def test_doctor_ok_when_numpy_present(monkeypatch, fake_provider_package, capsys):
    fake_provider_package._HAS_NUMPY = True
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.__version__ = "2.4.1"
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    out, f = _run_holographic_check(monkeypatch, _Provider(), capsys)
    assert "HRR semantic recall active" in out
    assert "numpy 2.4.1" in out
    assert not f.manual_issues


def test_doctor_falls_back_to_generic_when_plugin_missing(monkeypatch, capsys):
    from hermes_cli import doctor_state
    from hermes_cli.doctor_report import Finding

    called = {"generic": False}

    def _generic(name):
        called["generic"] = True

    monkeypatch.setattr(
        "plugins.memory.load_memory_provider", lambda name, **kw: None, raising=False)
    monkeypatch.setattr(doctor_state, "_memory_provider_generic", _generic)
    doctor_state._check_holographic_provider(Finding())
    assert called["generic"] is True


# --- memory status integration ------------------------------------------------

def test_memory_status_shows_degraded_hrr_line(monkeypatch, fake_provider_package, capsys):
    from hermes_cli import memory_setup

    fake_provider_package._HAS_NUMPY = False

    class _ProviderFull(_Provider):
        def is_available(self):
            return True

        def get_config_schema(self):
            return []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "holographic", "holographic": {}}})
    monkeypatch.setattr(memory_setup, "_get_available_providers",
                        lambda: [("holographic", "local", _ProviderFull())])
    monkeypatch.setattr(memory_setup, "_find_provider",
                        lambda providers, name: ("holographic", "local", _ProviderFull()))

    memory_setup.cmd_status(object())
    out = capsys.readouterr().out
    assert "HRR:" in out
    assert "degraded" in out
    assert "FTS-only" in out


def test_memory_status_shows_active_hrr_line(monkeypatch, fake_provider_package, capsys):
    from hermes_cli import memory_setup

    fake_provider_package._HAS_NUMPY = True
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.__version__ = "2.4.1"
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    class _ProviderFull(_Provider):
        def is_available(self):
            return True

        def get_config_schema(self):
            return []

    provider = _ProviderFull()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "holographic", "holographic": {}}})
    monkeypatch.setattr(memory_setup, "_get_available_providers",
                        lambda: [("holographic", "local", provider)])
    monkeypatch.setattr(memory_setup, "_find_provider",
                        lambda providers, name: ("holographic", "local", provider))

    memory_setup.cmd_status(object())
    out = capsys.readouterr().out
    assert "semantic recall active" in out
    assert "numpy 2.4.1" in out
