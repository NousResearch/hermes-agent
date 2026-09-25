"""Update provisions the Browser Use CLI when the default backend would silently downgrade."""

from unittest.mock import Mock

import pytest

import tools.browser_use_cli as browser_use_cli
from hermes_cli import tools_config_post_setup as post_setup
from hermes_cli import update_cmd_maint as update


@pytest.fixture
def quiet_update(monkeypatch):
    """Neutralize the PM-defaults loop so tests only exercise CLI provisioning."""
    import pm.defaults
    import pm.install
    import pm.lock
    import pm.paths

    monkeypatch.setattr(pm.install, "sealed", lambda: False)
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(pm.defaults, "default_packages", lambda names: [])
    monkeypatch.setattr(pm.paths, "lockfile_path", lambda: None)
    monkeypatch.setattr(
        pm.lock, "Lockfile", lambda path: Mock(names=Mock(return_value=[])))


@pytest.fixture
def cli_state(monkeypatch, quiet_update):
    """Default-backend user with no CLI; returns the _ensure_browser_use_cli stub."""
    monkeypatch.setattr(browser_use_cli, "get_browser_backend", lambda: "")
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: None)
    monkeypatch.setattr(browser_use_cli, "_camofox_active", lambda *a, **k: False)
    ensure = Mock()
    monkeypatch.setattr(post_setup, "_ensure_browser_use_cli", ensure)
    return ensure


def test_provisions_cli_when_backend_unset_and_missing(cli_state):
    update._install_default_tools_after_update()

    cli_state.assert_called_once()
    timeout_s = cli_state.call_args.kwargs.get("timeout_s")
    assert isinstance(timeout_s, int) and 0 < timeout_s <= 600


@pytest.mark.parametrize("backend", ["off", "browser-use", "agent-browser", "lightpanda"])
def test_skips_when_backend_explicit(quiet_update, monkeypatch, backend):
    monkeypatch.setattr(browser_use_cli, "get_browser_backend", lambda: backend)
    monkeypatch.setattr(
        browser_use_cli, "_find_cli", lambda: None,
    )
    monkeypatch.setattr(browser_use_cli, "_camofox_active", lambda *a, **k: False)
    ensure = Mock()
    monkeypatch.setattr(post_setup, "_ensure_browser_use_cli", ensure)

    update._install_default_tools_after_update()

    ensure.assert_not_called()


def test_skips_when_cli_present(quiet_update, monkeypatch):
    monkeypatch.setattr(browser_use_cli, "get_browser_backend", lambda: "")
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: ["/bin/browser-use"])
    ensure = Mock()
    monkeypatch.setattr(post_setup, "_ensure_browser_use_cli", ensure)

    update._install_default_tools_after_update()

    ensure.assert_not_called()


def test_skips_when_camofox(quiet_update, monkeypatch):
    monkeypatch.setattr(browser_use_cli, "get_browser_backend", lambda: "")
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: None)
    monkeypatch.setattr(browser_use_cli, "_camofox_active", lambda *a, **k: True)
    ensure = Mock()
    monkeypatch.setattr(post_setup, "_ensure_browser_use_cli", ensure)

    update._install_default_tools_after_update()

    ensure.assert_not_called()


def test_provision_failure_never_fails_update(cli_state):
    cli_state.side_effect = RuntimeError("boom")

    update._install_default_tools_after_update()  # must not raise


def test_config_read_failure_never_fails_update(quiet_update, monkeypatch):
    def _raise():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(browser_use_cli, "get_browser_backend", _raise)
    ensure = Mock()
    monkeypatch.setattr(post_setup, "_ensure_browser_use_cli", ensure)

    update._install_default_tools_after_update()  # must not raise
    ensure.assert_not_called()


@pytest.mark.parametrize("sealed,lazy", [(True, True), (False, False)])
def test_respects_sealed_and_lazy_opt_out(cli_state, monkeypatch, sealed, lazy):
    import pm.install

    monkeypatch.setattr(pm.install, "sealed", lambda: sealed)
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: lazy)

    update._install_default_tools_after_update()

    cli_state.assert_not_called()


def test_ensure_browser_use_cli_threads_bounded_timeout(monkeypatch):
    install = Mock(return_value=(True, "ok"))
    monkeypatch.setattr(browser_use_cli, "install_cli", install)

    post_setup._ensure_browser_use_cli(timeout_s=123)

    install.assert_called_once_with(timeout_s=123)
