"""Install and update provision the Browser Use CLI, the default browser driver."""

from unittest.mock import Mock

import pytest

import pm.defaults
import pm.install
import pm.lock
import pm.paths
import tools.browser_use_cli as browser_use_cli
from hermes_cli import tools_config_post_setup as post_setup
from hermes_cli import update_cmd_maint as update


@pytest.mark.parametrize("backend,declined,expected", [
    ("", frozenset(), True),                     # default backend: provision
    ("browser-use", frozenset(), True),          # explicit Browser Use mode needs it too
    ("off", frozenset(), False),                 # built-in tools chosen
    ("", frozenset({"agent-browser"}), False),   # --skip-browser declines the CLI as well
])
def test_default_tool_step_provisions_browser_use_cli(monkeypatch, backend, declined, expected):
    monkeypatch.setattr(pm.install, "sealed", lambda: False)
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(pm.defaults, "default_packages", lambda names: [])
    monkeypatch.setattr(pm.defaults, "declined", lambda: declined)
    monkeypatch.setattr(pm.paths, "lockfile_path", lambda: None)
    monkeypatch.setattr(pm.lock, "Lockfile", lambda path: Mock(names=Mock(return_value=[])))
    monkeypatch.setattr(browser_use_cli, "get_browser_backend", lambda: backend)
    monkeypatch.setattr(browser_use_cli, "_camofox_active", lambda *a, **k: False)
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: None)
    ensure = Mock()
    monkeypatch.setattr(post_setup, "_ensure_browser_use_cli", ensure)

    update._install_default_tools_after_update()

    assert ensure.called is expected
    if expected:
        assert 0 < ensure.call_args.kwargs["timeout_s"] <= 600
