from __future__ import annotations

import builtins
import os
from pathlib import Path
from unittest.mock import MagicMock

from hermes_cli import memory_setup


def test_find_provider_dir_falls_back_to_user_plugin():
    hermes_home = Path(os.environ["HERMES_HOME"])
    user_provider = hermes_home / "plugins" / "usermem"
    user_provider.mkdir(parents=True, exist_ok=True)

    found = memory_setup._find_provider_dir("usermem")

    assert found == user_provider


def test_find_provider_dir_prefers_bundled_over_user_plugin():
    hermes_home = Path(os.environ["HERMES_HOME"])
    user_provider = hermes_home / "plugins" / "holographic"
    user_provider.mkdir(parents=True, exist_ok=True)

    found = memory_setup._find_provider_dir("holographic")

    assert found is not None
    assert found.name == "holographic"
    assert "plugins/memory/holographic" in str(found)


def test_install_dependencies_reads_user_plugin_manifest(monkeypatch):
    hermes_home = Path(os.environ["HERMES_HOME"])
    user_provider = hermes_home / "plugins" / "usermem"
    user_provider.mkdir(parents=True, exist_ok=True)
    (user_provider / "plugin.yaml").write_text("pip_dependencies:\n  - fake-user-dep\n")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fake_user_dep":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    mock_run = MagicMock()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("subprocess.run", mock_run)

    memory_setup._install_dependencies("usermem")

    assert mock_run.called
    cmd = mock_run.call_args.args[0]
    assert "fake-user-dep" in cmd
