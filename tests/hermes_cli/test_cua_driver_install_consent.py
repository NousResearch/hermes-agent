"""cua-driver lands OUTSIDE ``$HERMES_HOME`` — Hermes must not put it there silently (#104413).

The upstream installer writes ``~/.cua-driver`` and ``~/.local/bin/cua-driver``, appends a PATH
line to ``~/.bashrc``/``~/.zshrc`` and fires a default-on install-event. A headless VPS user found
``~/.cua-driver`` weeks after an update with no notice. The Python side of the fix:

* the installer child env carries ``CUA_DRIVER_RS_NO_MODIFY_PATH=1`` (Hermes resolves
  ``~/.local/bin/cua-driver`` itself and install.sh's ``setup_path`` already manages that line);
* a fresh install says up front what it writes outside HERMES_HOME; refresh/repair runs stay quiet;
* ``hermes update`` only refreshes a driver that is already installed — resolved the way the runtime
  resolves it, so a NO_MODIFY_PATH install off a service account's PATH is not skipped forever;
* ``hermes uninstall`` names the cua-driver files it leaves behind and how to remove them (it never
  deletes them: the driver and skill pack may be shared with other agent harnesses).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _run_installer(label, *, verbose, env=None):
    """Run ``_run_cua_driver_installer`` with Popen stubbed; return (child env, info lines)."""
    from hermes_cli import tools_config_cua as tools_config

    captured, infos = {}, []
    fake_proc = MagicMock()
    fake_proc.pid = 1
    fake_proc.returncode = 1
    fake_proc.communicate.return_value = ("", None)

    def fake_popen(cmd, **kw):
        captured["env"] = kw.get("env")
        return fake_proc

    def fake_run(cmd, **kw):
        m = MagicMock(); m.returncode = 0; m.stderr = ""
        return m

    with patch("subprocess.run", side_effect=fake_run), \
         patch("subprocess.Popen", side_effect=fake_popen), \
         patch.object(tools_config, "_cua_driver_env",
                      return_value=dict(env if env is not None else {"PATH": "/usr/bin"})), \
         patch.object(tools_config, "_clear_stale_cua_install_lock"), \
         patch.object(tools_config, "_print_warning"), \
         patch.object(tools_config, "_print_success"), \
         patch.object(tools_config, "_print_info", side_effect=lambda m: infos.append(m)):
        tools_config._run_cua_driver_installer(label=label, verbose=verbose)
    return captured.get("env") or {}, infos


@pytest.mark.linux_only
class TestInstallerChildEnv:
    """``linux_only``: reaches Popen through the POSIX download-then-exec branch for real."""

    def test_no_modify_path_is_set_unless_the_user_chose_otherwise(self):
        for label, verbose in (("Installing", True), ("Refreshing", False), ("Repairing", False)):
            env, _ = _run_installer(label, verbose=verbose)
            assert env.get("CUA_DRIVER_RS_NO_MODIFY_PATH") == "1", label
        env, _ = _run_installer("Installing", verbose=True,
                                env={"PATH": "/usr/bin", "CUA_DRIVER_RS_NO_MODIFY_PATH": "0"})
        assert env.get("CUA_DRIVER_RS_NO_MODIFY_PATH") == "0"

    def test_only_a_fresh_install_announces_paths_outside_hermes_home(self):
        _, infos = _run_installer("Installing", verbose=True)
        notice = [m for m in infos if "outside HERMES_HOME" in m]
        assert notice and ".cua-driver" in notice[0] and "~/.local/bin/cua-driver" in notice[0]
        for label in ("Refreshing", "Repairing"):
            _, infos = _run_installer(label, verbose=False)
            assert not any("outside HERMES_HOME" in m for m in infos), label


class TestUpdateRefreshGate:
    """``hermes update`` refreshes only an already-installed driver, never installs one."""

    def _run(self, resolved):
        from hermes_cli import update_cmd_maint

        calls = []
        fake_tools_config = MagicMock()
        fake_tools_config.install_cua_driver = lambda **kw: calls.append(kw) or True
        with patch.object(update_cmd_maint, "_load_updates_cfg", return_value={}), \
             patch("tools.computer_use.cua_backend_driver.resolve_cua_driver_cmd",
                   return_value=resolved), \
             patch.dict(sys.modules, {"hermes_cli.tools_config": fake_tools_config}), \
             patch("builtins.print"):
            update_cmd_maint._refresh_cua_driver_after_update()
        return calls

    def test_no_driver_means_no_install(self):
        assert self._run(None) == []

    def test_installed_driver_is_refreshed_only_on_a_confirmed_newer_release(self):
        calls = self._run("/home/u/.local/bin/cua-driver")
        assert calls == [dict(upgrade=True, require_confirmed_update=True,
                              show_installer_progress=False)]


class TestUninstallLeftovers:
    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("CUA_DRIVER_RS_HOME", raising=False)
        return tmp_path

    def test_lists_only_what_exists_honours_the_home_override_and_deletes_nothing(self, home,
                                                                                  monkeypatch):
        from hermes_cli import uninstall

        assert uninstall.cua_driver_leftover_paths() == []
        keep = home / ".cua-driver" / "packages" / "keep"
        keep.parent.mkdir(parents=True)
        keep.write_text("x", encoding="utf-8")
        link = home / ".local" / "bin" / "cua-driver"
        link.parent.mkdir(parents=True)
        link.write_text("stub", encoding="utf-8")
        found = uninstall.cua_driver_leftover_paths()
        assert home / ".cua-driver" in found and link in found
        with patch("builtins.print"):
            uninstall._print_cua_driver_leftovers()
        assert keep.read_text(encoding="utf-8") == "x" and link.exists()

        custom = home / "elsewhere"
        custom.mkdir()
        monkeypatch.setenv("CUA_DRIVER_RS_HOME", str(custom))
        assert custom in uninstall.cua_driver_leftover_paths()
        assert home / ".cua-driver" not in uninstall.cua_driver_leftover_paths()

    def test_notice_names_paths_and_the_upstream_uninstaller_or_stays_silent(self, home, capsys):
        from hermes_cli import uninstall

        uninstall._print_cua_driver_leftovers()
        assert capsys.readouterr().out == ""
        (home / ".cua-driver").mkdir()
        uninstall._print_cua_driver_leftovers()
        out = capsys.readouterr().out
        assert str(home / ".cua-driver") in out
        assert "uninstall.sh" in out or "uninstall.ps1" in out
        assert "Other agents may share it" in out
