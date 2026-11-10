"""Every module that climbs from its own file to the install root lands on the checkout root.

These modules sit in ``hermes_agent/hermes_cli/``; a climb one level short ends in ``hermes_agent/``
and silently splits the update lock, the pause record and the host obligation from the ones an
older process keyed on the real root.
"""
import hashlib
import os
from pathlib import Path

from hermes_agent.hermes_cli._startup_fast import project_root_str

ROOT = Path(project_root_str()).resolve()


def test_install_root_helpers_resolve_to_the_checkout_root():
    from hermes_agent.hermes_cli import _early_recovery, linux_desktop_entry, update_lock, update_pause_record, venv_sync

    assert update_lock._default_install_root() == ROOT
    assert update_pause_record.install_root() == ROOT
    assert venv_sync._project_root() == ROOT
    assert _early_recovery._project_root() == ROOT
    assert _early_recovery._pytest_owns_live_checkout(ROOT)
    assert Path(os.path.realpath(linux_desktop_entry._project_root())) == ROOT


def test_old_path_desktop_entry_copy_resolves_to_the_checkout_root():
    """``nix/desktop.nix`` loads the byte copy at ``hermes_cli/`` on its own, one level higher."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("old_path_linux_desktop_entry", ROOT / "hermes_cli/linux_desktop_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(os.path.realpath(module._project_root())) == ROOT


def test_script_mode_main_puts_the_checkout_root_on_sys_path():
    from hermes_agent.hermes_cli import main

    assert Path(main._bootstrap_root) == ROOT


def test_host_obligation_is_keyed_on_the_checkout_root():
    from hermes_agent.hermes_cli.update_host_obligation import host_obligation_path

    key = hashlib.sha256(os.path.normcase(str(ROOT)).encode("utf-8")).hexdigest()[:16]
    assert host_obligation_path().stem.endswith(f"-{key}")


def test_moved_and_old_path_files_resolve_to_one_code_root():
    from hermes_agent.hermes_cli.update_receipt import _code_root_for_path
    from hermes_agent.hermes_cli.venv_sync import _is_tail_script

    for rel in ("hermes_agent/hermes_cli/update_receipt.py", "hermes_cli/main.py"):
        assert _code_root_for_path(str(ROOT / rel)) == ROOT
    for folder in ("hermes_agent/hermes_cli", "hermes_cli"):
        assert _is_tail_script(ROOT, str(ROOT / folder / "update_completion.py"))
