"""Native CLI ownership: foreign plugins cannot silently replace a command."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context(manager, key, name=None):
    return PluginContext(PluginManifest(name=name or key, key=key), manager)


def _handler(marker):
    return lambda args: marker


@pytest.mark.parametrize("shared_display_name", [False, True])
def test_foreign_cli_owner_is_rejected_without_a_registration(caplog, shared_display_name):
    manager = PluginManager()
    name = "same-display-name" if shared_display_name else None
    first = _context(manager, "owner-a", name)
    second = _context(manager, "owner-b", name)
    first_handle = first.register_cli_command("shared-cli", "A", lambda parser: None, _handler("A"))
    entry = manager._cli_commands["shared-cli"]
    registrations = list(manager._registration_order)

    with caplog.at_level("WARNING", logger="hermes_cli.plugins"):
        rejected = second.register_cli_command("shared-cli", "B", lambda parser: None, _handler("B"))

    assert rejected is None
    assert manager._cli_commands["shared-cli"] is entry
    assert entry["handler_fn"](None) == "A"
    assert manager._registration_order == registrations
    assert "owner-b" not in manager._ownership_ledger
    assert "collision" in caplog.text.lower()
    assert all(value in caplog.text for value in ("shared-cli", "owner-a", "owner-b"))
    first_handle.dispose()
    assert "shared-cli" not in manager._cli_commands
    # A rejected command is not queued to reappear after its owner's removal.
    assert "owner-b" not in manager._ownership_ledger


@pytest.mark.parametrize("dispose_old_first", [False, True])
def test_same_registry_owner_can_replace_and_dispose_its_command(dispose_old_first):
    manager = PluginManager()
    first = _context(manager, "same-owner", "old-display-name")
    reloaded = _context(manager, "same-owner", "new-display-name")
    old_handle = first.register_cli_command("shared-cli", "A", lambda parser: None, _handler("A"))
    new_handle = reloaded.register_cli_command("shared-cli", "B", lambda parser: None, _handler("B"))

    assert old_handle is not None and new_handle is not None
    assert manager._cli_commands["shared-cli"]["handler_fn"](None) == "B"
    if dispose_old_first:
        old_handle.dispose()
        assert manager._cli_commands["shared-cli"]["handler_fn"](None) == "B"
        new_handle.dispose()
    else:
        new_handle.dispose()
        assert manager._cli_commands["shared-cli"]["handler_fn"](None) == "A"
        old_handle.dispose()
    assert "shared-cli" not in manager._cli_commands


def test_rejected_owner_unload_does_not_remove_first_and_can_register_after_release():
    manager = PluginManager()
    first = _context(manager, "owner-a")
    second = _context(manager, "owner-b")
    first.register_cli_command("shared-cli", "A", lambda parser: None, _handler("A"))
    second.register_cli_command("shared-cli", "B", lambda parser: None, _handler("B"))
    second.register_cli_command("other-cli", "Other", lambda parser: None)

    assert manager.unload("owner-b") is True
    assert manager._cli_commands["shared-cli"]["plugin_key"] == "owner-a"
    assert "other-cli" not in manager._cli_commands
    assert manager.unload("owner-a") is True
    assert "shared-cli" not in manager._cli_commands
    assert second.register_cli_command("shared-cli", "B", lambda parser: None) is not None
    assert manager.unload("owner-b") is True
    assert "shared-cli" not in manager._cli_commands


def test_separate_managers_have_separate_cli_namespaces():
    manager_a, manager_b = PluginManager(), PluginManager()
    _context(manager_a, "owner-a").register_cli_command("shared-cli", "A", lambda parser: None)
    _context(manager_b, "owner-b").register_cli_command("shared-cli", "B", lambda parser: None)
    assert manager_a._cli_commands["shared-cli"]["plugin_key"] == "owner-a"
    assert manager_b._cli_commands["shared-cli"]["plugin_key"] == "owner-b"
    assert manager_a.unload("owner-a") is True
    assert manager_b._cli_commands["shared-cli"]["plugin_key"] == "owner-b"
    manager_b.unload("owner-b")


def test_concurrent_foreign_registrations_have_only_one_owner():
    manager = PluginManager()
    barrier = Barrier(2)

    def register(owner):
        context = _context(manager, owner)
        barrier.wait(timeout=10)
        return context.register_cli_command("shared-cli", owner, lambda parser: None)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(register, owner) for owner in ("owner-a", "owner-b")]
        handles = [future.result(timeout=10) for future in futures]

    assert sum(handle is not None for handle in handles) == 1
    winner = manager._cli_commands["shared-cli"]["plugin_key"]
    assert list(manager._ownership_ledger) == [winner]
    assert len(manager._registration_order) == 1
    assert manager.unload(winner) is True
    assert "shared-cli" not in manager._cli_commands


def _write_plugin(home, owner, marker):
    root = home / "plugins" / owner
    root.mkdir(parents=True)
    (root / "plugin.yaml").write_text(
        f"name: {owner}\nversion: 0.1.0\ndescription: Native CLI ownership probe\n",
        encoding="utf-8",
    )
    (root / "__init__.py").write_text(
        "def setup(parser):\n    pass\n"
        f"def handler(args):\n    print({marker!r})\n"
        "def register(ctx):\n"
        "    ctx.register_cli_command('modernise', 'Ownership probe', setup, handler)\n"
        f"    ctx.register_cli_command({owner + '-cli'!r}, 'Unique probe', setup, handler)\n",
        encoding="utf-8",
    )


def _invoke(home, enabled, command):
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n" + "".join(f"    - {owner}\n" for owner in enabled),
        encoding="utf-8",
    )
    environment = dict(
        os.environ,
        HOME=str(home),
        HERMES_HOME=str(home),
        HERMES_BUNDLED_PLUGINS=str(home / "no-bundled"),
        HERMES_ENABLE_PROJECT_PLUGINS="0",
    )
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", command],
        env=environment,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_native_cli_single_plugin_positive_control(tmp_path):
    home = tmp_path / "single"
    _write_plugin(home, "probe-a", "SDK_PROBE_A")
    result = _invoke(home, ["probe-a"], "modernise")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "SDK_PROBE_A"


def test_native_cli_foreign_collision_preserves_first_and_other_commands(tmp_path):
    home = tmp_path / "collision"
    _write_plugin(home, "probe-a", "SDK_PROBE_A")
    _write_plugin(home, "probe-b", "SDK_PROBE_B")
    result = _invoke(home, ["probe-a", "probe-b"], "modernise")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "SDK_PROBE_A", result.stdout + result.stderr
    assert "SDK_PROBE_B" not in result.stdout
    # Native CLI warnings use Hermes' normal file-backed logging, not stderr.
    warnings = (home / "logs" / "errors.log").read_text(encoding="utf-8")
    assert "collision" in warnings.lower(), warnings
    assert all(value in warnings for value in ("modernise", "probe-a", "probe-b"))
    # Refusing one registration must not disable the second plugin's other commands.
    other = _invoke(home, ["probe-a", "probe-b"], "probe-b-cli")
    assert other.returncode == 0, other.stdout + other.stderr
    assert other.stdout.strip() == "SDK_PROBE_B"
