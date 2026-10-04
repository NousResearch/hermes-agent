"""The detached host-gateway restart watcher must not follow the sticky ``active_profile``.

``hermes update`` respawns the multiplex host through ``_spawn_gateway_restart_watcher`` with a
selector-less ``gateway run --replace``. The respawn carries no supervisor marker, so
``_apply_profile_override`` honours ``active_profile`` (#22502): after ``hermes profile use
<named>`` the new process re-homed into that profile and was refused ("Profile '<named>' does not
get a gateway of its own"). The update still reported the restart as done and the host stayed
down. The watcher now names the host explicitly with ``--profile default``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import gateway


@pytest.mark.parametrize(
    "argv, expected",
    [
        (
            ["python", "-m", "hermes_cli.main", "gateway", "run", "--replace"],
            ["python", "-m", "hermes_cli.main", "--profile", "default", "gateway", "run", "--replace"],
        ),
        (
            ["hermes", "gateway", "run"],
            ["hermes", "--profile", "default", "gateway", "run"],
        ),
    ],
)
def test_selectorless_host_argv_is_pinned_to_default(argv, expected):
    assert gateway._pin_host_profile_selector(argv) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-m", "hermes_cli.main", "--profile", "default", "gateway", "run"],
        ["python", "-m", "hermes_cli.main", "-p", "worker", "gateway", "run"],
        ["python", "-m", "hermes_cli.main", "--profile=worker", "gateway", "run"],
        ["python", "-c", "print('not a gateway command')"],
    ],
)
def test_existing_selector_or_non_gateway_argv_is_left_alone(argv):
    assert gateway._pin_host_profile_selector(argv) == argv


def _capture_watcher_spawn(monkeypatch) -> list[list[str]]:
    spawned: list[list[str]] = []

    def _fake_popen(argv, *args, **kwargs):
        spawned.append(list(argv))
        return object()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(gateway, "_host_gateway_watcher_env", lambda: {})
    return spawned


def _respawn_argv(watcher_argv: list[str], old_pid: int) -> list[str]:
    """The command the watcher will respawn: everything after ``<old_pid>``."""
    return watcher_argv[watcher_argv.index(str(old_pid)) + 1:]


def test_host_restart_watcher_respawns_with_explicit_default_profile(monkeypatch):
    spawned = _capture_watcher_spawn(monkeypatch)

    assert gateway._spawn_gateway_restart_watcher(
        4242, [sys.executable, "-m", "hermes_cli.main", "gateway", "run", "--replace"], host=True,
    )

    respawn = _respawn_argv(spawned[0], 4242)
    gw = respawn.index("gateway")
    assert respawn[gw - 2:gw] == ["--profile", "default"], respawn
    assert respawn[gw:] == ["gateway", "run", "--replace"]


def test_named_profile_restart_watcher_argv_is_unchanged(monkeypatch):
    spawned = _capture_watcher_spawn(monkeypatch)

    assert gateway._spawn_gateway_restart_watcher(
        4242, [sys.executable, "-m", "hermes_cli.main", "--profile", "worker", "gateway", "run"],
        host=False,
    )

    respawn = _respawn_argv(spawned[0], 4242)
    assert respawn.count("--profile") == 1
    assert respawn[-4:] == ["--profile", "worker", "gateway", "run"]


@pytest.fixture
def _sticky_named_profile(tmp_path, monkeypatch):
    """A default root whose ``active_profile`` names ``worker`` and no supervisor marker."""
    root = tmp_path / ".hermes"
    worker = root / "profiles" / "worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text("worker", encoding="utf-8")
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: root)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    for var in ("HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD", "INVOCATION_ID",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(var, raising=False)
    return root, worker


def _home_after_override(monkeypatch, argv: list[str]) -> Path:
    monkeypatch.setattr(sys, "argv", argv)
    from hermes_cli.main import _apply_profile_override
    _apply_profile_override()
    return Path(os.environ["HERMES_HOME"]).resolve()


def test_premise_selectorless_respawn_follows_sticky_profile(_sticky_named_profile, monkeypatch):
    _root, worker = _sticky_named_profile
    assert _home_after_override(monkeypatch, ["hermes", "gateway", "run", "--replace"]) == worker.resolve()


def test_pinned_respawn_stays_on_the_default_root(_sticky_named_profile, monkeypatch):
    root, _worker = _sticky_named_profile
    pinned = gateway._pin_host_profile_selector(["hermes", "gateway", "run", "--replace"])
    assert _home_after_override(monkeypatch, pinned) == root.resolve()
