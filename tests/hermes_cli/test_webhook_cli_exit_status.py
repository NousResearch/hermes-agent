"""Webhook command refusals must reach the invoking shell (#139332)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _run_hermes(home: Path, *args: str, extra_env: dict | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_webhook_subscribe_disabled_exits_nonzero(tmp_path):
    """Subscribing when webhook is disabled must exit with code 1."""
    res = _run_hermes(tmp_path, "webhook", "subscribe", "my-route")
    assert res.returncode == 1
    assert "Webhook platform is not enabled" in res.stdout


def test_webhook_remove_disabled_exits_nonzero(tmp_path):
    """Removing when webhook is disabled must exit with code 1."""
    res = _run_hermes(tmp_path, "webhook", "remove", "my-route")
    assert res.returncode == 1
    assert "Webhook platform is not enabled" in res.stdout


def test_webhook_remove_missing_route_exits_nonzero(tmp_path):
    """Removing a non-existent route when enabled must exit with code 1."""
    cfg = {"platforms": {"webhook": {"enabled": True, "extra": {"port": 8644, "secret": "sec"}}}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")

    res = _run_hermes(tmp_path, "webhook", "remove", "nope-route")
    assert res.returncode == 1
    assert "No subscription named 'nope-route'" in res.stdout


def test_webhook_test_missing_route_exits_nonzero(tmp_path):
    """Testing a non-existent route when enabled must exit with code 1."""
    cfg = {"platforms": {"webhook": {"enabled": True, "extra": {"port": 8644, "secret": "sec"}}}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")

    res = _run_hermes(tmp_path, "webhook", "test", "nope-route")
    assert res.returncode == 1
    assert "No subscription named 'nope-route'" in res.stdout


def test_webhook_subscribe_invalid_name_exits_nonzero(tmp_path):
    """Subscribing with an invalid route name must exit with code 1."""
    cfg = {"platforms": {"webhook": {"enabled": True, "extra": {"port": 8644, "secret": "sec"}}}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")

    res = _run_hermes(tmp_path, "webhook", "subscribe", "Invalid Name With Spaces!")
    assert res.returncode == 1
    assert "Invalid name" in res.stdout


def test_webhook_subscribe_invalid_deliver_only_exits_nonzero(tmp_path):
    """Subscribing with --deliver-only and deliver=log must exit with code 1."""
    cfg = {"platforms": {"webhook": {"enabled": True, "extra": {"port": 8644, "secret": "sec"}}}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")

    res = _run_hermes(tmp_path, "webhook", "subscribe", "log-route", "--deliver-only", "--deliver", "log")
    assert res.returncode == 1
    assert "--deliver-only requires --deliver to be a real target" in res.stdout


def test_webhook_lifecycle_success_exits_zero(tmp_path):
    """A valid subscribe -> list -> remove lifecycle must all exit with code 0."""
    cfg = {"platforms": {"webhook": {"enabled": True, "extra": {"port": 8644, "secret": "sec"}}}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")

    # 1. Subscribe
    sub_res = _run_hermes(tmp_path, "webhook", "subscribe", "test-route")
    assert sub_res.returncode == 0
    assert "Created webhook subscription: test-route" in sub_res.stdout

    # 2. List
    list_res = _run_hermes(tmp_path, "webhook", "list")
    assert list_res.returncode == 0
    assert "test-route" in list_res.stdout

    # 3. Remove
    rm_res = _run_hermes(tmp_path, "webhook", "remove", "test-route")
    assert rm_res.returncode == 0
    assert "Removed webhook subscription: test-route" in rm_res.stdout


# ---------------------------------------------------------------------------
# _forward_command return propagation tests (#103257)
# ---------------------------------------------------------------------------


def test_forward_command_propagates_handler_exit_status(monkeypatch):
    """_forward_command must return the handler's status code so main() can exit with it."""
    from types import SimpleNamespace
    import hermes_cli.main as main_mod

    fake_module = SimpleNamespace(fake_action=lambda args: 42)
    monkeypatch.setitem(sys.modules, "hermes_cli.fake_subcommand", fake_module)

    forwarded = main_mod._forward_command("cmd_fake", "hermes_cli.fake_subcommand", "fake_action", forward_return=True)
    assert forwarded(None) == 42
