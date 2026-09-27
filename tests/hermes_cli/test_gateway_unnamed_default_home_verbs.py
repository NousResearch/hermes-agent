"""Unnamed dashboard gateway lifecycle verbs issued from the DEFAULT home must name
``-p default`` explicitly (``_own_profile_selector``), so the spawned child can never
re-read the sticky ``active_profile`` and restart another profile's gateway, and so the
action environment takes the named-target scrub branch like every other named profile.

Regression for the Enough1122 automated-review blocker on the ``-p default`` hub-action
argv PR: ``_gateway_subcommand(None, "restart")`` used to emit a bare
``["gateway", "restart"]``; with ``active_profile=worker_gamma`` the child then resolved
``HERMES_HOME`` to ``<root>/profiles/worker_gamma`` while the dashboard served the
default root, and the default profile's ``MY_PLATFORM_TOKEN`` survived into the child.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def default_home_process(tmp_path, monkeypatch):
    """Process whose own home is the DEFAULT one, with a sticky active_profile naming a
    named profile (the state that made the selector-less child resolve the wrong home)."""
    root = tmp_path / "hermes"
    (root / "profiles" / "coder").mkdir(parents=True)
    (root / "profiles" / "coder" / "config.yaml").write_text("{}\n")  # identity marker
    (root / "config.yaml").write_text("model: {default: x}\n")
    (root / "active_profile").write_text("coder")  # sticky: points at a named profile
    (root / ".env").write_text("MY_PLATFORM_TOKEN=dashboard-secret\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("MY_PLATFORM_TOKEN", "dashboard-secret")
    for var in ("HERMES_SUPERVISED_CHILD", "HERMES_S6_SUPERVISED_CHILD", "INVOCATION_ID",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR", "HERMES_UPDATE_POST_SWAP"):
        monkeypatch.delenv(var, raising=False)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    return root


def test_unnamed_lifecycle_verbs_from_default_home_always_name_default(default_home_process):
    """Every unnamed verb pins ``-p default``: the child argv names a profile, the env
    scrub branch treats it as a named target, and the dashboard token does not survive."""
    from hermes_cli.web_server_gateway import (
        _gateway_subcommand,
        _named_profile_from_action,
        _profile_action_environment,
    )

    for verb in ("start", "stop", "restart"):
        argv = _gateway_subcommand(None, verb)
        assert argv == ["-p", "default", "gateway", verb]
        assert _named_profile_from_action(argv) == "default"
    env = _profile_action_environment(_gateway_subcommand(None, "restart"))
    assert env["HERMES_HOME"] == str(default_home_process)
    assert "MY_PLATFORM_TOKEN" not in env


def test_unnamed_restart_child_ignores_sticky_active_profile(default_home_process, monkeypatch):
    """End to end: the spawned child's first act (``_apply_profile_override``) must land
    on the default root even though the sticky ``active_profile`` names ``coder``."""
    from hermes_cli.main import _apply_profile_override
    from hermes_cli.web_server_gateway import _gateway_subcommand, _profile_action_environment

    restart = _gateway_subcommand(None, "restart")
    for var, value in _profile_action_environment(restart).items():
        if var == "HERMES_HOME":
            monkeypatch.setenv(var, value)
    monkeypatch.setattr("sys.argv", ["hermes", *restart])
    _apply_profile_override()  # what the spawned child does first
    assert os.environ["HERMES_HOME"] == str(default_home_process)
