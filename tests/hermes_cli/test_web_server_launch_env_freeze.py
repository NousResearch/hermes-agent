"""The launch-env freeze is the LAST boot step, not the first.

Activation snapshots ``os.environ`` as the launch profile's credentials, and that snapshot is the
only source for launch keys with no ``.env`` to rebuild them from (systemd ``Environment=``,
``op run``, Compose). Taken at the TOP of ``start_server`` it missed everything the rest of boot
injected or rotated, for the process lifetime.
"""
import os

import pytest

from agent import secret_scope
from hermes_cli import web_server
from tui_gateway import launch_profile_policy

LATE_KEY = "LAUNCH_FREEZE_PROBE_TOKEN"


def test_boot_time_credential_injection_is_inside_the_frozen_launch_env(tmp_path, monkeypatch):
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.delenv(LATE_KEY, raising=False)
    # A two-profile host, without touching the live install's profiles/.
    monkeypatch.setattr(launch_profile_policy, "_servable_profile_homes",
                        lambda: {tmp_path / "a", tmp_path / "b"})
    monkeypatch.setattr(launch_profile_policy, "_multiplex_disabled_explicitly", lambda: False)

    # Stand-ins for the boot steps that follow the old (top-of-function) activation point. A
    # provider key injected by the auth gate / keepalive / a lifespan hook is the real case.
    def _inject(*_args, **_kwargs):
        os.environ[LATE_KEY] = "injected-during-boot"

    monkeypatch.setattr(web_server, "_configure_auth_gate", _inject)
    monkeypatch.setattr(web_server, "_build_uvicorn_server", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(web_server, "_port_bind_conflict", lambda *a, **k: False)
    served: list[bool] = []
    monkeypatch.setattr(web_server, "_run_serve", lambda *a, **k: served.append(True))

    try:
        web_server.start_server(open_browser=False, headless=True)
    finally:
        os.environ.pop(LATE_KEY, None)

    assert served, "start_server never reached the serve step"
    assert secret_scope.is_multiplex_active()
    assert launch_profile_policy.capture_launch_env().get(LATE_KEY) == "injected-during-boot", (
        "the launch env was frozen before boot finished injecting credentials")
