"""The Web Access pane's status poll never holds the process-wide skills lock across its gateway query.

``GET /api/webapp-access/status`` asks the gateway over its control socket what it serves. The pane
repeats that every 10 s, so a gateway slow to answer (mid-restart) must not stall every config,
skills and tools request, which all take ``_SKILLS_PROFILE_LOCK`` through ``_profile_scope``.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("fastapi")


def test_a_status_poll_waiting_on_the_gateway_does_not_block_a_skills_request(monkeypatch):
    from hermes_cli import webapp_launcher
    from tests.hermes_cli.native_gateway_client import native_desktop_client

    client = native_desktop_client(monkeypatch)
    in_control, release = threading.Event(), threading.Event()

    def _slow_gateway(_home, _command):
        in_control.set()
        release.wait(timeout=30)
        raise webapp_launcher.GatewayUnreachable("no gateway")

    monkeypatch.setattr(webapp_launcher, "_control", _slow_gateway)
    results: dict[str, int] = {}

    def _get(name, path):
        results[name] = client.get(path).status_code

    status = threading.Thread(target=_get, args=("status", "/api/webapp-access/status"), daemon=True)
    status.start()
    try:
        assert in_control.wait(timeout=10), "the status route never reached the gateway query"
        skills = threading.Thread(target=_get, args=("skills", "/api/skills"), daemon=True)
        skills.start()
        skills.join(timeout=10)
        assert not skills.is_alive(), "a skills request waited on the status poll's gateway query"
        assert results["skills"] == 200
    finally:
        release.set()
        status.join(timeout=10)
    assert results["status"] == 200
