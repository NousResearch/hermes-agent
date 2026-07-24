"""Tests for the Chronos cron-fire webhook ON THE DASHBOARD APP (web_server).

Regression guard for the relocation bug: the fire webhook MUST live on the
dashboard FastAPI app (`hermes_cli.web_server.app`) — the agent's public HTTP
surface on hosted deployments — not only on the aiohttp APIServerAdapter (which
hosted agents don't expose). It must:
  - be a registered route on the dashboard app,
  - be in PUBLIC_API_PATHS so the dashboard cookie gate doesn't 401 it before
    the JWT verifier runs,
  - reject a bad/missing NAS-JWT with 401 (the JWT is the real gate),
  - 400 on missing job_id,
  - on a valid token, resolve the job's profile and run fire_due in the
    background, returning 202.
"""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS


def _client(auth_required: bool):
    prev_auth = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = auth_required
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    return client, prev_auth, prev_host


def _restore(prev_auth, prev_host):
    if prev_auth is None:
        if hasattr(web_server.app.state, "auth_required"):
            delattr(web_server.app.state, "auth_required")
    else:
        web_server.app.state.auth_required = prev_auth
    if prev_host is None:
        if hasattr(web_server.app.state, "bound_host"):
            delattr(web_server.app.state, "bound_host")
    else:
        web_server.app.state.bound_host = prev_host


def test_route_registered_on_dashboard_app():
    """The fire webhook is served by the dashboard app (the hosted-agent public
    surface), not only the aiohttp adapter."""
    paths = {r.path for r in web_server.app.routes if hasattr(r, "path")}
    assert "/api/cron/fire" in paths


def test_fire_path_is_public():
    """Must bypass the dashboard cookie gate so the NAS bearer-JWT callback
    reaches the verifier (the JWT is the real auth)."""
    assert "/api/cron/fire" in PUBLIC_API_PATHS


def test_bad_token_401(monkeypatch):
    """Invalid NAS-JWT -> 401, even with the dashboard auth gate ENGAGED
    (proves the route is reachable past the cookie gate and the verifier is the
    gate). fire_due must NOT run."""
    fired = []
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: None),  # verification fails
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: fired.append((p, j)))

    client, pa, ph = _client(auth_required=True)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer forged"},
                           json={"job_id": "abc"})
        assert resp.status_code == 401
        assert fired == []
    finally:
        _restore(pa, ph)
        client.close()


def test_missing_job_id_400(monkeypatch):
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={})
        assert resp.status_code == 400
    finally:
        _restore(pa, ph)
        client.close()


def test_unknown_job_200_gone(monkeypatch):
    """Valid token but the job isn't found in any profile -> 200 'gone'
    (NAS shouldn't retry a fire for a cancelled/completed job)."""
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: None)
    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={"job_id": "ghost"})
        assert resp.status_code == 200
        assert resp.json().get("status") == "gone"
    finally:
        _restore(pa, ph)
        client.close()


def test_valid_token_accepts_and_fires(monkeypatch):
    """Valid token + known job -> 202 and fire_due invoked for the resolved
    profile."""
    fired = []
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire", "aud": "agent:x"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: fired.append((p, j)) or True)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={"job_id": "j1"})
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "j1"
    finally:
        _restore(pa, ph)
        client.close()
    # background task ran the fire for the resolved profile
    assert fired == [("default", "j1")]


# ── cron.provider: none per profile (#56103) ─────────────────────────────────
#
# The dashboard is ONE process serving MANY profiles. Provider resolution
# inside _fire_cron_job_for_profile must read the TARGET profile's config —
# via the scoped HERMES_HOME override — not the dashboard process's own
# profile. Otherwise a profile deliberately disabled with `cron.provider:
# none` (the standby half of an active/standby HA pair) would have its jobs
# executed anyway whenever the dashboard's own profile resolves to the
# built-in — the exact bypass flagged in the PR review of #56103.


def _profile_home_with_provider(tmp_path, name: str, provider) -> Path:
    """Create a minimal HERMES_HOME for ``name`` whose config selects
    ``provider`` (None → omit the cron section entirely = builtin default)."""
    home = tmp_path / name
    home.mkdir(parents=True, exist_ok=True)
    body = "" if provider is None else f'cron:\n  provider: "{provider}"\n'
    (home / "config.yaml").write_text(body, encoding="utf-8")
    return home


def test_fire_refuses_for_none_profile_even_when_dashboard_profile_is_builtin(
    tmp_path, monkeypatch, caplog
):
    """Cross-profile isolation, the review's direction: the TARGET profile is
    disabled (cron.provider: none) while the dashboard's own profile is
    builtin. The fire must be refused — resolution must come from the target
    profile's config, not the dashboard process's."""
    import logging

    import cron.jobs as cron_jobs
    import cron.scheduler as sched

    dashboard_home = _profile_home_with_provider(tmp_path, "dashboard", None)
    standby_home = _profile_home_with_provider(tmp_path, "standby", "none")
    monkeypatch.setenv("HERMES_HOME", str(dashboard_home))
    monkeypatch.setattr(
        web_server, "_cron_profile_home", lambda p: ("standby", standby_home)
    )

    claimed, ran = [], []
    monkeypatch.setattr(
        cron_jobs, "claim_job_for_fire",
        lambda jid: claimed.append(jid) or True, raising=False,
    )
    monkeypatch.setattr(
        sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True
    )

    with caplog.at_level(logging.WARNING, logger="cron.scheduler_provider"):
        result = web_server._fire_cron_job_for_profile("standby", "job-1")

    assert result is False
    assert claimed == [], "disabled profile's fire was claimed"
    assert ran == [], "disabled profile's job was executed"
    assert "refused" in caplog.text and "job-1" in caplog.text


def test_fire_executes_for_builtin_profile_even_when_dashboard_profile_is_none(
    tmp_path, monkeypatch
):
    """Mirror image (also proves the test above isn't vacuous): the TARGET
    profile is builtin while the dashboard's own profile is disabled. The fire
    must EXECUTE — a none dashboard profile must not block other profiles'
    jobs."""
    import cron.jobs as cron_jobs
    import cron.scheduler as sched

    dashboard_home = _profile_home_with_provider(tmp_path, "dashboard", "none")
    active_home = _profile_home_with_provider(tmp_path, "active", None)
    monkeypatch.setenv("HERMES_HOME", str(dashboard_home))
    monkeypatch.setattr(
        web_server, "_cron_profile_home", lambda p: ("active", active_home)
    )

    ran = []
    monkeypatch.setattr(cron_jobs, "claim_job_for_fire", lambda jid: True,
                        raising=False)
    monkeypatch.setattr(cron_jobs, "get_job",
                        lambda jid: {"id": jid, "name": "t"})
    monkeypatch.setattr(
        sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True
    )

    assert web_server._fire_cron_job_for_profile("active", "job-2") is True
    assert ran == ["job-2"], "builtin profile's fire did not execute"


def test_webhook_202_but_none_profile_never_runs(tmp_path, monkeypatch, caplog):
    """Endpoint-level: POST /api/cron/fire resolving to a none-disabled
    profile → 202 (NAS must not retry) but nothing is claimed or executed —
    the whole chain composed for real: route → background thread →
    _fire_cron_job_for_profile → scoped HERMES_HOME → resolver →
    NullCronScheduler.fire_due."""
    import logging
    import time

    import cron.jobs as cron_jobs
    import cron.scheduler as sched

    dashboard_home = _profile_home_with_provider(tmp_path, "dashboard", None)
    standby_home = _profile_home_with_provider(tmp_path, "standby", "none")
    monkeypatch.setenv("HERMES_HOME", str(dashboard_home))
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "standby")
    monkeypatch.setattr(
        web_server, "_cron_profile_home", lambda p: ("standby", standby_home)
    )
    # _fire_cron_job_for_profile itself stays REAL.

    claimed, ran = [], []
    monkeypatch.setattr(
        cron_jobs, "claim_job_for_fire",
        lambda jid: claimed.append(jid) or True, raising=False,
    )
    monkeypatch.setattr(
        sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True
    )

    client, pa, ph = _client(auth_required=False)
    with caplog.at_level(logging.WARNING, logger="cron.scheduler_provider"):
        try:
            resp = client.post("/api/cron/fire",
                               headers={"Authorization": "Bearer good"},
                               json={"job_id": "standby-job"})
            assert resp.status_code == 202
        finally:
            _restore(pa, ph)
            client.close()
        # The refusal happens in a background thread after the 202 — wait for
        # its warning (the only operator-visible trace) instead of a fixed
        # sleep.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "standby-job" not in caplog.text:
            time.sleep(0.01)

    assert "refused" in caplog.text and "standby-job" in caplog.text
    assert claimed == [], "disabled profile's webhook fire was claimed"
    assert ran == [], "disabled profile's webhook fire was executed"
