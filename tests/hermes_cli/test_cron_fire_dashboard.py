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
  - on a valid token, FORWARD the fire to the gateway api_server (which owns
    cron execution and the live delivery adapters) and pass its response
    through — 503 when the gateway is unreachable so NAS retries.
"""

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


def test_valid_fire_forwards_to_gateway(monkeypatch):
    """The dashboard is the public door only: a verified fire is FORWARDED to
    the gateway api_server (which owns the live adapters — relay/E2EE
    delivery), with the NAS bearer preserved, and the gateway's response is
    passed through. The deprecated in-dashboard execution must NOT run."""
    forwarded = []
    executed = []

    async def fake_forward(profile, job_id, authorization):
        forwarded.append((profile, job_id, authorization))
        return 202, {"status": "accepted", "job_id": job_id}

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: executed.append((p, j)))

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j1"})
        assert resp.status_code == 202
        assert resp.json().get("status") == "accepted"
        assert forwarded == [("default", "j1", "Bearer nas-jwt")]
        assert executed == []  # no local execution — gateway owns cron
    finally:
        _restore(pa, ph)
        client.close()


def test_gateway_unreachable_503_for_nas_retry(monkeypatch):
    """Gateway down (scale-to-zero wake window / restart) -> 503 so NAS
    retries per the Chronos contract. Deliberately NO local-execution
    fallback — delivering from the dashboard process cannot serve
    relay-fronted or E2EE targets."""
    executed = []

    async def fake_forward(profile, job_id, authorization):
        return None  # unreachable

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: executed.append((p, j)))

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j2"})
        assert resp.status_code == 503
        assert executed == []
    finally:
        _restore(pa, ph)
        client.close()


def test_gateway_error_status_passes_through(monkeypatch):
    """A non-2xx gateway response (e.g. its own 401 on a replayed JWT) passes
    through unchanged — the dashboard adds no interpretation of its own."""

    async def fake_forward(profile, job_id, authorization):
        return 401, {"error": "invalid fire token"}

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j3"})
        assert resp.status_code == 401
    finally:
        _restore(pa, ph)
        client.close()


# ── _gateway_fire_endpoint URL resolution ────────────────────────────────


def test_fire_endpoint_default_port(tmp_path, monkeypatch):
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    url = web_server._gateway_fire_endpoint("default", tmp_path)
    assert url == "http://127.0.0.1:8642/api/cron/fire"


def test_fire_endpoint_config_yaml_port_wins(tmp_path, monkeypatch):
    """The profile config.yaml port (read via the CANONICAL load_config, per
    the config-read guard) wins over the process env API_SERVER_PORT."""
    monkeypatch.setenv("API_SERVER_PORT", "9999")
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"platforms": {"api_server": {"extra": {"port": 8700}}}},
    )
    url = web_server._gateway_fire_endpoint("default", tmp_path)
    assert url == "http://127.0.0.1:8700/api/cron/fire"


def test_fire_endpoint_profile_env_port(tmp_path, monkeypatch):
    """A non-default profile reads API_SERVER_PORT from its own .env, not the
    dashboard process env (per-profile-gateway topology)."""
    monkeypatch.setenv("API_SERVER_PORT", "9999")  # dashboard process env
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    monkeypatch.setattr(web_server, "_cron_default_profile", lambda: "default")
    (tmp_path / ".env").write_text("API_SERVER_PORT=8701\n", encoding="utf-8")
    url = web_server._gateway_fire_endpoint("worker_alpha", tmp_path)
    assert url == "http://127.0.0.1:8701/api/cron/fire"


def test_fire_endpoint_multiplex_profile_prefix(tmp_path, monkeypatch):
    """Multiplex mode: a non-default profile routes through the default
    gateway's port with the /p/<profile>/ prefix mirror."""
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "1")
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    url = web_server._gateway_fire_endpoint("worker_alpha", tmp_path)
    assert url == "http://127.0.0.1:8642/p/worker_alpha/api/cron/fire"


def test_fire_endpoint_multiplex_reads_port_from_default_listener(tmp_path, monkeypatch):
    """Multiplex mode: only the DEFAULT profile's api_server is bound, so a
    secondary's fire URL must use the default home's port — not the
    secondary's own config.yaml/.env port, which nothing listens on
    (PR #84755). Real config files, real load_config()."""
    default_home = tmp_path / "root"
    worker_home = default_home / "profiles" / "worker_alpha"
    default_home.mkdir()
    worker_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n"
        "platforms:\n  api_server:\n    extra:\n      port: 8650\n",
        encoding="utf-8",
    )
    (worker_home / "config.yaml").write_text(
        "platforms:\n  api_server:\n    enabled: false\n    extra:\n      port: 8702\n",
        encoding="utf-8",
    )
    (worker_home / ".env").write_text("API_SERVER_PORT=8701\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(web_server, "_cron_default_profile", lambda: "default")

    url = web_server._gateway_fire_endpoint("worker_alpha", worker_home)

    assert url == "http://127.0.0.1:8650/p/worker_alpha/api/cron/fire"
    # The GATEWAY_MULTIPLEX_PROFILES env override is still honored (parity
    # with gateway/config.py): forcing it off restores per-profile routing.
    monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "0")
    assert web_server._gateway_fire_endpoint("worker_alpha", worker_home) == (
        "http://127.0.0.1:8702/api/cron/fire"
    )


# ── OOF-266: intentional-stop drop + Retry-After on transient 503 ─────────


def test_gateway_unreachable_503_carries_retry_after(monkeypatch):
    """Transient unreachable window (wake/restart) -> 503 WITH Retry-After so
    the scheduler spaces retries past the window instead of exhausting its
    budget inside it (OOF-266)."""

    async def fake_forward(profile, job_id, authorization):
        return None  # unreachable

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(
        web_server, "_gateway_intentionally_stopped", lambda p: False
    )

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j4"})
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "60"
    finally:
        _restore(pa, ph)
        client.close()


def test_gateway_intentionally_stopped_drops_with_200(monkeypatch):
    """Operator-stopped gateway (durable desired_state == "stopped") -> the
    fire is DROPPED with 200: retrying can never succeed until a human starts
    the gateway, so burning the scheduler retry budget only produces
    502-retry storms at NAS (OOF-266). Jobs re-arm via the Chronos provider's
    reconcile on the next gateway start."""
    executed = []

    async def fake_forward(profile, job_id, authorization):
        return None  # unreachable — gateway process is down

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(
        web_server, "_gateway_intentionally_stopped", lambda p: True
    )
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: executed.append((p, j)))

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j5"})
        assert resp.status_code == 200
        assert resp.json().get("status") == "gateway_stopped"
        assert executed == []  # dropped, never locally executed
    finally:
        _restore(pa, ph)
        client.close()


def test_stopped_check_only_consulted_when_gateway_unreachable(monkeypatch):
    """A reachable gateway's response passes through untouched — the
    intentional-stop check must not run (a stale desired_state file must
    never shadow a live gateway)."""
    consulted = []

    async def fake_forward(profile, job_id, authorization):
        return 202, {"status": "accepted", "job_id": job_id}

    def fake_stopped(profile):
        consulted.append(profile)
        return True  # would drop if (wrongly) consulted

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(web_server, "_gateway_intentionally_stopped", fake_stopped)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j6"})
        assert resp.status_code == 202
        assert consulted == []
    finally:
        _restore(pa, ph)
        client.close()


def test_gateway_own_503_also_carries_retry_after(monkeypatch):
    """The gateway's OWN 503 (draining / admission failure) is equally
    transient — the pass-through stamps the same Retry-After hint. Other
    statuses stay header-free."""

    async def fake_forward(profile, job_id, authorization):
        return 503, {"error": "gateway draining"}

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j7"})
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "60"
    finally:
        _restore(pa, ph)
        client.close()


# ── _gateway_intentionally_stopped (durable operator intent) ──────────────


def _stopped_check_home(monkeypatch, tmp_path):
    """Point the profile resolver at tmp_path so the check reads our file."""
    monkeypatch.setattr(
        web_server, "_cron_profile_home", lambda p: ("default", tmp_path)
    )


def test_intentionally_stopped_true_on_desired_state_stopped(tmp_path, monkeypatch):
    _stopped_check_home(monkeypatch, tmp_path)
    (tmp_path / "gateway_state.json").write_text(
        '{"gateway_state":"stopped","desired_state":"stopped"}', encoding="utf-8"
    )
    assert web_server._gateway_intentionally_stopped("default") is True


def test_intentionally_stopped_false_when_desired_running(tmp_path, monkeypatch):
    """Crash loop / transient outage: runtime says stopped but operator intent
    is running -> stay on the retryable path."""
    _stopped_check_home(monkeypatch, tmp_path)
    (tmp_path / "gateway_state.json").write_text(
        '{"gateway_state":"startup_failed","desired_state":"running"}',
        encoding="utf-8",
    )
    assert web_server._gateway_intentionally_stopped("default") is False


def test_intentionally_stopped_false_on_legacy_file_without_desired_state(
    tmp_path, monkeypatch
):
    """Legacy gateway_state.json (no desired_state) must NOT count as
    intentional — the volatile runtime field can read 'stopped' during
    transient windows. Fail open to the retryable path."""
    _stopped_check_home(monkeypatch, tmp_path)
    (tmp_path / "gateway_state.json").write_text(
        '{"gateway_state":"stopped"}', encoding="utf-8"
    )
    assert web_server._gateway_intentionally_stopped("default") is False


def test_intentionally_stopped_false_on_missing_or_bad_file(tmp_path, monkeypatch):
    _stopped_check_home(monkeypatch, tmp_path)
    assert web_server._gateway_intentionally_stopped("default") is False
    (tmp_path / "gateway_state.json").write_text("{not json", encoding="utf-8")
    assert web_server._gateway_intentionally_stopped("default") is False
    (tmp_path / "gateway_state.json").write_text('["list"]', encoding="utf-8")
    assert web_server._gateway_intentionally_stopped("default") is False


def test_intentionally_stopped_false_when_profile_resolution_fails(monkeypatch):
    """Exception-safe: any resolution failure (unknown profile, bad name)
    returns False — never blocks the retryable path."""
    def boom(profile):
        raise RuntimeError("no such profile")

    monkeypatch.setattr(web_server, "_cron_profile_home", boom)
    assert web_server._gateway_intentionally_stopped("ghost") is False


# ── last_fire_error stamp on forward failure (missed-fire visibility) ─────


def test_forward_failure_stamps_last_fire_error(monkeypatch):
    """Gateway unreachable -> the job record gets a durable last_fire_error
    stamp (via note_fire_forward_failure) so the miss is visible in
    `cronjob list` / the dashboard instead of only in gui.log."""
    stamped = []

    async def fake_forward(profile, job_id, authorization):
        return None  # unreachable

    def fake_call_cron(profile, func_name, *args, **kwargs):
        stamped.append((profile, func_name, args))
        return True

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(web_server, "_gateway_intentionally_stopped", lambda p: False)
    monkeypatch.setattr(web_server, "_call_cron_for_profile", fake_call_cron)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j8"})
        assert resp.status_code == 503  # retry contract unchanged
        assert len(stamped) == 1
        profile, func_name, args = stamped[0]
        assert profile == "default"
        assert func_name == "note_fire_forward_failure"
        assert args[0] == "j8"
        assert "api_server" in args[1]  # detail names the dead hop
    finally:
        _restore(pa, ph)
        client.close()


def test_forward_failure_stamp_error_never_breaks_retry_contract(monkeypatch):
    """Stamping is best-effort observability: if the jobs store write blows
    up, the webhook still returns the retryable 503 + Retry-After."""

    async def fake_forward(profile, job_id, authorization):
        return None

    def boom(profile, func_name, *args, **kwargs):
        raise RuntimeError("jobs.json locked")

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(web_server, "_gateway_intentionally_stopped", lambda p: False)
    monkeypatch.setattr(web_server, "_call_cron_for_profile", boom)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j9"})
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "60"
    finally:
        _restore(pa, ph)
        client.close()


def test_reachable_gateway_does_not_stamp(monkeypatch):
    """A successful forward must not touch the job record."""
    stamped = []

    async def fake_forward(profile, job_id, authorization):
        return 202, {"status": "accepted", "job_id": job_id}

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_forward_cron_fire_to_gateway", fake_forward)
    monkeypatch.setattr(
        web_server, "_call_cron_for_profile",
        lambda *a, **k: stamped.append(a),
    )

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer nas-jwt"},
                           json={"job_id": "j10"})
        assert resp.status_code == 202
        assert stamped == []
    finally:
        _restore(pa, ph)
        client.close()
