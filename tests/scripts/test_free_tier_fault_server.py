"""``scripts/free_tier_fault_server.py`` speaks the real free-tier wire contract: Hermes' own client
code, pointed at it, reaches the same ``anon_*`` codes and welcome refusals it reaches against
production. That is what makes a rehearsal against it worth anything."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "free_tier_fault_server.py"


def _load():
    spec = importlib.util.spec_from_file_location("free_tier_fault_server", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server():
    module = _load()
    srv = module.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    host, port = srv.server_address
    base = f"http://{host}:{port}"
    try:
        yield module, base
    finally:
        srv.shutdown()
        srv.server_close()


def _post(base: str, path: str, body=None, **kw) -> httpx.Response:
    return httpx.post(f"{base}{path}", json=body or {}, timeout=5.0, **kw)


class TestControlSurface:
    def test_catalogue_switches_and_reset(self, server):
        module, base = server
        catalogue = httpx.get(f"{base}/__scenarios", timeout=5.0).json()
        assert set(catalogue["nas"]) == set(module.NAS_SCENARIOS)
        assert set(catalogue["inference"]) == set(module.INFERENCE_SCENARIOS)

        state = _post(base, "/__scenario", {"inference": "rate_limited", "retry_after": 42}).json()
        assert state["inference"] == "rate_limited" and state["retry_after"] == 42
        assert httpx.get(f"{base}/__scenario?nas=paused", timeout=5.0).json()["nas"] == "paused"
        assert _post(base, "/__scenario", {"nas": "nope"}).status_code == 400

        assert _post(base, "/__reset").json() == {**_post(base, "/__reset").json(), "nas": "ok", "inference": "ok"}

    def test_the_control_surface_is_cors_open_for_a_renderer(self, server):
        _module, base = server
        preflight = httpx.options(f"{base}/__scenario", timeout=5.0)
        assert preflight.status_code == 204
        assert preflight.headers["access-control-allow-origin"] == "*"
        assert "POST" in preflight.headers["access-control-allow-methods"]

    def test_once_drops_back_to_the_happy_path_after_one_failure(self, server):
        _module, base = server
        _post(base, "/__scenario", {"inference": "upstream_503", "once": True})
        assert _post(base, "/v1/chat/completions", {"model": "nous/welcome", "messages": []}).status_code == 503
        assert _post(base, "/v1/chat/completions", {"model": "nous/welcome", "messages": []}).status_code == 200


class TestNasContract:
    def test_the_happy_path_mints_and_exchanges(self, server):
        _module, base = server
        minted = _post(base, "/api/anonymous/create").json()
        assert minted["token"].startswith("anon_") and minted["idle_ttl_days"] == 14
        exchanged = _post(base, "/api/anonymous/token", {"token": minted["token"]}).json()
        assert exchanged["access_token"].count(".") == 2 and exchanged["expires_in"] == 900
        assert exchanged["inference_base_url"] == f"{base}/v1"

    @pytest.mark.parametrize("scenario,status,error", [
        ("not_enabled", 404, "not_found"),
        ("paused", 503, "temporarily_disabled"),
        ("rate_limited", 429, "temporarily_unavailable"),
    ])
    def test_gate_and_limit_refusals_match_the_service(self, server, scenario, status, error):
        _module, base = server
        _post(base, "/__scenario", {"nas": scenario})
        response = _post(base, "/api/anonymous/create")
        assert response.status_code == status and response.json()["error"] == error
        if status == 429:
            assert int(response.headers["Retry-After"]) > 0

    def test_exchange_refusals_match_the_service(self, server):
        _module, base = server
        token = _post(base, "/api/anonymous/create").json()["token"]
        for scenario, status, error in (("pow", 428, "pow_required"), ("locked", 403, "account_locked")):
            _post(base, "/__scenario", {"nas": scenario})
            response = _post(base, "/api/anonymous/token", {"token": token})
            assert response.status_code == status and response.json()["error"] == error
        _post(base, "/__scenario", {"nas": "dead_once"})
        assert _post(base, "/api/anonymous/token", {"token": token}).json()["error"] == "unknown_token"
        fresh = _post(base, "/api/anonymous/create").json()["token"]
        assert _post(base, "/api/anonymous/token", {"token": fresh}).status_code == 200

    def test_hermes_own_client_reaches_the_same_codes_against_it(self, server, monkeypatch, tmp_path):
        """The point of the stand-in: ``ensure_portal_identity`` classifies its answers exactly as
        it classifies production's."""
        from hermes_cli import anon_auth, free_tier_bootstrap
        _module, base = server
        monkeypatch.setenv("HERMES_PORTAL_BASE_URL", base)
        monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
        monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
        anon_auth.reset_mint_memo_for_tests()
        free_tier_bootstrap.reset_for_tests()

        _post(base, "/__scenario", {"nas": "paused"})
        with pytest.raises(anon_auth.AuthError) as exc:
            anon_auth.ensure_portal_identity(explicit=True)
        assert exc.value.code == anon_auth.ANON_GATE_PAUSED

        _post(base, "/__scenario", {"nas": "ok"})
        state = anon_auth.ensure_portal_identity(explicit=True, force=True)
        assert anon_auth.is_guest_state(state)


class TestInferenceContract:
    def test_the_happy_path_answers_streaming_and_plain(self, server):
        _module, base = server
        plain = _post(base, "/v1/chat/completions", {"model": "nous/welcome", "messages": []}).json()
        assert plain["choices"][0]["message"]["content"]
        streamed = _post(base, "/v1/chat/completions", {"model": "nous/welcome", "messages": [], "stream": True})
        assert streamed.headers["content-type"].startswith("text/event-stream")
        assert streamed.text.rstrip().endswith("data: [DONE]")

    @pytest.mark.parametrize("scenario,reason,retry_after", [
        ("rate_limited", "rate_limited", 600), ("rate_limited_short", "rate_limited", 5),
        ("at_capacity", "at_capacity", 30), ("model_not_free", "model_not_free", 0),
    ])
    def test_structured_refusals_parse_as_welcome_refusals(self, server, scenario, reason, retry_after):
        from hermes_cli.anon_auth import parse_welcome_refusal
        _module, base = server
        _post(base, "/__scenario", {"inference": scenario})
        response = _post(base, "/v1/chat/completions", {"model": "gpt-5", "messages": []})
        assert response.status_code == 429
        refusal = parse_welcome_refusal(response.json())
        assert refusal["reason"] == reason and refusal["retry_after"] == retry_after
        if reason == "model_not_free":
            assert refusal["alternates"] == ["nous/welcome"]

    def test_route_refusals_and_outages_match_the_gateway(self, server):
        from hermes_cli.anon_auth import welcome_route_refusal
        _module, base = server
        _post(base, "/__scenario", {"inference": "wrong_host"})
        response = _post(base, "/v1/chat/completions", {"model": "nous/welcome", "messages": []})
        assert welcome_route_refusal(response.status_code, response.json()["message"]) == "anon_on_paid_host"
        _post(base, "/__scenario", {"inference": "tier_disabled"})
        response = _post(base, "/v1/chat/completions", {"model": "nous/welcome", "messages": []})
        assert response.status_code == 403 and "reason" not in response.json()
        assert welcome_route_refusal(403, response.json()["message"], f"{base}/v1") is None  # not a welcome host...
        _post(base, "/__scenario", {"inference": "bare_429"})
        response = _post(base, "/v1/chat/completions", {"model": "nous/welcome", "messages": []})
        assert response.headers["x-ratelimit-remaining-requests"] == "0"
        for scenario, status in (("upstream_503", 503), ("upstream_500", 500), ("invalid_token", 401)):
            _post(base, "/__scenario", {"inference": scenario})
            assert _post(base, "/v1/chat/completions", {"model": "nous/welcome", "messages": []}).status_code == status

    def test_the_dev_host_override_makes_the_stand_in_the_welcome_host(self, server, monkeypatch):
        from hermes_cli.anon_auth import route_is_welcome_host, welcome_route_refusal
        _module, base = server
        assert route_is_welcome_host(f"{base}/v1") is False
        monkeypatch.setenv("HERMES_EXTRA_WELCOME_HOSTS", "127.0.0.1, localhost")
        assert route_is_welcome_host(f"{base}/v1") is True
        assert route_is_welcome_host("http://localhost:9/v1") is True
        # ...and with it, the generic 403 reads as the tier refusing, exactly as in production.
        assert welcome_route_refusal(403, "You tried to access something", f"{base}/v1") == "tier_disabled"
        assert route_is_welcome_host("https://welcome-api.nousresearch.com/v1") is True
        assert route_is_welcome_host("https://inference-api.nousresearch.com/v1") is False


def test_the_script_runs_as_a_program():
    import subprocess
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "--inference" in result.stdout and "rate_limited" in result.stdout
    assert json.dumps(sorted(_load().INFERENCE_SCENARIOS))  # the catalogue is JSON-serialisable
