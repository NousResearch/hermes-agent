"""Doctor validates that configured models are actually served.

The static block in ``doctor.py`` checks that ``model.provider`` names a real
provider and that slug style suits it. It never asks the provider whether the
configured model exists, and ``fallback_providers`` is not read there at all.
A fleet ran ``fallback_providers: [lmstudio / hermes-4-14b]`` against an LM
Studio re-provisioned to serve only gemma builds: provider valid, credentials
fine, doctor green, and every fallback returned ``HTTP 400``.

These assert the relationship "configured model ∈ served models" rather than
freezing any model list, so a provider adding or renaming models cannot make
them fail.
"""

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from hermes_cli import doctor_live


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch):
    for var in ("LM_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _pooled(provider, *, token, base_url=None, auth_type="api_key"):
    """A real pool holding one credential — what ``load_pool(provider)`` returns."""
    entry = PooledCredential(
        provider=provider, id="c1", label="pooled", auth_type=auth_type,
        priority=0, source="manual", access_token=token, base_url=base_url,
    )
    return CredentialPool(provider, [entry])


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _serving(*model_ids, status_code=200, payload=None):
    """A models endpoint serving exactly ``model_ids`` (or a fixed payload)."""
    body = payload if payload is not None else {"data": [{"id": m} for m in model_ids]}
    return lambda url, headers=None, timeout=None: _Resp(body, status_code=status_code)


def _raising(exc):
    def _get(url, headers=None, timeout=None):
        raise exc
    return _get


def _statuses(results):
    return {r.name: r.status for r in results}


XAI_PRIMARY = {"model": {"provider": "xai", "default": "grok-4.6", "base_url": "https://api.x.ai/v1"}}
LMS_FALLBACK = {"fallback_providers": [
    {"provider": "lmstudio", "model": "hermes-4-14b", "base_url": "http://127.0.0.1:1234/v1"}]}


def _per_host(url, headers=None, timeout=None):
    if "x.ai" in url:
        return _Resp({"data": [{"id": "grok-4.6"}]})
    return _Resp({"data": [{"id": "gemma-4-e4b-it-mlx"}, {"id": "gemma-4-e2b-it-mlx"}]})


@pytest.mark.parametrize(
    ("config", "responder", "expected", "detail"),
    [
        (XAI_PRIMARY, _serving("grok-4.6"), ["pass"], None),
        # the reported shape: a healthy primary and a fallback nobody serves, judged on its OWN
        # provider's list, with the failure naming what is available
        ({**XAI_PRIMARY, **LMS_FALLBACK}, _per_host, ["pass", "fail"], "not served by lmstudio — available: gemma-4-e2b-it-mlx, gemma-4-e4b-it-mlx"),
        ({"model": {"provider": "lmstudio", "default": "lmstudio/gemma-4-e4b-it-mlx", "base_url": "http://127.0.0.1:1234/v1"}},
         _serving("gemma-4-e4b-it-mlx"), ["pass"], None),  # configured "provider/model", served bare
        ({"model": {"provider": "a", "default": "m0", "base_url": "http://a/v1"},
          "fallback_providers": [{"provider": "b", "model": "m1", "base_url": "http://b/v1"},
                                 {"provider": "c", "model": "m2", "base_url": "http://c/v1"}]},
         _serving("only-this"), ["fail", "fail", "fail"], None),
        # "could not ask" must never read as "absent": every offline run and auth-gated provider
        # would otherwise turn doctor red, which is how a useful check gets switched off
        (XAI_PRIMARY, _raising(OSError("connection refused")), ["warn"], "unreachable (OSError)"),
        (XAI_PRIMARY, _serving(status_code=401, payload={}), ["warn"], "HTTP 401"),
        (XAI_PRIMARY, _serving(payload=ValueError("not json")), ["warn"], "unparseable body"),
        (XAI_PRIMARY, _serving(payload={"object": "list"}), ["warn"], "no model list"),
        (XAI_PRIMARY, _serving(), ["warn"], "empty model list"),
        ({}, _serving("anything"), ["skip"], "no model configured"),
        ({"fallback_providers": [{"provider": "lmstudio", "model": "m"}]}, _serving("anything"), ["skip"], "no endpoint resolvable"),
    ],
    ids=["served", "fallback-absent", "prefixed-slug", "every-fallback-checked", "unreachable", "auth-gated",
         "unparseable", "not-openai-compatible", "empty-list", "no-model-configured", "no-endpoint"],
)
def test_a_route_fails_only_when_its_provider_provably_does_not_serve_the_model(monkeypatch, config, responder, expected, detail):
    monkeypatch.setattr(doctor_live, "_http_get", responder)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: CredentialPool(provider, []))

    results = doctor_live._probe_configured_models(config, 5.0)

    assert [r.status for r in results] == expected, _statuses(results)
    if detail:
        assert detail in results[-1].detail
    if detail and detail.startswith("not served"):
        assert "hermes-4-14b" in results[-1].name and "fallback" in results[-1].name


def _capturing(seen, *model_ids, status_code=200):
    def _get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = dict(headers or {})
        seen["calls"] = seen.get("calls", 0) + 1
        return _Resp({"data": [{"id": m} for m in model_ids]}, status_code=status_code)
    return _get


LMS_ROUTE = {"provider": "lmstudio", "model": "gemma-4-e4b-it-mlx", "base_url": "http://127.0.0.1:1234/v1"}


@pytest.mark.parametrize(
    ("config", "pool", "env", "stale", "status_code", "expected", "url", "bearer", "detail"),
    [
        # the reported case carried only provider+model: the endpoint is the pooled credential's
        ({"fallback_providers": [{"provider": "lmstudio", "model": "hermes-4-14b"}]},
         ("lm-secret", "http://192.168.2.55:1234/v1"), {}, False, 200, "pass", "http://192.168.2.55:1234/v1/models", "lm-secret", None),
        ({"model": {"provider": "xai-oauth", "default": "grok-4.6"}},
         ("xai-oauth-jwt", "https://api.x.ai/v1"), {}, False, 200, "pass", "https://api.x.ai/v1/models", "xai-oauth-jwt", None),
        # the runtime's order: the route's own key, then the pool, then the provider's env vars
        ({"fallback_providers": [{**LMS_ROUTE, "base_url": "http://route:1234/v1", "api_key": "route-key"}]},
         ("pooled-key", "http://pool:1234/v1"), {}, False, 200, "pass", "http://route:1234/v1/models", "route-key", None),
        ({"fallback_providers": [{**LMS_ROUTE, "key_env": "MY_LMS_KEY"}]},
         None, {"MY_LMS_KEY": "from-env"}, False, 200, "pass", "http://127.0.0.1:1234/v1/models", "from-env", None),
        ({"model": {"provider": "lmstudio", "default": "gemma-4-e4b-it-mlx", "base_url": "http://127.0.0.1:1234/v1"}},
         None, {"LM_API_KEY": "lm-env"}, False, 200, "pass", "http://127.0.0.1:1234/v1/models", "lm-env", None),
        ({"fallback_providers": [LMS_ROUTE]}, None, {}, False, 401, "warn", "http://127.0.0.1:1234/v1/models", None, "no credential resolved"),
        # a rejected key: the truth about the model list is unknown, never "absent"
        ({"model": {"provider": "xai-oauth", "default": "grok-4.6"}},
         ("stale", "https://api.x.ai/v1"), {}, False, 401, "warn", "https://api.x.ai/v1/models", "stale", "HTTP 401, with the resolved credential"),
        # a pooled OAuth token the runtime would refresh first: sending it reads as a bad
        # credential, refreshing it from doctor could rotate a single-use grant, so no request
        ({"model": {"provider": "xai-oauth", "default": "grok-4.6"}},
         ("expired-jwt", "https://api.x.ai/v1"), {}, True, 200, "warn", None, None, "due for refresh"),
        ({"fallback_providers": [{"provider": "xai-oauth", "model": "gemma-4-e4b-it-mlx", "api_key": "route-key"}]},
         ("expired-jwt", "https://api.x.ai/v1"), {}, True, 200, "pass", "https://api.x.ai/v1/models", "route-key", None),
    ],
    ids=["endpoint-from-pool", "pooled-bearer", "route-key-beats-pool", "route-key-env", "provider-env-var",
         "no-credential", "rejected-credential", "stale-pooled-oauth-not-probed", "route-key-probes-despite-stale-pool"],
)
def test_the_probe_sends_the_credential_the_runtime_would(monkeypatch, config, pool, env, stale, status_code, expected, url, bearer, detail):
    seen = {}
    monkeypatch.setattr(doctor_live, "_http_get", _capturing(seen, "gemma-4-e4b-it-mlx", "hermes-4-14b", "grok-4.6", status_code=status_code))
    monkeypatch.setattr("agent.credential_pool.load_pool",
                        (lambda provider: _pooled(provider, token=pool[0], base_url=pool[1], auth_type="oauth" if stale else "api_key"))
                        if pool else (lambda provider: CredentialPool(provider, [])))
    monkeypatch.setattr(CredentialPool, "_entry_needs_refresh", lambda self, entry: stale)
    for var, value in env.items():
        monkeypatch.setenv(var, value)

    [result] = doctor_live._probe_configured_models(config, 5.0)

    assert result.status == expected, result.detail
    assert seen.get("url") == url
    assert seen.get("headers", {}).get("Authorization") == (f"Bearer {bearer}" if bearer else None)
    if detail:
        assert detail in result.detail


def test_run_live_checks_actually_runs_the_configured_model_probe(monkeypatch):
    """The probe must be WIRED INTO ``run_live_checks``: every other test here calls
    ``_probe_configured_models`` directly and stays green if the call site disappears."""
    sentinel = doctor_live.ProbeResult("Model: sentinel", "fail", "(wired)")
    seen = {}

    def _fake(config, timeout):
        seen["called"] = True
        seen["timeout"] = timeout
        return [sentinel]

    monkeypatch.setattr(doctor_live, "_probe_configured_models", _fake)
    # run_live_checks reads config through hermes_cli.config (a local import on current main;
    # a module-level _load_config on older trees), so patch both.
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    monkeypatch.setattr(doctor_live, "_load_config", lambda: {}, raising=False)
    monkeypatch.setattr(doctor_live, "_run_one",
                        lambda name, fn, issues: doctor_live.ProbeResult(name, "skip", "(stubbed)"))
    monkeypatch.setattr(doctor_live, "_section", lambda *a, **k: None)
    monkeypatch.setattr(doctor_live, "_report", lambda *a, **k: None)

    results = doctor_live.run_live_checks([])

    assert seen.get("called"), "run_live_checks never called _probe_configured_models"
    assert sentinel in results, "the probe's results are dropped instead of reported"
    assert seen["timeout"] == doctor_live.DEFAULT_PROBE_TIMEOUT
