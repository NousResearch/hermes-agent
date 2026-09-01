"""Doctor validates that configured models are actually served.

The static block in ``doctor.py`` checks that ``model.provider`` names a real
provider and that slug style suits it. It never asks the provider whether the
configured model exists, and ``fallback_providers`` is not read there at all.

The gap is not theoretical. A fleet ran ``fallback_providers: [lmstudio /
hermes-4-14b]`` against an LM Studio that had been re-provisioned to serve only
gemma builds. Provider valid, credentials fine, doctor green — and every
fallback returned ``HTTP 400: Invalid model identifier``. The primary path hid
it until xAI faltered, at which point the safety net was already gone.

These assert the relationship "configured model ∈ served models" rather than
freezing any model list, so a provider adding or renaming models cannot make
them fail.
"""

import pytest

from hermes_cli import doctor_live


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _serving(*model_ids):
    """A models endpoint serving exactly ``model_ids``."""
    return lambda url, headers=None, timeout=None: _Resp(
        {"data": [{"id": m} for m in model_ids]}
    )


def _statuses(results):
    return {r.name: r.status for r in results}


# ── the reported failure ────────────────────────────────────────────────────


def test_fallback_model_absent_from_provider_fails(monkeypatch):
    """The exact reported shape: a fallback naming a model nobody serves.

    Per-host responder: the primary route is healthy, so the single failure
    proves the fallback is judged on its own provider's model list rather than
    inheriting the primary's verdict.
    """

    def _per_host(url, headers=None, timeout=None):
        if "x.ai" in url:
            return _Resp({"data": [{"id": "grok-4.6"}]})
        return _Resp({"data": [{"id": "gemma-4-e4b-it-mlx"},
                               {"id": "gemma-4-e2b-it-mlx"}]})

    monkeypatch.setattr(doctor_live, "_http_get", _per_host)
    config = {
        "model": {"provider": "xai", "default": "grok-4.6",
                  "base_url": "https://api.x.ai/v1"},
        "fallback_providers": [
            {"provider": "lmstudio", "model": "hermes-4-14b",
             "base_url": "http://127.0.0.1:1234/v1"}
        ],
    }

    results = doctor_live._probe_configured_models(config, 5.0)
    statuses = _statuses(results)

    failed = [n for n, s in statuses.items() if s == "fail"]
    assert len(failed) == 1, statuses
    assert "hermes-4-14b" in failed[0]
    assert "fallback" in failed[0]
    assert statuses["Model primary: xai/grok-4.6"] == "pass", statuses


def test_the_failure_message_names_what_is_available(monkeypatch):
    # A doctor line that says only "wrong" costs another debugging round trip.
    monkeypatch.setattr(doctor_live, "_http_get", _serving("gemma-4-e4b-it-mlx"))
    config = {
        "fallback_providers": [
            {"provider": "lmstudio", "model": "hermes-4-14b",
             "base_url": "http://127.0.0.1:1234"}
        ]
    }

    detail = doctor_live._probe_configured_models(config, 5.0)[0].detail

    assert "not served" in detail
    assert "gemma-4-e4b-it-mlx" in detail


def test_served_model_passes(monkeypatch):
    monkeypatch.setattr(doctor_live, "_http_get", _serving("grok-4.6"))
    config = {"model": {"provider": "xai", "default": "grok-4.6",
                        "base_url": "https://api.x.ai/v1"}}

    assert _statuses(doctor_live._probe_configured_models(config, 5.0)) == {
        "Model primary: xai/grok-4.6": "pass"
    }


# ── "cannot verify" must never read as "absent" ─────────────────────────────


@pytest.mark.parametrize(
    "responder",
    [
        pytest.param(
            lambda url, headers=None, timeout=None: (_ for _ in ()).throw(
                OSError("connection refused")
            ),
            id="unreachable",
        ),
        pytest.param(
            lambda url, headers=None, timeout=None: _Resp({}, status_code=401),
            id="auth_gated",
        ),
        pytest.param(
            lambda url, headers=None, timeout=None: _Resp(ValueError("not json")),
            id="unparseable",
        ),
        pytest.param(
            lambda url, headers=None, timeout=None: _Resp({"object": "list"}),
            id="not_openai_compatible",
        ),
    ],
)
def test_unverifiable_endpoints_warn_and_never_fail(monkeypatch, responder):
    """Absence must be proven, not inferred from a failed probe.

    Treating "could not ask" as "not there" would turn every offline run and
    every auth-gated provider into a red doctor, which is how a useful check
    gets switched off.
    """
    monkeypatch.setattr(doctor_live, "_http_get", responder)
    config = {"model": {"provider": "xai", "default": "grok-4.6",
                        "base_url": "https://api.x.ai/v1"}}

    statuses = list(_statuses(doctor_live._probe_configured_models(config, 5.0)).values())

    assert statuses == ["warn"]


def test_an_empty_model_list_is_unverifiable_not_absent(monkeypatch):
    # Some surfaces answer 200 with an empty list while still serving models.
    monkeypatch.setattr(doctor_live, "_http_get", _serving())
    config = {"model": {"provider": "xai", "default": "grok-4.6",
                        "base_url": "https://api.x.ai/v1"}}

    assert list(_statuses(doctor_live._probe_configured_models(config, 5.0)).values()) == [
        "warn"
    ]


# ── route enumeration ──────────────────────────────────────────────────────


def test_every_fallback_in_the_chain_is_checked(monkeypatch):
    monkeypatch.setattr(doctor_live, "_http_get", _serving("only-this"))
    config = {
        "model": {"provider": "a", "default": "m0", "base_url": "http://a/v1"},
        "fallback_providers": [
            {"provider": "b", "model": "m1", "base_url": "http://b/v1"},
            {"provider": "c", "model": "m2", "base_url": "http://c/v1"},
        ],
    }

    results = doctor_live._probe_configured_models(config, 5.0)

    assert len(results) == 3, _statuses(results)
    assert sum(1 for r in results if r.status == "fail") == 3


def test_a_provider_prefixed_slug_matches_the_bare_served_id(monkeypatch):
    # Configured as "lmstudio/gemma-4-e4b-it-mlx"; served bare. Reporting that
    # as missing would be a false alarm on a working route.
    monkeypatch.setattr(doctor_live, "_http_get", _serving("gemma-4-e4b-it-mlx"))
    config = {
        "model": {"provider": "lmstudio", "default": "lmstudio/gemma-4-e4b-it-mlx",
                  "base_url": "http://127.0.0.1:1234/v1"}
    }

    assert list(_statuses(doctor_live._probe_configured_models(config, 5.0)).values()) == [
        "pass"
    ]


def test_no_model_configured_skips(monkeypatch):
    monkeypatch.setattr(doctor_live, "_http_get", _serving("anything"))

    results = doctor_live._probe_configured_models({}, 5.0)

    assert [r.status for r in results] == ["skip"]


def test_unresolvable_endpoint_skips_rather_than_guessing(monkeypatch):
    # A fallback with no base_url and no pooled credential cannot be checked;
    # inventing a default endpoint would probe the wrong host.
    monkeypatch.setattr(doctor_live, "_http_get", _serving("anything"))
    monkeypatch.setattr(doctor_live, "_pool_base_url", lambda provider: None)
    config = {"fallback_providers": [{"provider": "lmstudio", "model": "m"}]}

    assert list(_statuses(doctor_live._probe_configured_models(config, 5.0)).values()) == [
        "skip"
    ]


def test_endpoint_comes_from_the_credential_pool_when_config_omits_it(monkeypatch):
    """The reported case carried only provider+model — the endpoint was pooled."""
    seen = {}

    def _capture(url, headers=None, timeout=None):
        seen["url"] = url
        return _Resp({"data": [{"id": "hermes-4-14b"}]})

    monkeypatch.setattr(doctor_live, "_http_get", _capture)
    monkeypatch.setattr(
        doctor_live, "_pool_base_url", lambda provider: "http://192.168.2.55:1234/v1"
    )
    config = {"fallback_providers": [{"provider": "lmstudio", "model": "hermes-4-14b"}]}

    results = doctor_live._probe_configured_models(config, 5.0)

    assert [r.status for r in results] == ["pass"]
    assert seen["url"] == "http://192.168.2.55:1234/v1/models"
