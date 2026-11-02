"""The auxiliary pricing probe carries the aux route's own credential (#146655).

Aux usage is priced through ``estimate_usage_cost``, which for an OpenAI-compatible endpoint asks
that endpoint's ``/models`` for its price table. ``record_aux_usage`` passed ``provider`` and
``base_url`` but no ``api_key``, so ``_auth_headers`` emitted no ``Authorization`` header and any
endpoint that guards ``/models`` answered the probe with 401: every auxiliary turn landed with an
empty ``estimated_cost_usd`` while the main turn — which passes ``agent.api_key`` to the very same
function — was priced. One function, two callers, one of them missing its credential.

The credential travels only with the route hints it belongs to. A record without ``base_url`` has
no route to authenticate to, and nothing here substitutes an ambient key for another host (#71410).

"""

from types import SimpleNamespace

import httpx
import pytest

from hermes_state import SessionDB

_BASE_URL = "http://127.0.0.1:59999/v1"
_ROUTE_KEY = "sk-aux-route"


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def probe_transport(monkeypatch):
    """Socket-free ``/models`` answering: records every request the real transport sends.

    Same shape as ``tests/agent/metadata_transport.py``; kept local so the fixture name cannot
    shadow its own import.
    """
    from agent import model_metadata

    responses, requests = [], []
    real_client = httpx.Client

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        **kw, transport=httpx.MockTransport(handle),
    ))
    monkeypatch.setattr(model_metadata, "detect_local_server_type", lambda *a, **kw: None)
    return responses, requests


@pytest.fixture(autouse=True)
def _clear_endpoint_caches():
    """The (base_url, api_key)-keyed memo must not leak between tests."""
    from agent import model_metadata

    caches = (model_metadata._endpoint_model_metadata_cache,
              model_metadata._endpoint_model_metadata_cache_time,
              model_metadata._endpoint_probe_path_cache)
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


def _mk_response(model="aux-m", prompt=1000, completion=500):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )


def _priced_models_payload():
    """$2/M input and $6/M output in the per-token units an OpenAI-compatible ``/models`` reports."""
    return {"data": [{"id": "aux-m", "pricing": {"prompt": "0.000002", "completion": "0.000006"}}]}


def _record(db, **hints):
    """Run one aux record through the ambient context and return the usage row it wrote."""
    from agent.aux_accounting import record_aux_usage, reset_accounting_context, set_accounting_context

    db.create_session("s1", source="cli")
    token = set_accounting_context(db, "s1")
    try:
        record_aux_usage(_mk_response(), "compression", **hints)
    finally:
        reset_accounting_context(token)
    with db._lock:
        rows = db._conn.execute(
            "SELECT task, model, estimated_cost_usd FROM session_model_usage WHERE session_id = ?",
            ("s1",),
        ).fetchall()
    return [dict(r) for r in rows]


def test_route_credential_reaches_the_pricing_probe(db, probe_transport):
    responses, requests = probe_transport
    responses.append(httpx.Response(200, json=_priced_models_payload()))

    rows = _record(db, provider="custom", base_url=_BASE_URL, api_key=_ROUTE_KEY)

    assert requests, "the pricing probe never reached the endpoint"
    assert requests[0].headers.get("authorization") == f"Bearer {_ROUTE_KEY}"
    # The price the authenticated probe returned is what lands on the row.
    assert rows[0]["estimated_cost_usd"] == pytest.approx(0.005)


def test_probe_stays_unauthenticated_when_no_key_is_given(db, probe_transport):
    """No key passed, no header invented — the pre-fix shape, kept observable as a guard."""
    responses, requests = probe_transport
    responses.append(httpx.Response(200, json=_priced_models_payload()))

    _record(db, provider="custom", base_url=_BASE_URL)

    assert requests[0].headers.get("authorization") is None


def test_hintless_fallback_record_probes_nothing(db, probe_transport):
    """Without route hints there is no route to authenticate to — and no host to leak a key to."""
    _responses, requests = probe_transport

    rows = _record(db)

    assert requests == []
    assert rows[0]["model"] == "aux-m"
    assert rows[0]["estimated_cost_usd"] == 0


def test_a_refused_pricing_probe_leaves_the_row_unpriced(db, probe_transport):
    """Authenticating is not inventing: an endpoint that still refuses gets no fabricated price."""
    responses, _requests = probe_transport
    responses.append(httpx.Response(401, json={"error": {"message": "invalid api key"}}))
    responses.append(httpx.Response(401, json={"error": {"message": "invalid api key"}}))

    rows = _record(db, provider="custom", base_url=_BASE_URL, api_key=_ROUTE_KEY)

    assert rows[0]["estimated_cost_usd"] == 0
