"""R3 every final cap alias is an exact positive integer before equality."""
import json

import httpx
import pytest
from openai import OpenAI

import agent.final_wire_admission as admission
from tests.agent.test_final_wire_review_regressions import identity
from tests.agent.test_final_wire_current_stream import Pool, body, replace_stream


BAD = [pytest.param(True, id="bool"), pytest.param(1.0, id="integral_float"),
       pytest.param("1", id="string"), pytest.param(None, id="null"),
       pytest.param(0, id="zero"), pytest.param(-1, id="negative"),
       pytest.param(1.5, id="fractional_float"), pytest.param(float("nan"), id="nan"),
       pytest.param(float("inf"), id="inf"), pytest.param([], id="array"), pytest.param({}, id="object")]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("bad_field", ["max_tokens", "max_completion_tokens"])
def test_every_alias_type_precedes_equality(bad, bad_field):
    final = body("hi")
    final.update(max_tokens=1, max_completion_tokens=1)
    final[bad_field] = bad
    snapshot = admission.project_final_body(final, identity())
    assert snapshot.reservation_state == admission.INVALID_RESERVATION
    assert snapshot.resolved_r is None
    with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundInvalidAccounting):
        admission.admit_final_json(final)


@pytest.mark.parametrize("bad", BAD[:7] + BAD[9:])
@pytest.mark.parametrize("bad_field", ["max_tokens", "max_completion_tokens"])
def test_sdk_final_alias_merge_refuses_zero_no_sleep(monkeypatch, bad, bad_field):
    pool = Pool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    final = []
    def hook(request):
        final.append(json.loads(request.content))
    with httpx.Client(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=identity().endpoint,
                                                        http_client=http, max_retries=2)
        with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundInvalidAccounting):
            sdk.chat.completions.create(**body("hi"), extra_body={"max_completion_tokens": 1, bad_field: bad})
        assert len(final) == 1
        assert type(final[0][bad_field]) is type(bad)
        assert pool.sent == []
        assert sleeps == []
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


@pytest.mark.parametrize("bad", BAD[7:9])
@pytest.mark.parametrize("bad_field", ["max_tokens", "max_completion_tokens"])
def test_nonfinite_post_sdk_final_stream_cap_typed_refusal(monkeypatch, bad, bad_field):
    # SDK rejects nonfinite builder values during JSON serialization. Reach
    # the final seam with a public hook instead; do not patch the dependency.
    pool = Pool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    def hook(request):
        final = body("hi")
        final.update(max_tokens=1, max_completion_tokens=1)
        final[bad_field] = bad
        replace_stream(request, final)
    with httpx.Client(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=identity().endpoint,
                                                        http_client=http, max_retries=2)
        with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundInvalidAccounting):
            sdk.chat.completions.create(**body("hi"))
    assert pool.sent == []
    assert sleeps == []
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("other", [1, 2])
def test_integer_aliases_reconcile_without_double_reservation(reverse, other):
    aliases = [("max_tokens", 1), ("max_completion_tokens", other)]
    if reverse:
        aliases.reverse()
    final = {"model": identity().model, "messages": body("hi")["messages"], **dict(aliases)}
    snap = admission.project_final_body(final, identity())
    if other == 2:
        assert snap.reservation_state == admission.INVALID_RESERVATION
        with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundInvalidAccounting):
            admission.admit_final_json(final)
    else:
        assert snap.reservation_state == admission.KNOWN_R
        assert snap.resolved_r == 1
        equality = admission.FinalAttemptIdentity(admission.COVERED_MAIN, "chat_completions", identity().model,
                                                 identity().endpoint, snap.estimated_input + 1, "equal")
        with admission.bind_attempt_identity(equality):
            admission.admit_final_json(final)


@pytest.mark.parametrize("other", [1, 2])
def test_sdk_equal_and_contradictory_integer_alias_controls(monkeypatch, other):
    pool = Pool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    with httpx.Client(transport=transport) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=identity().endpoint,
                                                        http_client=http, max_retries=2)
        with admission.bind_attempt_identity(identity()):
            if other == 2:
                with pytest.raises(admission.ProviderBoundInvalidAccounting):
                    sdk.chat.completions.create(**body("hi"), extra_body={"max_completion_tokens": other})
            else:
                sdk.chat.completions.create(**body("hi"), extra_body={"max_completion_tokens": other})
    assert len(pool.sent) == (1 if other == 1 else 0)
    assert sleeps == []
