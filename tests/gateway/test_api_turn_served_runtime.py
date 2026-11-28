"""Regression for #158798: an API turn without a model lock still reports the runtime that served it.

``/v1/runs`` reads the served ``{provider, model}`` pair off the turn's usage. Only a confirmed model
lock used to write it there, so an ordinary Run completed with ``runtime.provider`` and
``runtime.model`` both empty. The runtime is recorded for every API turn now, and only a confirmed
lock can fail the turn on a mismatch.
"""

from types import SimpleNamespace

from gateway.run_turn_runner import _settle_api_runtime_lock
from gateway.session_api_turn import api_execution

SERVED = {"provider": "custom", "model": "stub-model"}


def _settle(api, settings, agent=None, result=None):
    agent = agent or SimpleNamespace(**SERVED, _hermes_api_runtime={"provider": "custom"})
    result = {"final_response": "ok", "completed": True} if result is None else result
    captured = {"usage": {}}
    token = api_execution.set({"adapter": api, "settings": settings})
    try:
        _settle_api_runtime_lock(agent, result, captured)
    finally:
        api_execution.reset(token)
    return result, captured["usage"].get("runtime")


def test_an_unlocked_turn_reports_the_runtime_that_served_it(api):
    settings = {"requested_runtime": SERVED, "route_source": "raw_request"}

    result, runtime = _settle(api, settings)

    assert result == {"final_response": "ok", "completed": True}
    assert (runtime["provider"], runtime["model"]) == ("custom", "stub-model")
    assert runtime["requested"] == SERVED
    assert runtime["route_source"] == "raw_request"


def test_an_unlocked_turn_served_by_another_runtime_is_not_failed(api):
    # no lock, no refusal: the served pair is reported as it is, even where it differs from the request
    agent = SimpleNamespace(provider="custom", model="backup-model", _hermes_api_runtime={})
    settings = {"requested_runtime": SERVED, "route_source": "raw_request"}

    result, runtime = _settle(api, settings, agent=agent)

    assert result.get("failed") is not True and result["completed"] is True
    assert runtime["model"] == "backup-model"
    assert runtime["requested"]["model"] == "stub-model"


def test_a_locked_turn_on_its_own_runtime_reports_it_and_stays_completed(api):
    settings = {"confirmed_runtime_lock": True, "route": SERVED, "route_source": "raw_request"}

    result, runtime = _settle(api, settings)

    assert result["completed"] is True and "failed" not in result
    assert (runtime["provider"], runtime["model"]) == ("custom", "stub-model")
    assert runtime["model_lock"] == "confirmed"


def test_a_turn_with_no_api_admission_records_nothing():
    result, captured = {"final_response": "ok"}, {"usage": {}}

    _settle_api_runtime_lock(SimpleNamespace(**SERVED), result, captured)

    assert captured == {"usage": {}}
