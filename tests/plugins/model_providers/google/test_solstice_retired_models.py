"""A saved Solstice model the list no longer offers is sent as the default model, not as a failing id.

``gemini-4-argon``, ``gemini-3.1-pro`` and ``gemini-3.8-flash`` were on the old list and are not served by
the current endpoint. A user who picked one keeps it in config, so the profile maps exactly those ids (in any
spelling the client accepts) to the default model before thinking mapping and before the wire. Any other
id is left as typed.
"""

from __future__ import annotations

import json
import logging
import sys

import httpx
import pytest

import providers

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_REPLY = {"id": "i1", "status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]}
_RETIRED = ("gemini-4-argon", "gemini-3.1-pro", "gemini-3.8-flash")
_SPELLINGS = [f"{prefix}{model}" for model in _RETIRED for prefix in ("", "gemini/", "google/")]


def _profile():
    profile = providers.get_provider_profile("solstice")
    assert profile is not None
    return profile


def _default() -> str:
    return _profile().fallback_models[0]


def _send(model: str, reasoning_config=None) -> dict:
    """One turn through the real profile and client; the posted body."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    profile = _profile()
    client = profile.create_client(api_key="token", base_url=_BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    extra_body = profile.build_extra_body(model=model, reasoning_config=reasoning_config, base_url=_BASE)
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], extra_body=extra_body)
    (request,) = seen
    return json.loads(request.content)


@pytest.mark.parametrize("reasoning_config", [None, {"enabled": True, "effort": "xhigh"}, {"enabled": False}])
@pytest.mark.parametrize("model", _SPELLINGS)
def test_a_retired_model_gets_the_default_models_thinking_config(model, reasoning_config):
    profile = _profile()

    assert profile.build_extra_body(model=model, reasoning_config=reasoning_config, base_url=_BASE) == (
        profile.build_extra_body(model=_default(), reasoning_config=reasoning_config, base_url=_BASE)
    )


@pytest.mark.parametrize("model", _SPELLINGS)
def test_a_retired_model_goes_out_as_the_default_model(model):
    body = _send(model, {"enabled": True, "effort": "xhigh"})

    assert body["model"] == _default() == "gemini-4-hydrogen"
    assert body == _send(_default(), {"enabled": True, "effort": "xhigh"})


@pytest.mark.parametrize("model", _SPELLINGS)
def test_a_retired_model_goes_out_as_the_default_model_even_without_the_profiles_extra_body(model):
    """The client maps it too, so a caller that builds its own kwargs still reaches a served model."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    client = _profile().create_client(api_key="token", base_url=_BASE,
                                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])

    assert json.loads(seen[0].content)["model"] == _default()


@pytest.mark.parametrize("model", _SPELLINGS)
def test_the_profiles_model_normalization_upgrades_a_retired_model(model):
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert _profile().normalize_model_id(model) == _default()
    assert normalize_model_for_provider(model, "solstice") == _default()


@pytest.mark.parametrize("model", ["gemini-4-xenon", "gemini/gemini-4-xenon", "gemini-3.1-pro-preview", "models/gemini-3.1-pro"])
def test_any_other_id_is_left_as_typed(model):
    from agent.reasoning_effort import bare_gemini_model_id

    assert _profile().normalize_model_id(model) is None
    assert _send(model)["model"] == bare_gemini_model_id(model)


def test_the_upgrade_is_logged_once_per_process(monkeypatch, caplog):
    module = sys.modules[type(_profile()).__module__]
    monkeypatch.setattr(module, "_RETIRED_LOGGED", set())
    caplog.set_level(logging.INFO)

    for _ in range(3):
        _send("gemini-3.1-pro")

    lines = [r.getMessage() for r in caplog.records if "gemini-3.1-pro" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "gemini-3.1-pro" in lines[0] and "gemini-4-hydrogen" in lines[0]
