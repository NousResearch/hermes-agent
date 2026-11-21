"""Regression for #154435: an inline ``model.api_key`` (or legacy ``model.api``) is declared for
``model.base_url``. A ``CUSTOM_BASE_URL`` pointing elsewhere is a different endpoint and must not
receive it, the rule ``model.key_env`` already follows (#67453). When that leaves the request
with no key at all, a warning names the two settings, so the endpoint's 401 has an explanation."""

import logging

import pytest

from hermes_cli.config import save_config
from hermes_cli.runtime_provider import resolve_runtime_provider

_ENDPOINT_A = "http://127.0.0.1:9001/v1"
_ENDPOINT_B = "http://127.0.0.1:9002/v1"
_KEY = "sk-declared-for-endpoint-a"
_LOGGER = "hermes_cli.runtime_provider"


@pytest.fixture(autouse=True)
def _no_ambient_endpoint_or_key(monkeypatch):
    for var in ("CUSTOM_BASE_URL", "CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(("requested", "model_block"), [
    pytest.param("custom", {"provider": "custom", "api_key": _KEY}, id="bare-custom"),
    pytest.param("custom", {"provider": "custom", "api": _KEY}, id="legacy-api-field"),
    pytest.param("auto", {"api_key": _KEY}, id="provider-unset"),
    pytest.param("vllm", {"provider": "vllm", "api_key": _KEY}, id="local-server-alias"),
])
def test_inline_model_key_does_not_follow_custom_base_url_elsewhere(monkeypatch, requested, model_block):
    save_config({"model": {"base_url": _ENDPOINT_A, "default": "m", **model_block}})
    monkeypatch.setenv("CUSTOM_BASE_URL", _ENDPOINT_B)

    resolved = resolve_runtime_provider(requested=requested)

    assert resolved["base_url"] == _ENDPOINT_B
    assert resolved["api_key"] != _KEY


@pytest.mark.parametrize("custom_base_url", [
    pytest.param(None, id="no-override"),
    pytest.param(_ENDPOINT_A, id="override-is-the-configured-url"),
    pytest.param(_ENDPOINT_A + "/", id="override-with-trailing-slash"),
])
def test_inline_model_key_still_reaches_its_configured_base_url(monkeypatch, custom_base_url):
    save_config({"model": {"provider": "custom", "base_url": _ENDPOINT_A, "api_key": _KEY, "default": "m"}})
    if custom_base_url:
        monkeypatch.setenv("CUSTOM_BASE_URL", custom_base_url)

    resolved = resolve_runtime_provider(requested="custom")

    assert (resolved["base_url"], resolved["api_key"]) == (_ENDPOINT_A, _KEY)


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize(("requested", "model_block", "setting"), [
    pytest.param("custom", {"provider": "custom", "api_key": _KEY}, "model.api_key", id="inline-key"),
    pytest.param("auto", {"api_key": _KEY}, "model.api_key", id="inline-key-provider-unset"),
    pytest.param("custom", {"provider": "custom", "key_env": "ENDPOINT_A_KEY"}, "model.key_env", id="key-env"),
])
def test_a_withheld_model_key_is_named_with_the_override_that_withheld_it(
        monkeypatch, caplog, requested, model_block, setting):
    monkeypatch.setenv("ENDPOINT_A_KEY", _KEY)
    save_config({"model": {"base_url": _ENDPOINT_A, "default": "m", **model_block}})
    monkeypatch.setenv("CUSTOM_BASE_URL", _ENDPOINT_B)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resolve_runtime_provider(requested=requested)

    named = [m for m in _warnings(caplog) if setting in m and "CUSTOM_BASE_URL" in m and "model.base_url" in m]
    assert named, _warnings(caplog)
    # Setting names only: neither the key nor the URLs reach the log.
    assert not any(secret in m for m in named for secret in (_KEY, "127.0.0.1"))


@pytest.mark.parametrize(("custom_base_url", "explicit_api_key"), [
    pytest.param(_ENDPOINT_A + "/", None, id="override-is-the-configured-url"),
    pytest.param(_ENDPOINT_B, "sk-launch-key", id="launch-key-serves-the-override"),
])
def test_no_withheld_key_warning_while_the_request_has_its_key(monkeypatch, caplog, custom_base_url, explicit_api_key):
    save_config({"model": {"provider": "custom", "base_url": _ENDPOINT_A, "api_key": _KEY, "default": "m"}})
    monkeypatch.setenv("CUSTOM_BASE_URL", custom_base_url)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resolve_runtime_provider(requested="custom", explicit_api_key=explicit_api_key)

    assert not [m for m in _warnings(caplog) if "CUSTOM_BASE_URL" in m], _warnings(caplog)
