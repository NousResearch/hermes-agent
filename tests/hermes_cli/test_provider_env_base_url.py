"""Config ``model.base_url`` participates in API-key provider base-URL resolution.

The chat path routes a configured provider through ``config.yaml``'s ``model.base_url``
(the ``_openai_discovery_endpoint`` precedence: env var → config → canonical default),
but ``_provider_env_base_url`` used to read the env var alone (with ``actual`` as the
only config-aware exception). A provider whose registry default differs from the user's
endpoint (StepFun Step Plan regional ``.com`` vs the international ``.ai`` default) then
authenticated catalog/credential probes against the wrong host: chat worked, every
picker surface showed an empty model list.
"""

from __future__ import annotations

import sys
import types

import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import _provider_env_base_url, _registry_lookup


@pytest.fixture(autouse=True)
def _no_base_url_env(monkeypatch):
    for var in ("STEPFUN_BASE_URL", "ACTUAL_BASE_URL", "GMI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _config(monkeypatch, provider: str, base_url: str) -> None:
    monkeypatch.setattr(
        "hermes_cli.auth.read_raw_config",
        lambda: {"model": {"provider": provider, "base_url": base_url}},
    )


def test_env_var_wins_over_matching_config(monkeypatch):
    pconfig = _registry_lookup("stepfun")
    _config(monkeypatch, "stepfun", "https://config.example/step_plan/v1")
    monkeypatch.setenv("STEPFUN_BASE_URL", "https://env.example/step_plan/v1")
    assert _provider_env_base_url(pconfig) == "https://env.example/step_plan/v1"


def test_matching_config_base_url_is_used_without_env(monkeypatch):
    pconfig = _registry_lookup("stepfun")
    _config(monkeypatch, "stepfun", "https://api.stepfun.com/step_plan/v1")
    assert _provider_env_base_url(pconfig) == "https://api.stepfun.com/step_plan/v1"


def test_config_provider_alias_matches(monkeypatch):
    # ``step`` is a registry alias of ``stepfun``; a config written with the alias
    # must still route the catalog probe to the configured endpoint.
    pconfig = _registry_lookup("stepfun")
    _config(monkeypatch, "step", "https://alias.example/step_plan/v1")
    assert _provider_env_base_url(pconfig) == "https://alias.example/step_plan/v1"


def test_other_providers_config_is_ignored(monkeypatch):
    pconfig = _registry_lookup("stepfun")
    _config(monkeypatch, "gmi", "https://config.example/for-gmi/v1")
    assert _provider_env_base_url(pconfig) == ""


def test_config_without_base_url_falls_through(monkeypatch):
    pconfig = _registry_lookup("stepfun")
    monkeypatch.setattr(
        "hermes_cli.auth.read_raw_config", lambda: {"model": {"provider": "stepfun"}}
    )
    assert _provider_env_base_url(pconfig) == ""


def test_actual_keeps_config_routing(monkeypatch):
    # The pre-existing ``actual`` behaviour generalised: its configured appliance
    # URL still routes discovery, and an explicit env var still wins over it.
    pconfig = _registry_lookup("actual")
    _config(monkeypatch, "actual", "https://appliance.local/v1")
    assert _provider_env_base_url(pconfig) == "https://appliance.local/v1"
    monkeypatch.setenv("ACTUAL_BASE_URL", "https://env.example/v1")
    assert _provider_env_base_url(pconfig) == "https://env.example/v1"
