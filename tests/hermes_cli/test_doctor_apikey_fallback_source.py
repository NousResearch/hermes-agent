"""Fallback-variable API keys in ``hermes doctor`` provider probes (#136791).

A provider profile's ``env_vars`` chain may list another region's variable as a fallback
(``alibaba-token-plan-cn`` reads the international Token Plan key). When such a fallback key is
rejected by this profile's endpoint, the row must not demand action on a region the user never
configured: no hard ✗, no fix-list entry, and the detail names the variable the key really came
from plus the profile's own variable. A rejection of the profile's own first variable stays a
hard failure whose hint names that variable.
"""

from __future__ import annotations

import httpx

from hermes_cli import doctor_connectivity as dc

_CN_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/models"
_TP_CN_VARS = ("ALIBABA_TOKEN_PLAN_CN_API_KEY", "ALIBABA_TOKEN_PLAN_API_KEY")


def _probe_with_env(monkeypatch, env, status):
    def _get(url, headers=None, timeout=None):
        return httpx.Response(status)

    monkeypatch.setattr(httpx, "get", _get)
    for var in _TP_CN_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, val in env.items():
        monkeypatch.setenv(var, val)
    return dc._probe_apikey_provider("Alibaba Cloud (Token Plan, China)", _TP_CN_VARS, _CN_URL, None, True)


def test_fallback_key_rejected_here_warns_and_names_both_variables(monkeypatch):
    """Only the international key is set; its 401 on the China endpoint is a warning that names the
    source variable and the unset regional one, with no fix-list entry."""
    res = _probe_with_env(monkeypatch, {"ALIBABA_TOKEN_PLAN_API_KEY": "k"}, 401)
    assert "\u26a0" in res.lines[0][0]  # warn, not the hard ✗ of a region never configured
    detail = res.lines[0][2]
    assert "key from ALIBABA_TOKEN_PLAN_API_KEY" in detail
    assert "set ALIBABA_TOKEN_PLAN_CN_API_KEY" in detail
    assert res.issues == []


def test_own_key_rejected_stays_a_hard_failure_naming_its_variable(monkeypatch):
    """The profile's own first variable supplied the key; its 401 keeps the fail row and hint."""
    res = _probe_with_env(monkeypatch, {"ALIBABA_TOKEN_PLAN_CN_API_KEY": "k"}, 401)
    assert "\u2717" in res.lines[0][0]
    assert res.issues == ["Check ALIBABA_TOKEN_PLAN_CN_API_KEY in .env"]


def test_anthropic_gateway_403_with_fallback_key_warns(monkeypatch):
    """The Anthropic-style gateway's 403 with a fallback key also stays a warning (#136791)."""
    base = "https://res.services.ai.azure.com/anthropic"

    def _post(url, headers=None, params=None, json=None, timeout=None):
        return httpx.Response(403, json={"type": "error"})

    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(404))
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_FOUNDRY_FALLBACK_API_KEY", "k")
    monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", base)
    res = dc._probe_apikey_provider(
        "Azure Foundry", ("AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_FALLBACK_API_KEY"), None,
        "AZURE_FOUNDRY_BASE_URL", True)
    assert "\u26a0" in res.lines[0][0] and res.issues == []
    assert "key from AZURE_FOUNDRY_FALLBACK_API_KEY" in res.lines[0][2]
