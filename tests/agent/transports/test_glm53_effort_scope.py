"""The GLM-5.3 ``medium``→``high`` clamp is scoped to Z.AI's own *hosts*.

Measured 2026-11-20 against ``api.z.ai`` with a live key: ``reasoning_effort: "medium"`` is
refused with HTTP 400 and body code 1210 ("This model always engages in thinking and cannot be
disabled; please use low, high, or max") for ``glm-5.3``, ``glm-5.3-flash`` and
``glm-5.3-flashx``, on both ``/api/paas/v4`` and ``/api/coding/paas/v4``.

Measured 2026-10-03 against the Nous Portal reseller: ``z-ai/glm-5.3-prime`` with
``reasoning_effort: "medium"`` answered 200 with 63 reasoning tokens
(``reports/effort-knob-2026-10-03/probe-glm-5.3-prime.json``).

So the clamp belongs to the host, not the model name: clamping a reseller route would silently
buy more thinking tokens than the caller asked for. And the host test must read the *hostname* —
a URL that merely contains the marker is a different machine.
"""

from __future__ import annotations

import pytest

from agent.reasoning_effort import clamp_glm53_effort, is_zai_first_party_host
from agent.transports.chat_completions import _reasoning_config_for_model

ZAI_STANDARD = "https://api.z.ai/api/paas/v4"
ZAI_CODING = "https://api.z.ai/api/coding/paas/v4"
ZHIPU = "https://open.bigmodel.cn/api/paas/v4"
RESELLER = "https://inference-api.nousresearch.com/v1"


def _medium() -> dict:
    return {"enabled": True, "effort": "medium"}


@pytest.mark.parametrize("base_url", [ZAI_STANDARD, ZAI_CODING, ZHIPU])
@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"])
def test_first_party_hosts_upgrade_medium_to_high(base_url, model):
    """Every first-party endpoint refuses medium, so it must never reach the wire."""
    assert _reasoning_config_for_model(model, _medium(), base_url)["effort"] == "high"


def test_reseller_keeps_medium():
    """A reseller of the same model accepted medium, so the request stands as asked."""
    assert _reasoning_config_for_model("glm-5.3", _medium(), RESELLER)["effort"] == "medium"


def test_no_base_url_keeps_medium():
    """With no endpoint declared there is no evidence of Z.AI's vocabulary; do not guess."""
    assert _reasoning_config_for_model("glm-5.3", _medium(), None)["effort"] == "medium"


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_supported_efforts_pass_through_on_first_party(effort):
    cfg = {"enabled": True, "effort": effort}
    assert _reasoning_config_for_model("glm-5.3", cfg, ZAI_STANDARD)["effort"] == effort


def test_glm_5_2_is_not_touched_by_the_5_3_clamp():
    """5.2 has its own two-rung wire, handled by the provider profile."""
    assert _reasoning_config_for_model("glm-5.2", _medium(), ZAI_STANDARD)["effort"] == "medium"


def test_disabled_reasoning_is_left_alone():
    """The clamp rewrites an effort level only; it must not switch thinking back on."""
    cfg = {"enabled": False, "effort": "medium"}
    assert _reasoning_config_for_model("glm-5.3", cfg, ZAI_STANDARD)["enabled"] is False


@pytest.mark.parametrize(
    "base_url",
    [ZAI_STANDARD, ZAI_CODING, ZHIPU, "https://z.ai/api/paas/v4", "https://API.Z.AI/api/paas/v4",
     "https://api.z.ai./api/paas/v4", "api.z.ai"],
)
def test_first_party_spellings_are_recognised(base_url):
    assert is_zai_first_party_host(base_url) is True


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.z.ai.attacker.example/v1",
        "https://api.z.ai@attacker.example/v1",
        "https://user:pw@api.z.ai.attacker.example/v1",
        "https://attacker.example/?next=api.z.ai",
        "https://attacker.example/api.z.ai/v1",
        "https://open.bigmodel.cn.api.attacker.example/v1",
        "https://notz.ai/v1",
        RESELLER,
        "",
        None,
    ],
)
def test_lookalike_and_foreign_hosts_are_not_first_party(base_url):
    """A URL that merely mentions the marker is a different machine — never clamp its route."""
    assert is_zai_first_party_host(base_url) is False
    cfg = clamp_glm53_effort("glm-5.3", _medium(), base_url)
    assert cfg is not None and cfg["effort"] == "medium"


def test_auxiliary_and_moa_paths_apply_the_same_clamp():
    """An aux/MoA task to Z.AI must not send the level the host refuses (#112010 family)."""
    from agent.auxiliary_client import _build_call_kwargs

    kwargs = _build_call_kwargs(
        provider="zai",
        model="glm-5.3",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_config=_medium(),
        base_url=ZAI_STANDARD,
        task="moa_reference",
    )
    assert kwargs.get("reasoning_effort") == "high"


def test_auxiliary_reseller_path_keeps_medium():
    from agent.auxiliary_client import _build_call_kwargs

    kwargs = _build_call_kwargs(
        provider="nous",
        model="z-ai/glm-5.3-prime",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_config=_medium(),
        base_url=RESELLER,
        task="moa_reference",
    )
    # The reseller wire carries the level inside extra_body.reasoning; assert the level itself, not
    # merely "not high" — a route that stopped sending anything would pass that weaker check.
    assert str(kwargs.get("reasoning_effort") or "").lower() != "high"
    assert kwargs["extra_body"]["reasoning"]["effort"] == "medium"
