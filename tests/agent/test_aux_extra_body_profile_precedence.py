"""Caller auxiliary extra_body must win over provider-profile defaults.

Regression for #107879: `_merge_aux_extra_body` used to apply profile
`body` / `reasoning_extra` *after* the caller, so keys like DeepSeek
`thinking` were silently clobbered. Overlapping keys are shallow last-wins;
profile keys the caller omitted still apply (fail-open).
"""

from agent.auxiliary_client import _ProfileProjection, _build_call_kwargs, _merge_aux_extra_body


def _proj(body=None, reasoning_extra=None, handles_reasoning=True):
    return _ProfileProjection(body or {}, reasoning_extra or {}, {}, handles_reasoning)


def test_caller_thinking_disabled_beats_profile_enabled():
    out = _merge_aux_extra_body(
        {"thinking": {"type": "disabled"}},
        _proj(reasoning_extra={"thinking": {"type": "enabled"}}),
        None,
        "deepseek",
    )
    assert out["thinking"] == {"type": "disabled"}


def test_empty_caller_keeps_profile_thinking_enabled():
    out = _merge_aux_extra_body(
        {},
        _proj(reasoning_extra={"thinking": {"type": "enabled"}}),
        None,
        "openai",
    )
    assert out["thinking"] == {"type": "enabled"}


def test_caller_thinking_enabled_beats_profile_disabled():
    out = _merge_aux_extra_body(
        {"thinking": {"type": "enabled"}},
        _proj(reasoning_extra={"thinking": {"type": "disabled"}}),
        None,
        "deepseek",
    )
    assert out["thinking"] == {"type": "enabled"}


def test_body_key_clash_caller_wins_and_omitted_profile_keys_survive():
    out = _merge_aux_extra_body(
        {"foo": "caller"},
        _proj(body={"foo": "profile", "bar": 1}),
        None,
        "x",
    )
    assert out["foo"] == "caller" and out["bar"] == 1


def test_extra_body_none_fail_open_to_profile_body():
    out = _merge_aux_extra_body(None, _proj(body={"k": 1}), None, "x")
    assert out["k"] == 1


def test_reasoning_config_fallback_when_caller_silent():
    out = _merge_aux_extra_body({}, _proj(handles_reasoning=False), {"enabled": False}, "x")
    assert out["reasoning"] == {"enabled": False}


def test_caller_extra_body_reasoning_wins_over_reasoning_config():
    out = _merge_aux_extra_body(
        {"reasoning": {"effort": "none"}},
        _proj(handles_reasoning=False),
        {"enabled": True, "effort": "high"},
        "x",
    )
    assert out["reasoning"] == {"effort": "none"}


def test_build_call_kwargs_deepseek_vision_honors_caller_thinking_disabled():
    kwargs = _build_call_kwargs(
        provider="deepseek",
        model="deepseek-v4-flash-vision-exp",
        messages=[{"role": "user", "content": "test"}],
        extra_body={"thinking": {"type": "disabled"}},
    )
    assert kwargs["extra_body"]["thinking"]["type"] == "disabled"
