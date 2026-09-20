"""Reasoning config distinguishes omission from an explicit provider default."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _StopAfterPassthrough(RuntimeError):
    pass


def test_aiagent_forwarder_preserves_omitted_vs_explicit_none(monkeypatch):
    from agent import agent_init
    from run_agent import AIAgent

    captured = []
    monkeypatch.setattr(agent_init, "init_agent", lambda _agent, **kwargs: captured.append(kwargs))

    AIAgent.__init__(AIAgent.__new__(AIAgent))
    AIAgent.__init__(AIAgent.__new__(AIAgent), reasoning_config=None)

    assert "reasoning_config" not in captured[0]
    assert captured[1]["reasoning_config"] is None


def test_init_resolves_profile_scoped_omission_but_preserves_explicit_none(
    monkeypatch, tmp_path
):
    from agent import agent_init
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_a = tmp_path / "profiles" / "alpha"
    profile_b = tmp_path / "profiles" / "beta"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    (profile_a / "config.yaml").write_text(
        "agent:\n  reasoning_effort: high\n", encoding="utf-8"
    )
    (profile_b / "config.yaml").write_text(
        "agent:\n  reasoning_effort: low\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        agent_init,
        "_resolve_api_mode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_StopAfterPassthrough()),
    )

    for profile_home, effort in (
        (profile_a, "high"),
        (profile_b, "low"),
        (profile_a, "high"),
    ):
        token = set_hermes_home_override(profile_home)
        try:
            resolved = SimpleNamespace()
            with pytest.raises(_StopAfterPassthrough):
                agent_init.init_agent(resolved, model="test-model")
        finally:
            reset_hermes_home_override(token)
        assert resolved.reasoning_config == {"enabled": True, "effort": effort}

    token = set_hermes_home_override(profile_b)
    try:
        provider_default = SimpleNamespace()
        with pytest.raises(_StopAfterPassthrough):
            agent_init.init_agent(
                provider_default, model="test-model", reasoning_config=None
            )
    finally:
        reset_hermes_home_override(token)
    assert provider_default.reasoning_config is None
