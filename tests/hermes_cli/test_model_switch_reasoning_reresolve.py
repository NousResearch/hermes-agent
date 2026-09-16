"""Regression tests for the /model reasoning re-resolution (#96012).

``self.reasoning_config`` was resolved once at startup against the default
model. A mid-session ``/model`` switch updated the agent in place with the
correct per-model ``reasoning_overrides`` value, but the CLI kept the stale
startup-resolved config — and the next agent re-initialization injected it
back, clobbering the override. Providers that accept a narrower level set
(e.g. GLM-5.3-Flash: low/high/max only) then 400 on the stale level.

The switch path (``_stage_and_swap_model``, shared by the typed ``/model``
command and the picker) now re-runs the shared chokepoint
(``resolve_reasoning_config``) for the NEW model — outside the agent branch,
so the lazy-build path (switch before the first message) is covered too. An
explicit ``--reasoning`` stays authoritative for the whole run, and a failed
in-place agent swap rolls the reasoning config back with the rest of the CLI
snapshot. ``/new`` re-resolves through the same chokepoint for the model the
fresh session lands on.
"""

from __future__ import annotations

from cli import HermesCLI
from hermes_cli.model_switch import ModelSwitchResult


class _FakeModelInfo:
    context_window = 128_000
    max_output = 0

    def has_cost_data(self):
        return False

    def format_capabilities(self):
        return ""


class _StubCLI:
    """Minimum attrs ``_stage_and_swap_model`` reads on ``self`` (lazy path: no agent)."""

    agent = None
    model = "old-default-model"
    provider = ""
    requested_provider = ""
    api_key = ""
    _explicit_api_key = ""
    base_url = ""
    _explicit_base_url = ""
    api_mode = ""
    reasoning_config = {"effort": "medium"}
    _reasoning_cli_flag_applied = False


class _FailingAgent:
    def switch_model(self, **kwargs):
        raise RuntimeError("swap failed")


def _result(new_model="glm-5.3-flash"):
    return ModelSwitchResult(
        success=True,
        new_model=new_model,
        target_provider="zhipu",
        provider_changed=True,
        api_key="",
        base_url="",
        api_mode="",
        warning_message="",
        provider_label="",
        resolved_via_alias=False,
        capabilities=None,
        model_info=_FakeModelInfo(),
        is_global=False,
    )


def _stage(monkeypatch, stub):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: None)
    return HermesCLI._stage_and_swap_model(stub, _result(), "old-default-model")


def test_switch_reresolves_reasoning_for_new_model(monkeypatch):
    resolved_for: list[str] = []

    def _fake_resolve(cfg, model=""):
        resolved_for.append(model)
        return {"effort": "high"}  # the per-model override for glm-5.3-flash

    monkeypatch.setattr("hermes_constants.resolve_reasoning_config", _fake_resolve)
    stub = _StubCLI()
    # Lazy path: the switch lands before the first message (no live agent),
    # so only the CLI-level re-resolution can keep the effort correct.
    assert _stage(monkeypatch, stub) is True

    assert resolved_for == ["glm-5.3-flash"]
    assert stub.reasoning_config == {"effort": "high"}


def test_explicit_reasoning_flag_survives_switch(monkeypatch):
    def _fake_resolve(cfg, model=""):
        raise AssertionError("must not re-resolve: --reasoning is authoritative")

    monkeypatch.setattr("hermes_constants.resolve_reasoning_config", _fake_resolve)
    stub = _StubCLI()
    stub._reasoning_cli_flag_applied = True
    stub.reasoning_config = {"effort": "low"}  # from --reasoning

    assert _stage(monkeypatch, stub) is True

    assert stub.reasoning_config == {"effort": "low"}


def test_failed_agent_swap_rolls_back_reasoning_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config",
        lambda cfg, model="": {"effort": "high"},
    )
    stub = _StubCLI()
    stub.agent = _FailingAgent()

    assert _stage(monkeypatch, stub) is False

    # The swap failed → the CLI snapshot (reasoning included) is restored.
    assert stub.reasoning_config == {"effort": "medium"}
    assert stub.model == "old-default-model"


class _NewSessionStub:
    """Minimum attrs ``new_session`` reads on ``self`` (agent/db disabled)."""

    agent = None
    session_id = "old-session"
    _session_db = None
    model = "glm-5.3-flash"
    provider = "zai"
    api_key = ""
    base_url = ""
    api_mode = ""
    reasoning_config = {"enabled": True, "effort": "medium"}
    _reasoning_cli_flag_applied = False


def _config_with_override():
    return {
        "agent": {
            "reasoning_effort": "medium",
            "reasoning_overrides": {"glm-5-3-flash": "low"},
        },
        "model": {"default": "glm-5.3-flash"},
    }


def _run_new_session(monkeypatch, stub, config):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "CLI_CONFIG", config)
    monkeypatch.setattr(cli_mod, "_sync_process_session_id", lambda sid: None)
    HermesCLI.new_session(stub, silent=True)


def test_new_session_reresolves_reasoning_for_landed_model(monkeypatch):
    # /new must re-resolve through the chokepoint, not the global key only —
    # otherwise the per-model reasoning_overrides entry is lost (#96012).
    stub = _NewSessionStub()

    _run_new_session(monkeypatch, stub, _config_with_override())

    assert stub.reasoning_config == {"enabled": True, "effort": "low"}


def test_new_session_keeps_explicit_reasoning_flag(monkeypatch):
    stub = _NewSessionStub()
    stub._reasoning_cli_flag_applied = True
    stub.reasoning_config = {"enabled": True, "effort": "low"}  # --reasoning

    _run_new_session(monkeypatch, stub, _config_with_override())

    # An explicit --reasoning is authoritative for the whole run.
    assert stub.reasoning_config == {"enabled": True, "effort": "low"}


def test_new_session_reresolves_after_model_reset(monkeypatch):
    # Session had switched away from the config default; /new resets the
    # model back to it and must resolve reasoning for THAT model.
    stub = _NewSessionStub()
    stub.model = "session-switched-model"

    def _fake_switch_model(**kwargs):
        return ModelSwitchResult(
            success=True,
            new_model="glm-5.3-flash",
            target_provider="zai",
            provider_changed=True,
            api_key="",
            base_url="",
            api_mode="",
            warning_message="",
            provider_label="",
            resolved_via_alias=False,
            capabilities=None,
            model_info=_FakeModelInfo(),
            is_global=False,
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch_model)

    _run_new_session(monkeypatch, stub, _config_with_override())

    assert stub.model == "glm-5.3-flash"
    assert stub.reasoning_config == {"enabled": True, "effort": "low"}
