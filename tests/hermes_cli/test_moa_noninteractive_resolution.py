"""Non-interactive MoA resolution (#56828).

``hermes chat -m moa:<preset> -Q`` and ``hermes -z`` previously sent the
literal ``moa:<preset>`` string to a real provider API (HTTP 401). These
tests pin the resolution seam: model strings / --provider moa normalize to
the MoA virtual provider with a bare preset name BEFORE
resolve_runtime_provider runs.
"""

import pytest

from hermes_cli.moa_config import resolve_moa_request

MOA_CFG = {
    "moa": {
        "default_preset": "strategy",
        "presets": {
            "strategy": {
                "reference_models": [
                    {"provider": "openai-codex", "model": "gpt-5.5"},
                ],
                "aggregator": {"provider": "deepseek", "model": "deepseek-v4-pro"},
            },
            "review": {
                "reference_models": [
                    {"provider": "deepseek", "model": "deepseek-v4-pro"},
                ],
                "aggregator": {"provider": "openai-codex", "model": "gpt-5.5"},
            },
            "retired": {
                "enabled": False,
                "reference_models": [
                    {"provider": "deepseek", "model": "deepseek-v4-pro"},
                ],
                "aggregator": {"provider": "deepseek", "model": "deepseek-v4-pro"},
            },
        },
    }
}


# -- unit: resolve_moa_request -------------------------------------------------


def test_moa_prefix_resolves_regardless_of_provider():
    assert resolve_moa_request("auto", "moa:strategy", MOA_CFG, model_explicit=True) == (
        "moa", "strategy",
    )
    assert resolve_moa_request(None, "moa/review", MOA_CFG, model_explicit=True) == (
        "moa", "review",
    )
    # explicit prefix wins over a conflicting provider (issue repro #2)
    assert resolve_moa_request(
        "openai-codex", "moa:strategy", MOA_CFG, model_explicit=True
    ) == ("moa", "strategy")
    # case-insensitive scheme
    assert resolve_moa_request("auto", "MoA:strategy", MOA_CFG, model_explicit=True) == (
        "moa", "strategy",
    )


def test_moa_prefix_unknown_preset_raises_with_available_names():
    with pytest.raises(ValueError) as exc:
        resolve_moa_request("auto", "moa:bogus", MOA_CFG, model_explicit=True)
    message = str(exc.value)
    assert "bogus" in message
    assert "strategy" in message  # names the available presets


def test_explicit_selection_reaches_disabled_preset():
    # enabled: false only guards IMPLICIT bare-name matches (#55187);
    # explicit moa: prefix / --provider moa may still select it.
    assert resolve_moa_request("auto", "moa:retired", MOA_CFG, model_explicit=True) == (
        "moa", "retired",
    )
    assert resolve_moa_request("moa", "retired", MOA_CFG, model_explicit=True) == (
        "moa", "retired",
    )


def test_provider_moa_normalizes_model():
    # bare preset name accepted as-is
    assert resolve_moa_request("moa", "review", MOA_CFG, model_explicit=True) == (
        "moa", "review",
    )
    # no model / config-inherited non-preset model -> default preset
    assert resolve_moa_request("moa", "", MOA_CFG, model_explicit=False) == (
        "moa", "strategy",
    )
    assert resolve_moa_request("moa", "claude-x", MOA_CFG, model_explicit=False) == (
        "moa", "strategy",
    )
    # explicitly requested non-preset model with provider moa -> clear error
    with pytest.raises(ValueError):
        resolve_moa_request("moa", "claude-x", MOA_CFG, model_explicit=True)


def test_non_moa_requests_pass_through():
    assert resolve_moa_request("auto", "gpt-5.5", MOA_CFG, model_explicit=True) is None
    assert resolve_moa_request("deepseek", "", MOA_CFG, model_explicit=False) is None
    # bare preset name WITHOUT provider moa is not claimed (implicit match
    # is the interactive /model path's job, not this seam's)
    assert resolve_moa_request("auto", "strategy", MOA_CFG, model_explicit=True) is None


# -- integration: oneshot ------------------------------------------------------


class _CapturedAgent:
    last_kwargs = None

    def __init__(self, **kwargs):
        _CapturedAgent.last_kwargs = kwargs
        self.suppress_status_output = False
        self.stream_delta_callback = None
        self.tool_gen_callback = None

    def run_conversation(self, prompt):
        return {"final_response": "ok"}


@pytest.fixture
def oneshot_rig(monkeypatch, tmp_path):
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({**MOA_CFG, "model": {"default": "gpt-5.5", "provider": "openai-codex"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    import run_agent as run_agent_mod
    from hermes_cli import oneshot as oneshot_mod

    # _run_agent does `from run_agent import AIAgent` inside the function.
    monkeypatch.setattr(run_agent_mod, "AIAgent", _CapturedAgent)
    _CapturedAgent.last_kwargs = None
    return oneshot_mod


def test_oneshot_moa_prefix_builds_virtual_provider(oneshot_rig):
    text, _ = oneshot_rig._run_agent("hi", model="moa:review", provider=None)
    assert text == "ok"
    kwargs = _CapturedAgent.last_kwargs
    assert kwargs["provider"] == "moa"
    assert kwargs["model"] == "review"
    assert kwargs["base_url"] == "moa://local"


def test_oneshot_provider_moa_defaults_preset(oneshot_rig):
    oneshot_rig._run_agent("hi", model=None, provider="moa")
    kwargs = _CapturedAgent.last_kwargs
    assert kwargs["provider"] == "moa"
    assert kwargs["model"] == "strategy"


def test_oneshot_unknown_preset_exits_with_preset_list(oneshot_rig, capsys):
    with pytest.raises(SystemExit):
        oneshot_rig._run_agent("hi", model="moa:bogus", provider=None)
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "bogus" in combined
    assert "strategy" in combined


def test_run_oneshot_provider_moa_defaults_preset(oneshot_rig, capsys):
    """PUBLIC path: `hermes -z --provider moa` (no --model) must reach the
    default preset instead of being rejected by the provider/model guard."""
    rc = oneshot_rig.run_oneshot("hi", provider="moa")
    assert rc == 0
    kwargs = _CapturedAgent.last_kwargs
    assert kwargs["provider"] == "moa"
    assert kwargs["model"] == "strategy"
    assert "ok" in capsys.readouterr().out


def test_run_oneshot_unknown_preset_reports_on_real_stderr(oneshot_rig, capsys):
    """PUBLIC path: unknown-preset diagnostics must reach the real stderr —
    everything inside run_oneshot's redirect goes to devnull."""
    rc = oneshot_rig.run_oneshot("hi", model="moa:bogus")
    assert rc == 2
    err = capsys.readouterr().err
    assert "bogus" in err
    assert "strategy" in err


# -- integration: classic chat mixin -------------------------------------------


def test_chat_mixin_resolves_moa_prefix(monkeypatch, tmp_path):
    import yaml

    (tmp_path / "config.yaml").write_text(yaml.safe_dump(MOA_CFG), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import cli as cli_mod

    hc = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    hc.requested_provider = "auto"
    hc.model = "moa:review"
    hc._model_is_default = False
    hc.config = MOA_CFG
    hc._explicit_api_key = None
    hc._explicit_base_url = None
    hc._fallback_model = []
    hc.api_mode = "chat_completions"
    hc.api_key = None
    hc.base_url = None
    hc.provider = None
    hc.acp_command = None
    hc.acp_args = []
    hc.agent = None

    assert hc._ensure_runtime_credentials() is True
    assert hc.requested_provider == "moa"
    assert hc.model == "review"
    assert hc.provider == "moa"
    assert hc.base_url == "moa://local"
