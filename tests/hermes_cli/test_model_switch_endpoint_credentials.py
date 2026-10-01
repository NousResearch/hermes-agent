"""A key stored for one endpoint is never sent to the endpoint ``hermes model`` switches to.

Writers covered: the custom-endpoint flow (blank key), the named custom-provider flow (keyless
entry) and the Vercel AI Gateway flow. A same-endpoint re-entry with a blank key keeps the key.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.config import get_config_path, get_hermes_home

A_URL = "https://a.example.com/v1"
A_SECRET = "sk-ENDPOINT-A-MARKER"
B_URL = "http://127.0.0.1:18999/v1"


def _seed_endpoint_a(monkeypatch, key_shape: str) -> None:
    ref = {"api_key": "  api_key: ${CUSTOM_A_EXAMPLE_COM_API_KEY}\n",
           "key_env": "  key_env: CUSTOM_A_EXAMPLE_COM_API_KEY\n"}[key_shape]
    get_config_path().write_text(
        f"model:\n  provider: custom\n  default: a-model\n  base_url: {A_URL}\n{ref}")
    (get_hermes_home() / ".env").write_text(f"CUSTOM_A_EXAMPLE_COM_API_KEY={A_SECRET}\n")
    monkeypatch.setenv("CUSTOM_A_EXAMPLE_COM_API_KEY", A_SECRET)  # what the launch-time .env load does


def _run_custom_flow(monkeypatch, url: str) -> None:
    import hermes_cli.main_provider_setup as mps
    import hermes_cli.model_setup_flows_custom as flows
    import hermes_cli.secret_prompt as secret_prompt
    from hermes_cli.config import load_config

    answers = iter([url, "", "endpoint"])
    monkeypatch.setattr(flows, "line_input", lambda *a, **k: next(answers))
    monkeypatch.setattr(secret_prompt, "masked_secret_prompt", lambda *a, **k: "")
    monkeypatch.setattr(flows, "_probe_custom_endpoint", lambda key, u: ({"models": []}, u))
    monkeypatch.setattr(flows, "_pick_detected_model", lambda models: "picked-model")
    monkeypatch.setattr(flows, "_report_context_length_detection", lambda *a, **k: None)
    monkeypatch.setattr(mps, "_prompt_custom_api_mode_selection", lambda *a, **k: "")
    flows._model_flow_custom(load_config())


def _run_named_custom_flow(monkeypatch) -> None:
    from hermes_cli.model_setup_flows_custom import _model_flow_named_custom

    entry = {"name": "lab-b", "base_url": B_URL, "api_key": "", "model": "b-model", "discover_models": False}
    with patch("hermes_cli.curses_ui.curses_radiolist", side_effect=ImportError), \
            patch("builtins.input", return_value="1"):
        _model_flow_named_custom({}, entry)


def _run_ai_gateway_flow(monkeypatch) -> None:
    import hermes_cli.auth as auth
    import hermes_cli.main_provider_setup as mps
    import hermes_cli.model_setup_flows as flows
    import hermes_cli.models as models
    import hermes_cli.models_pricing as pricing

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "vck-GATEWAY-MARKER")
    monkeypatch.setattr(mps, "_prompt_api_key", lambda *a, **k: ("vck-GATEWAY-MARKER", False))
    monkeypatch.setattr(models, "ai_gateway_model_ids", lambda *a, **k: ["anthropic/claude-sonnet-4.5"])
    monkeypatch.setattr(pricing, "get_pricing_for_provider", lambda *a, **k: {})
    monkeypatch.setattr(auth, "_prompt_model_selection", lambda *a, **k: "anthropic/claude-sonnet-4.5")
    flows._model_flow_ai_gateway({}, "a-model")


def _key_sent_next_launch() -> str:
    from hermes_cli.runtime_provider import resolve_runtime_provider
    return str(resolve_runtime_provider().get("api_key") or "")


def _ai_gateway_secret() -> str:
    from hermes_cli.auth import PROVIDER_REGISTRY, _resolve_api_key_provider_secret
    return _resolve_api_key_provider_secret("ai-gateway", PROVIDER_REGISTRY["ai-gateway"])[0]


_SWITCHES = ["custom endpoint B, blank key", "named custom provider B without a key", "Vercel AI Gateway"]


@pytest.mark.parametrize(("switch", "key_shape", "keeps_a_key"), [
    *[(switch, shape, False) for switch in _SWITCHES for shape in ("api_key", "key_env")],
    ("custom endpoint A again, blank key", "api_key", True),
])
def test_stored_key_follows_the_endpoint_it_was_entered_for(monkeypatch, switch, key_shape, keeps_a_key):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    _seed_endpoint_a(monkeypatch, key_shape)

    if switch == "Vercel AI Gateway":
        _run_ai_gateway_flow(monkeypatch)
        sent = _ai_gateway_secret()
    else:
        if switch.startswith("named"):
            _run_named_custom_flow(monkeypatch)
        else:
            _run_custom_flow(monkeypatch, A_URL if keeps_a_key else B_URL)
        sent = _key_sent_next_launch()

    assert (sent == A_SECRET) is keeps_a_key, f"{switch}: next launch sends {sent!r}"
