"""Per-model ``context_length`` override must be honored even when the caller does not
supply ``custom_providers`` (self-resolved from config).

Regression: step 0c of agent.model_metadata.get_model_context_length was gated on the
caller having pre-loaded the provider list, so auxiliary fallback screening, CLI/TUI
context-reference estimators, gateway /status and vision auto-detect — which all pass
custom_providers=None — skipped the per-model override and fell to the 256K/272K
probe-down defaults, while the startup path honored the same setting.
"""
from __future__ import annotations

from pathlib import Path

BASE_URL = "https://cp-ctx-selfresolve.invalid/v1"
MODEL = "router/auto"
OVERRIDE = 999_999  # non-power-of-two so it cannot be a catalog value


def test_override_honored_when_caller_passes_no_custom_providers(tmp_path, monkeypatch):
    """The exact failing call shape (aux/CLI/TUI/gateway /status): override wins, probe-down default never reached."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(
        "custom_providers:\n"
        "  - name: test-route\n"
        f"    base_url: {BASE_URL}\n"
        "    key_env: TEST_FAKE_CONTEXT_ENV\n"
        f"    model: {MODEL}\n"
        "    api_mode: chat_completions\n"
        "    models:\n"
        f"      {MODEL}:\n"
        f"        context_length: {OVERRIDE}\n",
        encoding="utf-8",
    )

    from agent.model_metadata import get_model_context_length

    assert get_model_context_length(MODEL, base_url=BASE_URL, api_key="", provider="custom") == OVERRIDE
