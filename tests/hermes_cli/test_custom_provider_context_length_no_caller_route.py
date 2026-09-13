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


def _write_custom_provider(home: str, base_url: str, model: str, context_length: int) -> None:
    cfg = (
        ("custom_providers:\n"
         f"  - name: test-route\n"
         f"    base_url: {base_url}\n"
         f"    key_env: TEST_FAKE_CONTEXT_ENV\n"
         f"    model: {model}\n"
         "    api_mode: chat_completions\n"
         f"    models:\n"
         f"      {model}:\n"
         f"        context_length: {context_length}\n")
    )
    (Path(home) / "config.yaml").write_text(cfg, encoding="utf-8")


BASE_URL = "https://cp-ctx-selfresolve.invalid/v1"
MODEL = "router/auto"
OVERRIDE = 999_999  # non-power-of-two so it cannot be a catalog value


class TestGetModelContextLengthSelfResolvesConfigOverride:
    def test_honored_when_caller_passes_no_custom_providers(self, tmp_path, monkeypatch):
        """The exact failing call shape (aux/CLI/TUI/gateway /status): override wins, probe-down default never reached."""
        import os
        from pathlib import Path as _P

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(_P, "home", lambda: tmp_path)
        _write_custom_provider(str(tmp_path), BASE_URL, MODEL, OVERRIDE)

        from agent.model_metadata import get_model_context_length
        from hermes_constants import get_hermes_home, display_hermes_home  # noqa: F401  (cache isolation sanity)

        value = get_model_context_length(
            MODEL, base_url=BASE_URL, api_key="", provider="custom"
        )
        assert value == OVERRIDE

    def test_helper_self_resolves_without_any_args(self, tmp_path, monkeypatch):
        """get_custom_provider_context_length with neither list nor config dict."""
        from pathlib import Path as _P

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(_P, "home", lambda: tmp_path)
        _write_custom_provider(str(tmp_path), BASE_URL, MODEL, OVERRIDE)

        from hermes_cli.config_providers import get_custom_provider_context_length

        value = get_custom_provider_context_length(MODEL, BASE_URL)
        assert value == OVERRIDE
        value_slash = get_custom_provider_context_length(MODEL, BASE_URL + "/")
        assert value_slash == OVERRIDE

    def test_unknown_model_on_self_resolved_route_still_returns_none(self, tmp_path, monkeypatch):
        """A model with no per-model entry gets None from the helper — resolution continues downstream."""
        from pathlib import Path as _P

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(_P, "home", lambda: tmp_path)
        _write_custom_provider(str(tmp_path), BASE_URL, MODEL, OVERRIDE)

        from hermes_cli.config_providers import get_custom_provider_context_length

        assert get_custom_provider_context_length("some/other-model", BASE_URL) is None
