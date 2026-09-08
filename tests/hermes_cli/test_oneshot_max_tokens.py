"""Regression tests: one-shot mode must resolve the output cap like the interactive CLI.

``_run_agent`` used to build its AIAgent without ``max_tokens``, so neither
``HERMES_MAX_TOKENS`` nor (before AIAgent's own config fallback) an env-set cap ever
reached the request, and the endpoint's own default applied instead (#104485). These
tests pin ``_resolve_max_tokens`` precedence: env > ``model.max_tokens`` config >
provider-entry ``max_output_tokens``, mirroring ``gateway/run.py``.
"""

from hermes_cli.oneshot import _resolve_max_tokens


def _no_env(monkeypatch):
    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)


class TestResolveMaxTokens:
    def test_env_wins_over_config_and_runtime(self, monkeypatch):
        _no_env(monkeypatch)
        monkeypatch.setenv("HERMES_MAX_TOKENS", "25000")
        cfg = {"model": {"max_tokens": 9999}}
        assert _resolve_max_tokens(cfg, {"max_output_tokens": 1111}) == 25000

    def test_config_used_when_env_unset(self, monkeypatch):
        _no_env(monkeypatch)
        cfg = {"model": {"max_tokens": 9999}}
        assert _resolve_max_tokens(cfg, {"max_output_tokens": 1111}) == 9999

    def test_runtime_provider_entry_fills_when_global_keys_unset(self, monkeypatch):
        _no_env(monkeypatch)
        assert _resolve_max_tokens({"model": {}}, {"max_output_tokens": 1111}) == 1111

    def test_all_unset_returns_none(self, monkeypatch):
        _no_env(monkeypatch)
        assert _resolve_max_tokens({"model": {}}, {}) is None

    def test_invalid_env_falls_through_to_runtime_not_config(self, monkeypatch):
        # Gateway parity: a malformed HERMES_MAX_TOKENS is ignored (never crashes) and the
        # per-provider entry still applies; the config key stays skipped because the env
        # var was set, just unparseable.
        _no_env(monkeypatch)
        monkeypatch.setenv("HERMES_MAX_TOKENS", "not-a-number")
        cfg = {"model": {"max_tokens": 9999}}
        assert _resolve_max_tokens(cfg, {"max_output_tokens": 1111}) == 1111

    def test_non_positive_env_falls_through_to_runtime(self, monkeypatch):
        # A parseable but non-positive HERMES_MAX_TOKENS is treated as unset at that level
        # (a 0 cap would request an empty or erroring completion); like the unparseable
        # case above, the config key stays skipped and the provider entry applies.
        for raw in ("0", "-5"):
            _no_env(monkeypatch)
            monkeypatch.setenv("HERMES_MAX_TOKENS", raw)
            cfg = {"model": {"max_tokens": 9999}}
            assert _resolve_max_tokens(cfg, {"max_output_tokens": 1111}) == 1111

    def test_non_positive_config_ignored_runtime_fills(self, monkeypatch):
        # Same clamp at the config level: max_tokens: 0 is not a usable cap, so the
        # provider entry's max_output_tokens applies.
        _no_env(monkeypatch)
        cfg = {"model": {"max_tokens": 0}}
        assert _resolve_max_tokens(cfg, {"max_output_tokens": 1111}) == 1111

    def test_non_dict_model_section_ignored(self, monkeypatch):
        _no_env(monkeypatch)
        cfg = {"model": "custom-name"}
        assert _resolve_max_tokens(cfg, {"max_output_tokens": 1111}) == 1111

    def test_non_positive_runtime_entry_ignored(self, monkeypatch):
        _no_env(monkeypatch)
        assert _resolve_max_tokens({"model": {}}, {"max_output_tokens": 0}) is None
