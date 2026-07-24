"""Tests for the probe-verified model-limit override layer in models_dev.

``models.dev`` is a community catalog that under-reports context/output limits
for several providers (notably ``github-copilot``). ``_PROBE_VERIFIED_OVERRIDES``
plus the ``_resolve_probe_override`` matcher correct those numbers. These tests
exercise the matcher's normalization (alias providers, dot/dash version
equivalence, vendor-prefix stripping) and the override-merge behavior in
``get_model_info`` without touching the network.
"""

from dataclasses import replace

import agent.models_dev as md


class TestResolveProbeOverride:
    def test_exact_match_github_copilot_opus(self):
        o = md._resolve_probe_override("github-copilot", "claude-opus-4.8")
        assert o == {"context_window": 1_000_000, "max_output": 128_000}

    def test_provider_alias_copilot_to_github_copilot(self):
        # "copilot" is a Hermes provider id; it must normalize to github-copilot.
        a = md._resolve_probe_override("copilot", "claude-opus-4.8")
        b = md._resolve_probe_override("github-copilot", "claude-opus-4.8")
        assert a == b == {"context_window": 1_000_000, "max_output": 128_000}

    def test_dash_version_equivalence(self):
        # Anthropic SDK shape uses dash versions (claude-opus-4-8); the table is
        # keyed on dot form. The matcher must reconcile them.
        dot = md._resolve_probe_override("github-copilot", "claude-opus-4.8")
        dash = md._resolve_probe_override("github-copilot", "claude-opus-4-8")
        assert dot == dash

    def test_vendor_prefix_stripped(self):
        # "anthropic/claude-opus-4.7" → "claude-opus-4.7".
        o = md._resolve_probe_override("anthropic", "anthropic/claude-opus-4.7")
        assert o == {"context_window": 1_000_000, "max_output": 128_000}

    def test_gpt5_codex_backend_numbers(self):
        assert md._resolve_probe_override("openai-codex", "gpt-5.5") == {
            "context_window": 1_050_000,
            "max_output": 512_000,
        }

    def test_family_prefix_shrink_fallback(self):
        # An un-versioned alias should fall back to the family entry if present.
        # claude-opus-4.8 exists; a date-stamped variant must still resolve.
        o = md._resolve_probe_override("github-copilot", "claude-opus-4.8-20251101")
        assert o == {"context_window": 1_000_000, "max_output": 128_000}

    def test_unknown_provider_returns_none(self):
        assert md._resolve_probe_override("no-such-provider", "claude-opus-4.8") is None

    def test_unknown_model_returns_none(self):
        assert md._resolve_probe_override("github-copilot", "totally-made-up-model") is None

    def test_empty_inputs_return_none(self):
        assert md._resolve_probe_override("", "claude-opus-4.8") is None
        assert md._resolve_probe_override("github-copilot", "") is None


class TestCanonicalizeModelId:
    def test_lowercase_and_strip_vendor(self):
        assert md._canonicalize_model_id("Anthropic/Claude-Opus-4.7") == "claude-opus-4.7"

    def test_strip_date_stamp(self):
        assert md._canonicalize_model_id("claude-opus-4-7-20251101") == "claude-opus-4-7"

    def test_dots_preserved(self):
        assert md._canonicalize_model_id("gpt-5.5") == "gpt-5.5"


class TestGetModelInfoOverrideMerge:
    def test_override_replaces_limits_keeps_other_fields(self, monkeypatch):
        # Base ModelInfo from models.dev with WRONG (under-reported) limits.
        base = md.ModelInfo(
            id="claude-opus-4.8",
            name="Claude Opus 4.8",
            family="claude-opus",
            provider_id="github-copilot",
            context_window=200_000,   # under-reported
            max_output=64_000,        # under-reported
        )
        monkeypatch.setattr(md, "_parse_model_info", lambda mid, raw, pid: base)
        monkeypatch.setattr(
            md, "fetch_models_dev",
            lambda: {"github-copilot": {"models": {"claude-opus-4.8": {}}}},
        )

        info = md.get_model_info("github-copilot", "claude-opus-4.8")
        assert info is not None
        # Numeric limits replaced by the override...
        assert info.context_window == 1_000_000
        assert info.max_output == 128_000
        # ...while identity fields are preserved from the base.
        assert info.name == "Claude Opus 4.8"

    def test_override_synthesizes_when_modelsdev_missing(self, monkeypatch):
        # models.dev has no entry, but we DO have an override → synthesize.
        monkeypatch.setattr(md, "fetch_models_dev", lambda: {})
        info = md.get_model_info("github-copilot", "claude-opus-4.8")
        assert info is not None
        assert info.context_window == 1_000_000
        assert info.max_output == 128_000

    def test_no_override_no_modelsdev_returns_none(self, monkeypatch):
        monkeypatch.setattr(md, "fetch_models_dev", lambda: {})
        assert md.get_model_info("github-copilot", "totally-made-up-model") is None
