"""hermes_cli.compaction_preview: the published, read-only preview of a route's compaction trigger."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.compaction_preview import (
    CompactionPreview,
    ContextWindow,
    preview_compaction,
    resolve_context_window,
    threshold_rule_applies,
)

UNKNOWN_MODEL = "totally-unknown-model-x"


def _session_trigger(config: dict, model: str, provider: str, window: int) -> int:
    """The trigger a real session installs: the compressor built the way agent_init builds it."""
    from agent.agent_init import _compressor_max_tokens, _parse_compression_config
    from agent.context_compressor import ContextCompressor

    agent = SimpleNamespace(model=model, provider=provider, api_mode="", base_url="", max_tokens=None)
    settings = _parse_compression_config(agent, config)
    compressor = ContextCompressor(
        model=model, threshold_percent=settings.threshold, provider=provider,
        max_tokens=_compressor_max_tokens(agent), model_thresholds=settings.model_thresholds,
        threshold_tokens_cap=settings.threshold_tokens, quiet_mode=True,
    )
    return compressor.preview_threshold_tokens(model, window, provider)


@pytest.mark.parametrize(
    "compression, model, provider, window",
    [
        ({"threshold": 0.5}, "claude-opus-4-6", "anthropic", 1_000_000),
        ({"threshold": 0.5, "threshold_tokens": 256_000}, "claude-opus-4-6", "anthropic", 1_000_000),
        ({"threshold": 0.5, "model_thresholds": {"opus": 0.8}}, "claude-opus-4-6", "anthropic", 1_000_000),
        ({"threshold": 0.5, "model_thresholds": {"openrouter:opus": 0.8}}, "claude-opus-4-6", "anthropic", 1_000_000),
        ({"threshold": 0.5}, "claude-haiku-4-5", "anthropic", 200_000),
        ({"threshold": 0.5}, "gpt-5.5", "openai-codex", 272_000),
        ({"threshold": 0.5, "codex_gpt55_autoraise": False}, "gpt-5.5", "openai-codex", 272_000),
        ({"threshold": 0.9}, "tiny-local-model", "custom", 64_000),
        ({"threshold": 0.5}, "gemini-2.5-pro", "gemini", 1_048_576),
    ],
)
def test_trigger_is_the_one_a_session_installs(compression, model, provider, window):
    config = {"compression": compression}
    preview = preview_compaction(config, model=model, provider=provider, context_length=window)
    assert preview.threshold_tokens == _session_trigger(config, model, provider, window)
    assert (preview.context_length, preview.context_source) == (window, "caller")


def test_preview_reports_the_factors_behind_the_trigger():
    config = {"compression": {"threshold": 0.5, "threshold_tokens": 300_000, "model_thresholds": {"opus-4-6": 0.8, "opus": 0.6}}}
    capped = preview_compaction(config, model="claude-opus-4-6", provider="anthropic", context_length=1_000_000)
    assert isinstance(capped, CompactionPreview)
    assert capped.matched_rule == "opus-4-6"  # the longest matching key wins
    assert capped.base_percent == 0.8
    assert capped.threshold_cap == capped.threshold_tokens == 300_000

    small = preview_compaction({"compression": {"threshold": 0.5}}, model="claude-haiku-4-5", provider="anthropic", context_length=200_000)
    assert small.matched_rule is None and small.threshold_cap is None
    assert small.base_percent == 0.5 < small.effective_percent  # the small-window floor raised it


def test_gemini_route_reserves_the_native_adapters_default_output():
    from agent.gemini_native_adapter import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS

    gemini = preview_compaction({}, model="gemini-2.5-pro", provider="gemini", context_length=1_048_576)
    assert gemini.output_reservation == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    assert preview_compaction({}, model="claude-opus-4-6", provider="anthropic", context_length=1_000_000).output_reservation is None
    explicit = preview_compaction({}, model="claude-opus-4-6", provider="anthropic", context_length=1_000_000, max_tokens=32_000)
    assert explicit.output_reservation == 32_000


def test_provider_alias_and_model_id_are_previewed_as_the_session_runs_them():
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.models import normalize_provider

    provider = normalize_provider("claude")
    model = normalize_model_for_provider("claude-opus-4.6", provider)
    assert (model, provider) != ("claude-opus-4.6", "claude")  # the alias and the dotted id are both rewritten

    config = {"compression": {"threshold": 0.5, "model_thresholds": {f"{provider}:{model}": 0.9}}}
    preview = preview_compaction(config, model="claude-opus-4.6", provider="claude", context_length=1_000_000)
    assert (preview.model, preview.provider) == (model, provider)
    assert preview.matched_rule == f"{provider}:{model}"

    for unresolved in ("auto", ""):  # resolved from credentials at session start, never from the name
        untouched = preview_compaction({}, model="claude-opus-4.6", provider=unresolved, context_length=1_000_000)
        assert (untouched.model, untouched.provider) == ("claude-opus-4.6", unresolved)


def test_threshold_rule_applies_follows_the_compressors_ranking():
    assert threshold_rule_applies("opus", model="claude-opus-4-6", provider="anthropic")
    assert threshold_rule_applies("anthropic:opus", model="claude-opus-4-6", provider="anthropic")
    assert threshold_rule_applies("anthropic:opus", model="claude-opus-4.6", provider="claude")  # as the session runs it
    assert not threshold_rule_applies("openrouter:opus", model="claude-opus-4-6", provider="anthropic")
    assert not threshold_rule_applies("sonnet", model="claude-opus-4-6", provider="anthropic")


def test_garbage_compression_values_raise_as_they_do_at_session_start():
    with pytest.raises(ValueError):
        preview_compaction({"compression": {"threshold": "half"}}, model="claude-opus-4-6", provider="anthropic", context_length=1_000_000)


def test_window_source_tells_a_known_model_from_a_default():
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, DEFAULT_FALLBACK_CONTEXT

    catalogued = next(iter(DEFAULT_CONTEXT_LENGTHS))
    known = resolve_context_window({}, model=catalogued, provider="custom")
    assert isinstance(known, ContextWindow)
    assert known.source == "resolved" and known.tokens > 0

    unknown = resolve_context_window({}, model=UNKNOWN_MODEL, provider="custom")
    assert (unknown.tokens, unknown.source) == (DEFAULT_FALLBACK_CONTEXT, "default")


def test_model_context_length_pins_the_main_route_only():
    config = {"model": {"context_length": 400_000}}
    assert resolve_context_window(config, model=UNKNOWN_MODEL, provider="custom") == ContextWindow(400_000, "config")
    assert resolve_context_window(config, model=UNKNOWN_MODEL, provider="custom", main_route=False).source == "default"


def test_custom_provider_window_wins_over_the_catalog():
    config = {
        "model": {"base_url": "http://localhost:8000/v1"},
        "custom_providers": [
            {"name": "local", "base_url": "http://localhost:8000/v1", "models": {"my-model": {"context_length": 123_456}}}
        ],
    }
    assert resolve_context_window(config, model="my-model", provider="custom") == ContextWindow(123_456, "custom_provider")


def test_bedrock_window_never_probes_with_the_machines_credentials(monkeypatch):
    bedrock = pytest.importorskip("agent.bedrock_adapter")
    probed = []
    monkeypatch.setattr(bedrock, "probe_bedrock_context_length", lambda *a, **k: probed.append(a) or None)
    monkeypatch.setattr(bedrock, "_get_bedrock_runtime_client", lambda *a, **k: probed.append(a))

    tabled = next(iter(bedrock.BEDROCK_CONTEXT_LENGTHS))
    known = resolve_context_window({}, model=tabled, provider="bedrock")
    assert known == ContextWindow(bedrock.bedrock_static_context_length(tabled), "resolved")

    unknown = resolve_context_window({}, model="some-model-nobody-ships", provider="bedrock")
    assert unknown == ContextWindow(bedrock.BEDROCK_DEFAULT_CONTEXT_LENGTH, "default")
    assert probed == []


def _moa(provider: str, model: str) -> dict:
    return {"moa": {"default_preset": "duo", "presets": {"duo": {
        "aggregator": {"provider": provider, "model": model},
        "references": [{"provider": "openai-codex", "model": "gpt-5.5"}],
    }}}}


def test_moa_window_is_the_aggregators_and_resolves_no_credentials(monkeypatch):
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT
    from hermes_cli import runtime_provider

    resolved = []
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda *a, **k: resolved.append(k) or {})

    window = resolve_context_window(_moa("anthropic", "claude-opus-4-6"), model="duo", provider="moa")
    assert window == resolve_context_window({}, model="claude-opus-4-6", provider="anthropic", main_route=False)

    missing = resolve_context_window({}, model="no-such-preset", provider="moa")
    assert missing == ContextWindow(DEFAULT_FALLBACK_CONTEXT, "default")
    assert resolved == []


def test_preview_resolves_the_window_when_the_caller_has_none():
    config = {"model": {"context_length": 400_000}, "compression": {"threshold": 0.5}}
    preview = preview_compaction(config, model=UNKNOWN_MODEL, provider="custom")
    assert (preview.context_length, preview.context_source) == (400_000, "config")
    assert preview.threshold_tokens == _session_trigger(config, UNKNOWN_MODEL, "custom", 400_000)


def test_a_plugin_importing_the_preview_gets_no_private_import_warning(tmp_path):
    import hermes_yaml as yaml

    from hermes_cli.plugin_validate import validate_plugin_dir

    plugin = tmp_path / "advisor"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "advisor", "version": "1.0.0", "description": "x"}), encoding="utf-8"
    )
    (plugin / "__init__.py").write_text(
        "from hermes_cli.compaction_preview import (\n"
        "    CompactionPreview, ContextWindow, preview_compaction, resolve_context_window, threshold_rule_applies,\n"
        ")\n\n"
        "def register(ctx):\n    pass\n",
        encoding="utf-8",
    )
    report = validate_plugin_dir(plugin)
    assert report.ok, report.failures
    assert not [w for w in report.warnings if w.startswith("private core imports")]
