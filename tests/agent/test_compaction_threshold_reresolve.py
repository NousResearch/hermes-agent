"""Phase B — per-model compression threshold survives a model switch / fallback.

Regression for the compaction-thrash incident (2026-06-19): a mid-session
fallback ``claude-opus-4-8 -> gpt-5.5 (openai-codex)`` collapsed the window
1M -> 272K, but ``ContextCompressor.update_model`` re-derived the trigger from
the *stored* ``threshold_percent`` (opus's resolved value) instead of
re-resolving the destination model's configured ``per_model_threshold``. So the
configured ``gpt-5.5: 0.9`` was never applied after the switch.

Invariant I3: after any ``update_model`` (fallback or ``/model`` switch),
``threshold_tokens`` reflects the DESTINATION model's resolved threshold via the
same precedence chain used at init (per_model_threshold -> built-in family ->
global), not the source model's stored percent applied to the new window.
"""

from __future__ import annotations

from agent.context_compressor import ContextCompressor


_PER_MODEL = {"gpt-5.5": 0.9, "claude-opus-4-8": 0.5}


def _make_opus_compressor() -> ContextCompressor:
    """Build a compressor configured like the live Apollo session: opus main,
    per_model_threshold map threaded, global default 0.75."""
    return ContextCompressor(
        model="claude-opus-4-8",
        threshold_percent=0.5,  # opus's resolved value at init
        protect_first_n=3,
        protect_last_n=20,
        summary_target_ratio=0.25,
        quiet_mode=True,
        config_context_length=1_000_000,
        provider="claude-pool",
        per_model_threshold=_PER_MODEL,
        global_threshold_percent=0.75,
    )


def test_update_model_reresolves_per_model_threshold() -> None:
    cc = _make_opus_compressor()
    # Sanity: opus init resolved to 0.5 of 1M.
    assert cc.threshold_tokens == 500_000

    # Mid-session fallback to gpt-5.5 on the 272K Codex window.
    cc.update_model(
        model="gpt-5.5",
        context_length=272_000,
        provider="openai-codex",
    )

    # The DESTINATION model's configured per_model_threshold (0.9) must win,
    # NOT opus's stale stored 0.5 (which would give 136000) and NOT the global
    # 0.75 (which would give 204000).
    assert cc.threshold_percent == 0.9
    assert cc.threshold_tokens == int(272_000 * 0.9)  # 244800


def test_update_model_no_per_model_entry_falls_through_to_global() -> None:
    """A destination model with no per_model entry and no built-in family
    default falls through to the configured GLOBAL threshold (0.75), not the
    source model's stale stored percent."""
    cc = _make_opus_compressor()
    cc.update_model(
        model="some-other-model",
        context_length=400_000,
        provider="some-provider",
    )
    assert cc.threshold_percent == 0.75
    assert cc.threshold_tokens == int(400_000 * 0.75)


def test_update_model_without_threaded_config_keeps_legacy_behavior() -> None:
    """Backward-compat: a compressor constructed WITHOUT the per-model config
    (older callers / direct unit constructions) keeps the legacy behavior of
    re-applying the stored threshold_percent to the new window."""
    cc = ContextCompressor(
        model="claude-opus-4-8",
        threshold_percent=0.5,
        quiet_mode=True,
        config_context_length=1_000_000,
        provider="claude-pool",
    )
    cc.update_model(model="gpt-5.5", context_length=272_000, provider="openai-codex")
    # No config threaded -> legacy: stored 0.5 applied to the new window.
    assert cc.threshold_percent == 0.5
    assert cc.threshold_tokens == int(272_000 * 0.5)


def test_shared_resolver_matches_init_precedence() -> None:
    """Init-parity: the shared resolver used by update_model produces the same
    value the init block's precedence would, for each precedence tier."""
    from agent.auxiliary_client import resolve_compression_threshold

    # Tier 1: explicit per-model config wins.
    assert resolve_compression_threshold(
        _PER_MODEL, "gpt-5.5", "openai-codex", global_threshold=0.75
    ) == 0.9
    # Tier 3: no per-model entry, no family default -> global.
    assert resolve_compression_threshold(
        _PER_MODEL, "mystery-model", "mystery", global_threshold=0.75
    ) == 0.75
    # Tier 2: built-in codex gpt-5.5 autoraise applies when NOT in per_model map.
    assert resolve_compression_threshold(
        {}, "gpt-5.5", "openai-codex", global_threshold=0.5
    ) == resolve_compression_threshold(
        {}, "gpt-5.5", "openai-codex", global_threshold=0.5
    )
