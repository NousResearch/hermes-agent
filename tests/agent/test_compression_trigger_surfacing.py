"""#117915: the startup line must show the RATIO trigger and name the clamp that replaced it.

The old banner printed ``compress at 30% = 256,000 (capped at 256,000 tokens)``: the number
after ``% =`` was the already-capped trigger, so the line read as broken math (30% of 1M is
not 256,000) and nothing said what the configured ratio would have asked for, or which setting
overrode it. ``compression_trigger_report`` is the single derivation both the banner and
``hermes doctor`` quote, so the two can never disagree.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.context_compressor import compression_trigger_report


def test_default_cap_binds_and_is_named():
    rep = compression_trigger_report(
        context_length=1_000_000, ratio_percent=0.3, cap=256_000,
    )
    assert rep["ratio_tokens"] == 300_000, "the ratio trigger must be reported, not the capped value"
    assert rep["effective_tokens"] == 256_000
    assert rep["binding"] == "threshold_tokens"
    # The note names the setting and the way back to ratio-only behaviour.
    assert "compression.threshold_tokens" in rep["note"]
    assert "256,000" in rep["note"]
    assert "null" in rep["note"]


def test_no_cap_reports_ratio_only():
    rep = compression_trigger_report(context_length=1_000_000, ratio_percent=0.3, cap=None)
    assert rep["binding"] == "ratio"
    assert rep["effective_tokens"] == rep["ratio_tokens"] == 300_000
    assert rep["note"] == ""


def test_ratio_below_cap_is_not_reported_as_capped():
    # 128K window: the raise-only small-context floor puts the ratio trigger at 75% = 96,000,
    # well under the shipped 256,000 cap — the cap is present but must not be named as binding.
    rep = compression_trigger_report(context_length=128_000, ratio_percent=0.5, cap=256_000)
    assert rep["ratio_tokens"] == 96_000
    assert rep["effective_tokens"] == 96_000
    assert rep["binding"] == "ratio"
    assert rep["note"] == ""


def test_explicit_small_cap_binds():
    rep = compression_trigger_report(context_length=1_000_000, ratio_percent=0.3, cap=60_000)
    assert rep["effective_tokens"] == 60_000
    assert rep["binding"] == "threshold_tokens"
    assert "60,000" in rep["note"]


def test_cap_is_clamped_to_the_window():
    rep = compression_trigger_report(context_length=128_000, ratio_percent=0.5, cap=500_000)
    assert rep["cap"] == 128_000, "a cap above the window must be clamped to the window"


def test_auxiliary_ceiling_wins_over_the_ratio():
    rep = compression_trigger_report(
        context_length=1_000_000, ratio_percent=0.3, cap=None, aux_ceiling=200_000,
    )
    assert rep["binding"] == "auxiliary"
    assert rep["effective_tokens"] == 200_000
    assert "auxiliary" in rep["note"]
    assert "compression.threshold_tokens" not in rep["note"], (
        "the aux clamp must not be blamed on threshold_tokens"
    )


def _emit(agent, cs, capsys):
    from agent.agent_init import _emit_compression_summary

    _emit_compression_summary(agent, cs)
    return capsys.readouterr().out


def _agent(**compressor_attrs):
    cc_kwargs = dict(
        context_length=1_000_000,
        threshold_percent=0.3,
        threshold_tokens=256_000,
        threshold_tokens_cap=256_000,
        max_tokens=None,
        _aux_context_ceiling=None,
    )
    cc_kwargs.update(compressor_attrs)  # per-test overrides win over the shipped defaults
    cc = SimpleNamespace(**cc_kwargs)
    return SimpleNamespace(
        quiet_mode=False,
        context_compressor=cc,
        _compression_threshold_autoraised=None,
    )


def _settings(enabled=True, threshold=0.3):
    from agent.agent_init import CompressionSettings

    return CompressionSettings(enabled=enabled, threshold=threshold, autoraise_notice_enabled=False)


def test_banner_shows_ratio_trigger_and_names_the_binding_cap(capsys):
    out = _emit(_agent(), _settings(), capsys)
    assert "1,000,000 tokens" in out
    assert "300,000" in out, "the banner must show the ratio trigger the config asked for"
    assert "capped at 256,000" in out
    assert "compression.threshold_tokens" in out


def test_banner_without_a_binding_cap_keeps_the_plain_line(capsys):
    out = _emit(
        _agent(context_length=128_000, threshold_percent=0.75, threshold_tokens=96_000),
        _settings(threshold=0.75),
        capsys,
    )
    assert "compress at 75% = 96,000" in out
    assert "capped" not in out


def test_banner_still_reports_the_installed_trigger_for_an_unknown_clamp(capsys):
    # A clamp the report does not model must not make the line claim the ratio number.
    out = _emit(_agent(threshold_tokens=111_000), _settings(), capsys)
    assert "111,000" in out
    assert "300,000" not in out
