"""#117915: ``hermes doctor`` must name the effective compression trigger and which setting binds it.

Before this, doctor said nothing about ``compression.threshold_tokens``: a user who configured a
ratio (global ``threshold`` or ``model_thresholds``) only ever discovered the shipped 256,000 cap
by grepping ``threshold=`` out of ``agent.log``. The check is informational — a binding shipped
default is intent (#115986), so it must never add to ``Finding.issues`` or change the exit status.
"""

from __future__ import annotations

from hermes_cli.doctor_report import Finding


def _run(monkeypatch, capsys, *, config, raw=None):
    """Run the check against a mocked merged config (``load_config``) and raw file (``read_raw_config_readonly``)."""
    import hermes_cli.config as config_mod
    from hermes_cli import doctor_config

    monkeypatch.setattr(config_mod, "load_config", lambda: config)
    monkeypatch.setattr(config_mod, "read_raw_config_readonly", lambda: raw or {})
    f = doctor_config._check_compression_trigger(False)
    assert isinstance(f, Finding)
    return f, capsys.readouterr().out


def _merged(*, threshold_tokens, **compression):
    """What ``load_config()`` hands out: DEFAULT_CONFIG already merged in."""
    comp = {"enabled": True, "threshold": 0.5, **compression}
    if threshold_tokens is not ...:
        comp["threshold_tokens"] = threshold_tokens
    return {
        "model": {"default": "deepseek-flash", "provider": "deepseek"},
        "compression": comp,
    }


def test_default_cap_that_binds_is_reported_with_the_ratio_it_overrode(monkeypatch, capsys):
    # Merged view carries the shipped 256,000; the user's file never wrote the key.
    f, out = _run(
        monkeypatch,
        capsys,
        config=_merged(threshold_tokens=256_000, model_thresholds={"deepseek-flash": 0.3}),
        raw={"compression": {"model_thresholds": {"deepseek-flash": 0.3}}},
    )
    assert "256,000" in out, "the binding trigger must be shown"
    assert "300,000" in out, "the ratio the user configured must be shown next to it"
    assert "compression.threshold_tokens" in out
    assert "shipped default" in out, "an unset cap must be identified as the shipped default"
    assert "1,000,000" in out, "the window the ratio was computed against must be shown"
    assert f.issues == [] and f.manual_issues == [], (
        "surfacing a shipped default must not fail hermes doctor"
    )


def test_cap_written_by_the_user_is_identified_as_configured(monkeypatch, capsys):
    f, out = _run(
        monkeypatch,
        capsys,
        config=_merged(threshold_tokens=60_000),
        raw={"compression": {"threshold_tokens": 60_000}},
    )
    assert "60,000" in out
    assert "shipped default" not in out
    assert "config.yaml" in out
    assert f.issues == []


def test_null_cap_reports_ratio_only_trigger(monkeypatch, capsys):
    f, out = _run(
        monkeypatch,
        capsys,
        # Same 0.3 ratio as the binding case, but the cap nulled out: the trigger must fall back
        # to exactly what the ratio asks for instead of stopping at the shipped 256,000.
        config=_merged(threshold_tokens=None, model_thresholds={"deepseek-flash": 0.3}),
        raw={"compression": {"threshold_tokens": None}},
    )
    assert "300,000" in out, "ratio-only mode must report the ratio trigger"
    assert "shipped default" not in out
    assert f.issues == []


def test_ratio_below_the_cap_reports_no_binding_cap(monkeypatch, capsys):
    # 128K window (the bare "deepseek" catalog entry), 50% ratio floored to 75% -> 96,000,
    # comfortably under the 256,000 cap, so nothing may be reported as capped.
    f, out = _run(
        monkeypatch,
        capsys,
        config={
            "model": {"default": "deepseek", "provider": "deepseek"},
            "compression": {"enabled": True, "threshold": 0.5, "threshold_tokens": 256_000},
        },
        raw={},
    )
    assert "96,000" in out
    assert "shipped default" not in out
    assert f.issues == []


def test_disabled_compression_is_stated(monkeypatch, capsys):
    f, out = _run(monkeypatch, capsys, config=_merged(threshold_tokens=None, enabled=False), raw={})
    assert "disabled" in out.lower()
    assert f.issues == []


def test_check_is_registered_with_doctor():
    import hermes_cli.doctor as doctor

    titles = [title for title, _check in doctor.DOCTOR_CHECKS if title]
    assert any("Compression" in title for title in titles), (
        f"the compression trigger check must appear in DOCTOR_CHECKS, got {titles}"
    )


def _engine_installs(window, ratio_percent, cap, max_tokens):
    """The value ContextCompressor actually installs — the number doctor must agree with."""
    from agent.context_compressor import ContextCompressor

    pct = ContextCompressor._effective_threshold_percent(window, ratio_percent)
    ratio_tokens = ContextCompressor._compute_threshold_tokens(window, pct, max_tokens)
    cap_eff = min(cap, window) if cap is not None else None
    return cap_eff if cap_eff is not None and cap_eff < ratio_tokens else ratio_tokens


def test_doctor_matches_the_installed_trigger_when_max_tokens_is_configured(monkeypatch, capsys):
    """The trigger is pct x (window - max_tokens); doctor used the raw window, so it quoted a
    trigger the engine never installs and named the wrong setting as binding."""
    window, max_tokens, pct = 1_000_000, 131_072, 0.3
    f, out = _run(monkeypatch, capsys, config={
        "model": {"default": "deepseek-flash", "provider": "deepseek", "max_tokens": max_tokens},
        "compression": {"enabled": True, "threshold": pct, "threshold_tokens": None},
    }, raw={})
    installed = _engine_installs(window, pct, None, max_tokens)
    assert f"{installed:,}" in out, (
        f"doctor must quote the engine's installed trigger {installed:,}, got: {out}"
    )
    assert f"{int(window * pct):,}" not in out, "the raw-window number must not be shown as the trigger"
    assert f.issues == []


def test_doctor_matches_the_installed_trigger_on_native_gemini(monkeypatch, capsys):
    """No model.max_tokens is configured, but the native-Gemini route reserves 65,535 output
    tokens by default, so 30% of a 1M window is not 314,572."""
    window, pct = 1_048_576, 0.3
    from agent.gemini_native_adapter import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS

    f, out = _run(monkeypatch, capsys, config={
        "model": {"default": "gemini-2.5-pro", "provider": "gemini"},
        "compression": {"enabled": True, "threshold": pct, "threshold_tokens": None},
    }, raw={})
    installed = _engine_installs(window, pct, None, GEMINI_DEFAULT_MAX_OUTPUT_TOKENS)
    assert f"{installed:,}" in out, f"expected {installed:,} (65,535 reserved), got: {out}"
    assert f"{int(window * pct):,}" not in out
    assert f.issues == []


def test_doctor_blames_the_cap_only_when_the_cap_actually_binds(monkeypatch, capsys):
    """With the reservation applied the ratio asks for 491,520, so a 480,000 cap does bind —
    but with a larger reservation the same cap must not be blamed."""
    window, cap = 1_048_576, 480_000
    # 30% x (1,048,576 - 131,072) = 275,251, well under the cap: ratio wins, cap is not blamed.
    f, out = _run(monkeypatch, capsys, config={
        "model": {"default": "gemini-2.5-pro", "provider": "gemini", "max_tokens": 131_072},
        "compression": {"enabled": True, "threshold": 0.3, "threshold_tokens": cap},
    }, raw={})
    installed = _engine_installs(window, 0.3, cap, 131_072)
    assert installed == 275_251, f"premise: {installed:,}"
    assert "set in config.yaml" not in out, "a cap that does not bind must not be blamed"
    assert f"{installed:,}" in out
