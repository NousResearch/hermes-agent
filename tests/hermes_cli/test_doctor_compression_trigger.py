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
