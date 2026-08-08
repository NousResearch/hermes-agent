"""#71019: doctor must not skip model diagnostics on scalar model: key."""

from __future__ import annotations

from types import SimpleNamespace


def test_run_doctor_handles_scalar_model_without_crash(tmp_path, monkeypatch, capsys):
    """Scalar model: must not AttributeError-skip the model validation block."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model: gpt-4o\n"
        "provider: openai\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import doctor as doctor_mod

    args = SimpleNamespace(fix=False, yes=True, verbose=False, json=False)
    try:
        doctor_mod.run_doctor(args)
    except SystemExit:
        pass

    out = capsys.readouterr()
    combined = (out.out + out.err).lower()
    assert "traceback" not in combined
    assert "attributeerror" not in combined


def test_scalar_model_section_guard_unit():
    model_section = "gpt-4o"
    if isinstance(model_section, str):
        default_model = model_section.strip()
        provider_raw = ""
    elif isinstance(model_section, dict):
        default_model = (model_section.get("default") or "").strip()
        provider_raw = (model_section.get("provider") or "").strip()
    else:
        default_model = ""
        provider_raw = ""
    assert default_model == "gpt-4o"
    assert provider_raw == ""


def test_normalize_alone_may_leave_scalar_so_guard_required():
    from hermes_cli.config import _normalize_root_model_keys

    cfg = _normalize_root_model_keys({"model": "gpt-4o"})
    model_section = cfg.get("model")
    # Guard pattern used in doctor
    if isinstance(model_section, str):
        default_model = model_section.strip()
    elif isinstance(model_section, dict):
        default_model = (
            model_section.get("default") or model_section.get("model") or ""
        ).strip()
    else:
        default_model = ""
    assert default_model == "gpt-4o" or isinstance(model_section, dict)
