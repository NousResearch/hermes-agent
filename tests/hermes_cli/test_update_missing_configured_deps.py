"""Update fallback must name configured features whose optional deps stayed missing (#10651)."""

import subprocess
from types import SimpleNamespace
from unittest import mock

from hermes_cli import main_install_repair


def _run_fallback_with_failed_extra(monkeypatch, capsys, *, extra_fails: str, missing_features):
    def fake_install(cmd, **kwargs):
        target = cmd[-1]
        if target in (".[all]", f".[{extra_fails}]"):
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(main_install_repair, "_run_quarantined_install", fake_install)
    monkeypatch.setattr(main_install_repair, "_verify_console_scripts_installed", lambda *a, **k: None)
    monkeypatch.setattr(main_install_repair, "_verify_core_dependencies_installed", lambda *a, **k: None)
    monkeypatch.setattr(main_install_repair, "_venv_scripts_dir", lambda: None)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: False)
    monkeypatch.setattr(main_install_repair, "_load_installable_optional_extras", lambda group="all": [extra_fails, "mcp"])
    monkeypatch.setattr(main_install_repair, "_configured_features_missing_deps", lambda: missing_features)
    main_install_repair._install_python_dependencies_with_optional_fallback(["uv", "pip"])
    return capsys.readouterr().out


def test_fallback_names_configured_platform_whose_extra_failed(monkeypatch, capsys):
    out = _run_fallback_with_failed_extra(
        monkeypatch, capsys, extra_fails="feishu",
        missing_features=[("Feishu / Lark", "Run `hermes setup` to install Feishu support.")])
    assert "fail to load them on restart" in out
    assert "Feishu / Lark" in out and "hermes setup" in out
    # Unconfigured features that failed stay a plain "skipped" line, no scary warning.
    quiet = _run_fallback_with_failed_extra(monkeypatch, capsys, extra_fails="feishu", missing_features=[])
    assert "Skipped optional extras that still failed: feishu" in quiet
    assert "fail to load them on restart" not in quiet


def test_configured_features_reports_platform_with_missing_deps(monkeypatch):
    entry = SimpleNamespace(label="Feishu / Lark", check_fn=lambda: False, install_hint="Run `hermes setup`.")
    fake_registry = SimpleNamespace(get=lambda name: entry if name == "feishu" else None)
    fake_config = SimpleNamespace(get_connected_platforms=lambda: [SimpleNamespace(value="feishu")])
    with mock.patch("gateway.config.load_gateway_config", return_value=fake_config), \
         mock.patch("gateway.platform_registry.platform_registry", fake_registry), \
         mock.patch("hermes_cli.config.load_config", return_value={}):
        assert main_install_repair._configured_features_missing_deps() == [("Feishu / Lark", "Run `hermes setup`.")]
