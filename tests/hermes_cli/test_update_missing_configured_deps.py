"""Update fallback must name configured features whose optional deps stayed missing (#10651)."""

import os
import subprocess
import sys
from pathlib import Path

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
    monkeypatch.setattr(main_install_repair, "_configured_features_missing_deps", lambda *a, **k: missing_features)
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


def test_configured_features_probe_reads_the_fresh_target_interpreter(tmp_path, monkeypatch):
    """The check runs in the TARGET interpreter with a real config, so it sees the post-install
    truth rather than the updater's own pre-install import caches. Positive: the SDK is absent →
    the configured platform is named. Negative: only the child interpreter is told the dependency
    is present (the parent is untouched) → nothing is reported, proving the verdict comes from the
    fresh process."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  feishu:\n    enabled: true\n    extra:\n      app_id: cli_x\n      app_secret: y\n",
        encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(home)}
    monkeypatch.setattr(main_install_repair, "_resolve_install_target_python", lambda *a, **k: Path(sys.executable))
    real_probe = main_install_repair._venv_probe

    def probe(python, script, *args, env=None, prelude=""):
        return real_probe(python, prelude + script, *args, env=env)

    monkeypatch.setattr(main_install_repair, "_venv_probe",
                        lambda p, s, *a, env=None: probe(p, s, *a, env=env, prelude="import sys; sys.modules['lark_oapi'] = None\n"))
    missing = main_install_repair._configured_features_missing_deps(["uv", "pip"], env=env)
    assert [feature for feature, _hint in missing] == ["Feishu / Lark"]

    child_only_present = "import tools.lazy_deps as ld; ld.is_available = lambda *_a, **_k: True\n"
    monkeypatch.setattr(main_install_repair, "_venv_probe",
                        lambda p, s, *a, env=None: probe(p, s, *a, env=env, prelude=child_only_present))
    assert main_install_repair._configured_features_missing_deps(["uv", "pip"], env=env) == []
