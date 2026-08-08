"""Regression tests for ``hermes doctor --fix`` config-structure repairs."""

import contextlib
import io
import sys
import types
from argparse import Namespace
from pathlib import Path

import yaml

import hermes_cli.config as config_mod
import hermes_cli.doctor as doctor_mod


def _setup_doctor_env(monkeypatch, tmp_path, config_text: str):
    """Point doctor at an isolated config.yaml and stub unrelated probes."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config_text, encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    venv_bin_dir = project / "venv" / "bin"
    venv_bin_dir.mkdir(parents=True, exist_ok=True)
    hermes_bin = venv_bin_dir / "hermes"
    hermes_bin.write_text("#!/usr/bin/env python\n# entry point\n", encoding="utf-8")
    hermes_bin.chmod(0o755)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as auth_mod

        for name in (
            "get_nous_auth_status",
            "get_codex_auth_status",
            "get_gemini_oauth_auth_status",
            "get_minimax_oauth_auth_status",
        ):
            monkeypatch.setattr(auth_mod, name, lambda: {})
    except Exception:
        pass

    try:
        import httpx

        monkeypatch.setattr(
            httpx,
            "get",
            lambda *a, **kw: types.SimpleNamespace(status_code=200),
        )
    except Exception:
        pass

    return home


def _run_doctor(fix: bool) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=fix))
    return buf.getvalue()


def test_doctor_fix_converts_legacy_custom_providers_mapping(monkeypatch, tmp_path):
    home = _setup_doctor_env(
        monkeypatch,
        tmp_path,
        """
model:
  provider: custom
  default: qwen/qwen3
custom_providers:
  modelrelay:
    api: http://op3:7352/v1
    model: qwen/qwen3
    api_mode: openai-completions
""".lstrip(),
    )

    out = _run_doctor(fix=True)

    assert "Converted custom_providers from dict to list format" in out

    saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert saved["custom_providers"] == [
        {
            "name": "modelrelay",
            "base_url": "http://op3:7352/v1",
            "api_mode": "openai-completions",
            "model": "qwen/qwen3",
        }
    ]


def test_doctor_fix_converts_single_custom_provider_dict(monkeypatch, tmp_path):
    home = _setup_doctor_env(
        monkeypatch,
        tmp_path,
        """
model:
  provider: custom
  default: models/gemini-2.5-flash
custom_providers:
  name: GenerativeLanguage
  base_url: https://generativelanguage.googleapis.com/v1beta
  api_key: xxx
  model: models/gemini-2.5-flash
  rate_limit_delay: 2.0
""".lstrip(),
    )

    out = _run_doctor(fix=True)

    assert "Converted custom_providers from dict to list format" in out

    saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert saved["custom_providers"] == [
        {
            "name": "GenerativeLanguage",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "api_key": "xxx",
            "model": "models/gemini-2.5-flash",
            "rate_limit_delay": 2.0,
        }
    ]
