"""Tests for the Setup Codex Windows launcher prototype."""

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "setup_codex_launcher.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("setup_codex_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["setup_codex_launcher"] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_refuses_non_localhost_host(capsys):
    launcher = load_launcher()

    rc = launcher.main(["--host", "0.0.0.0", "--no-open"])

    assert rc == 2
    assert "Refusing non-localhost" in capsys.readouterr().err


def test_launcher_opens_setup_codex_when_dashboard_already_running(monkeypatch):
    launcher = load_launcher()
    opened = []

    monkeypatch.setattr(launcher, "probe_dashboard", lambda host, port: launcher.ProbeResult(True, "running"))
    monkeypatch.setattr(launcher, "start_dashboard", lambda host, port: (_ for _ in ()).throw(AssertionError("should not start")))
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: opened.append(url))

    rc = launcher.main([])

    assert rc == 0
    assert opened == ["http://127.0.0.1:9119/setup-codex"]
