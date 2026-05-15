"""Tests for acp_adapter.entry startup wiring."""

import sys

import acp
import pytest

from acp_adapter import entry


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_version_prints_without_starting_server(monkeypatch, capsys):
    monkeypatch.setattr(entry, "_setup_logging", lambda: (_ for _ in ()).throw(AssertionError("started server")))

    entry.main(["--version"])

    output = capsys.readouterr().out.strip()
    assert output
    assert "Starting hermes-agent ACP adapter" not in output


def test_main_check_prints_ok_without_starting_server(monkeypatch, capsys):
    monkeypatch.setattr(entry, "_setup_logging", lambda: (_ for _ in ()).throw(AssertionError("started server")))

    entry.main(["--check"])

    assert capsys.readouterr().out.strip() == "Hermes ACP check OK"


def test_main_setup_runs_model_configuration(monkeypatch):
    calls = {}

    def fake_hermes_main():
        calls["argv"] = sys.argv[:]

    monkeypatch.setattr("hermes_cli.main.main", fake_hermes_main)
    # Pretend stdin is not a TTY so the follow-up browser prompt is skipped.
    # That keeps this test focused on the model-setup wiring; the
    # browser-prompt path has its own test below.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    entry.main(["--setup"])

    assert calls["argv"][1:] == ["model"]


def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]


def test_main_setup_skips_browser_prompt_on_no(monkeypatch):
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "")

    called = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: called.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert called == []


def test_main_setup_browser_invokes_bundled_script(monkeypatch):
    """`hermes-acp --setup-browser` must shell out to the bundled bootstrap
    script — never reimplement the install logic inline."""
    monkeypatch.setattr("platform.system", lambda: "Linux")

    captured = {}

    def fake_run(cmd, check=False):
        captured["cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)

    entry.main(["--setup-browser"])

    assert captured["cmd"][0] == "bash"
    assert captured["cmd"][1].endswith("bootstrap_browser_tools.sh")
    # --yes is NOT passed when the flag is absent.
    assert "--yes" not in captured["cmd"]


def test_main_setup_browser_forwards_yes_flag(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")

    captured = {}

    def fake_run(cmd, check=False):
        captured["cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)

    entry.main(["--setup-browser", "--yes"])

    assert "--yes" in captured["cmd"]


def test_main_setup_browser_uses_powershell_on_windows(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")

    captured = {}

    def fake_run(cmd, check=False):
        captured["cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)

    entry.main(["--setup-browser", "--yes"])

    assert captured["cmd"][0] == "powershell.exe"
    assert any(part.endswith("bootstrap_browser_tools.ps1") for part in captured["cmd"])
    assert "-Yes" in captured["cmd"]


def test_main_setup_browser_propagates_failure(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")

    class _R:
        returncode = 7

    monkeypatch.setattr("subprocess.run", lambda cmd, check=False: _R())

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 7


def test_bootstrap_scripts_ship_with_package():
    """The package-data wiring (pyproject.toml) must include the bootstrap
    scripts — otherwise `--setup-browser` 404s at runtime."""
    from pathlib import Path

    bootstrap_dir = Path(entry.__file__).resolve().parent / "bootstrap"
    sh = bootstrap_dir / "bootstrap_browser_tools.sh"
    ps1 = bootstrap_dir / "bootstrap_browser_tools.ps1"

    assert sh.is_file(), f"missing bundled script: {sh}"
    assert ps1.is_file(), f"missing bundled script: {ps1}"

    sh_text = sh.read_text(encoding="utf-8")
    ps1_text = ps1.read_text(encoding="utf-8")

    # Sanity: scripts know how to find the Hermes-managed Node prefix.
    assert "HERMES_HOME" in sh_text
    assert "agent-browser" in sh_text
    assert "HermesHome" in ps1_text
    assert "agent-browser" in ps1_text
