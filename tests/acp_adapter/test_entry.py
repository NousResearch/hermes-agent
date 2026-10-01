"""Tests for acp_adapter.entry startup wiring."""

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


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []










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










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    import pm

    def refuse(name, **kwargs):
        raise pm.InstallError(name, "download failed")

    monkeypatch.setattr(pm, "ensure", refuse)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1


def test_setup_browser_is_one_explicit_package_request(monkeypatch):
    import pm

    calls = []
    monkeypatch.setattr(pm, "ensure", lambda name, **kwargs: calls.append((name, kwargs)))

    entry.main(["--setup-browser", "--yes"])

    assert calls == [("agent-browser", {"explicit": True})]


@pytest.mark.parametrize("argv", [[], ["-p", "work"]], ids=["sticky", "flag"])
def test_console_script_serves_the_selected_profile(tmp_path, argv):
    """``hermes-acp`` (and ``python -m acp_adapter``) start in ``entry.main()``, not hermes_cli.main:
    the server must still run under the profile ``hermes acp`` would pick (sticky or ``-p``)."""
    import os
    import subprocess
    import sys

    root = tmp_path / ".hermes"
    for name in ("work", "other"):
        (root / "profiles" / name).mkdir(parents=True)
        (root / "profiles" / name / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "active_profile").write_text("other" if argv else "work", encoding="utf-8")
    driver = (
        "import sys, acp\n"
        "async def fake_run_agent(agent, **kw):\n"
        "    from hermes_constants import get_hermes_home\n"
        "    print('SERVED', get_hermes_home())\n"
        "acp.run_agent = fake_run_agent\n"
        f"sys.argv = ['hermes-acp', *{argv!r}]\n"
        "from acp_adapter.entry import main\n"
        "main()\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
    env.update(HOME=str(tmp_path), HERMES_ACP_SKIP_CONFIGURED_MCP="1")
    out = subprocess.run([sys.executable, "-c", driver], env=env, cwd=tmp_path, capture_output=True,
                         text=True, timeout=120)

    served = [line for line in out.stdout.splitlines() if line.startswith("SERVED ")]
    assert served == [f"SERVED {root / 'profiles' / 'work'}"], out.stdout + out.stderr
