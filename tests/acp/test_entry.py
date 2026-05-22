"""Tests for acp_adapter.entry startup wiring."""

import os
import sys
from pathlib import Path

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


def test_main_setup_browser_calls_ensure_dependency(monkeypatch):
    """`hermes-acp --setup-browser` routes through dep_ensure.ensure_dependency."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append((dep, interactive))
        return True

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    entry.main(["--setup-browser"])

    assert ("node", True) in calls
    assert ("browser", True) in calls


def test_main_setup_browser_forwards_yes_flag(monkeypatch):
    """--yes suppresses interactive prompts in ensure_dependency."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append((dep, interactive))
        return True

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    entry.main(["--setup-browser", "--yes"])

    assert ("node", False) in calls
    assert ("browser", False) in calls


def test_main_setup_browser_stops_on_node_failure(monkeypatch):
    """If node install fails, browser install is not attempted."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append(dep)
        return dep != "node"  # node fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
    assert "node" in calls
    assert "browser" not in calls


def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# --profile propagation (#30571)
# ---------------------------------------------------------------------------


def _bootstrap_profile(tmp_path: Path, monkeypatch, name: str) -> Path:
    """Lay down ``<tmp>/.hermes/profiles/<name>`` and point Path.home() at it."""
    profile_dir = tmp_path / ".hermes" / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return profile_dir


def test_main_profile_flag_sets_hermes_home_before_load_env(monkeypatch, tmp_path):
    """``hermes-acp --profile <name>`` must repoint HERMES_HOME at the
    profile dir before ``_load_env`` runs, so the profile's .env is what
    gets loaded.
    """
    profile_dir = _bootstrap_profile(tmp_path, monkeypatch, "code-reviewer")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    observed = {}

    def fake_load_env():
        observed["HERMES_HOME"] = os.environ.get("HERMES_HOME")
        observed["HERMES_PROFILE"] = os.environ.get("HERMES_PROFILE")

    async def fake_run_agent(agent, **kwargs):
        observed["ran"] = True

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", fake_load_env)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main(["--profile", "code-reviewer"])

    assert observed["HERMES_HOME"] == str(profile_dir)
    assert observed["HERMES_PROFILE"] == "code-reviewer"
    assert observed.get("ran") is True


def test_main_short_profile_flag_is_equivalent(monkeypatch, tmp_path):
    """``-p`` is the same as ``--profile``."""
    profile_dir = _bootstrap_profile(tmp_path, monkeypatch, "fast-coder")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main(["-p", "fast-coder"])

    assert os.environ.get("HERMES_HOME") == str(profile_dir)
    assert os.environ.get("HERMES_PROFILE") == "fast-coder"


def test_main_profile_flag_applies_before_check_subcommand(monkeypatch, tmp_path, capsys):
    """``--check`` short-circuits the server start but still has to honour
    ``--profile`` so a check run under the requested profile uses the right
    config / dependencies."""
    profile_dir = _bootstrap_profile(tmp_path, monkeypatch, "research-agent")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.setattr(entry, "_setup_logging", lambda: (_ for _ in ()).throw(AssertionError("started server")))

    entry.main(["--check", "--profile", "research-agent"])

    assert os.environ.get("HERMES_HOME") == str(profile_dir)
    assert os.environ.get("HERMES_PROFILE") == "research-agent"
    assert capsys.readouterr().out.strip() == "Hermes ACP check OK"


def test_main_omitting_profile_leaves_environment_untouched(monkeypatch, tmp_path):
    """Without ``--profile`` the ACP entry must not clobber HERMES_HOME /
    HERMES_PROFILE — the parent ``hermes`` CLI or the spawning shell may
    have already set them."""
    inherited_home = tmp_path / "inherited"
    inherited_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(inherited_home))
    monkeypatch.setenv("HERMES_PROFILE", "inherited-profile")

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert os.environ.get("HERMES_HOME") == str(inherited_home)
    assert os.environ.get("HERMES_PROFILE") == "inherited-profile"


def test_main_profile_flag_rejects_unknown_profile(monkeypatch, tmp_path):
    """Missing profile directory exits with a clear error instead of silently
    falling back to the default profile."""
    _bootstrap_profile(tmp_path, monkeypatch, "exists")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--profile", "does-not-exist"])
    assert excinfo.value.code == 1
    assert os.environ.get("HERMES_PROFILE") is None


def test_main_profile_flag_does_not_overwrite_explicit_hermes_profile(monkeypatch, tmp_path):
    """A deliberate ``HERMES_PROFILE`` from the spawning shell wins over the
    setdefault inside ``_apply_profile_override`` — same contract as
    ``hermes_cli.main._apply_profile_override``."""
    profile_dir = _bootstrap_profile(tmp_path, monkeypatch, "named")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "operator-override")

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main(["--profile", "named"])

    # HERMES_HOME is repointed (filesystem-truth), HERMES_PROFILE preserved.
    assert os.environ.get("HERMES_HOME") == str(profile_dir)
    assert os.environ.get("HERMES_PROFILE") == "operator-override"
