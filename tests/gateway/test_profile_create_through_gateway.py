"""``hermes profile create`` through a live gateway process (real ``gateway.run``).

* local: the CLI on the gateway's own home has the profile made by the gateway (this process's
  ``provision_profile`` is made to fail, so only the gateway can have made it), and a refused request (an
  existing name) comes back as the CLI's usual error;
* named: a CLI whose own home has no gateway, pointed at one by ``HERMES_TUI_GATEWAY_URL`` (loopback),
  has the profile made in the GATEWAY's home and nothing in its own, and never prints the URL's token;
* an unreachable URL is an error, never a silent local create.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.test_native_http_auth import daemon

ROOT = Path(__file__).resolve().parents[2]


def _create(home: Path, user: Path, *argv, env=None):
    base = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    base.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(ROOT),
                PYTHONUNBUFFERED="1", **(env or {}))
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "profile", "create", *argv, "--no-alias"],
                          cwd=ROOT, env=base, capture_output=True, text=True, timeout=240, check=False)


@pytest.mark.platforms("linux", "macos")
def test_cli_creates_through_a_live_gateway_local_and_by_url(tmp_path, monkeypatch, capsys):
    with daemon(tmp_path) as (home, descriptor):
        user = tmp_path / "user"

        # Local: this process is on the gateway's home and cannot create a profile itself.
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HOME", str(user))
        monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
        from hermes_cli import profile_cmd, profile_provision
        from hermes_cli.subcommands.profile import build_profile_parser
        import argparse

        def refuse(*a, **k):
            raise AssertionError("created in the CLI process, not by the gateway")
        monkeypatch.setattr(profile_provision, "provision_profile", refuse)
        parser = argparse.ArgumentParser()
        build_profile_parser(parser.add_subparsers(), cmd_profile=lambda args: None)
        args = parser.parse_args(["profile", "create", "viagw", "--no-alias"])
        profile_cmd._profile_create(args)
        out = capsys.readouterr().out
        assert "Profile 'viagw' created at" in out
        assert (home / "profiles" / "viagw" / "config.yaml").is_file()

        with pytest.raises(SystemExit) as refused:
            profile_cmd._profile_create(args)
        assert refused.value.code == 1
        assert "A profile named 'viagw' already exists" in capsys.readouterr().out

        # Named: a different home with no gateway, pointed at this one by URL.
        terminal_home = tmp_path / "terminal-home"
        terminal_home.mkdir()
        url = descriptor["api_origin"].replace("http:", "ws:") + "/api/ws?token=normal-http-owner"
        remote = _create(terminal_home, user, "byurl", env={"HERMES_TUI_GATEWAY_URL": url})
        out = remote.stdout + remote.stderr
        assert remote.returncode == 0, out
        assert "normal-http-owner" not in out
        assert (home / "profiles" / "byurl" / "config.yaml").is_file(), out
        assert not (terminal_home / "profiles").exists(), out

        # Unreachable URL: an error, and still nothing anywhere.
        dead = _create(terminal_home, user, "nowhere", env={"HERMES_TUI_GATEWAY_URL": "ws://127.0.0.1:9/api/ws"})
        assert dead.returncode == 1
        assert "could not reach the gateway at ws://127.0.0.1:9 (" in dead.stdout + dead.stderr
        assert not (terminal_home / "profiles").exists()
        assert not (home / "profiles" / "nowhere").exists()


def _create_on_profile(home: Path, user: Path, profile: str, *argv):
    """``hermes -p <profile> profile create ...`` in a process that cannot create locally."""
    code = ("import sys\n"
            f"sys.argv = ['hermes', '-p', {profile!r}, 'profile', 'create', *{list(argv)!r}, '--no-alias']\n"
            "import hermes_cli.profile_provision as provision\n"
            "def local(request): raise SystemExit('created in the CLI process, not by the gateway')\n"
            "provision.provision_profile = local\n"
            "from hermes_cli.main import main\n"
            "main()\n")
    base = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    base.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(ROOT),
                PYTHONUNBUFFERED="1")
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=base, capture_output=True, text=True,
                          timeout=240, check=False)


@pytest.mark.platforms("linux", "macos")
def test_a_terminal_on_a_served_profile_creates_from_that_profile(tmp_path):
    """The multiplexer serves ``work``; ``hermes -p work profile create`` reaches it, and the new
    profile copies work's keys and model (``--clone``: its SOUL.md), never the launch profile's."""
    from tests.gateway.test_multiplex_cold_profile_sidebar_mutation import multiplex_daemon
    with multiplex_daemon(tmp_path, profiles=("work",)) as (home, _descriptor, _env):
        user = tmp_path / "user"
        work = home / "profiles" / "work"
        (work / ".env").write_text("OPENROUTER_API_KEY=sk-or-work\n")
        (work / "SOUL.md").write_text("work's soul\n")
        fresh = _create_on_profile(home, user, "work", "fresh")
        assert fresh.returncode == 0, fresh.stdout + fresh.stderr
        assert "sk-or-work" in (home / "profiles" / "fresh" / ".env").read_text()
        assert "work-model" in (home / "profiles" / "fresh" / "config.yaml").read_text()
        clone = _create_on_profile(home, user, "work", "copy", "--clone")
        assert clone.returncode == 0, clone.stdout + clone.stderr
        assert "Cloned config, .env, SOUL.md, and skills from work." in clone.stdout
        assert (home / "profiles" / "copy" / "SOUL.md").read_text() == "work's soul\n"
