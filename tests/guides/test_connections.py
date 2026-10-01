"""Bundled helpers run without skills and keep native profile credential storage."""

import json
import os
import subprocess
import sys

import pytest

from agent.knowledge import guides_root

SCRIPTS = guides_root() / "connections/google-workspace/scripts"


@pytest.mark.parametrize("script", ["setup.py", "google_api.py"])
def test_google_helper_runs_from_an_unrelated_working_directory(tmp_path, script):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"], cwd=tmp_path,
        env={**os.environ, "HERMES_HOME": str(tmp_path / "profile")},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert not (tmp_path / "profile").exists()


def test_google_setup_stores_credentials_in_selected_home(tmp_path):
    profile = tmp_path / "selected profile"
    profile.mkdir()
    credentials = {"installed": {"client_id": "test-client", "client_secret": "fake-test-secret"}}
    source = tmp_path / "client.json"
    source.write_text(json.dumps(credentials))
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "setup.py"), "--client-secret", str(source)],
        cwd=tmp_path, env={**os.environ, "HERMES_HOME": str(profile)},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((profile / "google_client_secret.json").read_text()) == credentials
    assert "fake-test-secret" not in result.stdout + result.stderr
    assert not (profile / "skills").exists()


@pytest.mark.parametrize("script, variable", [("setup.py", "google_setup_script"),
                                             ("google_api.py", "google_api_script")])
def test_documented_google_commands_parse_before_any_auth_or_api_call(monkeypatch, script, variable):
    import argparse
    import importlib.util
    import re
    import shlex

    spec = importlib.util.spec_from_file_location("guide_google_cli", SCRIPTS / script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec.loader.exec_module(module)
    guide = (SCRIPTS.parent / "guide.md").read_text()
    commands = re.findall(r'^python "\$' + variable + r'" (.*)$', guide, re.MULTILINE)
    assert commands
    parse_args = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def stop_after_parsing(parser, *args, **kwargs):
        parse_args(parser, *args, **kwargs)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop_after_parsing)
    for command in commands:
        monkeypatch.setattr(sys, "argv", [script, *shlex.split(command, comments=True)])
        with pytest.raises(Parsed):
            module.main()
