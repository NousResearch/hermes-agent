"""Tests for the optional google-workspace-cli skill."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "optional-skills/productivity/google-workspace-cli/SKILL.md"
WRAPPER_PATH = (
    REPO_ROOT
    / "optional-skills/productivity/google-workspace-cli/scripts/gws_wrapper.py"
)


@pytest.fixture
def wrapper_module(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    spec = importlib.util.spec_from_file_location("gws_wrapper_cli_test", WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_skill_description_is_short():
    text = SKILL_MD.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("description: "))
    description = line.split(": ", 1)[1].strip().strip('"')
    assert len(description) <= 60
    assert description.endswith(".")


def test_load_api_key_prefers_environment(wrapper_module, monkeypatch, tmp_path):
    (tmp_path / ".google_workspace_api_key").write_text("wk_file")
    monkeypatch.setenv("GWS_SKILL_API_KEY", "wk_env")
    assert wrapper_module.load_api_key() == "wk_env"


def test_load_api_key_reads_key_file(wrapper_module, monkeypatch, tmp_path):
    monkeypatch.delenv("GWS_SKILL_API_KEY", raising=False)
    key_path = tmp_path / ".google_workspace_api_key"
    key_path.write_text("wk_file\n", encoding="utf-8")
    assert wrapper_module.load_api_key() == "wk_file"


def test_get_token_posts_source_key_with_timeout(wrapper_module):
    captured = {}
    response = MagicMock()
    response.read.return_value = json.dumps({"access_token": "ya29.test"}).encode()
    response.__enter__ = lambda self: self
    response.__exit__ = MagicMock(return_value=False)

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return response

    with patch.object(wrapper_module.urllib.request, "urlopen", side_effect=fake_urlopen):
        token = wrapper_module.get_token("google_calendar", "wk_test")

    assert token == "ya29.test"
    assert captured["timeout"] == 15
    assert captured["request"].headers["X-api_key"] == "wk_test"
    assert json.loads(captured["request"].data.decode()) == {
        "source_key": "google_calendar"
    }


def test_main_injects_token_and_runs_gws(wrapper_module, monkeypatch):
    monkeypatch.setenv("GWS_SKILL_API_KEY", "wk_test")
    captured = {}

    def fake_run(cmd, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return MagicMock(returncode=0)

    monkeypatch.setattr(wrapper_module, "get_token", lambda source, key: "ya29.injected")
    monkeypatch.setattr(wrapper_module, "resolve_gws_binary", lambda: "/usr/bin/gws")

    with patch.object(subprocess, "run", side_effect=fake_run):
        result = wrapper_module.main(["calendar", "events", "list"])

    assert result == 0
    assert captured["cmd"] == ["/usr/bin/gws", "calendar", "events", "list"]
    assert captured["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "ya29.injected"


def test_main_requires_api_key(wrapper_module, monkeypatch):
    monkeypatch.delenv("GWS_SKILL_API_KEY", raising=False)
    with patch.object(sys, "stderr"):
        assert wrapper_module.main(["calendar", "events", "list"]) == 1
