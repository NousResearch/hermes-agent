"""Step-6: gateway curator-governance hook helper tests.

Covers gateway/run.py::_spawn_curator_governance_hook:
- still logs the curator summary via on_summary;
- missing $HERMES_HOME/scripts/curator-governance-hook.py → no spawn;
- present hook → spawns EXACT argv  [sys.executable, <hook>, --direct]
  detached (start_new_session), stdio devnull'd, no shell;
- Popen errors are contained (debug log path, no raise).
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import gateway.run as gateway_run  # noqa: E402


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


# ── missing hook → no spawn ────────────────────────────────────────────────


def test_missing_hook_logs_summary_and_does_not_spawn(
        hermes_home, caplog):
    delivered = []
    with patch.object(subprocess, "Popen",
                      side_effect=AssertionError("must not spawn")) as popen:
        gateway_run._spawn_curator_governance_hook(
            "curator summary text", delivered.append)
    assert delivered == ["curator summary text"]
    popen.assert_not_called()


# ── present hook → exact argv, detached, no shell ──────────────────────────


def test_present_hook_spawns_exact_argv_direct_detached(
        hermes_home):
    scripts = hermes_home / "scripts"
    scripts.mkdir()
    (scripts / "curator-governance-hook.py").write_text("# hook\n")
    delivered = []

    recorded = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = argv
        recorded["kwargs"] = kwargs

        class P:
            pid = 4242

        return P()

    with patch.object(subprocess, "Popen", side_effect=fake_popen):
        gateway_run._spawn_curator_governance_hook(
            "curator summary text", delivered.append)

    assert delivered == ["curator summary text"]
    hook = str(scripts / "curator-governance-hook.py")
    assert recorded["argv"] == [sys.executable, hook, "--direct"]
    kwargs = recorded["kwargs"]
    assert kwargs.get("start_new_session") is True, "must be detached"
    assert kwargs.get("shell", False) is False
    import subprocess as _s
    assert kwargs.get("stdin") == _s.DEVNULL
    assert kwargs.get("stdout") == _s.DEVNULL
    assert kwargs.get("stderr") == _s.DEVNULL


def test_present_hook_uses_hermes_home_not_cwd(
        hermes_home, monkeypatch, tmp_path):
    """The hook resolved must honor HERMES_HOME (get_hermes_home), not the
    repo checkout or CWD."""
    scripts = hermes_home / "scripts"
    scripts.mkdir()
    (scripts / "curator-governance-hook.py").write_text("# hook\n")
    monkeypatch.chdir(tmp_path)  # different CWD
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv

        class P:
            pid = 1
        return P()

    with patch.object(subprocess, "Popen", side_effect=fake_popen):
        gateway_run._spawn_curator_governance_hook("s", lambda m: None)
    assert str(scripts / "curator-governance-hook.py") in captured["argv"][1]


# ── Popen error is contained ───────────────────────────────────────────────


def test_popen_error_is_contained(hermes_home, caplog):
    scripts = hermes_home / "scripts"
    scripts.mkdir()
    (scripts / "curator-governance-hook.py").write_text("# hook\n")
    with patch.object(subprocess, "Popen",
                      side_effect=OSError("spawn failed")):
        # must not raise
        gateway_run._spawn_curator_governance_hook(
            "curator summary", lambda m: None)


def test_on_summary_error_does_not_prevent_hook_spawn(hermes_home):
    scripts = hermes_home / "scripts"
    scripts.mkdir()
    (scripts / "curator-governance-hook.py").write_text("# hook\n")

    def bad_summary(_msg):
        raise RuntimeError("logger exploded")

    recorded = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = argv

        class P:
            pid = 7
        return P()

    with patch.object(subprocess, "Popen", side_effect=fake_popen):
        gateway_run._spawn_curator_governance_hook("curator summary", bad_summary)
    assert "--direct" in recorded["argv"], (
        "summary-logging failure must not prevent governance hook spawn")