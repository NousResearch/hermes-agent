"""``check_runtime`` does not probe the venv again after a launch that just found it current."""

import sys
import types

import pytest

from hermes_cli import venv_sync


@pytest.fixture
def probes(monkeypatch, tmp_path):
    """Run check_runtime on a managed checkout and record the ``allow_incomplete`` each activate got."""
    seen = []
    fake_pm = types.SimpleNamespace(activate=lambda **kw: seen.append(kw.get("allow_incomplete", False)) or [])
    monkeypatch.setitem(sys.modules, "pm", fake_pm)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr("hermes_cli.steward.read_install_stamp", lambda root: {"updateMechanism": "self"})
    monkeypatch.setattr(venv_sync, "_launch_venv_verdicts", {})
    return tmp_path, seen


def test_a_launch_that_found_the_venv_current_skips_the_second_probe(probes):
    root, seen = probes
    venv_sync._launch_venv_verdicts[root.resolve()] = True
    assert venv_sync.check_runtime(root) is None
    assert seen == [True]


def test_a_stale_launch_verdict_keeps_the_full_check(probes):
    root, seen = probes
    venv_sync._launch_venv_verdicts[root.resolve()] = False
    venv_sync.check_runtime(root)
    assert seen == [False]


def test_without_a_launch_probe_in_this_process_the_full_check_runs(probes):
    root, seen = probes
    venv_sync.check_runtime(root)
    assert seen == [False]
