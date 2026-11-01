"""#87650: a no-agent cron script fired by the gateway is owned by the gateway's cron ticker,
not a delegate_task descendant. A gateway (re)started from a fenced lineage carries
HERMES_DELEGATED_CHILD_CONTEXT, and without the owner-boundary drop every `hermes kanban
<mutation>` the script makes is refused. Outside the gateway, or inside an in-process delegate
child, the fence must survive."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, delegated_child_context
from cron.scheduler_script import _run_job_script


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    script = home / "scripts" / "marker_probe.py"
    script.write_text(f"import os\nprint('MARKER=' + repr(os.environ.get({DELEGATED_CHILD_ENV_MARKER!r})))\n")
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, str(tmp_path / "fenced-root"))
    return script


def test_gateway_cron_script_runs_without_inherited_marker(probe, monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    ok, out = _run_job_script(str(probe))
    assert ok, out
    assert "MARKER=None" in out


def test_non_gateway_lineage_keeps_fence(probe, monkeypatch):
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    ok, out = _run_job_script(str(probe))
    assert ok, out
    assert "fenced-root" in out


def test_in_process_delegate_child_keeps_fence(probe, monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    with delegated_child_context("child"):
        ok, out = _run_job_script(str(probe))
    assert ok, out
    assert "MARKER=None" not in out
