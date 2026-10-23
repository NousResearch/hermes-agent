"""Dependency-gated gateway cases run in a job that has their dependency.

The ordinary pytest slices omit the ``messaging`` extra, so a gateway test that probes for the REAL
Discord/Telegram/Slack distribution skips there. Only the e2e job installs that extra; every such
file must be named by its messaging step (and start the e2e lane), or it is never run anywhere.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SDK_PROBE = re.compile(r"PathFinder\.find_spec\(\s*[\"'](discord|telegram|slack_bolt|slack_sdk)[\"']\s*\)")


def test_every_messaging_gated_gateway_file_runs_in_the_messaging_job():
    gated = sorted(str(path.relative_to(_REPO)) for path in (_REPO / "tests" / "gateway").glob("test_*.py")
                   if _SDK_PROBE.search(path.read_text(encoding="utf-8")))
    assert gated, "no messaging-gated gateway file found; the probe pattern drifted"
    yaml = pytest.importorskip("hermes_yaml")
    job = yaml.safe_load((_REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))["jobs"]["e2e"]
    extras = next(step["with"]["extras"] for step in job["steps"] if "setup-pm" in step.get("uses", ""))
    assert "messaging" in extras
    script = "\n".join(step.get("run", "") for step in job["steps"] if "messaging-gated" in step.get("name", ""))
    missing = [path for path in gated if path not in script]
    assert not missing, f"never run with their SDK installed: {missing}"
    # A skip in that step is a coverage hole, not a pass.
    assert "-rs" in script and "skipped" in script
    spec = importlib.util.spec_from_file_location("classify_changes", _REPO / "scripts" / "ci" / "classify_changes.py")
    classify_changes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classify_changes)
    assert all(classify_changes.classify([path])["e2e"] for path in gated), "editing one must start the e2e job"
