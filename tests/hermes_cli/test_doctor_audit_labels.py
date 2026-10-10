"""Doctor npm-audit row labels must name the tree that is actually audited.

Regression for #135928: the row auditing PROJECT_ROOT with --workspaces=false
was labelled "Browser tools (agent-browser)", but agent-browser is a pm-store
native executable with no npm dependency tree and no lockfile — it never
appears in the repo's package-lock.json. The mislabelled row attributed the
root workspace's own Electron/devtool findings to agent-browser, and its
remedy text pointed users at a lockfile that does not exist.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from hermes_cli import doctor, doctor_tools


def _audit_json(high=2):
    return json.dumps({
        "metadata": {"vulnerabilities": {
            "critical": 0, "high": high, "moderate": 0,
            "low": 0, "info": 0, "total": high,
        }},
    })


def test_root_row_is_labelled_for_the_tree_it_audits(tmp_path, capsys):
    """The PROJECT_ROOT row must be labelled "root workspace" — its count is
    resolved from the committed root package-lock.json — never "agent-browser"."""
    (tmp_path / "node_modules").mkdir()
    completed = subprocess.CompletedProcess([], 0, stdout=_audit_json(high=2), stderr="")
    with (
        patch.object(doctor_tools, "_pm_tool_path", return_value=Path("/fake/npm")),
        patch.object(doctor, "PROJECT_ROOT", tmp_path),
        # Force the WhatsApp-bridge import to fail so the resolver falls back to
        # PROJECT_ROOT/scripts/whatsapp-bridge (absent under tmp_path → row skipped).
        patch.dict(sys.modules, {"gateway.platforms.whatsapp_common": None}),
        patch.object(doctor_tools.subprocess, "run", return_value=completed),
    ):
        finding = doctor_tools._check_npm_audit(False)
    out = capsys.readouterr().out
    assert "agent-browser" not in out
    assert "root workspace has 2 npm vulnerabilities" in finding.issues
