"""The managed-node rung of the agent-browser probe must come from iter_hermes_node_dirs().

Behavioural pin for the hermes_cli/doctor_live.py consolidation in #49259: the probe's
managed-node candidate list is the canonical one, so a fix in the helper reaches doctor.
"""

import os
import shutil

from hermes_cli import doctor_live


def test_browser_available_probes_canonical_managed_node_dirs(tmp_path, monkeypatch):
    managed_bin = tmp_path / "managed" / "bin"
    managed_bin.mkdir(parents=True)
    browser = managed_bin / "agent-browser"
    browser.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    browser.chmod(0o755)

    def fake_which(cmd, path=None):
        if path is None:
            return None
        candidate = os.path.join(path, cmd)
        return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None

    def _not_found(*_args, **_kwargs):
        raise FileNotFoundError("agent-browser not installed")

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", tmp_path / "empty-home")
    monkeypatch.setattr("hermes_cli.doctor.PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr("hermes_constants.iter_hermes_node_dirs", lambda home=None: [managed_bin.parent, managed_bin])
    monkeypatch.setattr("tools.browser_tool_install._find_agent_browser", _not_found)

    assert doctor_live._browser_available() is True
