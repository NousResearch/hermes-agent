"""Dependency installation publication contracts."""

from pathlib import Path


def test_member_inputs_freezes_discovered_plugins(monkeypatch):
    import pm.install as install
    import pm.workspace

    members = [Path("/tmp/plugin-a")]
    monkeypatch.setattr(pm.workspace, "enabled_member_dirs", lambda: members)

    inputs = install._member_inputs(None)

    assert inputs == {"plugin_dirs": members}
