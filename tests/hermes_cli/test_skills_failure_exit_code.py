"""A failed ``hermes skills install|update|uninstall|snapshot import`` exits non-zero (#118301).

The Desktop spawns these as detached actions and learns their outcome only from the
exit code (``apps/desktop/src/store/hub-actions.ts`` toasts on non-zero), so a refused
action that exits 0 reads as "the button did nothing" while the reason sits unread in
the action log. The scan gate is the common case: a blocked install used to print
"Not installed: ..." and exit 0.
"""

import sys
import pytest

import hermes_cli.skills_hub as cli_hub


def _scan_gate_env(monkeypatch, tmp_path, *, verdict="dangerous", installed=None):
    """Wire ``do_install`` down to the real scan-policy decision for a community skill."""
    import tools.skills_guard as guard
    import tools.skills_hub as hub
    import tools.skills_hub_install as hub_install

    bundle = type("Bundle", (), {
        "name": "risky-skill", "files": {"SKILL.md": "---\nname: risky-skill\n---\n"},
        "source": "skills-sh", "identifier": "org/repo/risky-skill", "trust_level": "community",
        "metadata": {},
    })()
    q_path = tmp_path / "skills" / ".hub" / "quarantine" / "risky-skill"
    q_path.mkdir(parents=True)
    audit: list = []

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "append_audit_log", lambda *a, **k: audit.append(a))
    monkeypatch.setattr(hub, "HubLockFile", lambda: type(
        "Lock", (), {"get_installed": lambda self, n: installed})())
    monkeypatch.setattr(cli_hub, "_sources", lambda: [type("Source", (), {"source_id": lambda self: "skills-sh"})()])
    monkeypatch.setattr(cli_hub, "_resolve_source_meta_and_bundle",
                        lambda identifier, sources: (None, bundle, sources[0]))
    monkeypatch.setattr(cli_hub, "_record_skill_install", lambda *a, **k: None)
    monkeypatch.setattr(hub_install, "quarantine_bundle", lambda b: q_path)
    finding = guard.Finding("remote_fetch", "critical", "supply_chain", "SKILL.md", 40,
                            'curl -s "https://example.com"', "remote fetch")
    monkeypatch.setattr(cli_hub, "_scan_quarantined", lambda *a, **k: guard.ScanResult(
        skill_name="risky-skill", source="org/repo/risky-skill", trust_level="community",
        verdict=verdict, findings=[finding, finding]))
    installs: list = []

    def _install(q, name, category, b, result):
        installs.append(name)
        d = tmp_path / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(hub_install, "install_from_quarantine", _install)
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    return installs, audit


# --- the process exit code, through the real `hermes` entry point ---


@pytest.mark.parametrize("argv, target, outcome, code", [
    (["skills", "install", "org/repo/x", "--yes"], "do_install", False, 1),
    (["skills", "install", "org/repo/x", "--yes"], "do_install", True, 0),
    (["skills", "install", "org/repo/x", "--yes"], "do_install", None, 0),
    (["skills", "update"], "do_update", False, 1),
    (["skills", "update"], "do_update", None, 0),
    (["skills", "uninstall", "x", "--yes"], "do_uninstall", False, 1),
    (["skills", "uninstall", "x", "--yes"], "do_uninstall", True, 0),
    (["skills", "snapshot", "import", "snap.json"], "do_snapshot_import", False, 1),
    (["skills", "snapshot", "import", "snap.json"], "do_snapshot_import", None, 0),
    # The real scan gate, not a stubbed outcome: a refused community skill installs nothing and exits 1.
    (["skills", "install", "org/repo/risky-skill", "--yes"], "scan_gate", "dangerous", 1),
    (["skills", "install", "org/repo/risky-skill", "--yes"], "scan_gate", "caution", 1),
])
def test_cli_exit_code_follows_the_action_outcome(monkeypatch, tmp_path, capsys, argv, target, outcome, code):
    from hermes_cli.main import main

    if target == "scan_gate":
        installs, audit = _scan_gate_env(monkeypatch, tmp_path, verdict=outcome)
    else:
        monkeypatch.setattr(cli_hub, target, lambda *a, **k: outcome)
    if argv[1] == "update":
        # Never run hermes_cli.main with `update` in argv (it can turn into a real self-update): drive
        # the same router `hermes skills` returns its exit code from.
        from argparse import Namespace
        exit_code = cli_hub.skills_command(Namespace(skills_action="update", name=None, force=False)) or 0
    else:
        monkeypatch.setattr(sys, "argv", ["hermes", *argv])
        try:
            main()
            exit_code = 0
        except SystemExit as exc:
            exit_code = exc.code or 0
    assert exit_code == code
    if target == "scan_gate":
        assert installs == [] and "Not installed:" in capsys.readouterr().out
        assert audit and audit[0][0] == "BLOCKED"
