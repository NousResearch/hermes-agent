"""A failed ``hermes skills install|update|uninstall`` exits non-zero (#118301).

The Desktop spawns these as detached actions and learns their outcome only from the
exit code (``apps/desktop/src/store/hub-actions.ts`` toasts on non-zero), so a refused
action that exits 0 reads as "the button did nothing" while the reason sits unread in
the action log. The scan gate is the common case: a blocked install used to print
"Not installed: ..." and exit 0.
"""

import sys
from io import StringIO

import pytest
from rich.console import Console

import hermes_cli.skills_hub as cli_hub


def _console():
    sink = StringIO()
    return Console(file=sink, force_terminal=False, color_system=None, width=200), sink


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
    monkeypatch.setattr(cli_hub, "_sources", lambda: [object()])
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


@pytest.mark.parametrize("verdict", ["dangerous", "caution"])
def test_scan_blocked_install_reports_failure(monkeypatch, tmp_path, verdict):
    installs, audit = _scan_gate_env(monkeypatch, tmp_path, verdict=verdict)
    console, sink = _console()

    assert cli_hub.do_install("org/repo/risky-skill", console=console, skip_confirm=True) is False
    assert installs == []
    assert "Not installed:" in sink.getvalue()
    assert audit and audit[0][0] == "BLOCKED"


def test_successful_install_reports_success(monkeypatch, tmp_path):
    installs, _ = _scan_gate_env(monkeypatch, tmp_path, verdict="safe")
    monkeypatch.setattr(cli_hub, "_finish_change", lambda *a, **k: None)
    console, _ = _console()

    assert cli_hub.do_install("org/repo/risky-skill", console=console, skip_confirm=True) is True
    assert installs == ["risky-skill"]


def test_already_installed_is_not_a_failure(monkeypatch, tmp_path):
    """No-op the user owns: exits 0 (the Desktop never offers install on an installed row)."""
    _scan_gate_env(monkeypatch, tmp_path, installed={"install_path": "risky-skill"})
    console, _ = _console()

    assert cli_hub.do_install("org/repo/risky-skill", console=console, skip_confirm=True) is None


def test_unresolvable_source_reports_failure(monkeypatch):
    import tools.skills_hub as hub

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [object()])
    monkeypatch.setattr(cli_hub, "_resolve_source_meta_and_bundle",
                        lambda identifier, sources: (None, None, None))
    monkeypatch.setattr(cli_hub, "_record_skill_install", lambda *a, **k: None)
    console, sink = _console()

    assert cli_hub.do_install("org/repo/gone", console=console, skip_confirm=True) is False
    assert "Could not download" in sink.getvalue()


def _update_env(monkeypatch, outcomes):
    import tools.skills_hub as hub
    import tools.skills_hub_install as hub_install

    names = list(outcomes)
    monkeypatch.setattr(hub_install, "check_for_skill_updates", lambda **_k: [
        {"name": n, "identifier": f"org/{n}", "source": "github", "status": "update_available"}
        for n in names])
    monkeypatch.setattr(hub, "HubLockFile", lambda: type(
        "Lock", (), {"get_installed": lambda self, n: None})())
    monkeypatch.setattr(cli_hub, "do_install",
                        lambda identifier, **_k: outcomes[identifier.split("/", 1)[1]])


def test_update_reports_failure_when_a_new_version_is_refused(monkeypatch):
    _update_env(monkeypatch, {"good": True, "blocked": False})
    console, sink = _console()

    assert cli_hub.do_update(console=console) is False
    out = sink.getvalue()
    assert "Updated 1 skill(s)" in out
    assert "Not updated:" in out and "blocked" in out


def test_update_all_succeeding_is_not_a_failure(monkeypatch):
    _update_env(monkeypatch, {"good": True})
    console, sink = _console()

    assert cli_hub.do_update(console=console) is None
    assert "Not updated" not in sink.getvalue()


def test_uninstall_refusal_reports_failure(monkeypatch):
    import tools.skills_hub_install as hub_install

    monkeypatch.setattr(hub_install, "uninstall_skill",
                        lambda name: (False, f"'{name}' is not a hub-installed skill (may be a builtin)"))
    console, _ = _console()

    assert cli_hub.do_uninstall("builtin-skill", console=console, skip_confirm=True) is False


def test_uninstall_success(monkeypatch):
    import tools.skills_hub_install as hub_install

    monkeypatch.setattr(hub_install, "uninstall_skill", lambda name: (True, f"Uninstalled '{name}'"))
    monkeypatch.setattr(cli_hub, "_finish_change", lambda *a, **k: None)
    console, _ = _console()

    assert cli_hub.do_uninstall("hub-skill", console=console, skip_confirm=True) is True


# --- the process exit code, through the real `hermes` entry point ---


@pytest.mark.parametrize("argv, target, outcome, code", [
    (["skills", "install", "org/repo/x", "--yes"], "do_install", False, 1),
    (["skills", "install", "org/repo/x", "--yes"], "do_install", True, 0),
    (["skills", "install", "org/repo/x", "--yes"], "do_install", None, 0),
    (["skills", "update"], "do_update", False, 1),
    (["skills", "update"], "do_update", None, 0),
    (["skills", "uninstall", "x", "--yes"], "do_uninstall", False, 1),
    (["skills", "uninstall", "x", "--yes"], "do_uninstall", True, 0),
])
def test_cli_exit_code_follows_the_action_outcome(monkeypatch, argv, target, outcome, code):
    from hermes_cli.main import main

    monkeypatch.setattr(cli_hub, target, lambda *a, **k: outcome)
    monkeypatch.setattr(sys, "argv", ["hermes", *argv])
    try:
        main()
        exit_code = 0
    except SystemExit as exc:
        exit_code = exc.code or 0
    assert exit_code == code
