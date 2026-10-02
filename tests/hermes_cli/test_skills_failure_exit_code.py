"""A failed ``hermes skills install|update|uninstall|snapshot import`` exits non-zero (#118301).

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


def test_declined_install_confirm_is_not_a_failure(monkeypatch, tmp_path):
    """The user answering "no" at the install prompt is a no-op they own: None, exit 0."""
    installs, _ = _scan_gate_env(monkeypatch, tmp_path, verdict="safe")
    monkeypatch.setattr(cli_hub, "_confirm_install", lambda *a, **k: False)
    console, _ = _console()

    assert cli_hub.do_install("org/repo/risky-skill", console=console, skip_confirm=False) is None
    assert installs == []


def test_pinned_source_that_is_gone_reports_failure(monkeypatch):
    """``do_update`` pins the lockfile's registry; when no adapter matches, the install is refused
    ("Refusing to resolve ...") and must report False, not a success that over-counts the update."""
    import tools.skills_hub as hub

    class _Source:
        def source_id(self):
            return "skills-sh"

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [_Source()])
    monkeypatch.setattr(cli_hub, "_record_skill_install", lambda *a, **k: None)
    monkeypatch.setattr(cli_hub, "_resolve_source_meta_and_bundle",
                        lambda *a, **k: pytest.fail("a pinned-source refusal must not resolve elsewhere"))
    console, sink = _console()

    assert cli_hub.do_install("org/x", console=console, skip_confirm=True, source_id="gone-registry") is False
    assert "Refusing to resolve" in sink.getvalue()


def test_update_whose_registry_is_gone_is_not_updated(monkeypatch):
    """End to end through the real do_install: the pinned-source refusal surfaces as Not updated."""
    import tools.skills_hub as hub
    import tools.skills_hub_install as hub_install

    class _Source:
        def source_id(self):
            return "skills-sh"

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(cli_hub, "_sources", lambda: [_Source()])
    monkeypatch.setattr(cli_hub, "_record_skill_install", lambda *a, **k: None)
    monkeypatch.setattr(hub_install, "check_for_skill_updates", lambda **_k: [
        {"name": "orphan", "identifier": "org/orphan", "source": "gone-registry", "status": "update_available"}])
    monkeypatch.setattr(hub, "HubLockFile", lambda: type(
        "Lock", (), {"get_installed": lambda self, n: None})())
    console, sink = _console()

    assert cli_hub.do_update(console=console) is False
    out = sink.getvalue()
    assert "Not updated:" in out and "orphan" in out
    assert "Updated 1 skill(s)" not in out


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


@pytest.mark.parametrize("invalidate_cache", [True, False])
def test_uninstall_success_applies_the_change(monkeypatch, invalidate_cache):
    """A removal reports True and still reaches the cache step (invalidate now, or defer)."""
    import tools.skills_hub_install as hub_install

    monkeypatch.setattr(hub_install, "uninstall_skill", lambda name: (True, f"Uninstalled '{name}'"))
    finished: list = []
    monkeypatch.setattr(cli_hub, "_finish_change", lambda c, invalidate: finished.append(invalidate))
    console, _ = _console()

    assert cli_hub.do_uninstall("hub-skill", console=console, skip_confirm=True,
                                invalidate_cache=invalidate_cache) is True
    assert finished == [invalidate_cache]


def test_uninstall_refusal_does_not_touch_the_cache(monkeypatch):
    import tools.skills_hub_install as hub_install

    monkeypatch.setattr(hub_install, "uninstall_skill", lambda name: (False, "not a hub skill"))
    finished: list = []
    monkeypatch.setattr(cli_hub, "_finish_change", lambda *a, **k: finished.append(a))
    console, _ = _console()

    assert cli_hub.do_uninstall("builtin-skill", console=console, skip_confirm=True) is False
    assert finished == []


def test_declined_uninstall_is_not_a_failure(monkeypatch):
    import tools.skills_hub_install as hub_install

    monkeypatch.setattr(cli_hub, "_confirm_or_cancel", lambda *a, **k: False)
    monkeypatch.setattr(hub_install, "uninstall_skill",
                        lambda name: pytest.fail("a declined prompt must not uninstall"))
    console, _ = _console()

    assert cli_hub.do_uninstall("hub-skill", console=console, skip_confirm=False) is None


def _snapshot(tmp_path, names):
    import json

    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"taps": [], "skills": [
        {"name": n, "identifier": f"org/{n}"} for n in names]}), encoding="utf-8")
    return str(path)


def test_snapshot_import_reports_refused_entries(monkeypatch, tmp_path):
    """Restoring a snapshot whose entries the scan gate refuses must not end in "complete"."""
    outcomes = {"good": True, "blocked": False}
    monkeypatch.setattr(cli_hub, "do_install", lambda identifier, **_k: outcomes[identifier.split("/", 1)[1]])
    console, sink = _console()

    assert cli_hub.do_snapshot_import(_snapshot(tmp_path, ["good", "blocked"]), console=console) is False
    out = sink.getvalue()
    assert "Snapshot import complete" not in out
    assert "Not installed: blocked" in out


def test_snapshot_import_all_installed_is_not_a_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_hub, "do_install", lambda identifier, **_k: True)
    console, sink = _console()

    assert cli_hub.do_snapshot_import(_snapshot(tmp_path, ["good"]), console=console) is None
    assert "Snapshot import complete" in sink.getvalue()


def test_snapshot_import_unreadable_file_is_a_failure(tmp_path):
    console, _ = _console()

    assert cli_hub.do_snapshot_import(str(tmp_path / "missing.json"), console=console) is False


def test_snapshot_import_with_real_scan_gate_exits_non_zero(monkeypatch, tmp_path):
    """Two snapshot entries, both refused by the real ``should_allow_install`` policy, through
    the real ``hermes skills snapshot import`` entry point: exit 1, nothing installed."""
    from hermes_cli.main import main

    installs, audit = _scan_gate_env(monkeypatch, tmp_path, verdict="dangerous")
    monkeypatch.setattr(sys, "argv", ["hermes", "skills", "snapshot", "import",
                                      _snapshot(tmp_path, ["risky-skill", "risky-skill-2"])])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert installs == []
    assert [row[0] for row in audit] == ["BLOCKED", "BLOCKED"]


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
