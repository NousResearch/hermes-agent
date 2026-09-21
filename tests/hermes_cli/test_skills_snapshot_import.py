"""`hermes skills snapshot import` must not report success for a snapshot it did not restore.

Regression for #107640: unattended imports answer every `Confirm [y/N]:` with no,
printed `Snapshot import complete.` and exited 0 while `hermes skills list` showed zero
hub skills. The contract: `do_install` reports whether the skill is present afterwards,
`do_snapshot_import` reports whether every identified skill was restored, and the CLI
path turns that into the exit status a recovery script reads.
"""

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from hermes_cli import skills_hub
from hermes_cli.skills_hub import _snapshot_cli, do_install, do_snapshot_import


def _console() -> tuple[Console, StringIO]:
    buffer = StringIO()
    return Console(file=buffer, force_terminal=False, width=200), buffer


def _snapshot(tmp_path, identifiers):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({
        "taps": [],
        "skills": [{"identifier": ident, "name": ident.rsplit("/", 1)[-1], "category": ""}
                   for ident in identifiers],
    }), encoding="utf-8")
    return path


def test_snapshot_import_reports_unrestored_skills_and_the_cli_exits_nonzero(monkeypatch, tmp_path):
    present = {"official/a/alpha": True, "official/b/beta": False}
    installed = []

    def fake_install(identifier, category="", force=False, console=None, **_kwargs):
        installed.append(identifier)
        return present[identifier]

    monkeypatch.setattr(skills_hub, "do_install", fake_install)
    snapshot = _snapshot(tmp_path, list(present))
    console, out = _console()

    assert do_snapshot_import(str(snapshot), console=console) is False
    assert installed == list(present), "every identified skill is still attempted"
    text = out.getvalue()
    assert "1 of 2" in text and "official/b/beta" in text
    assert "Snapshot import complete" not in text

    with pytest.raises(SystemExit) as exit_info:
        _snapshot_cli(SimpleNamespace(snapshot_action="import", input=str(snapshot), force=False))
    assert exit_info.value.code == 1

    present["official/b/beta"] = True
    console, out = _console()
    assert do_snapshot_import(str(snapshot), console=console) is True
    assert "2 skill(s) restored" in out.getvalue()
    assert _snapshot_cli(SimpleNamespace(snapshot_action="import", input=str(snapshot), force=False)) is None


def test_do_install_reports_presence_after_a_cancelled_and_a_confirmed_prompt(monkeypatch, tmp_path):
    import tools.skills_guard as guard
    import tools.skills_hub as hub
    import tools.skills_hub_install as hub_install
    import tools.skills_hub_search as hub_search

    class _Source:
        def inspect(self, identifier):
            return type("Meta", (), {"extra": {}, "identifier": identifier, "name": "gamma", "path": "gamma"})()

        def fetch(self, identifier):
            return type("Bundle", (), {
                "name": "gamma", "files": {"SKILL.md": "---\ndescription: ok\n---\n# body\n"},
                "source": "github", "identifier": identifier, "trust_level": "community", "metadata": {},
            })()

    quarantine = tmp_path / "skills" / ".hub" / "quarantine" / "gamma"
    quarantine.mkdir(parents=True)
    installs = []

    def _install_from_quarantine(q_path, name, category, bundle, result):
        installs.append(name)
        target = tmp_path / "skills" / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("Lock", (), {"get_installed": lambda self, n: None})())
    monkeypatch.setattr(hub_search, "create_source_router", lambda auth: [_Source()])
    monkeypatch.setattr(hub_install, "quarantine_bundle", lambda bundle: quarantine)
    monkeypatch.setattr(hub_install, "install_from_quarantine", _install_from_quarantine)
    monkeypatch.setattr(guard, "scan_skill", lambda skill_path, source="community": guard.ScanResult(
        skill_name="gamma", source=source, trust_level="community", verdict="safe"))
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, "ok"))
    monkeypatch.setattr(skills_hub, "_finish_change", lambda *args, **kwargs: None)
    monkeypatch.setattr(skills_hub, "_announce_blueprint", lambda *args, **kwargs: None)

    answers = iter(["n", "y"])
    monkeypatch.setattr(skills_hub, "input", lambda prompt="": next(answers), raising=False)
    console, _out = _console()

    assert do_install("owner/repo/gamma", console=console) is False
    assert installs == [], "a cancelled prompt installs nothing"
    quarantine.mkdir(parents=True, exist_ok=True)
    assert do_install("owner/repo/gamma", console=console) is True
    assert installs == ["gamma"]
