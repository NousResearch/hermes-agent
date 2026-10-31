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


@pytest.mark.parametrize("first_present", [False, True])
def test_snapshot_import_reports_unrestored_skills_and_the_cli_exits_nonzero(
    monkeypatch, tmp_path, first_present
):
    present = {"official/a/alpha": first_present, "official/b/beta": False}
    installed = []

    def fake_install(identifier, category="", force=False, console=None, **_kwargs):
        installed.append(identifier)
        return present[identifier]

    monkeypatch.setattr(skills_hub, "do_install", fake_install)
    snapshot = _snapshot(tmp_path, list(present))
    original_snapshot = snapshot.read_bytes()
    console, out = _console()

    assert do_snapshot_import(str(snapshot), console=console) is False
    assert installed == list(present), "every identified skill is still attempted"
    text = out.getvalue()
    assert f"{int(first_present)} of 2" in text
    missing = ", ".join(identifier for identifier, restored in present.items() if not restored)
    assert f"Not installed (cancelled, blocked or unavailable): {missing}" in text
    assert "Snapshot import complete" not in text

    with pytest.raises(SystemExit) as exit_info:
        _snapshot_cli(SimpleNamespace(snapshot_action="import", input=str(snapshot), force=False))
    assert exit_info.value.code == 1

    console, out = _console()
    assert skills_hub.handle_skills_slash(f"/skills snapshot import {snapshot}", console=console) is None
    assert "Snapshot import failed" in out.getvalue()

    present["official/a/alpha"] = True
    present["official/b/beta"] = True
    console, out = _console()
    assert do_snapshot_import(str(snapshot), console=console) is True
    assert "2 skill(s) restored" in out.getvalue()
    assert _snapshot_cli(SimpleNamespace(snapshot_action="import", input=str(snapshot), force=False)) is None
    assert snapshot.read_bytes() == original_snapshot


@pytest.mark.parametrize("contents, expected", [
    (None, False),
    ("{", False),
    ('{"skills": [], "taps": []}', True),
    ('\ufeff{"skills": [], "taps": []}', True),
])
def test_snapshot_import_input_failures_and_empty_snapshot_exit_status(tmp_path, contents, expected):
    snapshot = tmp_path / "snapshot.json"
    if contents is not None:
        snapshot.write_text(contents, encoding="utf-8")
    console, _out = _console()
    assert do_snapshot_import(str(snapshot), console=console) is expected
    args = SimpleNamespace(snapshot_action="import", input=str(snapshot), force=False)
    if expected:
        assert _snapshot_cli(args) is None
    else:
        with pytest.raises(SystemExit) as exit_info:
            _snapshot_cli(args)
        assert exit_info.value.code == 1


def test_do_install_reports_presence_after_a_cancelled_and_a_confirmed_prompt(monkeypatch, tmp_path):
    import tools.skills_guard as guard
    import tools.skills_hub as hub
    import tools.skills_hub_install as hub_install
    import tools.skills_hub_search as hub_search
    from hermes_cli.observability import shared_metrics_events

    class _Source:
        def source_id(self):
            return "github"

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
    events = []
    installed_entry = None

    class _Lock:
        def get_installed(self, name):
            return installed_entry

    def record_install(*, kind, source, name, outcome, failure_class=None, registry=None, error=None):
        events.append({"kind": kind, "source": source, "name": name, "outcome": outcome,
                       "failure_class": failure_class, "registry": registry, "error": error})

    monkeypatch.setattr(shared_metrics_events, "record_extension_install", record_install)

    def _install_from_quarantine(q_path, name, category, bundle, result):
        nonlocal installed_entry
        installs.append(name)
        target = tmp_path / "skills" / name
        target.mkdir(parents=True, exist_ok=True)
        installed_entry = {"install_path": name}
        return target

    def _quarantine_bundle(bundle):
        quarantine.mkdir(parents=True, exist_ok=True)
        return quarantine

    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "HubLockFile", _Lock)
    monkeypatch.setattr(hub_search, "create_source_router", lambda auth: [_Source()])
    monkeypatch.setattr(hub_install, "quarantine_bundle", _quarantine_bundle)
    monkeypatch.setattr(hub_install, "install_from_quarantine", _install_from_quarantine)
    monkeypatch.setattr(guard, "scan_skill", lambda skill_path, source="community": guard.ScanResult(
        skill_name="gamma", source=source, trust_level="community", verdict="safe"))
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, "ok"))
    monkeypatch.setattr(skills_hub, "_finish_change", lambda *args, **kwargs: None)
    monkeypatch.setattr(skills_hub, "_announce_blueprint", lambda *args, **kwargs: None)

    monkeypatch.setattr(skills_hub, "input", lambda prompt="": "n", raising=False)
    console, _out = _console()

    assert do_install("owner/repo/gamma", console=console) is None
    assert installs == [], "a cancelled prompt installs nothing"
    assert events == [], "cancelled installs emit no install metric"
    snapshot = _snapshot(tmp_path, ["owner/repo/gamma"])
    assert do_snapshot_import(str(snapshot), console=console) is False
    with pytest.raises(SystemExit) as exit_info:
        _snapshot_cli(SimpleNamespace(snapshot_action="import", input=str(snapshot), force=False))
    assert exit_info.value.code == 1
    assert skills_hub.skills_command(SimpleNamespace(
        skills_action="install", identifier="owner/repo/gamma", category="", force=False,
    )) is None, "ordinary install cancellation keeps upstream's successful exit status"
    assert installs == []
    assert events == []
    monkeypatch.setattr(skills_hub, "input", lambda prompt="": "y")
    quarantine.mkdir(parents=True, exist_ok=True)
    assert do_install("owner/repo/gamma", console=console) is True
    assert installs == ["gamma"]
    assert events == [{"kind": "skill", "source": "hub", "name": "gamma", "outcome": "success",
                       "failure_class": None, "registry": "github", "error": None}]
    assert do_install("owner/repo/gamma", console=console) is True
    assert installs == ["gamma"], "an existing skill is not reinstalled without force"
    assert len(events) == 1, "an existing skill emits no duplicate install metric"
