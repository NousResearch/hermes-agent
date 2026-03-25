from io import StringIO

import pytest
from rich.console import Console

from hermes_cli.skills_hub import do_approve, do_audit, do_check, do_install, do_inspect, do_list, do_quarantine_list, do_reject, do_update, handle_skills_slash


class _DummyLockFile:
    def __init__(self, installed):
        self._installed = installed

    def list_installed(self):
        return self._installed


@pytest.fixture()
def hub_env(monkeypatch, tmp_path):
    """Set up isolated hub directory paths and return (monkeypatch, tmp_path)."""
    import tools.skills_hub as hub

    hub_dir = tmp_path / "skills" / ".hub"
    original_lock_file_cls = hub.HubLockFile
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "HUB_DIR", hub_dir)
    monkeypatch.setattr(hub, "LOCK_FILE", hub_dir / "lock.json")
    monkeypatch.setattr(hub, "QUARANTINE_DIR", hub_dir / "quarantine")
    monkeypatch.setattr(hub, "AUDIT_LOG", hub_dir / "audit.log")
    monkeypatch.setattr(hub, "TAPS_FILE", hub_dir / "taps.json")
    monkeypatch.setattr(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache")
    monkeypatch.setattr(hub, "HubLockFile", lambda: original_lock_file_cls(hub.LOCK_FILE))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    return hub_dir


# ---------------------------------------------------------------------------
# Fixtures for common skill setups
# ---------------------------------------------------------------------------

_HUB_ENTRY = {"name": "hub-skill", "source": "github", "trust_level": "community"}

_ALL_THREE_SKILLS = [
    {"name": "hub-skill", "category": "x", "description": "hub"},
    {"name": "builtin-skill", "category": "x", "description": "builtin"},
    {"name": "local-skill", "category": "x", "description": "local"},
]

_BUILTIN_MANIFEST = {"builtin-skill": "abc123"}


@pytest.fixture()
def three_source_env(monkeypatch, hub_env):
    """Populate hub/builtin/local skills for source-classification tests."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([_HUB_ENTRY]))
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: list(_ALL_THREE_SKILLS))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: dict(_BUILTIN_MANIFEST))

    return hub_env


def _capture(source_filter: str = "all") -> str:
    """Run do_list into a string buffer and return the output."""
    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_list(source_filter=source_filter, console=console)
    return sink.getvalue()


def _capture_check(monkeypatch, results, name=None) -> str:
    import tools.skills_hub as hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    monkeypatch.setattr(hub, "check_for_skill_updates", lambda **_kwargs: results)
    do_check(name=name, console=console)
    return sink.getvalue()


def _capture_update(monkeypatch, results) -> tuple[str, list[tuple[str, str, bool]]]:
    import tools.skills_hub as hub
    import hermes_cli.skills_hub as cli_hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    installs = []

    monkeypatch.setattr(hub, "check_for_skill_updates", lambda **_kwargs: results)
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("L", (), {
        "get_installed": lambda self, name: {"install_path": "category/" + name}
    })())
    monkeypatch.setattr(cli_hub, "do_install", lambda identifier, category="", force=False, console=None: installs.append((identifier, category, force)))

    do_update(console=console)
    return sink.getvalue(), installs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_do_list_initializes_hub_dir(monkeypatch, hub_env):
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: [])
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

    hub_dir = hub_env
    assert not hub_dir.exists()

    _capture()

    assert hub_dir.exists()
    assert (hub_dir / "lock.json").exists()
    assert (hub_dir / "quarantine").is_dir()
    assert (hub_dir / "index-cache").is_dir()


def test_do_list_distinguishes_hub_builtin_and_local(three_source_env):
    output = _capture()

    assert "hub-skill" in output
    assert "builtin-skill" in output
    assert "local-skill" in output
    assert "1 hub-installed, 1 builtin, 1 local" in output


def test_do_list_filter_local(three_source_env):
    output = _capture(source_filter="local")

    assert "local-skill" in output
    assert "builtin-skill" not in output
    assert "hub-skill" not in output


def test_do_list_filter_hub(three_source_env):
    output = _capture(source_filter="hub")

    assert "hub-skill" in output
    assert "builtin-skill" not in output
    assert "local-skill" not in output


def test_do_list_filter_builtin(three_source_env):
    output = _capture(source_filter="builtin")

    assert "builtin-skill" in output
    assert "hub-skill" not in output
    assert "local-skill" not in output


def test_do_check_reports_available_updates(monkeypatch):
    output = _capture_check(monkeypatch, [
        {"name": "hub-skill", "source": "skills.sh", "status": "update_available"},
        {"name": "other-skill", "source": "github", "status": "up_to_date"},
    ])

    assert "hub-skill" in output
    assert "update_available" in output
    assert "up_to_date" in output


def test_do_check_handles_no_installed_updates(monkeypatch):
    output = _capture_check(monkeypatch, [])

    assert "No hub-installed skills to check" in output


def test_do_update_reinstalls_outdated_skills(monkeypatch):
    output, installs = _capture_update(monkeypatch, [
        {"name": "hub-skill", "identifier": "skills-sh/example/repo/hub-skill", "status": "update_available"},
        {"name": "other-skill", "identifier": "github/example/other-skill", "status": "up_to_date"},
    ])

    assert installs == [("skills-sh/example/repo/hub-skill", "category", True)]
    assert "Updated 1 skill" in output


def test_do_install_quarantines_third_party_skill(monkeypatch, hub_env):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    class Meta:
        extra = {}

    bundle = hub.SkillBundle(
        name="demo-skill",
        files={"SKILL.md": "# Demo\n", "script.py": "print('hi')\n"},
        source="github",
        identifier="github/example/demo-skill",
        trust_level="community",
        metadata={},
    )
    result = guard.ScanResult(
        skill_name="demo-skill",
        source="github/example/demo-skill",
        trust_level="community",
        verdict="safe",
        findings=[],
        scanned_at="2026-03-25T00:00:00Z",
        summary="demo-skill: safe",
    )

    monkeypatch.setattr(hub, "GitHubAuth", lambda: object())
    monkeypatch.setattr(hub, "create_source_router", lambda _auth: [])
    monkeypatch.setattr("hermes_cli.skills_hub._resolve_source_meta_and_bundle", lambda identifier, sources: (Meta(), bundle, None))
    monkeypatch.setattr(guard, "scan_skill", lambda q_path, source=None: result)
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, "ok"))

    do_install("github/example/demo-skill", console=console, skip_confirm=True)
    output = sink.getvalue()

    assert "Quarantined:" in output
    assert "approve demo-skill" in output
    assert (hub.QUARANTINE_DIR / "demo-skill" / ".hermes-admission.json").exists()


def test_do_inspect_and_reject_quarantined_skill(monkeypatch, hub_env):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    class Meta:
        extra = {}

    bundle = hub.SkillBundle(
        name="demo-skill",
        files={"SKILL.md": "# Demo\n"},
        source="github",
        identifier="github/example/demo-skill",
        trust_level="community",
        metadata={},
    )
    result = guard.ScanResult(
        skill_name="demo-skill",
        source="github/example/demo-skill",
        trust_level="community",
        verdict="safe",
        findings=[],
        scanned_at="2026-03-25T00:00:00Z",
        summary="demo-skill: safe",
    )

    monkeypatch.setattr(hub, "GitHubAuth", lambda: object())
    monkeypatch.setattr(hub, "create_source_router", lambda _auth: [])
    monkeypatch.setattr("hermes_cli.skills_hub._resolve_source_meta_and_bundle", lambda identifier, sources: (Meta(), bundle, None))
    monkeypatch.setattr(guard, "scan_skill", lambda q_path, source=None: result)
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, "ok"))

    do_install("github/example/demo-skill", console=console, skip_confirm=True)

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_inspect("demo-skill", console=console)
    output = sink.getvalue()
    assert "Admission Report: demo-skill" in output

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_quarantine_list(console=console)
    output = sink.getvalue()
    assert "demo-skill" in output
    assert "quarantined" in output

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_reject("demo-skill", console=console)
    output = sink.getvalue()
    assert "Rejected:" in output


def test_do_approve_installs_quarantined_skill(monkeypatch, hub_env):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    class Meta:
        extra = {}

    bundle = hub.SkillBundle(
        name="demo-skill",
        files={"SKILL.md": "# Demo\n"},
        source="github",
        identifier="github/example/demo-skill",
        trust_level="community",
        metadata={},
    )
    result = guard.ScanResult(
        skill_name="demo-skill",
        source="github/example/demo-skill",
        trust_level="community",
        verdict="safe",
        findings=[],
        scanned_at="2026-03-25T00:00:00Z",
        summary="demo-skill: safe",
    )

    monkeypatch.setattr(hub, "GitHubAuth", lambda: object())
    monkeypatch.setattr(hub, "create_source_router", lambda _auth: [])
    monkeypatch.setattr("hermes_cli.skills_hub._resolve_source_meta_and_bundle", lambda identifier, sources: (Meta(), bundle, None))
    monkeypatch.setattr(guard, "scan_skill", lambda q_path, source=None: result)
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, "ok"))

    pre_sink = StringIO()
    pre_console = Console(file=pre_sink, force_terminal=False, color_system=None)
    do_install("github/example/demo-skill", console=pre_console, skip_confirm=True)

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_approve("demo-skill", console=console)
    output = sink.getvalue()

    assert "Installed:" in output
    assert (hub.SKILLS_DIR / "demo-skill" / "SKILL.md").exists()


def test_do_audit_revokes_drifted_skill(monkeypatch, hub_env):
    import tools.skills_guard as guard
    import tools.skills_hub as hub
    from agent.security.admission import (
        AdmissionRecord,
        CandidateKind,
        CandidateSource,
        mark_record_approved,
    )

    result = guard.ScanResult(
        skill_name="demo-skill",
        source="github/example/demo-skill",
        trust_level="community",
        verdict="safe",
        findings=[],
        scanned_at="2026-03-25T00:00:00Z",
        summary="demo-skill: safe",
    )

    monkeypatch.setattr(guard, "scan_skill", lambda q_path, source=None: result)
    installed_dir = hub.SKILLS_DIR / "demo-skill"
    installed_dir.mkdir(parents=True, exist_ok=True)
    installed = installed_dir / "SKILL.md"
    installed.write_text("# Demo\n", encoding="utf-8")

    class LockShim:
        def __init__(self):
            self._installed = {
                "demo-skill": {
                    "name": "demo-skill",
                    "source": "github",
                    "identifier": "github/example/demo-skill",
                    "trust_level": "community",
                    "scan_verdict": "safe",
                    "content_hash": "sha256:test",
                    "install_path": "demo-skill",
                    "files": ["SKILL.md"],
                    "metadata": {},
                }
            }

        def list_installed(self):
            return list(self._installed.values())

        def get_installed(self, name):
            return self._installed.get(name)

        def record_uninstall(self, name):
            self._installed.pop(name, None)

    monkeypatch.setattr(hub, "HubLockFile", LockShim)

    record = AdmissionRecord(
        record_id="skill-demo-skill",
        kind=CandidateKind.SKILL,
        source=CandidateSource(
            uri="github/example/demo-skill",
            display_name="demo-skill",
            installer="test",
        ),
    )
    mark_record_approved(record, approved_path=str(installed_dir), integrity_path=installed_dir)
    installed.write_text("# Tampered\n", encoding="utf-8")

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_audit("demo-skill", console=console)
    output = sink.getvalue()

    assert "Revoked:" in output
    assert not any(".hub" not in path.parts for path in hub.SKILLS_DIR.rglob("SKILL.md"))
