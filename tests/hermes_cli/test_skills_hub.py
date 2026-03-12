from io import StringIO

from rich.console import Console

from hermes_cli.skills_hub import do_list


class _DummyLockFile:
    def __init__(self, installed):
        self._installed = installed

    def list_installed(self):
        return self._installed


def test_do_list_initializes_hub_dir(monkeypatch, tmp_path):
    import tools.skills_hub as hub
    import tools.skills_tool as skills_tool

    hub_dir = tmp_path / "skills" / ".hub"
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "HUB_DIR", hub_dir)
    monkeypatch.setattr(hub, "LOCK_FILE", hub_dir / "lock.json")
    monkeypatch.setattr(hub, "QUARANTINE_DIR", hub_dir / "quarantine")
    monkeypatch.setattr(hub, "AUDIT_LOG", hub_dir / "audit.log")
    monkeypatch.setattr(hub, "TAPS_FILE", hub_dir / "taps.json")
    monkeypatch.setattr(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache")
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: [])

    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    assert not hub_dir.exists()

    do_list(console=console)

    assert hub_dir.exists()
    assert (hub_dir / "lock.json").exists()
    assert (hub_dir / "quarantine").is_dir()
    assert (hub_dir / "index-cache").is_dir()


def test_do_list_distinguishes_hub_builtin_and_local(monkeypatch, tmp_path):
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    hub_dir = tmp_path / "skills" / ".hub"
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "HUB_DIR", hub_dir)
    monkeypatch.setattr(hub, "LOCK_FILE", hub_dir / "lock.json")
    monkeypatch.setattr(hub, "QUARANTINE_DIR", hub_dir / "quarantine")
    monkeypatch.setattr(hub, "AUDIT_LOG", hub_dir / "audit.log")
    monkeypatch.setattr(hub, "TAPS_FILE", hub_dir / "taps.json")
    monkeypatch.setattr(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache")

    monkeypatch.setattr(
        hub,
        "HubLockFile",
        lambda: _DummyLockFile([
            {"name": "hub-skill", "source": "github", "trust_level": "community"},
        ]),
    )
    monkeypatch.setattr(
        skills_tool,
        "_find_all_skills",
        lambda: [
            {"name": "hub-skill", "category": "x", "description": "hub"},
            {"name": "builtin-skill", "category": "x", "description": "builtin"},
            {"name": "local-skill", "category": "x", "description": "local"},
        ],
    )
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {"builtin-skill": "abc123"})

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    do_list(console=console)

    output = sink.getvalue()
    assert "hub-skill" in output
    assert "builtin-skill" in output
    assert "local-skill" in output
    assert "1 hub-installed, 1 builtin, 1 local" in output


def test_do_list_local_filter(monkeypatch, tmp_path):
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    hub_dir = tmp_path / "skills" / ".hub"
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "HUB_DIR", hub_dir)
    monkeypatch.setattr(hub, "LOCK_FILE", hub_dir / "lock.json")
    monkeypatch.setattr(hub, "QUARANTINE_DIR", hub_dir / "quarantine")
    monkeypatch.setattr(hub, "AUDIT_LOG", hub_dir / "audit.log")
    monkeypatch.setattr(hub, "TAPS_FILE", hub_dir / "taps.json")
    monkeypatch.setattr(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache")

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([]))
    monkeypatch.setattr(
        skills_tool,
        "_find_all_skills",
        lambda: [
            {"name": "builtin-skill", "category": "x", "description": "builtin"},
            {"name": "local-skill", "category": "x", "description": "local"},
        ],
    )
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {"builtin-skill": "abc123"})

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    do_list(source_filter="local", console=console)

    output = sink.getvalue()
    assert "local-skill" in output
    assert "builtin-skill" not in output
