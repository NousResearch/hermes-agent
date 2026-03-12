from io import StringIO

from rich.console import Console

from hermes_cli.skills_hub import do_list


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


def test_do_list_distinguishes_builtin_hub_and_local(monkeypatch):
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    class _FakeLock:
        def list_installed(self):
            return [{"name": "hub-skill", "source": "github", "trust_level": "community"}]

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "HubLockFile", _FakeLock)
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: [
        {"name": "builtin-skill", "category": "content"},
        {"name": "hub-skill", "category": "content"},
        {"name": "local-skill", "category": "content"},
    ])
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {"builtin-skill": "abc123"})

    out = StringIO()
    console = Console(file=out, force_terminal=False, color_system=None)

    do_list(console=console)

    rendered = out.getvalue()
    assert "builtin-skill" in rendered and "builtin" in rendered
    assert "hub-skill" in rendered and "github" in rendered
    assert "local-skill" in rendered and "local" in rendered
    assert "1 hub-installed, 1 builtin, 1 local" in rendered


def test_do_list_source_filter_local(monkeypatch):
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    class _FakeLock:
        def list_installed(self):
            return [{"name": "hub-skill", "source": "github", "trust_level": "community"}]

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "HubLockFile", _FakeLock)
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: [
        {"name": "builtin-skill", "category": "content"},
        {"name": "hub-skill", "category": "content"},
        {"name": "local-skill", "category": "content"},
    ])
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {"builtin-skill": "abc123"})

    out = StringIO()
    console = Console(file=out, force_terminal=False, color_system=None)

    do_list(source_filter="local", console=console)

    rendered = out.getvalue()
    assert "local-skill" in rendered
    assert "builtin-skill" not in rendered
    assert "hub-skill" not in rendered
