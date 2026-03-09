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
