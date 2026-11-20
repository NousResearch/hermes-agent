"""Skill sync keeps user edits and tells the updater's user how to inspect them."""

import shutil

from tools import skills_sync
from tools.skills_sync_optional import restore_official_optional_skill
from hermes_cli.update_cmd_maint import _print_bundled_skills_sync_report, _profile_skill_sync_status


def test_kept_skill_edits_have_an_actionable_update_report(tmp_path, monkeypatch, capsys):
    bundled = tmp_path / "bundled"
    source = bundled / "example/SKILL.md"
    source.parent.mkdir(parents=True)
    source.write_text("---\nname: example\ndescription: fixture\n---\nOriginal\n", encoding="utf-8")
    installed = tmp_path / "skills"
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", installed)
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", installed / ".manifest.json")
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(skills_sync, "_get_optional_dir", lambda: tmp_path / "optional")
    monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", list)

    _print_bundled_skills_sync_report()
    user_skill = installed / "example/SKILL.md"
    assert user_skill.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert "list-modified" not in capsys.readouterr().out
    user_skill.write_text("My local instructions\n", encoding="utf-8")
    source.write_text(source.read_text(encoding="utf-8") + "Upstream revision\n", encoding="utf-8")

    _print_bundled_skills_sync_report()
    out = capsys.readouterr().out
    assert "1 user-modified (kept)" in out
    assert "hermes skills list-modified" in out
    assert user_skill.read_text(encoding="utf-8") == "My local instructions\n"


def test_update_reports_retained_copies_that_no_longer_receive_bundled_updates(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profile"
    home.mkdir()
    bundled = tmp_path / "bundled"
    optional = tmp_path / "optional"
    optional.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: bundled)
    monkeypatch.setattr(skills_sync, "_get_optional_dir", lambda: optional)
    for name in ("optional-local-edit", "optional-pristine", "dropped-local", "dropped-archived", "still-bundled"):
        source = bundled / name / "SKILL.md"
        source.parent.mkdir(parents=True)
        source.write_text(f"---\nname: {name}\ndescription: fixture\n---\nOriginal\n", encoding="utf-8")

    assert "dropped-local" in skills_sync.sync_skills(quiet=True)["copied"]
    installed = home / "skills"
    edited = installed / "optional-local-edit" / "SKILL.md"
    edited.write_text(edited.read_text(encoding="utf-8") + "Local edits\n", encoding="utf-8")
    archive = installed / ".archive"
    archive.mkdir()
    (installed / "dropped-archived").rename(archive / "dropped-archived")
    for name in ("optional-local-edit", "optional-pristine"):
        (bundled / name).rename(optional / name)
    for name in ("dropped-local", "dropped-archived"):
        (bundled / name).rename(tmp_path / name)
    manifest = installed / ".bundled_manifest"
    manifest_before = manifest.read_bytes()

    _print_bundled_skills_sync_report()
    out = capsys.readouterr().out
    assert "1 moved to optional, local copy kept (no bundled updates): optional-local-edit" in out
    assert "hermes skills repair-official <name> --restore" in out
    assert "1 removed from the bundle, local copy kept (no bundled updates): dropped-local" in out
    assert "Skills are up to date" not in out
    for name in ("optional-pristine", "still-bundled", "dropped-archived"):
        assert name not in out
    # The unedited optional copy is adopted by the official-provenance backfill, so it is Hub-managed.
    assert skills_sync._read_hub_install_paths() == {"optional-pristine"}
    profile_status = _profile_skill_sync_status(skills_sync.sync_skills(quiet=True))
    assert "!1 moved to optional" in profile_status and "!1 removed from the bundle" in profile_status
    assert edited.read_text(encoding="utf-8").endswith("Local edits\n")
    assert (installed / "dropped-local" / "SKILL.md").is_file()
    assert (archive / "dropped-archived" / "SKILL.md").is_file()
    assert manifest.read_bytes() == manifest_before

    # The exits the notice recommends work: the restored official copy is Hub-managed, and a deleted
    # dropped copy stops being reported.
    assert restore_official_optional_skill("optional-local-edit", restore=True)["restored"]
    assert "optional-local-edit" in skills_sync._read_hub_install_paths()
    shutil.rmtree(installed / "dropped-local")
    result = skills_sync.sync_skills(quiet=True)
    assert "dropped-local" not in result["retired_kept"] and "dropped-local" in result["cleaned"]
