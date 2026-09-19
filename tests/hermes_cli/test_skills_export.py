"""Portable skill content export, a bounded slice of #102277."""
import argparse
import hashlib
import json
import tarfile

import pytest


def test_export_preserves_selected_skill_contents(tmp_path, monkeypatch):
    from hermes_cli.subcommands.skills import build_skills_parser
    from hermes_cli.main_agent_cmds import cmd_skills

    home = tmp_path / "home"
    skill = home / "skills" / "writing" / "notes"
    (skill / "references").mkdir(parents=True)
    (skill / "empty").mkdir()
    contents = {"SKILL.md": b"---\nname: notes\n---\nWrite notes.\n",
                "references/example.txt": b"Example notes\n"}
    for name, data in contents.items():
        (skill / name).write_bytes(data)
    monkeypatch.setenv("HERMES_HOME", str(home))
    output = tmp_path / "notes.tar.gz"
    parser = argparse.ArgumentParser()
    build_skills_parser(parser.add_subparsers(), cmd_skills=cmd_skills)
    args = parser.parse_args(["skills", "export", "writing/notes", "-o", str(output)])
    args.func(args)
    with tarfile.open(output, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        assert archive.getmember("skills/writing/notes/empty").isdir()
        assert all(not member.uname and not member.gname for member in archive.getmembers())
        for name, data in contents.items():
            archived = "skills/writing/notes/" + name
            assert archive.extractfile(archived).read() == data
            assert manifest["files"][archived] == hashlib.sha256(data).hexdigest()
    assert (skill / "SKILL.md").read_bytes() == contents["SKILL.md"]


def test_export_refuses_unsafe_sources_and_preserves_existing_output(tmp_path, monkeypatch):
    from hermes_cli.skills_export import export_skills

    home = tmp_path / "home"
    skill = home / "skills" / "notes"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("Write notes.")
    monkeypatch.setenv("HERMES_HOME", str(home))
    output = tmp_path / "bundle.tar.gz"
    output.write_bytes(b"existing archive")
    with pytest.raises(ValueError, match="already exists"):
        export_skills(["notes"], output)
    assert output.read_bytes() == b"existing archive"
    output.unlink()
    with pytest.raises(ValueError, match="outside"):
        export_skills(["notes"], skill / "bundle.tar.gz")
    with pytest.raises(ValueError, match="Unsafe"):
        export_skills(["../outside"], output)
    external = tmp_path / "external.txt"
    external.write_text("not selected")
    (skill / "redirect").symlink_to(external)
    with pytest.raises(ValueError, match="link"):
        export_skills(["notes"], output)
    assert not output.exists()
    (skill / "redirect").unlink()
    (home / "skills" / "alias").symlink_to(skill, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_skills(["alias"], output)
    with pytest.raises(ValueError, match="Not a skill"):
        export_skills(["missing"], output)
    assert not output.exists()
    (skill / ".env").write_text("synthetic credential fixture")
    with pytest.raises(ValueError, match="credential"):
        export_skills(["notes"], output)
    assert not output.exists()


def test_export_explains_missing_parent_and_hardlink_failure(tmp_path, monkeypatch):
    from hermes_cli.skills_export import export_skills

    home = tmp_path / "home"
    skill = home / "skills" / "notes"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("Write notes.")
    monkeypatch.setenv("HERMES_HOME", str(home))

    with pytest.raises(ValueError, match="parent directory does not exist"):
        export_skills(["notes"], tmp_path / "missing" / "bundle.tar.gz")

    def no_hardlink(*_args):
        raise OSError("hard links disabled")

    monkeypatch.setattr("hermes_cli.skills_export.os.link", no_hardlink)
    with pytest.raises(ValueError, match="Cannot publish export.*hard links are unavailable"):
        export_skills(["notes"], tmp_path / "bundle.tar.gz")
