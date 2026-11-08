"""Archive capture must not split the group-sync and presentation transaction."""
from pathlib import Path
import shutil
import zipfile

import pytest

import hermes_yaml as yaml


@pytest.mark.parametrize("kind", ["export", "backup", "clone"])
def test_archive_captures_coherent_ui_pair_during_a_mixed_update(tmp_path, monkeypatch, kind):
    import tui_gateway.server as srv
    from hermes_cli.backup import create_pre_update_backup
    from hermes_cli.profiles import create_profile, export_profile, import_profile
    from tui_gateway.group_chat_storage import migrate_group_chat_state

    root = tmp_path / ".hermes"
    source = root / "profiles" / "helper"
    source.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    key = "hermes-bots-groups"
    state = Path("state/group-chat.yaml")
    (root / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (source / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (source / "profile.yaml").write_text("display_name: Example\n", encoding="utf-8")

    def update(label, revision):
        result = srv._methods["profiles.configure"]("write", {
            "name": "helper", "ui_meta": {key: {"label": label}, "example-settings": {"label": label}},
            "ui_meta_expected_revisions": {key: revision, "example-settings": revision},
        })
        assert result["result"]["applied"]["ui_meta"] is True

    update("old", 0)
    triggered = False

    def interleave():
        nonlocal triggered
        if not triggered:
            triggered = True
            update("new", 1)

    paired = {source / "profile.yaml", source / state}
    real_copy = shutil.copyfile
    real_zip_write = zipfile.ZipFile.write
    real_writestr = zipfile.ZipFile.writestr

    def copyfile(src, dst, **kwargs):
        result = real_copy(src, dst, **kwargs)
        if Path(src) in paired:
            interleave()
        return result

    def zip_write(zf, filename, *args, **kwargs):
        result = real_zip_write(zf, filename, *args, **kwargs)
        if Path(filename) in paired:
            interleave()
        return result

    def writestr(zf, arcname, data, *args, **kwargs):
        result = real_writestr(zf, arcname, data, *args, **kwargs)
        name = arcname.filename if isinstance(arcname, zipfile.ZipInfo) else str(arcname)
        if name in {"profile.yaml", str(state)}:
            interleave()
        return result

    with monkeypatch.context() as race:
        race.setattr(shutil, "copyfile", copyfile)
        race.setattr(zipfile.ZipFile, "write", zip_write)
        race.setattr(zipfile.ZipFile, "writestr", writestr)
        if kind == "export":
            archive = export_profile("helper", str(tmp_path / "profile.tar.gz"))
            restored = import_profile(str(archive), name="restored")
        elif kind == "clone":
            restored = create_profile("restored", clone_from="helper", clone_all=True, no_alias=True)
        else:
            archive = create_pre_update_backup(hermes_home=source)
            assert archive is not None
            restored = tmp_path / "restored-backup"
            restored.mkdir()
            with zipfile.ZipFile(archive) as zf:
                for rel in ("config.yaml", "profile.yaml", str(state)):
                    target = restored / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(rel))
    assert triggered
    migrate_group_chat_state(restored)
    metadata = yaml.safe_load((restored / "profile.yaml").read_text())
    runtime = yaml.safe_load((restored / state).read_text())
    assert runtime["snapshot"] == metadata["ui_meta"]["example-settings"]
    assert runtime["revision"] == metadata["_ui_meta_revisions"]["example-settings"]


def test_backup_inventory_includes_state_published_after_the_directory_walk(tmp_path):
    from hermes_cli.profile_ui_archive import capture_backup_ui_pairs

    (tmp_path / "state").mkdir()
    (tmp_path / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (tmp_path / "profile.yaml").write_text("display_name: Example\n", encoding="utf-8")
    files = [(tmp_path / name, Path(name)) for name in ("config.yaml", "profile.yaml")]
    state = Path("state/group-chat.yaml")
    content = yaml.safe_dump({"revision": 1, "snapshot": {"rooms": {}}}).encode()
    (tmp_path / state).write_bytes(content)

    captured = capture_backup_ui_pairs(files)
    assert captured[state] == content
    assert {relative for _, relative in files} == {Path("config.yaml"), Path("profile.yaml"), state}


@pytest.mark.platforms("posix")
def test_ui_archive_capture_refuses_symlinks_without_touching_the_target(tmp_path):
    from hermes_cli.profile_ui_archive import stage_profile_ui_pair

    source = tmp_path / "source"
    staged = tmp_path / "staged"
    source.mkdir()
    staged.mkdir()
    target = tmp_path / "target.yaml"
    target.write_text("display_name: Example\n", encoding="utf-8")
    (source / "profile.yaml").symlink_to(target)
    (staged / "profile.yaml").symlink_to(target)
    before = target.read_bytes()
    with pytest.raises(OSError, match="symlinked"):
        stage_profile_ui_pair(source, staged)
    assert target.read_bytes() == before
    assert (staged / "profile.yaml").is_symlink()
