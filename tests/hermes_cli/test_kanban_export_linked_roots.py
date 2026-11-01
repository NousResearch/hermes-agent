"""Board exports skip linked asset roots as well as linked descendants."""

import tarfile

import pytest


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("linked", ["attachments", "logs"])
def test_export_skips_linked_root_and_preserves_regular_assets(tmp_path, monkeypatch, linked):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_transfer import export_board

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_ATTACHMENTS_ROOT"):
        monkeypatch.delenv(key, raising=False)
    kb.create_board("alpha")
    external = tmp_path / "external"
    external.mkdir()
    outside = external / "outside.txt"
    outside.write_text("not a board asset")
    roots = {"attachments": kb.attachments_root("alpha"), "logs": kb.worker_logs_dir("alpha")}
    roots[linked].symlink_to(external, target_is_directory=True)
    regular = "logs" if linked == "attachments" else "attachments"
    roots[regular].mkdir()
    (roots[regular] / "owned.txt").write_text("board asset")
    (roots[regular] / "linked.txt").symlink_to(outside)

    result = export_board("alpha", str(tmp_path / "export"), include_logs=True)

    with tarfile.open(result["archive"], "r:gz") as archive:
        assert not any(name.endswith("outside.txt") for name in archive.getnames())
        assert not any(name.endswith("linked.txt") for name in archive.getnames())
        exported = archive.extractfile(f"alpha/{regular}/owned.txt")
        assert exported is not None
        assert exported.read() == b"board asset"
    count = {"attachments": "attachment_files", "logs": "log_files"}
    assert result["counts"][count[linked]] == 0
    assert result["counts"][count[regular]] == 1
    assert outside.read_text() == "not a board asset"
