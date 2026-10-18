"""Full restores keep durable cache artifacts without importing another machine's runtime."""

import struct
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import backup


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "source"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Exercise the real archive and filesystem operations without installing host services/aliases.
    monkeypatch.setattr(backup, "_revive_gateway_after_import", lambda _home: None)
    monkeypatch.setattr("hermes_cli.profiles.create_wrapper_script", lambda _name: None)
    return home


@pytest.mark.parametrize("archive_prefix", ["", ".hermes/"])
def test_backup_restore_keeps_durable_cache_and_preserves_runtime(
    hermes_home, tmp_path, monkeypatch, archive_prefix
):
    portable = {"config.yaml": b"model: restored\n", "profiles/coder/config.yaml": b"model: coder\n"}
    for profile in ("", "profiles/coder/"):
        for path in (
            "cache/images/photo.png", "cache/audio/voice.ogg", "cache/videos/clip.mp4",
            "cache/documents/notes.pdf", "cache/screenshots/page.png", "cache/citations/ledger.json",
            "cache/generated/images/drawing.png", "cache/generated/videos/movie.mp4",
            "skills/demo/cache/notes.txt", "skills/demo/tools/custom.py", "skills/demo/installs/facts.json",
        ):
            portable[profile + path] = b"original user data: " + (profile + path).encode()
    runtime = [
        profile + path
        for profile in ("", "profiles/coder/")
        for path in (
            "installs/facts.json", "tools/python/bin/python", "cache/model_catalog.json",
            "cache/browser-use/session.json", "cache/.uv_self_update_stamp", "gateway.pid",
        )
    ]
    for rel, content in {**portable, **dict.fromkeys(runtime, b"source runtime")}.items():
        path = hermes_home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    archive = tmp_path / "backup.zip"
    assert backup.run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        archived = {name: zf.read(name) for name in zf.namelist()}
    assert portable.items() <= archived.items()
    assert set(runtime).isdisjoint(archived)
    # Older archives can contain runtime files; importing them must keep the destination's copies.
    with zipfile.ZipFile(archive, "w") as zf:
        for rel, content in {**archived, **dict.fromkeys(runtime, b"source runtime")}.items():
            zf.writestr(archive_prefix + rel, content)

    target = tmp_path / "destination"
    for rel in runtime:
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"destination runtime")
    monkeypatch.setenv("HERMES_HOME", str(target))

    assert backup.run_import(Namespace(zipfile=str(archive), force=True)) is None
    for rel, content in portable.items():
        assert (target / rel).read_bytes() == content, rel
    for rel in runtime:
        assert (target / rel).read_bytes() == b"destination runtime", rel


@pytest.mark.parametrize("archive_prefix", ["", ".hermes/"])
@pytest.mark.parametrize("member", [
    "cache/images/photo.png", "profiles/coder/cache/citations/ledger.json",
    "cache/generated/images/drawing.png", "profiles/coder/cache/generated/videos/movie.mp4",
])
def test_corrupt_durable_cache_refuses_restore_before_writing(
    hermes_home, tmp_path, capsys, archive_prefix, member
):
    config = hermes_home / "config.yaml"
    config.write_bytes(b"model: live\n")
    archive = tmp_path / "damaged.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(archive_prefix + "config.yaml", b"model: restored\n")
        zf.writestr(archive_prefix + member, b"original artifact")
        info = zf.getinfo(archive_prefix + member)
    # Keep a valid ZIP directory but break the artifact's CRC, as with damaged backup media.
    with archive.open("r+b") as stream:
        stream.seek(info.header_offset + 26)
        name_len, extra_len = struct.unpack("<HH", stream.read(4))
        stream.seek(info.header_offset + 30 + name_len + extra_len)
        stream.write(b"X")

    assert backup.run_import(Namespace(zipfile=str(archive), force=True)) == 1
    output = capsys.readouterr().out
    assert "backup archive is damaged" in output
    assert member in output
    assert config.read_bytes() == b"model: live\n"
    assert list(hermes_home.iterdir()) == [config]
