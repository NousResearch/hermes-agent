"""Regression tests for safe_extract_targz symlink materialization (#149830).

A profile archive produced by ``export_profile`` keeps symlinks (``shutil.copytree(...,
symlinks=True)`` -> ``tarfile.add``), so ``import_profile`` used to fail on any profile
containing one (for example the LSP bundle links under ``lsp/bin``). The extractor now
materializes links that resolve inside the destination and refuses the rest.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from hermes_cli.archive_safe import safe_extract_targz


def _tar_with(entries: list[tuple[str, bytes | None, int, str]]) -> io.BytesIO:
    """Build an in-memory tar.gz. ``entries`` are (name, data, type, linkname)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data, mtype, linkname in entries:
            info = tarfile.TarInfo(name)
            info.type = mtype
            info.linkname = linkname
            if data is None:
                tf.addfile(info)
            else:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def _extract(buf: io.BytesIO, tmp_path: Path) -> Path:
    archive = tmp_path / "payload.tar.gz"
    archive.write_bytes(buf.read())
    out = tmp_path / "out"
    safe_extract_targz(archive, out)
    return out


def test_internal_relative_symlink_is_materialized(tmp_path):
    buf = _tar_with([
        ("payload/", None, tarfile.DIRTYPE, ""),
        ("payload/lsp/", None, tarfile.DIRTYPE, ""),
        ("payload/lsp/bin/", None, tarfile.DIRTYPE, ""),
        ("payload/lsp/bin/tool", None, tarfile.SYMTYPE, "../../plugins/eng/engine.py"),
        ("payload/plugins/", None, tarfile.DIRTYPE, ""),
        ("payload/plugins/eng/", None, tarfile.DIRTYPE, ""),
        ("payload/plugins/eng/engine.py", b"print('engine')\n", tarfile.REGTYPE, ""),
    ])
    out = _extract(buf, tmp_path)
    link = out / "payload" / "lsp" / "bin" / "tool"
    assert link.is_symlink()
    assert link.readlink() == Path("../../plugins/eng/engine.py")
    # The link resolves to the sibling member that was also extracted.
    assert link.resolve() == (out / "payload" / "plugins" / "eng" / "engine.py").resolve()


def test_symlink_with_dotdot_that_stays_inside_is_materialized(tmp_path):
    buf = _tar_with([
        ("payload/a/", None, tarfile.DIRTYPE, ""),
        ("payload/a/b/", None, tarfile.DIRTYPE, ""),
        ("payload/a/b/link", None, tarfile.SYMTYPE, "../target.txt"),
        ("payload/a/target.txt", b"hi\n", tarfile.REGTYPE, ""),
    ])
    out = _extract(buf, tmp_path)
    link = out / "payload" / "a" / "b" / "link"
    assert link.is_symlink()
    assert (out / "payload" / "a" / "target.txt").read_text() == "hi\n"


def test_absolute_symlink_is_refused(tmp_path):
    buf = _tar_with([
        ("payload/", None, tarfile.DIRTYPE, ""),
        ("payload/evil", None, tarfile.SYMTYPE, "/etc/passwd"),
    ])
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(buf.read())
    with pytest.raises(ValueError, match="escapes the destination"):
        safe_extract_targz(archive, tmp_path / "out")


def test_symlink_escaping_the_destination_is_refused(tmp_path):
    buf = _tar_with([
        ("payload/", None, tarfile.DIRTYPE, ""),
        ("payload/evil", None, tarfile.SYMTYPE, "../../outside.txt"),
    ])
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(buf.read())
    with pytest.raises(ValueError, match="escapes the destination"):
        safe_extract_targz(archive, tmp_path / "out")


def test_windows_style_absolute_symlink_is_refused(tmp_path):
    buf = _tar_with([
        ("payload/", None, tarfile.DIRTYPE, ""),
        ("payload/evil", None, tarfile.SYMTYPE, "C:\\Windows\\system32\\evil.dll"),
    ])
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(buf.read())
    with pytest.raises(ValueError, match="escapes the destination"):
        safe_extract_targz(archive, tmp_path / "out")


def test_hardlink_is_still_refused(tmp_path):
    buf = _tar_with([
        ("payload/", None, tarfile.DIRTYPE, ""),
        ("payload/real.txt", b"hi\n", tarfile.REGTYPE, ""),
        ("payload/hard.txt", None, tarfile.LNKTYPE, "payload/real.txt"),
    ])
    archive = tmp_path / "hard.tar.gz"
    archive.write_bytes(buf.read())
    with pytest.raises(ValueError, match="Unsupported archive member"):
        safe_extract_targz(archive, tmp_path / "out")


def test_exported_profile_with_a_symlink_round_trips(tmp_path):
    """End-to-end: export_profile keeps links, import_profile now accepts them (#149830)."""
    import os

    from hermes_cli import profiles as profiles_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    fake_home = tmp_path / "home"
    profile_dir = fake_home / "profiles" / "rlink"
    (profile_dir / "lsp" / "bin").mkdir(parents=True)
    (profile_dir / "plugins" / "eng").mkdir(parents=True)
    (profile_dir / "plugins" / "eng" / "engine.py").write_text("print('engine')\n")
    os.symlink("../../plugins/eng/engine.py", profile_dir / "lsp" / "bin" / "tool")
    # A real Hermes home is recognized by its marker files; export_profile resolves the
    # profile through the default root, which reads HERMES_HOME directly.
    (fake_home / "config.yaml").write_text("version: 1\n")
    os.environ["HERMES_HOME"] = str(fake_home)

    try:
        archive = profiles_mod.export_profile("rlink", str(tmp_path / "rlink"))
        assert archive.exists()
        imported = profiles_mod.import_profile(str(archive), name="rlink2")
    finally:
        os.environ.pop("HERMES_HOME", None)

    link = imported / "lsp" / "bin" / "tool"
    assert link.is_symlink()
    assert link.resolve() == (imported / "plugins" / "eng" / "engine.py").resolve()
