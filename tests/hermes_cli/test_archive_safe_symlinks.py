"""``hermes profile import`` recreates the symlinks ``hermes profile export`` keeps (#149830).

Export stages with ``copytree(symlinks=True)``, so a profile with a link (the LSP bundle links
under ``lsp/bin``) used to export fine and then fail to import. Import now recreates links that
stay inside the imported profile and refuses every link that could reach outside it.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from hermes_cli.archive_safe import safe_extract_targz

D, F, L, H = tarfile.DIRTYPE, tarfile.REGTYPE, tarfile.SYMTYPE, tarfile.LNKTYPE


def _write_archive(path: Path, entries: list[tuple[str, int, object]]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, mtype, payload in entries:
            info = tarfile.TarInfo(name)
            info.type = mtype
            if mtype == F:
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            else:
                info.linkname = payload
                tf.addfile(info)
    return path


@pytest.mark.require_symlinks
def test_exported_profile_with_symlinks_imports_with_the_links_kept(tmp_path, monkeypatch):
    from hermes_cli.profiles import export_profile, import_profile

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    src = home / "profiles" / "rlink"
    (src / "lsp" / "bin").mkdir(parents=True)
    (src / "plugins" / "eng").mkdir(parents=True)
    (src / "plugins" / "eng" / "engine.py").write_text("print('engine')\n")
    os.symlink(os.path.join("..", "..", "plugins", "eng", "engine.py"), src / "lsp" / "bin" / "tool")
    os.symlink("plugins", src / "plugins-link", target_is_directory=True)
    # The LSP installer links by absolute path; export must ship it as a relative link.
    os.symlink(src / "plugins" / "eng" / "engine.py", src / "lsp" / "bin" / "abs-tool")
    os.symlink(src, src / "plugins" / "abs-root", target_is_directory=True)
    (src / "plugins").chmod(0o555)  # export must not need to write its staged copy
    try:
        imported = import_profile(str(export_profile("rlink", str(tmp_path / "rlink"))), name="rlink2")
    finally:
        (src / "plugins").chmod(0o755)

    assert (imported / "lsp" / "bin" / "tool").is_symlink()
    assert (imported / "lsp" / "bin" / "tool").read_text() == "print('engine')\n"
    assert (imported / "plugins-link").is_symlink()
    assert (imported / "plugins-link" / "eng" / "engine.py").is_file()
    abs_tool = imported / "lsp" / "bin" / "abs-tool"
    assert not os.path.isabs(os.readlink(abs_tool))
    assert abs_tool.resolve() == (imported / "plugins" / "eng" / "engine.py").resolve()
    assert (imported / "plugins" / "abs-root").resolve() == imported.resolve()


@pytest.mark.require_symlinks
@pytest.mark.parametrize("entries", [
    pytest.param([("p/", D, ""), ("p/e", L, "/etc/passwd")], id="absolute"),
    pytest.param([("p/sub/", D, ""), ("p/dl", L, "sub"), ("p/dl/x", L, ".")], id="link-inside-a-dir-link"),
    pytest.param([("p/sub/", D, ""), ("p/dl/x", L, "."), ("p/dl", L, "sub")], id="dir-link-over-an-implied-dir"),
    pytest.param([("p/", D, ""), ("p/e", L, "C:\\Windows\\system32")], id="windows-drive"),
    pytest.param([("p/", D, ""), ("p/e", L, "../sibling")], id="dotdot-to-a-sibling-of-the-root"),
    # Publishing renames p, so ../p/x would then point into whatever profile is named p.
    pytest.param([("p/", D, ""), ("p/x", F, b"x"), ("p/e", L, "../p/x")], id="walks-back-in-by-root-name"),
    # z only escapes through y, so a lexical check of z's target alone passes it.
    pytest.param([("p/a/b/", D, ""), ("p/a/b/y", L, "../.."), ("p/a/b/z", L, "y/../../.."),
                  ("p/a/b/z/PWNED", F, b"x")], id="chained-link-then-file-through-it"),
    pytest.param([("p/a/", D, ""), ("p/a/y", L, "../../.."), ("p/a/y/PWNED", L, ".")],
                 id="link-created-through-an-escaping-link"),
    pytest.param([("p/a/b/", D, ""), ("p/a/b/z", L, "y/../../.."), ("p/a/b/y", L, "../..")],
                 id="escape-completed-by-a-later-link"),
    pytest.param([("p/d/", D, ""), ("p/d/x", F, b"x"), ("p/d", L, "x")], id="link-over-an-existing-dir"),
    pytest.param([("p/", D, ""), ("p/r", F, b"r"), ("p/h", H, "p/r")], id="hardlink"),
])
def test_a_link_that_could_reach_outside_the_root_is_refused(tmp_path, entries):
    dest = tmp_path / "outer" / "dest"
    dest.mkdir(parents=True)
    archive = _write_archive(tmp_path / "evil.tar.gz", entries)

    with pytest.raises(ValueError):
        safe_extract_targz(archive, dest, allow_symlinks=True)

    # The caller discards the staging dir on error; nothing may have landed beside it.
    assert [p.name for p in (tmp_path / "outer").iterdir()] == ["dest"]


@pytest.mark.require_symlinks
@pytest.mark.platforms("posix")  # file modes
def test_imported_env_link_becomes_an_owner_only_file(tmp_path, monkeypatch):
    from hermes_cli.profiles import import_profile

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    archive = tmp_path / "p.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("p/keys.txt")
        info.size, info.mode = 4, 0o644
        tf.addfile(info, io.BytesIO(b"K=v\n"))
        info = tarfile.TarInfo("p/.env.import")  # an archived sibling the conversion must not clobber
        info.size = 5
        tf.addfile(info, io.BytesIO(b"keep\n"))
        link = tarfile.TarInfo("p/.env")
        link.type, link.linkname = L, "keys.txt"
        tf.addfile(link)

    env = import_profile(str(archive), name="p2") / ".env"

    assert (env.parent / ".env.import").read_text() == "keep\n"
    assert not env.is_symlink()
    assert env.read_text() == "K=v\n"
    assert env.stat().st_mode & 0o777 == 0o600
