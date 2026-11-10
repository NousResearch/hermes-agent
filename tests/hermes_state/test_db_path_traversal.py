"""A SessionDB opens the file its path names on the filesystem.

A path such as ``A/link/../state.db`` where ``link`` is a directory symlink into ``B/child``
names ``B/state.db``: the kernel follows ``link`` before applying ``..``. Collapsing ``..``
lexically would open (and create) ``A/state.db`` instead and show an empty history. These pin the
constructor to today's behaviour: it keeps the path it is given and lets the kernel walk it.
"""
import os
import subprocess
import sys
from contextlib import closing

import pytest
from pathlib import Path

from hermes_state import SessionDB


def _layout(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    (b / "child").mkdir(parents=True)
    (a / "link").symlink_to(b / "child", target_is_directory=True)
    with closing(SessionDB(b / "state.db")) as db:
        db.create_session("expected-b", "cli")
    return a, b


def test_an_absolute_path_through_a_link_and_dotdot_opens_the_file_it_names(tmp_path):
    a, b = _layout(tmp_path)
    with closing(SessionDB(a / "link" / ".." / "state.db")) as db:
        row = db.get_session("expected-b")
        assert row is not None and row["id"] == "expected-b"
        db.append_message("expected-b", "user", "through the link")
    assert not (a / "state.db").exists()
    with closing(SessionDB(b / "state.db")) as db:
        assert [m["content"] for m in db.get_messages("expected-b")] == ["through the link"]


def test_a_relative_path_through_a_link_and_dotdot_opens_the_file_it_names(tmp_path, monkeypatch):
    a, b = _layout(tmp_path)
    monkeypatch.chdir(a)
    with closing(SessionDB(Path("link") / ".." / "state.db")) as db:
        row = db.get_session("expected-b")
        assert row is not None and row["id"] == "expected-b"
        db.append_message("expected-b", "user", "through the link")
    assert not (a / "state.db").exists()
    with closing(SessionDB(b / "state.db")) as db:
        assert [m["content"] for m in db.get_messages("expected-b")] == ["through the link"]

def test_a_database_file_link_is_kept_as_given(tmp_path):
    """The handle keeps the path as given: the port's refusal of a file link into another
    profile needs to see the link."""
    real = tmp_path / "b" / "state.db"
    real.parent.mkdir()
    (tmp_path / "a").mkdir()
    link = tmp_path / "a" / "state.db"
    link.symlink_to(real)
    with closing(SessionDB(link)) as db:
        assert Path(db.db_path).is_symlink()


_FROM_REMOVED_CWD = """
import os, sys
from contextlib import closing
from pathlib import Path
from hermes_state import SessionDB
file, gone, read_only = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
os.chdir(gone)
os.rmdir(gone)
with closing(SessionDB(file, read_only=read_only)) as db:
    assert db.get_session("s")["id"] == "s"
    if not read_only:
        db.append_message("s", "user", "cwd removed")
print("ok")
"""


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("read_only", [False, True])
def test_an_absolute_open_works_after_the_working_directory_was_removed(tmp_path, read_only):
    """Run in a child: the open must not need the (deleted) working directory at all, and this
    process's test filesystem guard itself resolves paths against it."""
    home = tmp_path / "home"
    home.mkdir()
    with closing(SessionDB(home / "state.db")) as db:
        db.create_session("s", "cli")
    gone = tmp_path / "gone"
    gone.mkdir()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    done = subprocess.run(
        [sys.executable, "-c", _FROM_REMOVED_CWD, str(home / "state.db"), str(gone), "1" if read_only else "0"],
        capture_output=True, text=True, env=env, timeout=60, check=False)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
    if not read_only:
        with closing(SessionDB(home / "state.db")) as db:
            assert [m["content"] for m in db.get_messages("s")] == ["cwd removed"]
