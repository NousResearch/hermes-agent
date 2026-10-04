"""A same-process status reader cannot donate a descriptor to checkout custody."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from hermes_cli import update_lock


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("descriptor_owner", ["incidental-open", "status-reader"])
def test_checkout_custody_never_borrows_a_local_readers_descriptor(tmp_path, descriptor_owner):
    root = tmp_path / "checkout"
    root.mkdir()
    lock_path = update_lock.checkout_lock_path(root)
    lock_path.touch()
    opened, resume = threading.Event(), threading.Event()
    thread = None
    reader_fd = None

    def observe():
        def pause(frame, event, arg):
            if event == "call" and frame.f_code is update_lock._try_lock.__code__:
                sys.settrace(None)
                opened.set()
                assert resume.wait(20), "acquisition never released the status reader"
            return pause
        sys.settrace(pause)
        try:
            update_lock.checkout_lock_held(root)
        finally:
            sys.settrace(None)

    if descriptor_owner == "status-reader":
        thread = threading.Thread(target=observe)
        thread.start()
        assert opened.wait(20), "status reader never opened its descriptor"
    else:
        reader_fd = os.open(lock_path, os.O_RDONLY)

    contender = tmp_path / "contender.py"
    contender.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, sys.argv[1])\nfrom hermes_cli.update_lock import UpdateLock\n"
        "lock = UpdateLock(path=Path(sys.argv[2]), install_root=Path(sys.argv[3]))\n"
        "print(json.dumps({'acquired': lock.acquire()}), flush=True)\nlock.release()\n",
        encoding="utf-8",
    )
    lock = update_lock.UpdateLock(path=tmp_path / "owner-marker", install_root=root)
    try:
        assert lock.acquire(), lock.holder
        resume.set()
        if thread is not None:
            thread.join(20)
            assert not thread.is_alive()
        else:
            os.close(reader_fd)
            reader_fd = None
        result = subprocess.run(
            [sys.executable, "-I", str(contender), str(Path(update_lock.__file__).parents[1]),
             str(tmp_path / "other-home-marker"), str(root)],
            capture_output=True, text=True, encoding="utf-8", timeout=20, check=True,
        )
        assert json.loads(result.stdout)["acquired"] is False, (
            "a contender entered while the first owner was alive: its custody descriptor "
            "was borrowed from a local reader that has now closed it"
        )
    finally:
        resume.set()
        if thread is not None:
            thread.join(20)
        if reader_fd is not None:
            os.close(reader_fd)
        lock.release()
    assert not update_lock.checkout_lock_held(root), "release must leave no orphaned custody"


@pytest.mark.platforms("posix")
def test_exec_child_reenters_the_lock_it_explicitly_inherited(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\n"
        "from hermes_cli.update_lock import UpdateLock\n"
        "lock=UpdateLock(path=Path(sys.argv[2])/'child-marker',install_root=Path(sys.argv[2]))\n"
        "assert lock.acquire(), lock.holder\nlock.release()\n"
        "print('JOINED',flush=True)\nsys.stdin.readline()\n", encoding="utf-8")
    owner = update_lock.UpdateLock(path=tmp_path / "parent-marker", install_root=root)
    contender = update_lock.UpdateLock(path=tmp_path / "contender-marker", install_root=root)
    assert owner.acquire()
    child = subprocess.Popen(
        [sys.executable, "-I", str(child_script), str(Path(update_lock.__file__).parents[1]), str(root)],
        pass_fds=update_lock.checkout_lock_fds(root), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    try:
        assert child.stdout.readline().strip() == "JOINED"
        owner.release()
        assert child.poll() is None
        assert not contender.acquire(), "the inherited OFD must outlive the original owner"
        child.communicate("exit\n", timeout=20)
        assert child.returncode == 0
        assert contender.acquire(), "the last descendant's exit must release the lock"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        contender.release()
        owner.release()
