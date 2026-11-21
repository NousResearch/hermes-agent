"""A profile delete (or multiplexed rename) never removes/moves a home whose gateway.lock a live runtime still holds (N60).

A multiplexer answers the delete's rescan ``pending`` after 5 s and keeps tearing the profile down;
it releases the profile's ``gateway.lock`` reservation last. The holder here is a real separate
process reserving through the same primitive the multiplexer uses (``process_ownership.reserve``).
"""
import argparse
import errno
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import profile_cmd, profiles, profiles_remove
from hermes_constants import named_profile_has_identity

_HOLDER = (
    "import sys\nfrom pathlib import Path\nfrom gateway.runtime_ownership import process_ownership\n"
    "home = Path(sys.argv[1])\nprocess_ownership.reserve([home])\nprint('held', flush=True)\n"
    "sys.stdin.readline()\nprocess_ownership.release(home)\n")


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service", "_stop_profile_backends",
                 "_notify_multiplexer", "_stop_gateway_process", "_maybe_register_gateway_service"):
        monkeypatch.setattr(profiles, name, lambda *_a, **_k: None)
    return profiles.create_profile("owned-delete", no_alias=True)


def _hold(profile_dir):
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(profile_dir)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, cwd=str(Path(profiles.__file__).parent.parent))
    assert holder.stdout.readline().strip() == "held"
    return holder


def _release(holder):
    holder.stdin.write("\n")
    holder.stdin.flush()
    assert holder.wait(timeout=10) == 0


# Every site that removes or moves a profile home after unrouting it from a live multiplexer.
CASES = [("delete", True), ("delete", False), ("rename", False), ("recreate", False)]


@pytest.mark.parametrize(("op", "released"), CASES, ids=["delete-owner-releases", "delete-owner-keeps-it",
                                                       "rename-owner-keeps-it", "recreate-over-owned-shell"])
def test_profile_home_waits_for_the_live_owner_to_release_it(profile, monkeypatch, op, released):
    monkeypatch.setattr(profiles_remove, "OWNER_RELEASE_WAIT_SECONDS", 30.0 if released else 1.5)
    monkeypatch.setattr(profiles, "_live_default_multiplexer", lambda: op == "rename")  # rename's unroute arm
    if op == "recreate":
        shutil.rmtree(profile)
        profile.mkdir()
        profiles.mark_named_profile_deleted(profile)
    holder = _hold(profile)
    outcome = {}

    def delete():
        try:
            if op == "rename":
                profiles.rename_profile("owned-delete", "owned-renamed")
            elif op == "recreate":  # the tombstoned, marker-less shell an owner's reserve() re-made
                profiles.create_profile("owned-delete", no_alias=True)
            else:
                profiles.delete_profile("owned-delete", yes=True)
            outcome["ok"] = True
        except (RuntimeError, FileExistsError) as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=delete)
    try:
        worker.start()
        worker.join(timeout=1.0)
        assert profile.is_dir(), "the profile was removed while a live gateway still owned it"
        if released:
            _release(holder)
        worker.join(timeout=30)
        assert not worker.is_alive()
    finally:
        if holder.poll() is None:
            _release(holder)
    if released:
        assert outcome == {"ok": True} and not profile.exists()
    else:
        assert "still owns" in outcome["error"] and (profile / "gateway.lock").is_file()


def test_a_refused_delete_is_shown_as_delete_incomplete_until_retried(profile, monkeypatch, capsys):
    """#141466: the refused delete keeps its tombstone over a populated home; every surface names
    that state and the retry instead of hiding the profile or calling it missing/existing."""
    import argparse
    from hermes_cli import main as hermes_main, profile_cmd
    from hermes_constants import named_profile_delete_incomplete
    monkeypatch.setattr(profiles_remove, "OWNER_RELEASE_WAIT_SECONDS", 1.0)
    holder = _hold(profile)
    try:
        with pytest.raises(RuntimeError) as failed:
            profiles.delete_profile("owned-delete", yes=True)
    finally:
        _release(holder)
    assert named_profile_delete_incomplete(profile) and (profile / "SOUL.md").is_file()
    unfinished = f"Deletion of profile 'owned-delete' did not finish: {profile} still holds its files."
    assert "still owns" in str(failed.value) and unfinished in str(failed.value)
    capsys.readouterr()
    profile_cmd._profile_list(argparse.Namespace())
    rows = capsys.readouterr().out
    assert any(line.split()[:1] == ["owned-delete"] and "delete incomplete" in line for line in rows.splitlines())
    assert unfinished in rows
    for refused in (lambda: profiles.resolve_profile_env("owned-delete"),  # the -p resolution
                    lambda: profile_cmd._profile_use(argparse.Namespace(profile_name="owned-delete")),
                    lambda: profiles.create_profile("owned-delete", no_alias=True)):
        with pytest.raises((FileNotFoundError, FileExistsError, SystemExit)) as exc:
            refused()
        text = str(exc.value) if exc.type is not SystemExit else capsys.readouterr().out
        assert unfinished in text and "hermes profile delete owned-delete" in text
        assert "does not exist" not in text and "profile use" not in text
    monkeypatch.setattr(hermes_main.sys, "argv", ["hermes", "-p", "owned-delete", "config", "show"])
    with pytest.raises(SystemExit):
        hermes_main._apply_profile_override()
    assert unfinished in capsys.readouterr().err
    profiles.delete_profile("owned-delete", yes=True)  # the retry finishes it
    assert not profile.exists() and not named_profile_delete_incomplete(profile)


@pytest.mark.parametrize(("record", "replaced"), [(b"", False), (b"incom", False), (b"deleted\r\n", True)],
                         ids=["empty", "torn", "crlf-deleted"])
def test_only_a_finished_delete_record_lets_create_replace_the_leftovers(profile, monkeypatch, record, replaced):
    """A retry rewrites a partial delete's record (truncate, then write). Killed in between, the
    leftovers must not read as a finished delete's shell, which create replaces."""
    from hermes_constants import profile_tombstone_path
    note = profile / "preserved-note.txt"
    note.write_text("keep me")

    def partial(path, _onexc):
        for filename in ("config.yaml", ".env", "SOUL.md", "profile.yaml", "auth.json", "state.db"):
            (path / filename).unlink(missing_ok=True)
        raise OSError(errno.EBUSY, "file is in use", str(note))

    monkeypatch.setattr(profiles, "_rmtree_with_retry", partial)
    with pytest.raises(RuntimeError, match="did not finish"):
        profiles.delete_profile("owned-delete", yes=True)
    profile_tombstone_path(profile).write_bytes(record)
    if replaced:  # a finished delete's record, LF or CRLF: the dir is a shell create replaces
        profiles.create_profile("owned-delete", no_alias=True, no_skills=True)
    else:
        with pytest.raises(FileExistsError, match="did not finish"):
            profiles.create_profile("owned-delete", no_alias=True, no_skills=True)
    assert note.exists() is not replaced


def test_delete_incomplete_checks_the_tombstone_before_probing_identity(tmp_path, monkeypatch):
    """A colliding directory that is NOT tombstoned must never be probed for identity files: on Python
    3.11-3.13 ``Path.is_file()`` re-raises a ``PermissionError`` from an unreadable directory, which would
    replace the caller's "already exists" with a raw error (import, rename onto the name)."""
    import hermes_constants
    home = tmp_path / "profiles" / "locked"
    home.mkdir(parents=True)

    def unreadable(_home):
        raise PermissionError(13, "Permission denied", str(_home / "config.yaml"))
    monkeypatch.setattr(hermes_constants, "named_profile_has_identity", unreadable)
    assert hermes_constants.named_profile_delete_incomplete(home) is False


def test_partial_delete_preserves_markerless_files_for_rename(profile, monkeypatch, capsys):
    note = profile / 'preserved-note.txt'
    note.write_text('keep me', encoding='utf-8')
    unlink = os.unlink

    def fail_on_note(path, *args, **kwargs):
        if Path(path).name == note.name:
            # Model the order in which rmtree can remove all the root identity files
            # before encountering an open file, independently of scandir order.
            for filename in ('config.yaml', '.env', 'SOUL.md', 'profile.yaml', 'auth.json', 'state.db'):
                (profile / filename).unlink(missing_ok=True)
            raise OSError(errno.EBUSY, 'file is in use', path)
        return unlink(path, *args, **kwargs)

    profiles.set_active_profile('owned-delete')
    with monkeypatch.context() as fail:
        fail.setattr(os, 'unlink', fail_on_note)
        for _ in range(2):
            with pytest.raises(RuntimeError) as failed:
                profiles.delete_profile('owned-delete', yes=True)
            assert not named_profile_has_identity(profile) and note.exists()
            assert 'did not finish' in str(failed.value)

    assert profiles.get_active_profile_name() == 'default'
    assert 'owned-delete' not in profiles.list_profile_names()
    capsys.readouterr()
    profile_cmd._profile_list(argparse.Namespace())
    assert 'delete incomplete' in capsys.readouterr().out
    with pytest.raises(FileExistsError, match='did not finish'):
        profiles.create_profile('owned-delete', no_alias=True)
    assert note.read_text(encoding='utf-8') == 'keep me'
    monkeypatch.setattr(profiles, 'check_alias_collision', lambda _name: 'test alias not installed')
    kept = profiles.rename_profile('owned-delete', 'kept')
    assert (kept / note.name).read_text(encoding='utf-8') == 'keep me'
    assert profiles.create_profile('owned-delete', no_alias=True, no_skills=True).is_dir()


@pytest.mark.parametrize("attempt", ["first_delete", "retry"])
def test_an_interrupted_delete_keeps_its_leftovers_from_create(profile, monkeypatch, attempt):
    """Ctrl-C (or a kill) inside rmtree, after the identity files are gone, skips every ``except
    Exception``: the tombstone written before rmtree must already say the delete did not finish."""
    def interrupted(path, _onexc):
        for filename in ("config.yaml", ".env", "SOUL.md", "profile.yaml", "auth.json", "state.db"):
            (path / filename).unlink(missing_ok=True)
        raise KeyboardInterrupt

    (profile / "preserved-note.txt").write_text("keep me")
    if attempt == "retry":
        def failed(path, _onexc):
            (path / "config.yaml").unlink(missing_ok=True)
            raise OSError(errno.EBUSY, "file is in use", str(path / "preserved-note.txt"))
        with monkeypatch.context() as fail:
            fail.setattr(profiles, "_rmtree_with_retry", failed)
            with pytest.raises(RuntimeError, match="did not finish"):
                profiles.delete_profile("owned-delete", yes=True)
    with monkeypatch.context() as stop:
        stop.setattr(profiles, "_rmtree_with_retry", interrupted)
        with pytest.raises(KeyboardInterrupt):
            profiles.delete_profile("owned-delete", yes=True)
    from hermes_constants import named_profile_delete_incomplete
    assert named_profile_delete_incomplete(profile)
    with pytest.raises(FileExistsError, match="did not finish"):
        profiles.create_profile("owned-delete", no_alias=True, no_skills=True)
    assert (profile / "preserved-note.txt").read_text() == "keep me"


def test_a_refused_rename_keeps_a_partial_deletes_record(profile, monkeypatch):
    """The message offers rename to keep a partial delete's leftovers. Rename must leave the delete's
    ``incomplete`` record as it found it, during its owner wait (a kill there skips the finally) and
    after a refusal: read as ``deleted``, or cleared, the leftovers become a shell create erases."""
    from hermes_constants import named_profile_delete_incomplete
    note = profile / "preserved-note.txt"
    note.write_text("keep me")

    def partial(path, _onexc):
        for filename in ("config.yaml", ".env", "SOUL.md", "profile.yaml", "auth.json", "state.db"):
            (path / filename).unlink(missing_ok=True)
        raise OSError(errno.EBUSY, "file is in use", str(note))

    with monkeypatch.context() as fail:
        fail.setattr(profiles, "_rmtree_with_retry", partial)
        with pytest.raises(RuntimeError, match="did not finish"):
            profiles.delete_profile("owned-delete", yes=True)
    registered, in_wait, wait = [], [], profiles.wait_profile_unowned
    monkeypatch.setattr(profiles_remove, "OWNER_RELEASE_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(profiles, "_live_default_multiplexer", lambda: True)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", registered.append)
    monkeypatch.setattr(profiles, "wait_profile_unowned",
                        lambda home, **kw: in_wait.append(named_profile_delete_incomplete(home)) or wait(home, **kw))
    holder = _hold(profile)
    try:
        with pytest.raises(RuntimeError, match="retry the rename"):
            profiles.rename_profile("owned-delete", "owned-kept")
    finally:
        _release(holder)
    assert in_wait == [True] and named_profile_delete_incomplete(profile) and registered == []
    with pytest.raises(FileExistsError, match="did not finish"):
        profiles.create_profile("owned-delete", no_alias=True, no_skills=True)
    assert note.read_text() == "keep me"


def test_a_finished_delete_never_tombstones_a_create_that_cleared_it(profile, monkeypatch):
    """Another process's same-name create can clear the tombstone right after rmtree; recording the
    finished delete must not write it back over that create's new profile."""
    from hermes_constants import clear_named_profile_deleted, named_profile_is_deleted
    rmtree = profiles._rmtree_with_retry

    def removed_then_created_elsewhere(path, onexc):
        rmtree(path, onexc)
        clear_named_profile_deleted(path)  # what create_profile does before staging its new tree

    monkeypatch.setattr(profiles, "_rmtree_with_retry", removed_then_created_elsewhere)
    profiles.delete_profile("owned-delete", yes=True)
    assert not named_profile_is_deleted(profile)


def test_an_interrupted_delete_of_the_active_profile_resets_it(profile, monkeypatch):
    """Ctrl-C skips everything after rmtree. The sticky profile must not keep naming a home whose
    delete did not finish, or every command, the advised retry included, refuses to start."""
    from hermes_cli import main as hermes_main
    profiles.set_active_profile("owned-delete")

    def interrupted(_path, _onexc):
        raise KeyboardInterrupt

    monkeypatch.setattr(profiles, "_rmtree_with_retry", interrupted)
    with pytest.raises(KeyboardInterrupt):
        profiles.delete_profile("owned-delete", yes=True)
    assert profiles.get_active_profile() == "default"  # the sticky file, not this process's home
    monkeypatch.setattr(hermes_main.sys, "argv", ["hermes", "profile", "delete", "owned-delete", "--yes"])
    hermes_main._apply_profile_override()  # the advised retry reaches its handler


def test_a_retried_partial_delete_leaves_a_replaceable_post_delete_shell(profile, monkeypatch):
    """The retry that finishes a partial delete records a plain delete again, so a shell a stale
    writer re-creates afterwards is replaced by ``create`` instead of refused as unfinished."""
    def partial(path, _onexc):
        for filename in ("config.yaml", ".env", "SOUL.md", "profile.yaml", "auth.json", "state.db"):
            (path / filename).unlink(missing_ok=True)
        raise OSError(errno.EBUSY, "file is in use", str(path / "preserved-note.txt"))

    with monkeypatch.context() as fail:
        fail.setattr(profiles, "_rmtree_with_retry", partial)
        with pytest.raises(RuntimeError, match="did not finish"):
            profiles.delete_profile("owned-delete", yes=True)
    profiles.delete_profile("owned-delete", yes=True)
    assert not profile.exists()
    profile.mkdir()  # a stale writer re-creates the deleted home
    assert profiles.create_profile("owned-delete", no_alias=True, no_skills=True).is_dir()


def test_a_finished_delete_whose_record_cannot_be_rewritten_says_how_to_finish(profile, monkeypatch, capsys):
    """The rmtree succeeded but rewriting the record failed (e.g. a Windows reader holding it): the
    record stays ``incomplete``, so the delete must say so and name the command that finishes it."""
    from hermes_constants import TOMBSTONE_INCOMPLETE, profile_tombstone_path
    marker, real_open = profile_tombstone_path(profile), os.open

    def held(path, flags, *args, **kwargs):
        if Path(path) == marker:
            raise PermissionError(errno.EACCES, "held by another process", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(profiles_remove.os, "open", held)
    profiles.delete_profile("owned-delete", yes=True)
    monkeypatch.undo()
    assert not profile.exists() and marker.read_bytes() == TOMBSTONE_INCOMPLETE
    assert "finish it with `hermes profile delete owned-delete`" in capsys.readouterr().out
