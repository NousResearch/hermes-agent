"""A profile delete (or multiplexed rename) never removes/moves a home whose gateway.lock a live runtime still holds (N60).

A multiplexer answers the delete's rescan ``pending`` after 5 s and keeps tearing the profile down;
it releases the profile's ``gateway.lock`` reservation last. The holder here is a real separate
process reserving through the same primitive the multiplexer uses (``process_ownership.reserve``).
"""
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import profiles, profiles_remove

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
