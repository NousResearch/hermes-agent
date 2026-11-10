"""``make_user_dir`` contract off Windows and its grant selection (the Windows ACL effect itself is
proved by ``test_hermes_dir_acl_windows.py`` on Windows CI).

No test pretends to be on Windows: elevation is a host fact the functions take as data (or the
process_is_elevated probe stands in for it on a production path), and only the native writer
(``_grant_with_pywin32``) is replaced, to record what an elevated grant would have touched."""
import os
from pathlib import Path

import pytest

import hermes_dir_acl


@pytest.mark.platforms("posix")
def test_off_windows_it_is_plain_mkdir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hermes_dir_acl, "grant_user_access", lambda p, **_k: calls.append(p))
    target = hermes_dir_acl.make_user_dir(tmp_path / "a" / "b", 0o700, parents=True)
    assert target.is_dir() and calls == []
    assert (target.stat().st_mode & 0o777) == 0o700
    with pytest.raises(FileExistsError):
        hermes_dir_acl.make_user_dir(target)
    with pytest.raises(FileNotFoundError):
        hermes_dir_acl.make_user_dir(tmp_path / "x" / "y")


def test_elevated_grants_every_created_dir_and_an_existing_leaf_once(tmp_path, monkeypatch):
    granted = []
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "grant_user_access", lambda p, **_k: granted.append(p) or True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())

    leaf = hermes_dir_acl.make_user_dir(tmp_path / "state" / "gateway-locks", parents=True, exist_ok=True)
    assert granted == [tmp_path / "state", leaf]  # parents first, so the leaf inherits from a granted parent
    hermes_dir_acl.make_user_dir(leaf, exist_ok=True)
    assert granted == [tmp_path / "state", leaf]  # memoized per process

    existing = tmp_path / "left-by-an-earlier-elevated-run"
    existing.mkdir()
    hermes_dir_acl.make_user_dir(existing, exist_ok=True)
    assert granted[-1] == existing  # the repair case: re-grant an existing leaf


def test_elevated_existing_leaf_without_exist_ok_is_repaired_then_raises(tmp_path, monkeypatch):
    """Path.mkdir contract stays (FileExistsError), but the left-behind dir is granted first."""
    granted = []
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "grant_user_access", lambda p, **_k: granted.append(p) or True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    left = tmp_path / "state"
    left.mkdir()
    with pytest.raises(FileExistsError):
        hermes_dir_acl.make_user_dir(left)
    assert granted == [left]
    hermes_dir_acl.make_user_dir(left, parents=True, exist_ok=True)
    assert granted == [left]  # memoized: no second ACE


def test_a_failed_grant_never_fails_the_mkdir_and_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    attempts = []

    def boom(path, *_a, **_k):
        attempts.append(path)
        raise OSError("access denied")

    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32", boom)
    target = hermes_dir_acl.make_user_dir(tmp_path / "d", exist_ok=True)
    hermes_dir_acl.make_user_dir(target, exist_ok=True)
    assert target.is_dir() and attempts == [target, target]


def test_remedy_names_the_platform_repair():
    remedy = hermes_dir_acl.unusable_dir_remedy
    assert remedy("C:\\x", windows=True) == 'from an elevated prompt run: icacls "C:\\x" /reset /t'
    assert "chown" in remedy("/x", windows=False)


# ── P2-5: an elevated grant never follows a junction/symlink out of the Hermes home ──────────────


@pytest.fixture()
def elevated_windows(tmp_path, monkeypatch):
    """Fake an elevated Windows process; record what the native grant would have touched."""
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "Program Files" / "X"
    outside.mkdir(parents=True)
    (outside / "app.exe").write_text("x")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    touched = []
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32",
                        lambda path, *_a, **_k: touched.append(os.path.realpath(path)))
    return home, outside, touched


def _escapes(touched, outside):
    return [p for p in touched if p == str(outside) or p.startswith(str(outside) + os.sep)]


def test_make_user_dir_refuses_an_existing_leaf_that_is_a_link(elevated_windows, caplog):
    home, outside, touched = elevated_windows
    (home / "cache").mkdir()
    planted = home / "cache" / "images"
    planted.symlink_to(outside, target_is_directory=True)
    with caplog.at_level("WARNING", logger="hermes_dir_acl"):
        hermes_dir_acl.make_user_dir(planted, parents=True, exist_ok=True)
        hermes_dir_acl.make_user_dir(planted, parents=True, exist_ok=True)
    assert touched == []
    assert [r for r in caplog.records if "never follows a junction" in r.getMessage()]


def test_make_user_dir_refuses_a_link_among_its_ancestors_in_the_home(elevated_windows):
    home, outside, touched = elevated_windows
    (home / "cache").symlink_to(outside, target_is_directory=True)
    made = hermes_dir_acl.make_user_dir(home / "cache" / "images" / "x", parents=True, exist_ok=True)
    assert made.is_dir()  # the mkdir itself still happens (it lands where the user's own link points)
    assert touched == [], "a grant went through a planted link"


def test_grant_user_access_refuses_a_link_directly(elevated_windows):
    home, outside, touched = elevated_windows
    link = home / "logs"
    link.symlink_to(outside, target_is_directory=True)
    assert hermes_dir_acl.grant_user_access(link) is False
    assert hermes_dir_acl.grant_user_access(link / "deeper") is False
    assert touched == []


def test_make_user_dir_still_grants_real_dirs_under_a_linked_home(tmp_path, monkeypatch):
    """HERMES_HOME itself is the user's own choice; only entries INSIDE it are swappable."""
    real = tmp_path / "real-home"
    real.mkdir()
    linked = tmp_path / "linked-home"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(linked))
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    calls = []
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32",
                        lambda path, is_dir, expected=None: calls.append((path, expected)))
    hermes_dir_acl.make_user_dir(linked / "state", exist_ok=True)
    assert calls == [(linked / "state", os.path.join(str(real), "state"))]


def test_tree_walk_skips_symlinks_and_never_enters_them(tmp_path):
    root = tmp_path / "home"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "f.txt").write_text("x")
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents=True)
    (root / "a" / "jn").symlink_to(outside, target_is_directory=True)
    (root / "flink").symlink_to(outside / "deep")
    skipped = []
    seen = {p.relative_to(root).as_posix(): d for p, d in hermes_dir_acl.iter_tree_without_reparse_points(root, skipped)}
    assert seen == {".": True, "a": True, "a/b": True, "a/f.txt": False}
    assert sorted(p.name for p in skipped) == ["flink", "jn"]


def test_tree_walk_treats_a_reparse_attribute_as_a_link(tmp_path, monkeypatch):
    """A Windows directory junction is not S_ISLNK on every Python; the reparse attribute decides."""
    root = tmp_path / "home"
    (root / "junction").mkdir(parents=True)
    real_lstat = os.lstat

    class Junction:
        def __init__(self, st):
            self._st = st
            self.st_file_attributes = hermes_dir_acl.FILE_ATTRIBUTE_REPARSE_POINT | hermes_dir_acl.FILE_ATTRIBUTE_DIRECTORY

        def __getattr__(self, name):
            return getattr(self._st, name)

    monkeypatch.setattr(hermes_dir_acl.os, "lstat",
                        lambda p, *a, **k: Junction(real_lstat(p)) if str(p).endswith("junction") else real_lstat(p))
    skipped = []
    seen = [p for p, _ in hermes_dir_acl.iter_tree_without_reparse_points(root, skipped)]
    assert seen == [root] and skipped == [root / "junction"]


def test_tree_grant_never_touches_a_target_outside_the_home(elevated_windows):
    home, outside, touched = elevated_windows
    (home / "sessions").mkdir()
    (home / "sessions" / "s.json").write_text("{}")
    (home / "sessions" / "jn").symlink_to(outside, target_is_directory=True)
    (home / "hardlink.exe").hardlink_to(outside / "app.exe")
    granted, skipped, failed = hermes_dir_acl.grant_tree_user_access(home)
    assert _escapes(touched, outside) == [] and failed == []
    assert sorted(p.name for p in skipped) == ["hardlink.exe", "jn"]
    assert granted == 3  # home, sessions, s.json


def test_tree_grant_is_a_noop_unelevated(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32", lambda *_a, **_k: pytest.fail("granted"))
    assert hermes_dir_acl.grant_tree_user_access(tmp_path, elevated=False) == (0, [], [])
    assert hermes_dir_acl.grant_tree_user_access(tmp_path) == (0, [], [])  # the probe off an elevated Windows host


@pytest.mark.parametrize("shape", ["symlink-only", "symlink-before-a-file", "hardlink-only", "nested-hardlink"])
def test_icacls_fallback_refuses_any_link_in_the_subtree(tmp_path, monkeypatch, shape):
    """Review-9 (b): the prewalk checked inside the loop, so a link met after the last yield
    (a root holding only a link) was admitted, and a hard-linked file was never checked."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("sentinel")
    root = tmp_path / "home"
    root.mkdir()
    if shape.startswith("symlink"):
        (root / "a-link").symlink_to(outside, target_is_directory=True)
        if shape == "symlink-before-a-file":
            (root / "z-file").write_text("x")
    elif shape == "hardlink-only":
        (root / "alias").hardlink_to(outside / "file")
    else:
        (root / "d").mkdir()
        (root / "d" / "alias").hardlink_to(outside / "file")
    monkeypatch.setattr(hermes_dir_acl, "_whoami_sid", lambda: "S-1-5-21-1-2-3-1001")
    calls = []
    monkeypatch.setattr(hermes_dir_acl.subprocess, "run", lambda argv, **_k: calls.append(argv))
    with pytest.raises(hermes_dir_acl.ReparsePointRefused):
        hermes_dir_acl._grant_with_icacls(root, True, str(root))
    assert calls == []


def test_icacls_fallback_still_grants_a_plain_subtree(tmp_path, monkeypatch):
    root = tmp_path / "home"
    (root / "d").mkdir(parents=True)
    (root / "d" / "f").write_text("x")
    monkeypatch.setattr(hermes_dir_acl, "_whoami_sid", lambda: "S-1-5-21-1-2-3-1001")
    calls = []
    monkeypatch.setattr(hermes_dir_acl.subprocess, "run", lambda argv, **_k: calls.append(argv))
    hermes_dir_acl._grant_with_icacls(root, True, str(root))
    assert [c[0] for c in calls] == ["icacls"] and "/L" in calls[0]


def test_a_recreated_dir_is_granted_again(tmp_path, monkeypatch):
    """Review-9 (c): the grant memo was keyed by pathname, so media release deleting a digest dir
    and the next capture recreating it at the same path left the new dir without the grant."""
    granted = []
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "grant_user_access", lambda p, **_k: granted.append(p) or True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    target = tmp_path / "native-inputs" / ("0" * 64)
    hermes_dir_acl.make_user_dir(target, 0o700, parents=True, exist_ok=True)
    hermes_dir_acl.make_user_dir(target, 0o700, exist_ok=True)
    assert granted.count(target) == 1  # existing object: memoized
    target.rmdir()
    hermes_dir_acl.make_user_dir(target, 0o700, exist_ok=True)
    assert granted.count(target) == 2, "the recreated directory never got the grant"


def test_media_release_then_recapture_grants_the_recreated_digest_dir(tmp_path, monkeypatch):
    """The production lifecycle behind (c): capture -> release_unheld_media -> capture again."""
    from gateway import session_ingress_media as media
    from hermes_state import SessionDB

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    grants = []
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32", lambda path, *_a, **_k: grants.append(str(path)))
    source = tmp_path / "sample.txt"
    source.write_text("same attachment bytes")
    db = SessionDB(db_path=home / "state.db")
    try:
        first = media.capture_native_media([source])
        directory = str(Path(first[0]["path"]).parent)
        assert media.release_unheld_media(db, first, retain_history=False) == 1
        assert not Path(directory).exists()
        assert media.capture_native_media([source]) == first
    finally:
        db.close()
    assert grants.count(directory) == 2


def test_repairing_an_existing_dir_also_grants_what_is_already_inside_it(tmp_path, monkeypatch):
    """Review #140490 (pastels minor 1): the pywin32 write (SetKernelObjectSecurity) does not
    propagate, so an old token lock already inside a dir an earlier elevated run left stayed
    Administrators-only after make_user_dir repaired the dir. A write on the dir now walks its
    existing children (never through a link); an already-granted dir (no write) walks nothing."""
    home = tmp_path / "home"
    locks = home / "gateway-locks"
    (locks / "old").mkdir(parents=True)
    (locks / "telegram-bot-token-x.lock").write_text("{}")
    outside = tmp_path / "outside"
    outside.mkdir()
    (locks / "jn").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    writes = []

    def native(path, is_dir, expected=None):
        writes.append((Path(path), expected))
        return True  # the DACL lacked the ACE: written, not propagated

    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32", native)
    hermes_dir_acl.make_user_dir(locks, parents=True, exist_ok=True, elevated=True)
    touched = sorted(p.relative_to(home).as_posix() for p, _ in writes)
    assert touched == ["gateway-locks", "gateway-locks/old", "gateway-locks/telegram-bot-token-x.lock"]
    assert all(expected == os.path.join(os.path.realpath(home), *p.relative_to(home).parts) for p, expected in writes)

    writes.clear()
    monkeypatch.setattr(hermes_dir_acl, "_granted", set())
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32", lambda p, *_a, **_k: writes.append(p) and False)
    hermes_dir_acl.make_user_dir(locks, exist_ok=True, elevated=True)
    assert writes == [locks]  # already granted: no write, so no child walk


def test_icacls_tree_fallback_walks_the_tree_once(tmp_path, monkeypatch):
    """Review #140490 (pastels minor 2): without pywin32 the tree repair re-walked the whole subtree
    of every directory it granted (quadratic). One walk now decides which directories hold a link;
    those are skipped as before, and every other entry gets its own icacls call (propagation alone
    misses children whose DACL is protected, which is exactly what mkdir(0o700) makes on 3.13+)."""
    home = tmp_path / "home"
    for d in ("a/b/c", "dirty/sub"):
        (home / d).mkdir(parents=True)
    for f in ("a/f1", "a/b/f2", "a/b/c/f3", "dirty/f4", "dirty/sub/f5"):
        (home / f).write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / "dirty" / "jn").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32", _no_pywin32)
    monkeypatch.setattr(hermes_dir_acl, "_whoami_sid", lambda: "S-1-5-21-1-2-3-1001")
    calls = []
    monkeypatch.setattr(hermes_dir_acl.subprocess, "run", lambda argv, **_k: calls.append(Path(argv[1])))
    walks = []
    real_iter = hermes_dir_acl.iter_tree_without_reparse_points
    monkeypatch.setattr(hermes_dir_acl, "iter_tree_without_reparse_points",
                        lambda root, skipped: walks.append(root) or real_iter(root, skipped))

    granted, skipped, failed = hermes_dir_acl.grant_tree_user_access(home, elevated=True)

    assert walks == [home], "one walk, not one per directory"
    clean = ["a", "a/b", "a/b/c", "a/b/c/f3", "a/b/f2", "a/f1", "dirty/f4", "dirty/sub", "dirty/sub/f5"]
    assert sorted(p.relative_to(home).as_posix() for p in calls) == clean
    assert sorted(p.relative_to(tmp_path).as_posix() for p in skipped) == ["home", "home/dirty", "home/dirty/jn"]
    assert failed == [] and granted == len(clean)


def _no_pywin32(*_a, **_k):
    raise ImportError("pywin32 missing")
