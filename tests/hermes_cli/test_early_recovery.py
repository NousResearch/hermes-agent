"""Startup triggers PM recovery without importing the damaged dependencies.

Real generation rebuilds and pre-activation startup live in tests/pm; these
checks keep marker ownership, retry limits and single-flight behavior intact.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import _early_recovery as er
from hermes_cli import _early_recovery_zip as erz
from pm import recovery

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("prefix", [[], ["-p", "default"], ["--profile=default"]])
def test_bootstrap_and_pm_cli_work_without_site_packages(tmp_path, prefix):
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home"), "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-S", "-m", "hermes_cli.main", *prefix, "pm", "repair", "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "hermes pm repair" in result.stdout


@pytest.mark.parametrize("marker_name", [".update-incomplete", ".lazy-refresh-incomplete"])
def test_marker_requests_pm_repair_then_clears(tmp_path, monkeypatch, capsys, marker_name):
    root = _project(tmp_path)
    marker = root / marker_name
    marker.write_text("interrupted", encoding="utf-8")
    calls = []
    monkeypatch.setattr(recovery, "repair_dependencies", calls.append)
    assert er.recover_if_needed(root, argv=[]) is True
    assert calls == [root]
    assert not marker.exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("body", ["", "not json", '{"attempts": 1}', "started=1\npid=0\n"])
def test_failed_repair_keeps_marker_and_stops_at_retry_limit(tmp_path, monkeypatch, capsys, body):
    root = _project(tmp_path)
    marker = root / ".update-incomplete"
    marker.write_text(body, encoding="utf-8")
    attempts = er._read_marker_attempts(marker)
    calls = []
    def fail(project):
        calls.append(project)
        raise RuntimeError("dependency build failed")
    monkeypatch.setattr(recovery, "repair_dependencies", fail)
    for expected in range(attempts + 1, er._EARLY_CORE_INSTALL_MAX_ATTEMPTS + 1):
        assert er.recover_if_needed(root, argv=[]) is False
        assert er._read_marker_attempts(marker) == expected
        if "pid=" in body:
            assert marker.read_text(encoding="utf-8").startswith(body)
    before = len(calls)
    assert er.recover_if_needed(root, argv=[]) is False
    assert len(calls) == before
    output = capsys.readouterr()
    assert output.out == ""
    assert "hermes pm repair" in output.err


def test_recovery_obeys_live_owner_and_single_flight(tmp_path, monkeypatch):
    root = _project(tmp_path)
    marker = root / ".update-incomplete"
    marker.write_text(f"started=1\npid={os.getpid()}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(recovery, "repair_dependencies", calls.append)
    assert er.recover_if_needed(root, argv=[]) is False
    assert calls == []
    assert marker.exists()
    marker.write_text("interrupted", encoding="utf-8")
    fd = er._claim_recovery_lock(root)
    assert fd is not None
    try:
        assert er.recover_if_needed(root, argv=[]) is False
        assert calls == []
        assert marker.exists()
    finally:
        os.close(fd)
    assert er.recover_if_needed(root, argv=["update"]) is True
    assert calls == [root]


def test_missing_environment_cannot_write_a_retry_marker_without_lock(tmp_path, monkeypatch):
    from pm.environments import install_state_dir, runtime_facts_path
    from pm.lock import Facts

    root = _project(tmp_path)
    state = install_state_dir(root)
    Facts(runtime_facts_path(root)).record_state("venv", "old", [], environment=state / "environments" / "old" / "venv")
    fd = er._claim_recovery_lock(root)
    assert fd is not None
    try:
        assert er.recover_if_needed(root, argv=[]) is False
        assert not (state / ".repair-incomplete").exists()
    finally:
        os.close(fd)


def test_pm_commands_and_healthy_startup_do_not_repair(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setattr(recovery, "repair_dependencies", lambda _: pytest.fail("unexpected install"))
    assert er.recover_if_needed(root, argv=[]) is False
    marker = root / ".update-incomplete"
    marker.write_text("interrupted", encoding="utf-8")
    assert er.recover_if_needed(root, argv=["pm", "repair"]) is False
    assert marker.exists()



def test_pid_liveness_recognizes_current_process():
    assert er._pid_is_running(os.getpid()) is True
    assert er._pid_is_running(0) is False


@pytest.mark.platforms("posix")
def test_pid_liveness_counts_a_zombie_as_dead():
    """A crashed stage lingering unreaped must not read as a live owner.

    ``os.kill(pid, 0)`` succeeds for a zombie, so before the state probe a
    crashed updater under an un-reaping parent pinned every lock keyed on its
    pid (update marker, recovery marker) for the full age ceiling.
    """
    pid = os.fork()
    assert pid >= 0
    if pid == 0:
        os._exit(0)  # noqa: P111 — child exits without running pytest teardown

    # Do NOT wait() yet: the child must linger unreaped (a zombie). Poll until
    # the state probe actually reports 'Z' so the assertion can't race the exit.
    became_zombie = False
    for _ in range(40):
        state = er._process_state(pid)
        if state is not None and state.upper().startswith("Z"):
            became_zombie = True
            break
        time.sleep(0.05)
    assert became_zombie, "child never reached zombie state on this platform"

    assert er._pid_is_running(pid) is False, "a zombie is not a live owner"

    os.waitpid(pid, 0)  # reap so the test leaks no children


@pytest.mark.platforms("linux", "macos")
def test_process_state_reports_a_letter_for_a_live_pid():
    state = er._process_state(os.getpid())
    assert state is not None and len(state) == 1, "a live pid must expose a state"
    assert not state.upper().startswith("Z"), "this process is not a zombie"


@pytest.mark.platforms("posix")
def test_process_state_is_none_for_a_dead_pid():
    state = er._process_state(4294967294)
    assert state is None, "an unprobeable pid must degrade to None (unknown)"


def test_marker_owner_liveness_uses_recorded_pid(tmp_path, monkeypatch):
    marker = tmp_path / ".update-incomplete"
    marker.write_text("started=1\npid=4321\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        er, "_pid_is_running", lambda pid: seen.append(pid) or True
    )

    assert er._marker_owner_is_live(marker) is True
    assert seen == [4321]

def _project(tmp_path: Path, *, pyproject: bool = True) -> Path:
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    if pyproject:
        (root / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndependencies = [\n'
            '  "ruamel.yaml==0.18.17",\n'
            '  "python-dotenv==1.2.2",\n'
            '  "PyJWT[crypto]==2.13.0",\n'
            "]\n",
            encoding="utf-8",
        )
    return root


@pytest.mark.parametrize("link", ["hardlink", "symlink"])
def test_zip_journal_writer_never_writes_through_a_preexisting_temp(tmp_path, link):
    """A temp left at the journal's fixed temp name as a link must not carry the journal into its target."""
    root = tmp_path / "install"
    root.mkdir()
    secret = root / ".env"
    secret.write_text("API_KEY=keep\n", encoding="utf-8")
    tmp = root / (er.ZIP_SWAP_JOURNAL + ".tmp")
    if link == "hardlink":
        os.link(secret, tmp)
    else:
        try:
            tmp.symlink_to(secret)
        except OSError:
            pytest.skip("symlinks need privileges here")
    erz.write_zip_swap_journal(root, "staging", [["a.py", True, "", ""]], "0123456789ab")
    assert secret.read_text(encoding="utf-8-sig") == "API_KEY=keep\n"
    journal = root / er.ZIP_SWAP_JOURNAL
    assert not journal.is_symlink() and '"phase": "staging"' in journal.read_text(encoding="utf-8-sig")


def test_zip_journal_publication_never_deletes_a_file_at_its_old_fixed_temp_name(tmp_path):
    """The writer unlinked ``.hermes-update-zip-swap.tmp`` before its exclusive create: a user's file by
    that name was erased (review Z5). Its temp is now unpredictable: the file survives, byte for byte."""
    root = tmp_path / "install"
    root.mkdir()
    user = root / (er.ZIP_SWAP_JOURNAL + ".tmp")
    user.write_text("USER NOTES\n", encoding="utf-8")
    erz.write_zip_swap_journal(root, "staging", [["a.py", True, "", ""]], "0123456789ab")
    erz.write_zip_swap_journal(root, "swapping", [["a.py", True, "", ""]], "0123456789ab")
    assert user.read_text(encoding="utf-8-sig") == "USER NOTES\n"
    assert '"phase": "swapping"' in (root / er.ZIP_SWAP_JOURNAL).read_text(encoding="utf-8-sig")
    assert sorted(p.name for p in root.iterdir()) == sorted([er.ZIP_SWAP_JOURNAL, user.name])


def _symlink_or_skip(link: Path, target: str) -> None:
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks need privileges here")


@pytest.mark.parametrize("kind", ["dangling-symlink", "live-symlink", "regular"])
def test_an_interrupted_swap_restores_its_backup_by_entry_not_by_target(tmp_path, kind):
    """A swap killed after it moved a tracked symlink aside and installed a regular file: recovery took
    a DANGLING backup for "no backup" (exists() follows it), then deleted it and retired the journal
    (review Z3). The entry itself comes back, type and target intact; regular/live controls too."""
    root = tmp_path / "install"
    root.mkdir()
    (root / "target.txt").write_text("T", encoding="utf-8")
    old = root / "alpha.hermes-update-old"
    if kind == "regular":
        old.write_text("OLD", encoding="utf-8")
    else:
        _symlink_or_skip(old, "gone.txt" if kind == "dangling-symlink" else "target.txt")
    (root / "alpha").write_text("NEW", encoding="utf-8")
    # As the swap journals it: the staged copy (now live) and the moved-aside original by identity.
    erz.write_zip_swap_journal(root, "swapping", [["alpha", True, erz.zip_entry_identity(root / "alpha"),
                                                  erz.zip_entry_identity(old)]], "0123456789ab")

    assert erz.restore_interrupted_zip_swap(root) is True
    alpha = root / "alpha"
    if kind == "regular":
        assert not alpha.is_symlink() and alpha.read_text(encoding="utf-8") == "OLD"
    else:
        assert alpha.is_symlink() and os.readlink(alpha) == ("gone.txt" if kind == "dangling-symlink" else "target.txt")
    assert not os.path.lexists(old) and not (root / er.ZIP_SWAP_JOURNAL).exists()
    assert (root / "target.txt").read_text(encoding="utf-8") == "T"


@pytest.mark.parametrize("body", ["", "{not json", '{"phase": "swap", "entries": [["a.py", true]]}',
                                  '{"phase": "swapping", "entries": [["../a.py", true]]}', "read-error"])
def test_an_unparsed_zip_journal_keeps_itself_and_every_backup(tmp_path, monkeypatch, body):
    """A mid-swap tree (a.py new with its backup, b.py old with its staging copy) whose journal cannot
    be read or understood is never settled by guesswork: the journal and every sibling stay."""
    root = tmp_path / "install"
    root.mkdir()
    for name, text in {"a.py": "NEW_A", "a.py.hermes-update-old": "OLD_A", "b.py": "OLD_B",
                       "b.py.hermes-update-staging": "NEW_B"}.items():
        (root / name).write_text(text, encoding="utf-8")
    journal = root / er.ZIP_SWAP_JOURNAL
    journal.write_text('{"phase": "swapping", "gen": "0123456789ab", "entries": [["a.py", true, "", ""], '
                       '["b.py", true, "", ""]]}' if body == "read-error" else body, encoding="utf-8")
    if body == "read-error":  # one transient read failure (AV scan, sharing violation)
        real = Path.read_text

        def flaky(self, *args, **kwargs):
            if self == journal:
                raise PermissionError(13, "in use")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky)
    assert erz.restore_interrupted_zip_swap(root) is False
    assert journal.is_file()
    assert {p.name: p.read_text(encoding="utf-8-sig") for p in root.iterdir() if p != journal and p.suffix != ".lock"} == {
        "a.py": "NEW_A", "a.py.hermes-update-old": "OLD_A", "b.py": "OLD_B", "b.py.hermes-update-staging": "NEW_B"}


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 0)() == 0,
                    reason="POSIX modes; root ignores them")
def test_dropping_a_staged_tree_never_changes_a_hardlinked_live_files_mode(tmp_path):
    live = tmp_path / "release"
    live.mkdir()
    (live / "Hermes.exe").write_bytes(b"app")
    staging = tmp_path / "apps.hermes-update-staging" / "release"
    staging.mkdir(parents=True)
    os.link(live / "Hermes.exe", staging / "Hermes.exe")
    (live / "Hermes.exe").chmod(0o555)
    staging.chmod(0o555)  # a read-only dir refuses rmtree: the retry path runs
    erz._drop_path(staging.parent)
    assert not staging.parent.exists()
    assert (live / "Hermes.exe").stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize("body", ['{"attempts": Infinity}', '{"attempts": -Infinity}', '{"attempts": NaN}',
                                  '{"attempts": true}', '{"attempts": -4}'])
def test_marker_attempts_from_a_hand_edited_body_are_a_nonnegative_int(tmp_path, body):
    marker = tmp_path / ".update-incomplete"
    marker.write_text(body, encoding="utf-8")
    attempts = er._read_marker_attempts(marker)
    assert type(attempts) is int and attempts >= 0


def test_a_failed_repair_counts_its_attempt_even_on_an_undecodable_marker(tmp_path):
    """The attempt bump read the marker strictly, inside only suppress(OSError): a non-UTF-8 marker
    raised UnicodeDecodeError out of every launch's failed repair (review C6)."""
    marker = tmp_path / ".update-incomplete"
    marker.write_bytes(b"pid=1\nstash=\xff\xfe\n")

    er._count_failed_attempt(marker)

    assert er._read_marker_attempts(marker) == 1


def test_the_foreign_lock_record_is_fsynced_under_a_per_writer_temp(tmp_path, monkeypatch):
    """_remember_foreign_lock rewrote the restore's only record through a fixed-name ``.tmp`` with no
    fsync (review C12, invariant 5): a crash could leave an empty marker, and two writers shared one temp."""
    marker, lock = tmp_path / "hermes-update-pull", tmp_path / "update.lock"
    marker.write_text("pid=1\npre=abc\n", encoding="utf-8")
    lock.write_text("x", encoding="utf-8")
    synced, replaced = [], []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    monkeypatch.setattr(os, "replace", lambda src, dst: replaced.append((Path(src), Path(dst), len(synced)))
                        or real_replace(src, dst))

    er._remember_foreign_lock(marker, lock)

    assert [(dst, count > 0) for _src, dst, count in replaced] == [(marker, True)]
    assert replaced[0][0].name != marker.name + ".tmp"
    assert marker.read_text(encoding="utf-8").endswith(f"foreign_lock={er._lock_identity(lock)}\n")
