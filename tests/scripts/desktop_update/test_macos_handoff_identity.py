"""Native shell/Python custody boundary; the fixture updater never updates code."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import pytest

from hermes_cli.update_lock import UpdateLock, process_create_time
from tests.scripts.desktop_update.test_desktop_update_posix_marker import POSIX

pytestmark = pytest.mark.platforms("macos")
ROOT = POSIX.parents[2]


def _frozen_recovery(tmp_path):
    from hermes_cli._early_recovery import RECOVERY_CLOSURE

    frozen = tmp_path / "frozen"
    for rel in RECOVERY_CLOSURE:
        target = frozen / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    (frozen / "hermes_cli" / "__init__.py").write_text("")
    return frozen


def test_loaded_shell_identity_survives_source_removal_during_custody(tmp_path):
    source = tmp_path / "checkout"
    script = source / "scripts" / "desktop-update" / "marker.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(POSIX.with_name("marker.sh"), script)
    helper = ROOT / "hermes_cli" / "_darwin_process_identity.py"
    if helper.exists():
        (source / "hermes_cli").mkdir()
        shutil.copyfile(helper, source / "hermes_cli" / helper.name)
    moved = tmp_path / "moved"
    marker = tmp_path / "marker"
    command = (
        f'MARKER={shlex.quote(str(marker))}; INSTALL_ROOT={shlex.quote(str(tmp_path))}; '
        "DESKTOP_PID=0 HANDOFF_RUN='' MARKER_CLAIMED=1; log() { :; }; "
        f'. {shlex.quote(str(script))}; '
        'MY_CT=$(proc_ct "$MY_PID"); STARTED_AT=$(marker_now); '
        'marker_own_body "$STARTED_AT"; marker_locked marker_publish_new "$MARKER_BODY"; '
        f'/bin/mv {shlex.quote(str(source))} {shlex.quote(str(moved))}; '
        f'/bin/rm -f {shlex.quote(str(moved / "scripts/desktop-update/marker.sh"))} '
        f'{shlex.quote(str(moved / "hermes_cli/_darwin_process_identity.py"))}; '
        f'proc_ct {os.getpid()}; marker_add_delegate {os.getpid()}; '
        'marker_locked marker_release_locked'
    )
    result = subprocess.run(["/bin/bash", "-c", command], capture_output=True,
                            text=True, timeout=20, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert not source.exists()
    assert not (moved / "scripts/desktop-update/marker.sh").exists()
    assert not (moved / "hermes_cli/_darwin_process_identity.py").exists()
    assert result.stdout.strip(), "loaded proc_ct lost its source dependency"
    assert float(result.stdout) == pytest.approx(process_create_time(), abs=0.0005)
    # Releasing the shell's custody preserves the precise live delegate.
    assert f'delegate:{os.getpid()} ct:' not in marker.read_text()
    lock = UpdateLock(path=marker)
    assert lock.acquire()
    lock.release()
    assert not marker.exists()


@pytest.mark.parametrize("no_site", [False, True], ids=["psutil", "stdlib"])
def test_shell_delegate_acquires_and_releases_under_sibling_custodian(tmp_path, no_site):
    home = tmp_path / "home"
    install = tmp_path / "install"
    launcher = install / ".hermes" / "bin" / "hermes"
    launcher.parent.mkdir(parents=True)
    (install / "pm").mkdir()
    witness = tmp_path / "witness.json"
    consumer = _frozen_recovery(tmp_path) if no_site else ROOT
    code = f'''import json, os, sys
from pathlib import Path
sys.path.insert(0, {str(consumer)!r})
from hermes_cli.update_lock import UpdateLock, process_create_time
if '--help' in sys.argv:
    print('--keep-stash'); sys.exit(0)
p = Path(os.environ['HERMES_HOME']) / '.hermes-update-in-progress'
before = p.read_text()
lock = UpdateLock(path=p, install_root=Path.cwd())
ok = lock.acquire()
lock.release()
Path({str(witness)!r}).write_text(json.dumps(dict(ok=ok, before=before, after=p.read_text(), pid=os.getpid(), ct=process_create_time())))
sys.exit(0 if ok else 2)
'''
    payload = tmp_path / "inert_updater.py"
    payload.write_text(code)
    launcher.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} {"-I -S" if no_site else ""} {shlex.quote(str(payload))} "$@"\n')
    launcher.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(home),
           "TMPDIR": str(tmp_path), "HERMES_UPDATE_SHIM_GRACE_SECONDS": "0"}
    result = subprocess.run(["/bin/bash", str(POSIX), "--daemonized", "--no-ui",
                             "--install-root", str(install)], env=env, cwd=tmp_path,
                            text=True, capture_output=True, timeout=45)
    assert witness.exists(), result.stdout + result.stderr
    observed = json.loads(witness.read_text())
    assert observed["ok"], observed
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'delegate:{observed["pid"]} ct:' in observed["before"]
    assert "delegate:" not in observed["after"], "Python release must retire its shell-written identity"
    assert observed["before"].splitlines()[0] == observed["after"].splitlines()[0]
    assert not (home / ".hermes-update-in-progress").exists(), "shell releases custody last"


def test_darwin_unreadable_identity_never_falls_back_to_coarse_time(monkeypatch):
    import ctypes
    from hermes_cli.update_lock import _stdlib_create_time

    def unavailable(*args, **kwargs):
        raise OSError("libproc unavailable")

    monkeypatch.setattr(ctypes, "CDLL", unavailable)
    assert _stdlib_create_time(os.getpid()) is None


def test_frozen_no_site_identity_is_precise_and_dead_pid_is_unknown(tmp_path):
    frozen = _frozen_recovery(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pid = child.pid
    try:
        expected = process_create_time(pid)
        code = (f'import sys; sys.path.insert(0, {str(frozen)!r}); '
                'from hermes_cli.update_lock import process_create_time; '
                f'print(process_create_time({pid}))')
        result = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                                capture_output=True, text=True, check=True, timeout=10)
        assert float(result.stdout) == pytest.approx(expected, abs=0.000001)
    finally:
        child.kill()
        child.wait(timeout=10)
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                            capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.strip() == "None"


def test_shell_timestamp_matches_fractional_kernel_identity(tmp_path):
    # Choose a real process demonstrably outside the strict own-incarnation
    # epsilon; do not fake Darwin or substitute a timestamp probe.
    for _ in range(20):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            ct = process_create_time(child.pid)
            assert ct is not None
            if ct % 1 <= 0.005:
                continue
            command = f'. {shlex.quote(str(POSIX.with_name("marker.sh")))}; proc_ct {child.pid}'
            result = subprocess.run(["/bin/bash", "-c", command], capture_output=True,
                                    text=True, timeout=10, cwd=tmp_path)
            assert result.returncode == 0, result.stderr
            assert float(result.stdout) == pytest.approx(ct, abs=0.0005)
            # Completion children run without site-packages, too.
            from hermes_cli.update_lock import _stdlib_create_time
            assert _stdlib_create_time(child.pid) == pytest.approx(ct, abs=0.000001)
            break
        finally:
            child.kill()
            child.wait(timeout=10)
    else:
        pytest.fail("could not obtain a fractional native process identity")


def test_foreign_contender_and_recycled_self_delegate_preserve_custody(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_UPDATE_HANDOFF_PID", raising=False)
    marker = tmp_path / "marker"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        command = (
            f'MARKER={shlex.quote(str(marker))}; MY_PID={child.pid}; '
            f'INSTALL_ROOT={shlex.quote(str(tmp_path))}; '
            "DESKTOP_PID=0 HANDOFF_RUN='' MARKER_CLAIMED=1; log() { :; }; "
            f'. {shlex.quote(str(POSIX.with_name("marker.sh")))}; '
            'STARTED_AT=$(marker_now); marker_own_body "$STARTED_AT"; '
            'marker_locked marker_publish_new "$MARKER_BODY"; '
            f'marker_add_delegate {os.getpid()}'
        )
        subprocess.run(["/bin/bash", "-c", command], check=True, timeout=15)
        original = marker.read_bytes()
        frozen = _frozen_recovery(tmp_path)
        contender = (
            # Detach from pytest: otherwise its live delegate is our ancestor,
            # legitimately authorizing a completion child rather than a rival.
            'import os, time; child=os.fork(); '
            'exec("if child: os._exit(0)\\nwhile os.getppid() != 1: time.sleep(0.01)"); '
            f'import sys; sys.path.insert(0, {str(frozen)!r}); '
            'from pathlib import Path; from hermes_cli.update_lock import UpdateLock; '
            f'lock=UpdateLock(path=Path({str(marker)!r})); '
            'ok=lock.acquire(); lock.release(); print(ok)'
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-c", contender],
                                text=True, capture_output=True, check=True, timeout=15)
        assert result.stdout.strip() == "False"
        assert marker.read_bytes() == original, "foreign release must not delete custody"
        # A nearby old incarnation of our PID is NOT an authorized delegate.
        ct = process_create_time()
        assert ct is not None
        stale = original.decode().replace(f'delegate:{os.getpid()} ct:{ct:.3f}',
                                          f'delegate:{os.getpid()} ct:{ct - 0.034:.3f}')
        assert stale != original.decode()
        marker.write_text(stale)
        lock = UpdateLock(path=marker)
        assert not lock.acquire(), "never widen the own-incarnation epsilon"
        lock.release()
        assert marker.read_text() == stale
        marker.write_bytes(original)
        assert lock.acquire(), "the exact shell-written delegate is authorized"
        lock.release()
        assert "delegate:" not in marker.read_text()
        assert int(marker.read_text().splitlines()[0]) == child.pid
        # Shell remains the custodian; only its release retires the marker.
        subprocess.run(["/bin/bash", "-c", command.split('STARTED_AT=')[0] +
                        'marker_locked marker_release_locked'], check=True, timeout=15)
        assert not marker.exists()
    finally:
        child.kill()
        child.wait(timeout=10)
