
import os
import sys
import time
import errno
import pytest
import shutil
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from hermes_cli.doctor_state import _check_deleted_sidecars, ScannerDisposition
from hermes_cli.doctor_report import Finding

def _wait_for_ready(proc, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if "READY" in line:
            return True
        if proc.poll() is not None:
            raise RuntimeError(f"Process exited early with {proc.returncode}")
    raise TimeoutError("Process did not print READY in time")

def _write_holder_script(script_path):
    content = """
import sys
import time
import sqlite3

db_path = sys.argv[1]
conn = sqlite3.connect(db_path, isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("CREATE TABLE sentinel (id INT)")
conn.execute("INSERT INTO sentinel VALUES (42)")
conn.commit()

print("READY", flush=True)

while True:
    time.sleep(1)
"""
    script_path.write_text(content)
    if sys.platform.startswith("linux"):
        os.chmod(script_path, 0o755)


def mock_platform(monkeypatch, hermes_home):
    monkeypatch.setattr('hermes_cli.doctor.HERMES_HOME', hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import os
    original_listdir = os.listdir
    import hermes_cli.doctor_state as ds
    monkeypatch.setattr(ds, "_has_cooperative_barrier", lambda p: True, raising=False)
    def mock_listdir(path):
        if path == "/proc":
            import psutil
            pids = [str(os.getpid())]
            for child in psutil.Process().children(recursive=True):
                try:
                    if child.status() != psutil.STATUS_ZOMBIE:
                        pids.append(str(child.pid))
                except Exception:
                    pass
            return pids
        return original_listdir(path)
        return original_listdir(path)
    monkeypatch.setattr(os, "listdir", mock_listdir)

def test_real_subprocess_preservation(monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    if not hasattr(os, "pidfd_open"):
        pytest.skip("Test requires pidfd_open support in Python/OS")
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        db_path = hermes_home / "state.db"
        
        mock_platform(monkeypatch, hermes_home)
        
        script_path = tmp_path / "hermes"
        _write_holder_script(script_path)
        
        env = os.environ.copy()
        env["HERMES_HOME"] = str(hermes_home)
        
        
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(db_path)],
            env=env, stdout=subprocess.PIPE, text=True
        )
        
        try:
            _wait_for_ready(proc)
            wal_path = hermes_home / "state.db-wal"
            wal_path.unlink()
            
            f = Finding()
            disp = _check_deleted_sidecars(f, True, db_path)
            
            assert disp == ScannerDisposition.CLEAR, f"Issues: {f.issues}"
            assert getattr(f, "checkpoint_verified", False)
            
            artifacts = list(hermes_home.glob("*.retired-wal-*"))
            assert len(artifacts) == 1
            artifact_dir = artifacts[0]
            
            recovery_dir = tmp_path / "recovery"
            shutil.copytree(artifact_dir, recovery_dir)
            
            r_conn = sqlite3.connect(recovery_dir / "state.db")
            res = r_conn.execute("SELECT * FROM sentinel").fetchall()
            assert res == [(42,)]
            
        finally:
            proc.kill()
            proc.wait()

def test_capture_failure_aborts_all(monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    if not hasattr(os, "pidfd_open"):
        pytest.skip("Test requires pidfd_open support in Python/OS")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        db_path = hermes_home / "state.db"
        
        mock_platform(monkeypatch, hermes_home)

        script_path = tmp_path / "hermes"
        _write_holder_script(script_path)

        env = os.environ.copy()
        env["HERMES_HOME"] = str(hermes_home)
        

        proc1 = subprocess.Popen(
            [sys.executable, str(script_path), str(db_path)],
            env=env, stdout=subprocess.PIPE, text=True
        )
        
        script2 = tmp_path / "hermes-agent"
        script2.write_text("""
import sys, time, sqlite3
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
while True:
    try:
        conn.execute("SELECT * FROM sentinel")
        break
    except sqlite3.OperationalError:
        time.sleep(0.1)
print("READY", flush=True)
while True:
    time.sleep(1)
""")
        proc2 = subprocess.Popen(
            [sys.executable, str(script2), str(db_path)],
            env=env, stdout=subprocess.PIPE, text=True
        )

        try:
            _wait_for_ready(proc1)
            _wait_for_ready(proc2)

            wal_path = hermes_home / "state.db-wal"
            wal_path.unlink()

            import hermes_state_dbfile
            original_capture = hermes_state_dbfile.capture_retired_wal_generation
            
            call_count = 0
            def mock_capture(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("Mocked second capture failure")
                return original_capture(*args, **kwargs)

            monkeypatch.setattr(hermes_state_dbfile, "capture_retired_wal_generation", mock_capture)

            f = Finding()
            disp = _check_deleted_sidecars(f, True, db_path)

            assert disp == ScannerDisposition.HOLDERS, f"Issues: {f.issues}"
            assert any("Failed to durably capture WAL" in iss for iss in f.issues), f"Issues: {f.issues}"
            
            assert proc1.poll() is None
            assert proc2.poll() is None

        finally:
            proc1.kill()
            proc2.kill()
            proc1.wait()
            proc2.wait()

def test_checkpoint_busy_handled(monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    if not hasattr(os, "pidfd_open"):
        pytest.skip("Test requires pidfd_open support in Python/OS")
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        db_path = hermes_home / "state.db"
        
        mock_platform(monkeypatch, hermes_home)
        
        script_path = tmp_path / "hermes"
        _write_holder_script(script_path)
        env = os.environ.copy()
        env["HERMES_HOME"] = str(hermes_home)
        
        
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(db_path)],
            env=env, stdout=subprocess.PIPE, text=True
        )
        
        try:
            _wait_for_ready(proc)
            wal_path = hermes_home / "state.db-wal"
            wal_path.unlink()
            
            import hermes_state_repair
            original_guard = hermes_state_repair._exclusive_repair_db_guard
            
            class MockConnection:
                def __init__(self, real_conn):
                    self.real_conn = real_conn
                def execute(self, sql, *args, **kwargs):
                    if "wal_checkpoint" in sql:
                        m = MagicMock()
                        m.fetchone.return_value = (1, 10, 10)
                        return m
                    return getattr(self.real_conn, "execute")(sql, *args, **kwargs)
                def __getattr__(self, name):
                    return getattr(self.real_conn, name)

            class MockGuardContext:
                def __init__(self, real_guard):
                    self.real_guard = real_guard
                def __enter__(self):
                    self.conn, self.err = self.real_guard.__enter__()
                    if self.conn:
                        self.conn = MockConnection(self.conn)
                    return self.conn, self.err
                def __exit__(self, *args):
                    return self.real_guard.__exit__(*args)
            def mock_guard_factory(path):
                return MockGuardContext(original_guard(path))
                
            monkeypatch.setattr(hermes_state_repair, "_exclusive_repair_db_guard", mock_guard_factory)

            f = Finding()
            disp = _check_deleted_sidecars(f, True, db_path)
            
            assert disp == ScannerDisposition.HOLDERS, f"Issues: {f.issues}"
            assert any("WAL checkpoint returned busy flag" in iss for iss in f.issues), f"Issues: {f.issues}"
            
        finally:
            proc.kill()
            proc.wait()

def test_macOS_like_no_pidfd_reports_holders(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        hermes_home = tmp_path / "hermes_home"
        db_path = hermes_home / "state.db"
        
        import hermes_state_dbfile
        def mock_darwin_holders(path):
            return [(99999, str(path) + "-wal")]
        monkeypatch.setattr(hermes_state_dbfile, "_iter_darwin_sidecar_holders", mock_darwin_holders)
        
        f = Finding()
        disp = _check_deleted_sidecars(f, True, db_path)
        
        assert disp == ScannerDisposition.HOLDERS, f"Issues: {f.issues}"

def test_r5_01_unrelated_db_scoped():
    """R5-01: Scan target DB while child holds unrelated deleted WAL."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        unrelated_db = tmp_path / "unrelated.db"
        
        conn = sqlite3.connect(unrelated_db, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE sentinel (id INT)")
        conn.execute("INSERT INTO sentinel VALUES (42)")
        
        unrelated_wal = tmp_path / "unrelated.db-wal"
        unrelated_shm = tmp_path / "unrelated.db-shm"
        if unrelated_wal.exists(): unrelated_wal.unlink()
        if unrelated_shm.exists(): unrelated_shm.unlink()
            
        f = Finding()
        import hermes_cli.doctor_state as ds
        original_listdir = os.listdir
        def mock_listdir(path):
            if path == "/proc":
                return [str(os.getpid())]
            return original_listdir(path)
        
        try:
            os.listdir = mock_listdir
            disp, holders = ds._strict_scan_deleted_sidecars(db_path)
        finally:
            os.listdir = original_listdir
        
        assert disp == ScannerDisposition.CLEAR, f"Holders: {holders}"

def test_r5_03_current_process_holder():
    """R5-03: Current-process retired holder must not be CLEAR."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        
        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE sentinel (id INT)")
        conn.execute("INSERT INTO sentinel VALUES (42)")
        
        wal_path = tmp_path / "state.db-wal"
        wal_path.unlink()
        
        import hermes_cli.doctor_state as ds
        original_listdir = os.listdir
        def mock_listdir(path):
            if path == "/proc":
                return [str(os.getpid())]
            return original_listdir(path)
        
        try:
            os.listdir = mock_listdir
            disp, holders = ds._strict_scan_deleted_sidecars(db_path)
        finally:
            os.listdir = original_listdir
        assert disp == ScannerDisposition.HOLDERS
        assert any(h["pid"] == os.getpid() for h in holders)

def test_r5_03_permission_denied():
    """R5-03: Per-process permission denial must return UNKNOWN."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        
        import hermes_cli.doctor_state as ds
        original_listdir = os.listdir
        def mock_listdir(path):
            if path.startswith("/proc/") and path.endswith("/fd"):
                raise OSError(13, "Permission denied")
            return original_listdir(path)
            
        def mock_listdir_root(path):
            if path == "/proc":
                return ["1", "2"] 
            return mock_listdir(path)
            
        try:
            os.listdir = mock_listdir_root
            disp, holders = ds._strict_scan_deleted_sidecars(db_path)
            assert disp == ScannerDisposition.UNKNOWN
        finally:
            os.listdir = original_listdir

def test_r5_02_caller_propagation_holders(monkeypatch):
    """R5-02: Caller receives HOLDERS -> returns early."""
    from hermes_cli.doctor_state import _check_state_db
    import hermes_cli.doctor_state as ds
    
    f = Finding()
    should_fix = True
    
    mock_health_called = False
    def mock_health(*args):
        nonlocal mock_health_called
        mock_health_called = True
        
    monkeypatch.setattr(ds, "_state_db_health", mock_health)
    
    def mock_check_deleted(*args):
        return ScannerDisposition.HOLDERS
        
    monkeypatch.setattr(ds, "_check_deleted_sidecars", mock_check_deleted)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        monkeypatch.setattr('hermes_cli.doctor.HERMES_HOME', tmp_path)
        (tmp_path / "state.db").write_text("fake")
        
        _check_state_db.__wrapped__(should_fix, f)
        assert not mock_health_called, "Health check should not be called when blocked"

def test_r5_05_shm_only_blocked(monkeypatch):
    """R5-05: SHM-only candidate returns blocked finding."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        
        monkeypatch.setattr('hermes_cli.doctor.HERMES_HOME', tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        
        def mock_pidfd_open(pid, flags=0): return pid
        import os
        if not hasattr(os, "pidfd_open"): monkeypatch.setattr(os, "pidfd_open", mock_pidfd_open, raising=False)
        
        def mock_strict_scan(*args, **kwargs):
            return ScannerDisposition.HOLDERS, [{
                "pid": os.getpid(),
                "canonical": str(db_path) + "-shm",
                "fd_path": "/fake",
                "dev": 1,
                "ino": 2
            }]
            
        import hermes_cli.doctor_state as ds
        monkeypatch.setattr(ds, "_strict_scan_deleted_sidecars", mock_strict_scan)
        
        import psutil
        original_process = psutil.Process
        class MockProcess(original_process):
            def exe(self): return "/usr/bin/hermes"
            def cmdline(self): return ["/usr/bin/hermes", "--profile", "default"]
            def environ(self): return {"HERMES_HOME": str(tmp_path)}
                
        monkeypatch.setattr(psutil, "Process", MockProcess)

        f = Finding()
        disp = _check_deleted_sidecars(f, True, db_path)
        assert disp == ScannerDisposition.HOLDERS
        assert any("has no WAL descriptor" in issue for issue in f.issues), f.issues

def test_r5_06_entrypoint_identity(monkeypatch):
    """R5-06: Misleading executable path returns blocked."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        
        monkeypatch.setattr('hermes_cli.doctor.HERMES_HOME', tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        
        def mock_pidfd_open(pid, flags=0): return pid
        import os
        if not hasattr(os, "pidfd_open"): monkeypatch.setattr(os, "pidfd_open", mock_pidfd_open, raising=False)
        
        def mock_strict_scan(*args, **kwargs):
            return ScannerDisposition.HOLDERS, [{
                "pid": os.getpid(),
                "canonical": str(db_path) + "-wal",
                "fd_path": "/fake",
                "dev": 1,
                "ino": 2
            }]
            
        import hermes_cli.doctor_state as ds
        monkeypatch.setattr(ds, "_strict_scan_deleted_sidecars", mock_strict_scan)
        
        import psutil
        original_process = psutil.Process
        class MockProcess(original_process):
            def exe(self): return "/tmp/hermes-tools/fake_script.py"
            def cmdline(self): return ["/usr/bin/python3", "/tmp/hermes-tools/fake_script.py"]
            def environ(self): return {"HERMES_HOME": str(tmp_path)}
                
        monkeypatch.setattr(psutil, "Process", MockProcess)
        
        f = Finding()
        disp = _check_deleted_sidecars(f, True, db_path)
        assert disp == ScannerDisposition.HOLDERS
        assert any("lacks verified Hermes entrypoint" in issue for issue in f.issues), f.issues

def test_r5_04_cooperative_barrier_disabled(monkeypatch):
    """R5-04: Legacy processes lack cooperative barrier -> auto termination disabled."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Test requires Linux")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        
        monkeypatch.setattr('hermes_cli.doctor.HERMES_HOME', tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        
        def mock_pidfd_open(pid, flags=0): return pid
        import os
        if not hasattr(os, "pidfd_open"): monkeypatch.setattr(os, "pidfd_open", mock_pidfd_open, raising=False)
        
        def mock_strict_scan(*args, **kwargs):
            return ScannerDisposition.HOLDERS, [{
                "pid": os.getpid(),
                "canonical": str(db_path) + "-wal",
                "fd_path": "/fake",
                "dev": 1,
                "ino": 2
            }]
            
        import hermes_cli.doctor_state as ds
        monkeypatch.setattr(ds, "_strict_scan_deleted_sidecars", mock_strict_scan)
        
        import psutil
        original_process = psutil.Process
        class MockProcess(original_process):
            def exe(self): return "/usr/bin/hermes"
            def cmdline(self): return ["/usr/bin/hermes", "--profile", "default"]
            def environ(self): return {"HERMES_HOME": str(tmp_path)}
                
        monkeypatch.setattr(psutil, "Process", MockProcess)
        
        f = Finding()
        disp = _check_deleted_sidecars(f, True, db_path)
        assert disp == ScannerDisposition.HOLDERS
        assert any("lacks a verified write-quarantine" in issue for issue in f.issues), f.issues


def test_r6_03_caller_propagation_unknown(monkeypatch):
    from hermes_cli.doctor_state import _check_state_db, ScannerDisposition
    import hermes_cli.doctor_state as ds
    from hermes_cli.doctor import Finding
    import tempfile
    from pathlib import Path
    
    f = Finding()
    
    mock_health_called = False
    def mock_health(*args):
        nonlocal mock_health_called
        mock_health_called = True
    monkeypatch.setattr(ds, "_state_db_health", mock_health)
    
    def mock_strict_scan(*args, **kwargs):
        return ScannerDisposition.UNKNOWN, []
    monkeypatch.setattr(ds, "_strict_scan_deleted_sidecars", mock_strict_scan)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Patch exists() to always return True for state.db so we don't rely on HERMES_HOME matching
        original_exists = Path.exists
        def mock_exists(self):
            if self.name == "state.db": return True
            return original_exists(self)
        monkeypatch.setattr(Path, "exists", mock_exists)
        
        _check_state_db.__wrapped__(True, f)
        assert not mock_health_called, "Health check should not be called when UNKNOWN"
        assert any("was incomplete" in iss for iss in f.issues), f"Expected UNKNOWN finding, got: {f.issues}"

def test_r6_03_caller_propagation_wal(monkeypatch):
    from hermes_cli.doctor_state import _check_state_db, ScannerDisposition
    import hermes_cli.doctor_state as ds
    from hermes_cli.doctor import Finding
    import tempfile
    from pathlib import Path
    
    f = Finding()
    
    mock_health_called = False
    def mock_health(*args):
        nonlocal mock_health_called
        mock_health_called = True
    monkeypatch.setattr(ds, "_state_db_health", mock_health)
    
    mock_wal_called = False
    def mock_wal(*args):
        nonlocal mock_wal_called
        mock_wal_called = True
    monkeypatch.setattr(ds, "_state_db_wal", mock_wal)
    
    def mock_check_deleted(*args):
        return ScannerDisposition.HOLDERS
    monkeypatch.setattr(ds, "_check_deleted_sidecars", mock_check_deleted)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Patch exists() to always return True for state.db so we don't rely on HERMES_HOME matching
        original_exists = Path.exists
        def mock_exists(self):
            if self.name == "state.db": return True
            return original_exists(self)
        monkeypatch.setattr(Path, "exists", mock_exists)
        
        _check_state_db.__wrapped__(True, f)
        assert not mock_health_called, "Health check should not be called when HOLDERS"
        assert not mock_wal_called, "WAL check should not be called when HOLDERS"

def test_r6_03_exemptions_strict(monkeypatch):
    import hermes_cli.doctor_state as ds
    from hermes_cli.doctor_state import ScannerDisposition
    import tempfile
    from pathlib import Path
    import sys, os
    if not sys.platform.startswith("linux"):
        import pytest
        pytest.skip("Test requires Linux")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        
        original_listdir = os.listdir
        def mock_listdir(path):
            if path == "/proc":
                return ["9999"]
            if path == "/proc/9999/fd":
                return ["3", "4"]
            return original_listdir(path)
            
        original_readlink = os.readlink
        def mock_readlink(path, *args, **kwargs):
            if str(path) == "/proc/9999/fd/3" or str(path) == "/proc/9999/fd/4":
                return str(db_path) + "-wal (deleted)"
            return original_readlink(path, *args, **kwargs)
            
        original_stat = os.stat
        def mock_stat(path, *args, **kwargs):
            if str(path) == "/proc/9999/fd/3" or str(path) == "/proc/9999/fd/4":
                class Stat:
                    st_dev = 1
                    st_ino = 1234
                return Stat()
            return original_stat(path, *args, **kwargs)
            
        monkeypatch.setattr(os, "listdir", mock_listdir)
        monkeypatch.setattr(os, "readlink", mock_readlink)
        monkeypatch.setattr(os, "stat", mock_stat)
        
        disp, holders = ds._strict_scan_deleted_sidecars(db_path, exempt_identities={"/proc/9999/fd/3"})
        
        assert disp == ScannerDisposition.HOLDERS
        assert len(holders) == 1
        assert holders[0]["fd_path"] == "/proc/9999/fd/4", "Only exact descriptors should be exempted"



def test_r9_linux_eio_ebadf(monkeypatch):
    import hermes_cli.doctor_state as ds
    from hermes_cli.doctor_state import ScannerDisposition
    import tempfile
    from pathlib import Path
    import sys, os
    if not sys.platform.startswith("linux"):
        import pytest
        pytest.skip("Test requires Linux")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        
        original_listdir = os.listdir
        def mock_listdir(path):
            if path == "/proc":
                return ["9999"]
            if path == "/proc/9999/fd":
                raise OSError(5, "Input/output error") # EIO
            return original_listdir(path)
            
        monkeypatch.setattr(os, "listdir", mock_listdir)
        
        disp, holders = ds._strict_scan_deleted_sidecars(db_path)
        assert disp == ScannerDisposition.UNKNOWN
        
        # Test EBADF
        def mock_listdir_ebadf(path):
            if path == "/proc":
                return ["9999"]
            if path == "/proc/9999/fd":
                raise OSError(9, "Bad file descriptor") # EBADF
            return original_listdir(path)
            
        monkeypatch.setattr(os, "listdir", mock_listdir_ebadf)
        
        disp, holders = ds._strict_scan_deleted_sidecars(db_path)
        assert disp == ScannerDisposition.UNKNOWN


def test_r9_macos_empty_unknown(monkeypatch):
    import hermes_cli.doctor_state as ds
    from hermes_cli.doctor_state import ScannerDisposition
    import tempfile
    from pathlib import Path
    import sys
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "state.db"
        
        monkeypatch.setattr(sys, "platform", "darwin")
        
        def mock_iter_darwin(*args, **kwargs):
            return [] # Returns empty
            
        monkeypatch.setattr("hermes_state_dbfile._iter_darwin_sidecar_holders", mock_iter_darwin, raising=False)
        
        disp, holders = ds._strict_scan_deleted_sidecars(db_path)
        assert disp == ScannerDisposition.UNKNOWN
