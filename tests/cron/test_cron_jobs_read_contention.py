"""``jobs.json`` reads must survive a concurrent atomic replace of the file.

Windows: CPython opens files WITHOUT FILE_SHARE_DELETE, so a plain ``open`` racing the ticker's
``os.replace`` onto jobs.json is denied with ERROR_ACCESS_DENIED/ERROR_SHARING_VIOLATION — both
reported as ``[Errno 13] Permission denied``. Before ``utils.read_text_shared`` that killed the
running job (measured on the host: the read during a multi-job dispatch second failed and the run
was recorded FAILED, with the ticker saving jobs.json in the same second).

The denial is transient — the rename window is sub-millisecond — so the fix is a bounded retry on
the read. A FILE_SHARE_DELETE reader handle was measured to leave the writer's ``os.replace`` denied
anyway, so it buys none of the expected benefit; see the rationale comment in ``utils``.
"""

import errno
import json

import pytest

import utils
from cron import jobs as cronjobs


def _denial():
    """The errno Windows reports for a denied open during a replace."""
    return PermissionError(errno.EACCES, "Permission denied")


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Route the cron store to an isolated temp dir."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(cronjobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(cronjobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(cronjobs, "OUTPUT_DIR", cron_dir / "output")
    return cron_dir


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Keep the retry budget but lose its wall-clock cost."""
    monkeypatch.setattr(utils, "_read_retry_delay", lambda attempt: 0.0)


# =========================================================================
# 1. Read retry semantics
# =========================================================================


class TestReadTextSharedRetry:
    def test_transient_denial_is_retried_then_succeeds(self, tmp_path, monkeypatch):
        target = tmp_path / "jobs.json"
        target.write_text('{"jobs": []}', encoding="utf-8")
        real_read = utils._read_text_once
        calls = []

        def flaky(path, encoding):
            calls.append(path)
            if len(calls) == 1:
                raise _denial()
            return real_read(path, encoding)

        monkeypatch.setattr(utils, "_read_text_once", flaky)

        assert utils.read_text_shared(target) == '{"jobs": []}'
        assert len(calls) == 2

    def test_persistent_denial_costs_exactly_the_budget(self, tmp_path, monkeypatch):
        target = tmp_path / "jobs.json"
        target.write_text("{}", encoding="utf-8")
        calls = []

        def always_denied(path, encoding):
            calls.append(path)
            raise _denial()

        monkeypatch.setattr(utils, "_read_text_once", always_denied)

        with pytest.raises(PermissionError):
            utils.read_text_shared(target)
        assert len(calls) == utils._READ_RETRY_ATTEMPTS + 1

    def test_missing_file_is_not_retried(self, tmp_path, monkeypatch):
        calls = []

        def counting(path, encoding):
            calls.append(path)
            raise FileNotFoundError(errno.ENOENT, "No such file", str(path))

        monkeypatch.setattr(utils, "_read_text_once", counting)

        with pytest.raises(FileNotFoundError):
            utils.read_text_shared(tmp_path / "absent.json")
        assert len(calls) == 1

    def test_unrelated_oserror_is_not_retried(self, tmp_path, monkeypatch):
        calls = []

        def counting(path, encoding):
            calls.append(path)
            raise OSError(errno.EIO, "I/O error")

        monkeypatch.setattr(utils, "_read_text_once", counting)

        with pytest.raises(OSError):
            utils.read_text_shared(tmp_path / "jobs.json")
        assert len(calls) == 1

    def test_bom_and_crlf_semantics_match_plain_open(self, tmp_path):
        """The retry must not change how the bytes decode: BOM stripped, CRLF normalized."""
        target = tmp_path / "jobs.json"
        target.write_bytes('\ufeff{"jobs": []}\r\n'.encode("utf-8"))

        assert utils.read_text_shared(target) == '{"jobs": []}\n'


# =========================================================================
# 2. jobs.json read path uses it (and keeps its old failure contract)
# =========================================================================


class TestJobsFileReadContention:
    def test_parse_jobs_file_survives_a_contended_open(self, cron_store, monkeypatch):
        cron_store.mkdir(parents=True, exist_ok=True)
        jobs_file = cron_store / "jobs.json"
        jobs_file.write_text(json.dumps({"jobs": [{"id": "alpha"}]}), encoding="utf-8")
        real_read = utils._read_text_once
        calls = []

        def flaky(path, encoding):
            calls.append(path)
            if len(calls) == 1:
                raise _denial()
            return real_read(path, encoding)

        monkeypatch.setattr(utils, "_read_text_once", flaky)

        data, strict_retry = cronjobs._parse_jobs_file(jobs_file)
        assert data["jobs"][0]["id"] == "alpha"
        assert strict_retry is False
        assert len(calls) == 2

    def test_load_jobs_reports_a_real_denial_as_a_runtime_error(self, cron_store, monkeypatch):
        """A non-transient denial must still surface — the retry must not swallow it."""
        cron_store.mkdir(parents=True, exist_ok=True)
        (cron_store / "jobs.json").write_text('{"jobs": []}', encoding="utf-8")

        def always_denied(path, encoding):
            raise _denial()

        monkeypatch.setattr(utils, "_read_text_once", always_denied)

        with pytest.raises(RuntimeError, match="Failed to read cron database"):
            cronjobs.load_jobs()

    def test_load_jobs_reads_a_healthy_store(self, cron_store):
        cron_store.mkdir(parents=True, exist_ok=True)
        (cron_store / "jobs.json").write_text(
            json.dumps({"jobs": [{"id": "beta", "prompt": "x"}]}), encoding="utf-8")

        loaded = cronjobs.load_jobs()
        assert [job["id"] for job in loaded] == ["beta"]


# =========================================================================
# 3. A real Windows denial (not a mock) must be absorbed
# =========================================================================


@pytest.mark.platforms("windows")
def test_read_text_shared_absorbs_a_real_windows_sharing_denial(tmp_path, monkeypatch):
    """A share-mode-0 handle denies our open with the production errno; the retry must outlive it."""
    import ctypes
    import threading
    import time
    from ctypes import wintypes

    target = tmp_path / "jobs.json"
    target.write_text('{"jobs": [{"id": "locked"}]}', encoding="utf-8")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.HANDLE]
    held = threading.Event()

    def blocker():
        handle = kernel32.CreateFileW(str(target), 0x80000000, 0, None, 3, 0x80, None)
        held.set()
        time.sleep(0.02)
        kernel32.CloseHandle(handle)

    real_read = utils._read_text_once
    denials = []

    def counting(path, encoding):
        try:
            return real_read(path, encoding)
        except PermissionError as exc:
            denials.append(exc.errno)
            raise

    monkeypatch.setattr(utils, "_read_text_once", counting)
    monkeypatch.setattr(utils, "_read_retry_delay", lambda attempt: 0.05)  # outlive the 20 ms hold

    thread = threading.Thread(target=blocker, daemon=True)
    thread.start()
    assert held.wait(2.0)

    assert utils.read_text_shared(target) == '{"jobs": [{"id": "locked"}]}'
    thread.join(timeout=3)
    assert denials == [errno.EACCES], "the denial must be a real one the retry absorbed"
