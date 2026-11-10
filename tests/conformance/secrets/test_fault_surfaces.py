"""A fault at each seam of today's auth record write publishes nothing an observer can keep.

The conformance suite breaks a write at the failure points the fixture's hook names; these
tests go one level deeper, into the seams of today's own writer behind the local backend:
the merge into the loaded store, the write note, the save, the JSON write, the serialization
of a record that cannot be serialized, a disk that fills mid-write, the atomic rename, the
fsync, the chmod, and the directory fsync — each failing while the caller's record is bound
in the write's frames. Whatever observer keeps what the write leaves behind — the error the
caller reads, an exception a log handler keeps with its live traceback, a warning — finds
no trace of the record, raw or escaped, in any frame it can render.
"""

import errno
import logging
import warnings

import pytest

from hermes_conformance.secrets.suite import _plaintext_forms, assert_no_frame_holds
from hermes_secret_port import SecretWriteFailed
from tests.conformance.secrets.test_local import local_backend

#: One record's plaintext, in the forms a writer or reporter might hold it in.
MARKER = 'zFault\nSurface "Record" 7c31'


def _planted_here(exc: BaseException) -> BaseException:
    """*exc* with this module's own frames trimmed from its traceback: these tests plant the
    record themselves, and the suite's chain walker skips only its own frames."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
        tb = tb.tb_next
    exc.__traceback__ = tb
    return exc


class _KeepingHandler(logging.Handler):
    """An observer's logger: keeps every LogRecord that carries ``exc_info`` — the live
    exception, frames and all — for later rendering."""

    def __init__(self):
        super().__init__()
        self.exceptions: list[BaseException] = []

    def emit(self, record: logging.LogRecord) -> None:
        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None:
            self.exceptions.append(exc)


#: Each seam of today's auth record write, and how a fault there fails: raising where the
#: seam is called, or (for the full-disk seam) failing the file object's write in flight.
#: The final chmod seam fails after the file is written, a failure today's writer swallows.
SEAMS = ("merge", "note", "save", "json-write", "serialization", "full-disk",
         "rename", "fsync", "chmod", "final-chmod", "directory-fsync")


def _break_the_seam(monkeypatch, seam, reached):
    """Make today's auth write fail at *seam*, naming the record in the failure it raises
    (the failure is today's code raising it, so nothing vouches its text)."""
    import utils
    from hermes_cli import auth

    def fail(*args, **kwargs):
        reached.append(True)
        raise OSError(errno.EIO, "the writer saw " + MARKER, MARKER)

    if seam == "serialization":
        return {"invalid": object()}  # today's JSON dumper cannot serialize it
    if seam == "full-disk":
        original_fdopen = utils.os.fdopen

        class _FullDisk:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def write(self, text):
                reached.append(True)
                raise OSError(errno.ENOSPC, "the disk filled mid-write")

        monkeypatch.setattr(utils.os, "fdopen", lambda *a, **kw: _FullDisk(original_fdopen(*a, **kw)))
        return {}
    if seam == "final-chmod":
        original_chmod = utils.os.chmod

        def chmod(path, mode, *args, **kwargs):
            if str(path).endswith("auth.json"):
                reached.append(True)
                raise OSError(errno.EIO, "the writer saw " + MARKER, MARKER)
            return original_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(utils.os, "chmod", chmod)
        return {}
    patches = {
        "merge": (auth, "_store_provider_state"),
        "note": (auth, "note_store_write"),
        "save": (auth, "_save_auth_store"),
        "json-write": (auth, "atomic_json_write"),
        "rename": (utils, "atomic_replace"),
        "fsync": (utils.os, "fsync"),
        "chmod": (utils.os, "fchmod"),
        "directory-fsync": (utils, "fsync_directory"),
    }
    monkeypatch.setattr(*patches[seam], fail)
    return {}


@pytest.mark.parametrize("seam", SEAMS)
def test_a_fault_at_each_seam_of_the_auth_write_publishes_nothing(tmp_path, monkeypatch, seam):
    """A fault at any seam of today's auth record write — raised while the caller's record is
    bound in the write's own frames, and naming the record in the failing text, the worst a
    real writer could do — leaves the record nowhere an observer can keep it: the fresh error
    the caller reads carries none of its text, no frame the error's whole chain passes
    through holds it, no exception a log handler keeps with its live traceback unwinds a
    record-holding frame, and no warning carries it on the way out. The seam after the file
    is written today's writer swallows: the write completes there, and still nothing is
    published."""
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    record = dict(_break_the_seam(monkeypatch, seam, reached := []), token=MARKER)
    kept = _KeepingHandler()
    logging.getLogger().addHandler(kept)
    observed_warnings: list[str] = []
    original_warn = warnings.warn

    def warned(*args, **kwargs):
        observed_warnings.append(" ".join(str(arg) for arg in args))
        return original_warn(*args, **kwargs)

    monkeypatch.setattr(warnings, "warn", warned)
    raised = None
    try:
        with backend.bind("a"):
            if seam == "final-chmod":
                store.set_auth("probe", record)  # today's writer swallows this fault
            else:
                with pytest.raises(SecretWriteFailed) as failed:
                    store.set_auth("probe", record)
                raised = failed.value
    finally:
        logging.getLogger().removeHandler(kept)
    # This test planted the record: its own frame must not read as a leak to the walker.
    del record
    if seam != "serialization":
        assert reached, f"the {seam!r} seam was never reached"
    for text in observed_warnings:
        for form in _plaintext_forms(MARKER):
            assert form not in text, "a warning carried the record's plaintext"
    for kept_exc in kept.exceptions:
        assert_no_frame_holds(_planted_here(kept_exc), MARKER)
    if raised is not None:
        for form in _plaintext_forms(MARKER):
            assert form not in str(raised) + repr(raised) + repr(raised.args)
        assert_no_frame_holds(_planted_here(raised), MARKER)


def test_the_global_auth_stores_failed_load_is_logged_clean_of_the_record(tmp_path, monkeypatch):
    """Today's global auth store loader logs a load failure with its live exception —
    ``exc_info`` and all — and a record's write touches that path when the profile merges into
    the global root's per-provider state. The kept log record is the real artifact the port's
    structure answers: the load runs before the record is bound in any frame below the write,
    so the kept exception's traceback unwinds no record-holding frame, however it is
    rendered."""
    from hermes_cli import auth
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    root = tmp_path / "global"
    root.mkdir()
    (root / "auth.json").mkdir()  # today's loader cannot read it, and logs that with exc_info
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root / "auth.json")
    monkeypatch.setattr(auth, "_global_auth_store_cache", None)
    kept = _KeepingHandler()
    logging.getLogger().addHandler(kept)
    try:
        with backend.bind("a"):
            store.set_auth("nous", {"token": MARKER})
    finally:
        logging.getLogger().removeHandler(kept)
    assert kept.exceptions, "today's loader no longer logs the failed global load with its exception"
    for kept_exc in kept.exceptions:
        assert_no_frame_holds(_planted_here(kept_exc), MARKER)
        for form in _plaintext_forms(MARKER):
            assert form not in str(kept_exc) + repr(kept_exc) + repr(kept_exc.args)
