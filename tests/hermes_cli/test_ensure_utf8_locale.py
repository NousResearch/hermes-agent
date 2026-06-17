"""Regression tests for hermes_cli._ensure_utf8().

Covers the crash class where the setup wizard (and other banner-printing
commands) emit box-drawing characters and the ⚕ glyph, which raise
UnicodeEncodeError when stdout/stderr are bound to a non-UTF-8 codec.

Historically the repair was gated on ``sys.platform == "win32"`` and only
caught the Windows cp1252 case. Linux hosts with a latin-1 / C / POSIX locale
(common on minimal Debian installs and Raspberry Pi) hit the identical crash
in ``hermes setup`` because the repair returned early. See the Raspberry Pi
report: latin-1 locale → UnicodeEncodeError before the wizard could start.
"""

import io
import os
import sys

import hermes_cli


# The exact glyphs the setup wizard / banners print (setup.py ~line 2962+).
_BANNER = "┌─────┐\n│ ⚕ Hermes │\n└─────┘"


class _FakeStream:
    """Minimal text stream backed by an in-memory byte buffer with a codec.

    Mirrors how CPython binds sys.stdout to the locale encoding: writes that
    can't be encoded raise UnicodeEncodeError, just like a real latin-1 TTY.
    """

    def __init__(self, encoding, *, supports_reconfigure=True):
        self.encoding = encoding
        self._supports_reconfigure = supports_reconfigure
        self.errors = "strict"
        self._buf = io.BytesIO()

    def write(self, s):
        self._buf.write(s.encode(self.encoding, self.errors))
        return len(s)

    def flush(self):
        pass

    def reconfigure(self, *, encoding=None, errors=None):
        if not self._supports_reconfigure:
            raise AttributeError("reconfigure")
        if encoding is not None:
            self.encoding = encoding
        if errors is not None:
            self.errors = errors

    def getvalue(self):
        return self._buf.getvalue()


def _run_with_streams(monkeypatch, out, err):
    monkeypatch.setattr(sys, "stdout", out, raising=False)
    monkeypatch.setattr(sys, "stderr", err, raising=False)
    hermes_cli._ensure_utf8()


def test_latin1_stdout_is_repaired_to_utf8(monkeypatch):
    """A latin-1 stdout (the Raspberry Pi case) becomes UTF-8 capable."""
    out = _FakeStream("latin-1")
    err = _FakeStream("latin-1")

    # Sanity: before the fix, the banner cannot be encoded.
    try:
        out.write(_BANNER)
        pre_fix_crashes = False
    except UnicodeEncodeError:
        pre_fix_crashes = True
    assert pre_fix_crashes, "fixture should reproduce the original crash"

    out = _FakeStream("latin-1")
    err = _FakeStream("latin-1")
    _run_with_streams(monkeypatch, out, err)

    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
    assert sys.stderr.encoding.lower().replace("-", "") == "utf8"
    # The banner now encodes without raising.
    sys.stdout.write(_BANNER)
    assert "⚕".encode("utf-8") in sys.stdout.getvalue()


def test_ascii_posix_locale_is_repaired(monkeypatch):
    """C/POSIX locale resolves to ascii stdout — also must be repaired."""
    out = _FakeStream("ascii")
    err = _FakeStream("ascii")
    _run_with_streams(monkeypatch, out, err)
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
    sys.stdout.write(_BANNER)  # no raise


def test_utf8_stream_left_untouched(monkeypatch):
    """Already-UTF-8 streams are a no-op: object identity preserved AND the
    process environment is left untouched (no PYTHONUTF8/PYTHONIOENCODING
    burned in on a healthy UTF-8 host)."""
    out = _FakeStream("utf-8")
    err = _FakeStream("utf-8")
    sentinel_out, sentinel_err = out, err
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    _run_with_streams(monkeypatch, out, err)
    assert sys.stdout is sentinel_out
    assert sys.stderr is sentinel_err
    # Healthy UTF-8 host: no environment mutation (minimal footprint).
    assert "PYTHONUTF8" not in os.environ
    assert "PYTHONIOENCODING" not in os.environ


def test_repair_sets_child_process_env(monkeypatch):
    """When a real repair happens, child-process UTF-8 hints are set."""
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    _run_with_streams(monkeypatch, _FakeStream("latin-1"), _FakeStream("latin-1"))
    assert os.environ.get("PYTHONUTF8") == "1"
    assert os.environ.get("PYTHONIOENCODING") == "utf-8"


def test_repair_does_not_override_explicit_env(monkeypatch):
    """A user's explicit PYTHONIOENCODING is respected (setdefault, not set)."""
    monkeypatch.setenv("PYTHONIOENCODING", "utf-16")
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    _run_with_streams(monkeypatch, _FakeStream("latin-1"), _FakeStream("latin-1"))
    assert os.environ["PYTHONIOENCODING"] == "utf-16"


def test_fallback_when_reconfigure_unavailable(monkeypatch, tmp_path):
    """Streams without reconfigure() fall back to reopening the fd as UTF-8."""
    real_path = tmp_path / "out.txt"
    fh = open(real_path, "w", encoding="latin-1")

    class _NoReconfigure:
        """latin-1 stream exposing a real fileno() but no reconfigure()."""

        encoding = "latin-1"

        def fileno(self):
            return fh.fileno()

    stream = _NoReconfigure()
    monkeypatch.setattr(sys, "stdout", stream, raising=False)
    monkeypatch.setattr(sys, "stderr", stream, raising=False)
    hermes_cli._ensure_utf8()

    # Replaced with a new UTF-8 stream object (not reconfigured in place).
    assert sys.stdout is not stream
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
    sys.stdout.write(_BANNER)
    sys.stdout.flush()
    fh.close()
    assert "⚕".encode("utf-8") in real_path.read_bytes()


def test_broken_stream_does_not_raise(monkeypatch):
    """A stream whose repair raises must be swallowed, never crash import."""

    class _Hostile:
        encoding = "latin-1"

        def reconfigure(self, *a, **k):
            raise OSError("nope")

        def fileno(self):
            raise OSError("no fd")

    monkeypatch.setattr(sys, "stdout", _Hostile(), raising=False)
    monkeypatch.setattr(sys, "stderr", _Hostile(), raising=False)
    # Must not propagate.
    hermes_cli._ensure_utf8()


def test_none_streams_do_not_raise(monkeypatch):
    """pythonw / detached streams (sys.stdout is None) must be tolerated."""
    monkeypatch.setattr(sys, "stdout", None, raising=False)
    monkeypatch.setattr(sys, "stderr", None, raising=False)
    hermes_cli._ensure_utf8()
