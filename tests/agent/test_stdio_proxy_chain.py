"""Regression tests for #139687: stdio proxy installers must not stack.

Three installers wrap ``sys.stdout``/``sys.stderr``:

* ``agent.process_bootstrap._install_safe_stdio``   -> ``_SafeWriter``
* ``hermes_logging.redact_detached_stdio``           -> ``_RedactingStdio``
* ``agent.thread_scoped_output._ensure_installed``   -> ``_ThreadRoutingStream``

Each used to check only the *top* of the chain, so a gateway that calls them
alternately (per agent build / per background worker) appended a new layer every
cycle until ``getattr(sys.stdout, ...)`` in ``_line_buffer_piped_stdout`` raised
``RecursionError`` and every message failed.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import sys
import threading

import pytest

import hermes_logging
from agent import process_bootstrap
from agent import thread_scoped_output as tso


def _depth(stream) -> int:
    return sum(1 for _ in tso.stdio_chain(stream)) - 1


@pytest.fixture
def piped_stdio(monkeypatch):
    """Install pipe streams during the test call phase and release every sink."""
    originals = []
    streams = []
    sinks = []

    def install():
        if not originals:
            originals.append((sys.stdout, sys.stderr))
        out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        err = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        streams.extend((out, err))
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        monkeypatch.setattr(tso, "_installed", {})
        monkeypatch.setattr(tso, "_sinks", {})
        monkeypatch.setattr(tso, "_routing_states", {})
        monkeypatch.setattr(hermes_logging, "_line_buffered_stdout", None, raising=False)

        def fake_open(*_args, **_kwargs):
            sink = io.StringIO()
            sinks.append(sink)
            return sink

        monkeypatch.setattr(tso, "open", fake_open, raising=False)
        return out, err

    yield install
    if originals:
        sys.stdout, sys.stderr = originals[0]
    for stream in (*streams, *sinks):
        stream.close()


def _one_cycle() -> None:
    process_bootstrap._install_safe_stdio()
    hermes_logging.redact_detached_stdio()
    with tso.thread_scoped_silence():
        pass


def test_alternating_installers_keep_chain_bounded(piped_stdio):
    piped_stdio()
    _one_cycle()
    baseline = _depth(sys.stdout)
    for _ in range(500):
        _one_cycle()
    assert _depth(sys.stdout) == baseline
    assert _depth(sys.stderr) == baseline


def test_every_install_order_stays_bounded(piped_stdio):
    """Silence-first and redaction-first orders must not stack either."""
    def silence():
        with tso.thread_scoped_silence():
            pass

    steps = {
        "safe": process_bootstrap._install_safe_stdio,
        "redact": hermes_logging.redact_detached_stdio,
        "silence": silence,
    }
    for order in (
        ("silence", "safe", "redact"),
        ("redact", "silence", "safe"),
        ("safe", "silence", "redact"),
    ):
        piped_stdio()
        for name in order:
            steps[name]()
        baseline = _depth(sys.stdout)
        for _ in range(300):
            for name in order:
                steps[name]()
        assert _depth(sys.stdout) == baseline, order


def test_line_buffering_survives_many_agent_builds(piped_stdio):
    piped_stdio()
    for _ in range(500):
        _one_cycle()
        hermes_logging._line_buffer_piped_stdout()
    assert getattr(sys.stdout, "line_buffering", False) is True


def test_silencing_still_routes_per_thread(piped_stdio):
    out, _ = piped_stdio()
    for _ in range(50):
        _one_cycle()

    with tso.thread_scoped_silence():
        print("hidden-from-silenced-thread")
        other = threading.Thread(target=lambda: print("visible-from-other-thread"))
        other.start()
        other.join()
    print("visible-after-unsilence")
    sys.stdout.flush()

    written = out.buffer.getvalue().decode("utf-8")
    assert "hidden-from-silenced-thread" not in written
    assert "visible-from-other-thread" in written
    assert "visible-after-unsilence" in written


def test_redirect_restoring_an_older_proxy_is_adopted(piped_stdio):
    piped_stdio()
    _one_cycle()
    baseline = _depth(sys.stdout)
    for _ in range(20):
        with contextlib.redirect_stdout(io.StringIO()):
            _one_cycle()
        _one_cycle()
    assert _depth(sys.stdout) == baseline


def test_stdio_chain_stops_at_unknown_wrapper():
    class Unknown:
        def __init__(self):
            self._inner = self

    stream = Unknown()
    assert list(tso.stdio_chain(stream)) == [stream]


def test_unknown_wrapper_cannot_claim_nested_redactor(monkeypatch, piped_stdio):
    out, _ = piped_stdio()
    secret = "sk-proj-" + "A" * 30

    class UnknownWrapper:
        def __init__(self, stream):
            self._inner = hermes_logging._RedactingStdio(io.StringIO())
            self.stream = stream

        def write(self, text):
            return self.stream.write(text)

        def flush(self):
            return self.stream.flush()

    monkeypatch.setattr(sys, "stdout", UnknownWrapper(out))
    hermes_logging.redact_detached_stdio()
    sys.stdout.write(secret + "\n")
    sys.stdout.flush()
    assert secret not in out.buffer.getvalue().decode("utf-8")


def test_every_proxy_order_redacts_all_write_paths(piped_stdio):
    secret = "sk-proj-" + "B" * 30

    def silence():
        with tso.thread_scoped_silence():
            pass

    installers = (process_bootstrap._install_safe_stdio, hermes_logging.redact_detached_stdio, silence)
    for order in itertools.permutations(installers):
        out, err = piped_stdio()
        for install in order:
            install()
        print("plain-print")
        sys.stdout.write(secret + "\n")
        sys.stdout.writelines([secret, "\n", "plain-lines\n"])
        sys.stderr.writelines([secret, "\n", "plain-error\n"])
        sys.stdout.flush()
        sys.stderr.flush()
        stdout = out.buffer.getvalue().decode("utf-8")
        stderr = err.buffer.getvalue().decode("utf-8")
        assert secret not in stdout + stderr, [step.__name__ for step in order]
        assert "plain-print" in stdout and "plain-lines" in stdout
        assert "plain-error" in stderr


def test_known_proxy_cycles_and_excess_depth_are_reported(monkeypatch):
    loop = process_bootstrap._SafeWriter(None)
    object.__setattr__(loop, "_inner", loop)
    with pytest.raises(ValueError, match="cycle"):
        list(tso.stdio_chain(loop))
    with pytest.raises(ValueError, match="cycle"):
        tso.stdio_chain_find(loop, process_bootstrap._SafeWriter)
    monkeypatch.setattr(sys, "stdout", loop)
    for install in (process_bootstrap._install_safe_stdio, hermes_logging.redact_detached_stdio):
        with pytest.raises(ValueError, match="cycle"):
            install()
        assert sys.stdout is loop
    with pytest.raises(ValueError, match="cycle"):
        with tso.thread_scoped_silence():
            pass
    assert sys.stdout is loop

    deep = io.StringIO()
    for _ in range(tso._MAX_CHAIN + 1):
        deep = process_bootstrap._SafeWriter(deep)
    with pytest.raises(ValueError, match="depth"):
        list(tso.stdio_chain(deep))


def test_broken_pipes_still_tolerate_write_flush_and_writelines(monkeypatch, piped_stdio):
    piped_stdio()

    class BrokenPipe:
        def write(self, _text):
            raise OSError(5, "pipe closed")

        def flush(self):
            raise ValueError("pipe closed")

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdout", BrokenPipe())
    monkeypatch.setattr(sys, "stderr", BrokenPipe())
    _one_cycle()
    print("safe print")
    sys.stdout.write("safe write")
    sys.stdout.writelines(["safe", " lines"])
    sys.stdout.flush()
    sys.stderr.writelines(["safe", " errors"])
    sys.stderr.flush()


def test_concurrent_installers_keep_both_chains_bounded(piped_stdio):
    from concurrent.futures import ThreadPoolExecutor

    out, err = piped_stdio()
    start = threading.Barrier(6)

    def install_many(index):
        start.wait(timeout=10)
        for _ in range(100):
            if index % 3 == 0:
                process_bootstrap._install_safe_stdio()
            elif index % 3 == 1:
                hermes_logging.redact_detached_stdio()
            else:
                with tso.thread_scoped_silence():
                    pass

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(install_many, range(6)))
    assert _depth(sys.stdout) <= 3
    assert _depth(sys.stderr) <= 3
    sys.stdout.write("still visible\n")
    sys.stderr.write("still visible\n")
    sys.stdout.flush()
    sys.stderr.flush()
    assert b"still visible" in out.buffer.getvalue()
    assert b"still visible" in err.buffer.getvalue()


def test_silence_path_does_not_import_bootstrap(tmp_path):
    import os
    import subprocess

    code = """
import io
import sys
from agent.thread_scoped_output import thread_scoped_silence
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
with thread_scoped_silence():
    pass
assert 'agent.process_bootstrap' not in sys.modules
assert 'hermes_bootstrap' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env={**os.environ, "HERMES_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unexpected_write_errors_are_not_silently_discarded():
    class FailingStream:
        def write(self, _text):
            raise RuntimeError("write failed")

        def flush(self):
            raise RuntimeError("flush failed")

        def writelines(self, _lines):
            raise RuntimeError("writelines failed")

    proxy = tso._ThreadRoutingStream(FailingStream(), tso._RoutingState(io.StringIO()))
    for operation in (lambda: proxy.write("text"), proxy.flush, lambda: proxy.writelines(["text"])):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
