"""Regression: per-turn safety and background silence must not grow stdio chains."""

import contextlib
import io
import itertools
import logging
import sys
import threading

import pytest

import agent.thread_scoped_output as routing
import hermes_logging
from agent.process_bootstrap import _install_safe_stdio


@pytest.fixture
def streams(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(routing, "_installed", {})
    monkeypatch.setattr(routing, "_sinks", {"stdout": io.StringIO(), "stderr": io.StringIO()})
    monkeypatch.setattr(routing, "_routing_states", {})
    monkeypatch.setattr(hermes_logging, "_line_buffered_stdout", None)
    return out, err


@pytest.mark.parametrize("redact", [False, True])
def test_repeated_agent_and_background_worker_startup(streams, redact):
    """The real installers used by agent startup and background reviews compose."""
    sys.stdout, sys.stderr = streams
    for _ in range(600):
        _install_safe_stdio()
        hermes_logging._line_buffer_piped_stdout()
        if redact:
            hermes_logging.redact_detached_stdio()
        with routing.thread_scoped_silence():
            sys.stdout.write("hidden-out\n")
            sys.stderr.write("hidden-err\n")
    sys.stdout.write("visible-out\n")
    sys.stderr.write("visible-err\n")
    assert streams[0].getvalue() == "visible-out\n"
    assert streams[1].getvalue() == "visible-err\n"


def _install_routing():
    with routing.thread_scoped_silence():
        pass


@pytest.mark.parametrize("installers", list(itertools.permutations([
    _install_safe_stdio, _install_routing, hermes_logging.redact_detached_stdio,
])))
def test_all_install_orders_remain_bounded_and_redact(streams, installers):
    sys.stdout, sys.stderr = streams
    for install in installers:
        install()
    installed = sys.stdout, sys.stderr
    for _ in range(20):
        for install in installers:
            install()
    assert (sys.stdout, sys.stderr) == installed
    secret = "sk-" + "a" * 40
    for stream in installed:
        stream.write(secret + "\n")
        stream.flush()
    for stream in streams:
        assert secret not in stream.getvalue()
        assert stream.getvalue().strip()


def test_outer_layers_preserved_during_concurrent_silence_and_redirect(streams):
    sys.stdout, sys.stderr = streams
    _install_routing()
    _install_safe_stdio()
    hermes_logging.redact_detached_stdio()
    installed = sys.stdout, sys.stderr
    entered, release = threading.Event(), threading.Event()
    errors = []

    def worker():
        try:
            with routing.thread_scoped_silence():
                entered.set()
                assert release.wait(10)
                sys.stdout.write("hidden\n")
                sys.stderr.write("hidden\n")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(10)
        with contextlib.redirect_stdout(io.StringIO()) as redirected:
            _install_routing()
            sys.stdout.write("loud-redirect\n")
        assert (sys.stdout, sys.stderr) == installed
        sys.stdout.write("loud-main\n")
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert not errors
    assert redirected.getvalue() == "loud-redirect\n"
    assert streams[0].getvalue() == "loud-main\n"
    assert streams[1].getvalue() == ""


def test_metadata_errors_are_not_hidden(streams):
    class FailingStream(io.StringIO):
        @property
        def line_buffering(self):
            raise RuntimeError("metadata failed")

    sys.stdout = FailingStream()
    _install_routing()
    _install_safe_stdio()
    hermes_logging.redact_detached_stdio()
    with pytest.raises(RuntimeError, match="metadata failed"):
        hermes_logging._line_buffer_piped_stdout()


def test_gateway_logging_startup_with_real_piped_stream(streams, tmp_path):
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    secret = "sk-" + "b" * 40
    log_dir = tmp_path / "logs"
    try:
        with (tmp_path / "stdout.log").open("w", encoding="utf-8") as output:
            sys.stdout, sys.stderr = output, streams[1]
            hermes_logging.redact_detached_stdio()
            for _ in range(600):
                _install_safe_stdio()
                log_dir = hermes_logging.setup_logging(hermes_home=tmp_path, mode="gateway")
                with routing.thread_scoped_silence():
                    sys.stdout.write("hidden\n")
            assert output.line_buffering is True
            sys.stdout.write("visible " + secret + "\n")
            logging.getLogger("gateway.stdio-regression").warning("logged %s", secret)
            hermes_logging.flush_log_queue()
            assert log_dir == tmp_path / "logs"
        written = (tmp_path / "stdout.log").read_text()
        assert "visible" in written
        assert "hidden" not in written
        assert secret not in written
        logged = (log_dir / "gateway.log").read_text()
        assert "logged" in logged
        assert secret not in logged
    finally:
        hermes_logging._reset_queued_handlers()
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()


def test_wrapper_search_is_iterative_and_rejects_real_cycles():
    from agent.process_bootstrap import _SafeWriter
    from agent.stdio_wrappers import find_stdio_wrapper

    stream = io.StringIO()
    for _ in range(2000):
        stream = _SafeWriter(stream)
    assert find_stdio_wrapper(stream, routing._ThreadRoutingStream) is None
    stream._inner = stream
    with pytest.raises(ValueError, match="Cycle in Hermes stdio wrappers"):
        find_stdio_wrapper(stream, routing._ThreadRoutingStream)
