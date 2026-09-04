"""Regression tests for non-fatal gateway traceback signals."""

from __future__ import annotations

from gateway import faulthandler_setup


def test_traceback_signal_does_not_chain_to_fatal_default(monkeypatch, tmp_path):
    observed = {}

    def register(signum, **kwargs):
        observed["signum"] = signum
        observed.update(kwargs)

    monkeypatch.setattr(faulthandler_setup.faulthandler, "register", register)

    with (tmp_path / "tracebacks.log").open("a") as output:
        faulthandler_setup.register_traceback_signal(12, file=output)

    assert observed["signum"] == 12
    assert observed["all_threads"] is True
    assert observed["chain"] is False
