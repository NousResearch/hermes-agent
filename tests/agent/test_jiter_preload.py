from __future__ import annotations

import importlib
import logging
import sys

from agent import jiter_preload


def test_preload_jiter_native_extension_loads_sdk_parser_dependency():
    assert jiter_preload.preload_jiter_native_extension() is True
    assert "jiter.jiter" in sys.modules


def test_preload_jiter_native_extension_is_best_effort(monkeypatch):
    monkeypatch.setattr(jiter_preload, "_JITER_PRELOADED", False)

    def _raise_missing(name: str):
        assert name == "jiter.jiter"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _raise_missing)

    assert jiter_preload.preload_jiter_native_extension() is False
    assert jiter_preload._JITER_PRELOADED is False
    assert isinstance(jiter_preload._JITER_PRELOAD_ERROR, ModuleNotFoundError)


def test_preload_jiter_native_extension_logs_debug_on_best_effort_failure(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(jiter_preload, "_JITER_PRELOADED", False)

    def _raise_missing(name: str):
        assert name == "jiter.jiter"
        raise ModuleNotFoundError("jiter fixture missing")

    monkeypatch.setattr(importlib, "import_module", _raise_missing)

    with caplog.at_level(logging.DEBUG, logger="agent.jiter_preload"):
        assert jiter_preload.preload_jiter_native_extension() is False

    records = [
        record
        for record in caplog.records
        if record.name == "agent.jiter_preload"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert "Failed to preload optional jiter native extension" in records[0].getMessage()
    assert "jiter fixture missing" in records[0].getMessage()
    assert jiter_preload._JITER_PRELOADED is False
    assert isinstance(jiter_preload._JITER_PRELOAD_ERROR, ModuleNotFoundError)
