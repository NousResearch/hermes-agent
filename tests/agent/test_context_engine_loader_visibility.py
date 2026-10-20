"""Loader visibility for context-engine construction failures.

Regression test: when a context engine's ``register(ctx)`` raises during
construction, the loader must log at WARNING with the traceback — not
``debug``. A failed construction silently downgrades every session to the
built-in compressor, and this log line is the only breadcrumb pointing at
the real cause (e.g. a SQLite schema error in the engine's storage init).
At debug level the exception is invisible in production and the degradation
presents as an unexplained behavior change.
"""

import logging

import plugins.context_engine as ce_loader


def _write_failing_engine(plugin_dir, name="boomengine"):
    engine_dir = plugin_dir / name
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    raise RuntimeError('storage init exploded: no such column: search_content')\n",
        encoding="utf-8",
    )
    return engine_dir


def test_register_failure_logs_warning_with_traceback(tmp_path, monkeypatch, caplog):
    """A register() crash must surface at WARNING with the real exception."""
    plugin_dir = tmp_path / "context_engine_plugins"
    _write_failing_engine(plugin_dir)
    monkeypatch.setattr(ce_loader, "_CONTEXT_ENGINE_PLUGINS_DIR", plugin_dir)

    with caplog.at_level(logging.DEBUG, logger="plugins.context_engine"):
        engine = ce_loader.load_context_engine("boomengine")

    assert engine is None
    warning_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "register() failed" in r.getMessage()
    ]
    assert warning_records, (
        "register() construction failure must be logged at WARNING+ "
        f"(records seen: {[(r.levelname, r.getMessage()) for r in caplog.records]})"
    )
    rec = warning_records[0]
    # The real exception message must be present so operators can act on it.
    assert "storage init exploded" in rec.getMessage()
    # And the traceback must ride along (exc_info), not just the str(e).
    assert rec.exc_info is not None


def test_healthy_engine_still_loads_without_warning(tmp_path, monkeypatch, caplog):
    """A working register() path must not emit the failure warning."""
    plugin_dir = tmp_path / "context_engine_plugins"
    engine_dir = plugin_dir / "okengine"
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(
        "from agent.context_engine import ContextEngine\n"
        "\n"
        "class _OkEngine(ContextEngine):\n"
        "    @property\n"
        "    def name(self):\n"
        "        return 'okengine'\n"
        "    def update_from_response(self, usage):\n"
        "        pass\n"
        "    def should_compress(self, prompt_tokens=None):\n"
        "        return False\n"
        "    def compress(self, messages, current_tokens=None):\n"
        "        return messages\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_context_engine(_OkEngine())\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ce_loader, "_CONTEXT_ENGINE_PLUGINS_DIR", plugin_dir)

    with caplog.at_level(logging.DEBUG, logger="plugins.context_engine"):
        engine = ce_loader.load_context_engine("okengine")

    assert engine is not None
    assert not [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "register() failed" in r.getMessage()
    ]
