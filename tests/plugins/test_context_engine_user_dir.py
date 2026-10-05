"""``context.engine: <name>`` names the active engine, so an engine dropped into
``$HERMES_HOME/plugins/<name>/`` loads without a ``plugins.enabled`` entry and never trips the
"Context engine 'X' not found — falling back to built-in compressor" warning (#61839)."""

import logging
from pathlib import Path
from textwrap import dedent

_ENGINE_SRC = dedent('''
    from agent.context_engine import ContextEngine

    class Demo(ContextEngine):
        @property
        def name(self):
            return "ctx_demo"
        def update_from_response(self, usage):
            pass
        def should_compress(self, prompt_tokens=None):
            return False
        def compress(self, messages, current_tokens=None):
            return messages

    def register(ctx):
        ctx.register_context_engine(Demo())
''')


def test_user_installed_engine_is_selected_by_name(tmp_path: Path, monkeypatch, caplog):
    engine_dir = tmp_path / "plugins" / "ctx_demo"
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(_ENGINE_SRC)
    (tmp_path / "plugins" / "notes").mkdir()  # unrelated user plugin: never treated as an engine
    (tmp_path / "plugins" / "notes" / "__init__.py").write_text("def register(ctx): pass\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.agent_init import _select_context_engine
    from plugins.context_engine import discover_context_engines, load_context_engine

    assert load_context_engine("notes") is None
    assert "ctx_demo" in {name for name, _, _ in discover_context_engines()}
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        engine = _select_context_engine({"context": {"engine": "ctx_demo"}})
    assert engine is not None and engine.name == "ctx_demo"
    assert "not found" not in caplog.text


def test_missing_engine_produces_a_user_notice_and_a_working_engine_does_not(tmp_path: Path, monkeypatch):
    """The fallback to the built-in compressor must be visible, not only a log line."""
    engine_dir = tmp_path / "plugins" / "ctx_demo"
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(_ENGINE_SRC)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.agent_init import _context_engine_fallback_notice, _select_context_engine

    for cfg in ({"context": {"engine": "compressor"}}, {}):
        assert _context_engine_fallback_notice(cfg, _select_context_engine(cfg)) is None
    ok = {"context": {"engine": "ctx_demo"}}
    assert _context_engine_fallback_notice(ok, _select_context_engine(ok)) is None
    # A configured engine that did not load (_select_context_engine returned None).
    notice = _context_engine_fallback_notice({"context": {"engine": "not_installed"}}, None)
    assert notice and "not_installed" in notice


def test_fallback_notice_survives_a_later_compression_warning_write():
    """Compression code reassigns ``_compression_warning`` before turn 1; the engine notice must
    not ride on that slot or it is dropped and the fallback goes silent again."""
    from types import SimpleNamespace

    from agent.agent_init import _emit_compression_summary
    from agent.status_output import StatusOutputMixin

    class Agent(StatusOutputMixin):
        platform = "telegram"
        quiet_mode = True
        notice_callback = status_callback = None
        _compression_threshold_autoraised = None
        _context_engine_fallback_notice = "Context engine 'cmi' failed to load"

    agent = Agent()
    _emit_compression_summary(agent, SimpleNamespace(enabled=True, autoraise_notice_enabled=True))
    agent._compression_warning = "unrelated compression warning"  # e.g. conversation_compression
    agent._compression_warning = None
    notices = []
    agent.notice_callback = notices.append  # the gateway wires this per turn, after __init__
    agent._replay_startup_warnings()
    assert [n.text for n in notices] == ["Context engine 'cmi' failed to load"]
