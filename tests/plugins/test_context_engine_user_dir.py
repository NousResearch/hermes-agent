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


def test_each_profile_loads_its_own_copy_of_a_user_engine(tmp_path: Path, monkeypatch):
    """One process serving two profiles (multiplex / Desktop backend): profile B's
    ``plugins/<name>/`` must not resolve to profile A's already-imported module (A -> B -> A)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.context_engine import load_context_engine

    homes = {}
    for tag in ("A", "B"):
        engine_dir = tmp_path / tag / "plugins" / "ctx_per_home"
        engine_dir.mkdir(parents=True)
        (engine_dir / "__init__.py").write_text(_ENGINE_SRC.replace("class Demo(ContextEngine):",
                                                                   f"class Demo(ContextEngine):\n    TAG = {tag!r}"))
        homes[tag] = tmp_path / tag
    monkeypatch.setenv("HERMES_HOME", str(homes["A"]))

    loaded = []
    for tag in ("A", "B", "A"):
        token = set_hermes_home_override(str(homes[tag]))
        try:
            loaded.append(type(load_context_engine("ctx_per_home")).TAG)
        finally:
            reset_hermes_home_override(token)
    assert loaded == ["A", "B", "A"]
