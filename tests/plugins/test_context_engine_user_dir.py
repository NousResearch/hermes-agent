"""``context.engine: <name>`` names the active engine, so an engine dropped into
``$HERMES_HOME/plugins/<name>/`` loads without a ``plugins.enabled`` entry and never trips the
"Context engine 'X' not found — falling back to built-in compressor" warning (#61839)."""

import logging
import sys
from pathlib import Path
from textwrap import dedent

_ENGINE_SRC = dedent('''
    from agent.context_engine import ContextEngine

    HOME_TAG = "{tag}"

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
    (engine_dir / "__init__.py").write_text(_ENGINE_SRC.format(tag="ctx_demo"))
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


def test_user_engine_module_is_scoped_to_the_profile_home(tmp_path: Path, monkeypatch):
    """One process, two profiles, same engine name: each profile must run its own module (#134347)."""
    from plugins.context_engine import load_context_engine

    for tag in ("alpha", "beta"):
        engine_dir = tmp_path / tag / "plugins" / "ctx_demo"
        engine_dir.mkdir(parents=True)
        (engine_dir / "__init__.py").write_text(_ENGINE_SRC.format(tag=tag))

    engines = []
    for tag in ("alpha", "beta"):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / tag))
        engine = load_context_engine("ctx_demo")
        assert engine is not None
        engines.append(engine)

    modules = [sys.modules[type(engine).__module__] for engine in engines]
    assert modules[0] is not modules[1]
    assert modules[0].HOME_TAG == "alpha"
    assert modules[1].HOME_TAG == "beta"  # "alpha" before the home-scoped cache key
    assert modules[0].__file__ == str(
        tmp_path / "alpha" / "plugins" / "ctx_demo" / "__init__.py"
    )
    assert modules[1].__file__ == str(
        tmp_path / "beta" / "plugins" / "ctx_demo" / "__init__.py"
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "alpha"))
    again = load_context_engine("ctx_demo")
    assert (
        sys.modules[type(again).__module__] is modules[0]
    )  # same home still reuses the module
