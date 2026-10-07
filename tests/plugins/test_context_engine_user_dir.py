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


def test_failed_import_is_reported_with_original_exception(tmp_path: Path, monkeypatch):
    name = "ctx_import_failure"
    engine_dir = tmp_path / "plugins" / name
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(
        "from agent.context_engine import ContextEngine\n"
        "raise RuntimeError('import boom')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from plugins.context_engine import context_engine_load_errors, load_context_engine

    assert load_context_engine(name) is None
    assert context_engine_load_errors()[name] == "import boom"


def test_register_and_constructor_failures_are_reported(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.context_engine import context_engine_load_errors, load_context_engine

    register_name = "ctx_register_failure"
    register_dir = tmp_path / "plugins" / register_name
    register_dir.mkdir(parents=True)
    (register_dir / "__init__.py").write_text(
        "from agent.context_engine import ContextEngine\n"
        "def register(ctx):\n    raise ValueError('register boom')\n",
        encoding="utf-8",
    )
    assert load_context_engine(register_name) is None
    assert context_engine_load_errors()[register_name] == "register boom"

    constructor_name = "ctx_constructor_failure"
    constructor_dir = tmp_path / "plugins" / constructor_name
    constructor_dir.mkdir(parents=True)
    (constructor_dir / "__init__.py").write_text(
        "from agent.context_engine import ContextEngine\n"
        "class Broken(ContextEngine):\n"
        "    @property\n    def name(self): return 'broken'\n"
        "    def __init__(self): raise RuntimeError('constructor boom')\n"
        "    def update_from_response(self, usage): pass\n"
        "    def should_compress(self, prompt_tokens=None): return False\n"
        "    def compress(self, messages, current_tokens=None): return messages\n",
        encoding="utf-8",
    )
    assert load_context_engine(constructor_name) is None
    assert context_engine_load_errors()[constructor_name] == "constructor boom"


def test_load_errors_are_scoped_to_home_and_cleared_when_not_found(tmp_path: Path, monkeypatch):
    name = "ctx_scoped_failure"
    first_home = tmp_path / "first"
    engine_dir = first_home / "plugins" / name
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(
        "from agent.context_engine import ContextEngine\n"
        "raise RuntimeError('first home boom')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(first_home))
    from plugins.context_engine import context_engine_load_errors, load_context_engine

    assert load_context_engine(name) is None
    assert context_engine_load_errors()[name] == "first home boom"

    second_home = tmp_path / "second"
    second_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(second_home))
    assert context_engine_load_errors() == {}
    assert load_context_engine(name) is None
    assert context_engine_load_errors() == {}

    monkeypatch.setenv("HERMES_HOME", str(first_home))
    assert context_engine_load_errors()[name] == "first home boom"
    (engine_dir / "__init__.py").unlink()
    assert load_context_engine(name) is None
    assert context_engine_load_errors() == {}
