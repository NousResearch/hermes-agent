from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path

import hermes_cli.kanban_db as kb
import hermes_cli.kanban_db_pg as kb_pg
import hermes_cli.kanban_db_sqlite as kb_sqlite
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLERS = [
    REPO_ROOT / "hermes_cli/kanban.py",
    REPO_ROOT / "gateway/run.py",
    REPO_ROOT / "tools/kanban_tools.py",
]


def _required_surface() -> list[str]:
    names: set[str] = set()
    for path in CALLERS:
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "hermes_cli":
                for alias in node.names:
                    if alias.name == "kanban_db":
                        aliases.add(alias.asname or alias.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in aliases:
                    names.add(node.func.attr)
    return sorted(names)


@pytest.fixture(autouse=True)
def _restore_default_selector_after_test():
    yield
    import hermes_cli.config as config_mod

    original = config_mod.load_config
    try:
        config_mod.load_config = lambda: {}
        os.environ.pop("HERMES_KANBAN_BACKEND", None)
        importlib.reload(kb)
    finally:
        config_mod.load_config = original


def _reload_selector(monkeypatch, config: dict | None = None):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {} if config is None else config,
    )
    return importlib.reload(kb)


def _normalized_runtime_signature(func) -> str:
    sig = inspect.signature(func)
    parts: list[str] = []
    for param in sig.parameters.values():
        text = param.name
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            parts.append(text)
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{text}")
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{text}")
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY and "*" not in parts and not any(p.startswith("*") for p in parts):
            parts.append("*")
        if param.default is not inspect._empty:
            text = f"{text}={param.default!r}"
        parts.append(text)
    return f"{func.__name__}({', '.join(parts)})"


def test_selector_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_BACKEND", raising=False)
    selected = _reload_selector(monkeypatch, config={})

    assert selected.get_selected_backend_name() == "sqlite"
    assert selected.get_selected_backend_module() == "hermes_cli.kanban_db_sqlite"
    assert selected.connect.__code__.co_filename.endswith("kanban_db_sqlite.py")



def test_selector_honors_config_backend(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_BACKEND", raising=False)
    selected = _reload_selector(monkeypatch, config={"kanban": {"backend": "postgres"}})

    assert selected.get_selected_backend_name() == "postgres"
    assert selected.get_selected_backend_module() == "hermes_cli.kanban_db_pg"
    assert selected.connect.__code__.co_filename.endswith("kanban_db_pg.py")



def test_selector_env_override_beats_config(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BACKEND", "postgres")
    selected = _reload_selector(monkeypatch, config={"kanban": {"backend": "sqlite"}})

    assert selected.get_selected_backend_name() == "postgres"
    assert selected.connect.__code__.co_filename.endswith("kanban_db_pg.py")



def test_wrapper_exports_required_surface_with_sqlite_signatures(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_BACKEND", raising=False)
    selected = _reload_selector(monkeypatch, config={})

    required = _required_surface()
    missing = [name for name in required if not hasattr(selected, name)]
    assert not missing

    mismatched = []
    for name in required:
        wrapper_sig = _normalized_runtime_signature(getattr(selected, name))
        sqlite_sig = _normalized_runtime_signature(getattr(kb_sqlite, name))
        if wrapper_sig != sqlite_sig:
            mismatched.append((name, wrapper_sig, sqlite_sig))
    assert not mismatched

    from hermes_cli.kanban_db import connect, resolve_workspace

    assert connect is selected.connect
    assert resolve_workspace is selected.resolve_workspace
