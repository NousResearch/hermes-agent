from __future__ import annotations

import ast
from pathlib import Path

from scripts.code_health import measure, py_rules

REPO = Path(__file__).resolve().parents[2]


def _hits(rule: str, source: str) -> list[int]:
    tree = py_rules.canonical_tree(ast.parse(source))
    return sorted(set(py_rules.CHECKERS[rule](tree, py_rules.Ctx())))


def _pattern_hits(rule: str, source: str) -> list[int]:
    patterns = {rule_id: pattern for rule_id, pattern, _ in measure._regex_rules(REPO)}
    pattern = patterns[rule]
    return sorted({line for line, operation in measure._operations(ast.parse(source))
                   if pattern.match(operation)})


def test_hx004_ignores_unused_nested_fallback_helper():
    source = """import os

def f():
    try:
        return get_secret("TOKEN")
    except UnscopedSecretError:
        def unused():
            return os.getenv("TOKEN")
        raise
"""
    assert _hits("HX004", source) == []


def test_hx004_follows_called_nested_fallback_helper():
    source = """import os

def f():
    try:
        return get_secret("TOKEN")
    except UnscopedSecretError:
        def fallback():
            return os.getenv("TOKEN")
        return fallback()
"""
    assert _hits("HX004", source) == [6]


def test_hx007_requires_a_known_config_loader():
    unrelated = """class Counter:
    def load_config(self):
        return 1

async def f(counter):
    return counter.load_config()
"""
    imported = """from hermes_cli.config import load_config

async def f():
    return load_config()
"""
    assert _hits("HX007", unrelated) == []
    assert _hits("HX007", imported) == [4]


def test_hx008_respects_asyncio_shadowing():
    shadowed = """import asyncio

def f(asyncio):
    return asyncio.get_event_loop()
"""
    imported = """import asyncio

def f():
    return asyncio.get_event_loop()
"""
    assert _hits("HX008", shadowed) == []
    assert _hits("HX008", imported) == [4]


def test_hx009_unconditional_overwrite_clears_result_provenance():
    source = """import asyncio

async def f(tasks):
    results = await asyncio.gather(*tasks, return_exceptions=True)
    results = [1, 2]
    for result in results:
        if isinstance(result, Exception):
            raise result
"""
    assert _hits("HX009", source) == []


def test_hx009_saved_alias_keeps_result_provenance():
    source = """import asyncio

async def f(tasks):
    results = await asyncio.gather(*tasks, return_exceptions=True)
    saved = results
    results = [1, 2]
    for result in saved:
        if isinstance(result, Exception):
            raise result
"""
    assert _hits("HX009", source) == [8]


def test_hx009_conditional_overwrite_keeps_may_provenance():
    source = """import asyncio

async def f(tasks, replace):
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if replace:
        results = [1, 2]
    for result in results:
        if isinstance(result, Exception):
            raise result
"""
    assert _hits("HX009", source) == [8]


def test_hx012_accepts_the_context_bound_thread_helper_shape():
    source = """import contextvars
import threading

def ctx_bound(fn):
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)

def spawn_context_thread(target):
    return threading.Thread(target=ctx_bound(target))
"""
    assert _hits("HX012", source) == []


def test_hx012_rejects_a_lookalike_wrapper_that_does_not_copy_context():
    source = """import threading

def ctx_bound(fn):
    return fn

def spawn_context_thread(target):
    return threading.Thread(target=ctx_bound(target))
"""
    assert _hits("HX012", source) == [7]


def test_ps_p05_matches_multiline_operation_not_warning_string():
    multiline = """import os
import subprocess

def f():
    subprocess.run(
        ["x"],
        env=os.environ,
        timeout=5,
    )
"""
    prose = """WARNING = "subprocess.run(['x'], env=os.environ, timeout=5)"
"""
    assert _pattern_hits("PS-P05", multiline) == [5]
    assert _pattern_hits("PS-P05", prose) == []


def test_ps_p06_matches_multiline_operation_not_warning_string():
    multiline = """import os

def f():
    token = os.getenv(
        "DISCORD_TOKEN",
    )
    return token
"""
    prose = """WARNING = "os.getenv('DISCORD_TOKEN')"
"""
    assert _pattern_hits("PS-P06", multiline) == [4]
    assert _pattern_hits("PS-P06", prose) == []
