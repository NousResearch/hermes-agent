"""Unit tests for the Discord adapter's opt-in outbound mention resolution.

The method is AST-extracted and executed in isolation so the test needs no live
Discord client or the adapter's heavier imports.
"""
import ast
import asyncio
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_root = _HERE
for _ in range(6):
    if os.path.isdir(os.path.join(_root, "plugins", "platforms", "discord")):
        break
    _root = os.path.dirname(_root)
_ADAPTER = os.path.join(_root, "plugins", "platforms", "discord", "adapter.py")


def _load_resolver():
    fn = next(
        n for n in ast.walk(ast.parse(open(_ADAPTER).read()))
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_resolve_outbound_mentions"
    )
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {"os": os, "re": re, "Any": object}
    exec(compile(mod, _ADAPTER, "exec"), ns)
    return ns["_resolve_outbound_mentions"]


class _Member:
    def __init__(self, id, display_name=None, name=None, global_name=None):
        self.id = id
        self.display_name = display_name
        self.name = name
        self.global_name = global_name


class _Guild:
    def __init__(self, members):
        self.members = members


class _Channel:
    def __init__(self, guild):
        self.guild = guild


def _run(content, flag="true"):
    resolve = _load_resolver()
    guild = _Guild([
        _Member(200, display_name="Support Bot", name="supportbot"),
        _Member(300, display_name="Alice", name="alice"),
        _Member(400, display_name="Al", name="al"),
    ])
    ch = _Channel(guild)
    if flag is None:
        os.environ.pop("DISCORD_RESOLVE_MENTIONS", None)
    else:
        os.environ["DISCORD_RESOLVE_MENTIONS"] = flag
    return asyncio.run(resolve(object(), content, ch))


@pytest.mark.parametrize("content,expected", [
    ("@Support Bot can you take this?", "<@200> can you take this?"),   # multi-word name
    ("@support bot pls", "<@200> pls"),                                 # case-insensitive
    ("hey @Alice and @Support Bot", "hey <@300> and <@200>"),           # multiple, longest-first
    ("<@200> already tagged", "<@200> already tagged"),                 # already a real mention
    ("mail me@example.com", "mail me@example.com"),                     # not a mention (word-char before @)
    ("@Nobody here", "@Nobody here"),                                   # unknown name untouched
])
def test_resolves_when_enabled(content, expected):
    assert _run(content, "true") == expected


def test_noop_when_disabled():
    assert _run("@Support Bot hi", "false") == "@Support Bot hi"


def test_noop_when_unset():
    assert _run("@Support Bot hi", None) == "@Support Bot hi"


def test_shorter_name_still_resolves_alone():
    assert _run("ping @Al now", "true") == "ping <@400> now"
