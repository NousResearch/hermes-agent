"""The plugin docs' Python code blocks import only names on the published plugin API.

``hermes_cli/data/plugin_api.json`` is the list ``hermes plugins validate`` enforces; a docs example
that imports anything else teaches plugin authors an import validate warns on. Blocks that still need
a seam are named in ``plugin_docs_import_seams.json``, which may only shrink.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path

from hermes_cli.plugin_validate_private_imports import (
    _core_module_exists,
    _hermes_packages,
    _imported,
    _private_target,
    _type_checking_nodes,
)

_ROOT = Path(__file__).resolve().parents[2]
_SEAMS = Path(__file__).with_name("plugin_docs_import_seams.json")
_FENCE = re.compile(r"^([ \t]*)```(\w*)[^\n]*\n(.*?)^\1```", re.MULTILINE | re.DOTALL)
_IMPORT_STMT = re.compile(r"^[ \t]*(from[ \t]+[\w.]+[ \t]+import[ \t]+(?:\([^)]*\)|[^\n]+)|import[ \t]+[^\n]+)", re.MULTILINE)


def _plugin_docs() -> list[Path]:
    docs = _ROOT / "website" / "docs"
    return [*sorted((docs / "developer-guide").rglob("*.md*")),
            *sorted((docs / "user-guide" / "features").glob("plugins*.md")), _ROOT / "plugins" / "AGENTS.md"]


def _import_trees(block: str) -> list[ast.AST]:
    """The block as one tree, or, when its prose placeholders keep it from parsing, each import alone."""
    try:
        return [ast.parse(block)]
    except SyntaxError:
        trees = []
        for stmt in _IMPORT_STMT.findall(block):
            try:
                trees.append(ast.parse(stmt.strip()))
            except SyntaxError:
                continue
        return trees


def unlisted_imports(markdown: str) -> set[str]:
    """Dotted Hermes targets the markdown's Python blocks import that the plugin API does not list.
    A module core does not ship is the example plugin's own package; ``TYPE_CHECKING`` imports are
    skipped, as validate skips them."""
    found = set()
    for _indent, lang, body in _FENCE.findall(markdown):
        if lang.lower() not in ("python", "py", "python3"):
            continue
        for tree in _import_trees(textwrap.dedent(body)):
            skipped = _type_checking_nodes(tree)
            for node in ast.walk(tree):
                for module, name in [] if id(node) in skipped else _imported(node):
                    target = _private_target(module, name, _hermes_packages(), set())
                    if target and _core_module_exists(module):
                        found.add(target)
    return found


def test_scanner_flags_unlisted_imports_even_in_blocks_that_do_not_parse():
    markdown = textwrap.dedent("""\
        ```python title="plugin.py"
        from agent.memory_provider import MemoryProvider
        from tools.approval_smart import judge
        from plugins.web.my_backend.provider import MyProvider
        class P(MemoryProvider): …  # prose placeholder, not Python
        ```

        ```bash
        python -c 'from tools.approval import _private'
        ```
        """)

    assert unlisted_imports(markdown) == {"tools.approval_smart.judge"}


def test_plugin_docs_import_only_the_published_plugin_api():
    found = {f"{doc.relative_to(_ROOT).as_posix()} {target}"
             for doc in _plugin_docs() for target in unlisted_imports(doc.read_text(encoding="utf-8"))}
    seams = set(json.loads(_SEAMS.read_text(encoding="utf-8"))["sites"])

    assert not found - seams, (
        "plugin docs import Hermes names outside hermes_cli/data/plugin_api.json; rewrite the example "
        f"onto ctx.* or a listed name: {sorted(found - seams)}")
    assert not seams - found, f"no longer in the docs, delete from {_SEAMS.name}: {sorted(seams - found)}"
