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
_FENCE = re.compile(r"(?P<indent>[ \t]*)(?P<fence>(?P<ch>[`~])(?P=ch){2,})(?P<info>.*)")
# An ``...`` or ``…`` standing alone as an argument / item / body, e.g. ``f(name="x", ...)``.
_PLACEHOLDER = re.compile(r"(?<![\w.])(?:\.\.\.|…)(?![\w.])")


def _plugin_docs() -> list[Path]:
    docs = _ROOT / "website" / "docs"
    return [*sorted((docs / "developer-guide").rglob("*.md*")),
            *sorted((docs / "user-guide" / "features").glob("plugins*.md")), _ROOT / "plugins" / "AGENTS.md"]


def _fences(markdown: str):
    """``(opening line, language, body)`` per fenced block, as CommonMark reads it: the next line of only the same
    character, at least as long, closes it at any indent up to 3 columns past the opener's (so the two indents need
    not match, and a fence nested in a list item still closes); an unclosed fence runs to the end of the page."""
    lines = markdown.splitlines(keepends=True)
    fences = [_FENCE.fullmatch(line.rstrip("\r\n").expandtabs(4)) for line in lines]
    i = 0
    while i < len(lines):
        opener, i = fences[i], i + 1
        if not opener or (opener["ch"] == "`" and "`" in opener["info"]):
            continue
        start, end = i, len(lines)
        for j in range(start, len(lines)):
            closer = fences[j]
            if (closer and closer["fence"][0] == opener["ch"] and len(closer["fence"]) >= len(opener["fence"])
                    and not closer["info"].strip() and len(closer["indent"]) <= len(opener["indent"]) + 3):
                end = j
                break
        i = end + 1
        yield start, re.match(r"\w*", opener["info"].lstrip())[0], "".join(lines[start:end])


def _parse(block: str) -> ast.AST | None:
    """The whole block as one tree; when prose placeholders keep it from parsing, the whole block again
    with each placeholder swapped for an expression that is valid in that slot. ``None`` if neither parses."""
    for swap in (None, "...", "*()", "**{}"):
        try:
            return ast.parse(block if swap is None else _PLACEHOLDER.sub(swap, block))
        except SyntaxError:
            continue
    return None


def scan(markdown: str) -> tuple[set[str], list[int]]:
    """Dotted Hermes targets the markdown's Python blocks import that the plugin API does not list, and
    the opening line of every Python block that mentions ``import`` but does not parse even after
    placeholder repair (so it cannot be checked). A module core does not ship is the example plugin's
    own package; ``TYPE_CHECKING`` imports are skipped, as validate skips them."""
    found, unparsable = set(), []
    for line, lang, body in _fences(markdown):
        if lang.lower() not in ("python", "py", "python3"):
            continue
        tree = _parse(textwrap.dedent(body))
        if tree is None:
            if "import" in body:
                unparsable.append(line)
            continue
        skipped = _type_checking_nodes(tree)
        for node in ast.walk(tree):
            for module, name in [] if id(node) in skipped else _imported(node):
                target = _private_target(module, name, _hermes_packages(), set())
                if target and _core_module_exists(module):
                    found.add(target)
    return found, unparsable


def test_scanner_flags_unlisted_imports_even_in_blocks_with_placeholders():
    markdown = textwrap.dedent("""\
        ```python title="plugin.py"
        from agent.memory_provider import MemoryProvider
        from tools.approval_smart import judge
        from plugins.web.my_backend.provider import MyProvider
        class P(MemoryProvider): …  # prose placeholder
        ctx.register(name="p", ...)
        ```

        ```py
        from tools.approval import \\
            _continued
        class P: …
        ```

        ```python
        if enabled: from tools.approval import _inline
        class P: …
        ```

        ```python
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from tools.approval import _typing_only
        class P: …
        ```

        ```bash
        python -c 'from tools.approval import _private'
        ```
        """)

    assert scan(markdown) == ({"tools.approval_smart.judge", "tools.approval._continued",
                               "tools.approval._inline"}, [])


def test_scanner_reports_blocks_it_cannot_parse_instead_of_guessing():
    markdown = textwrap.dedent("""\
        ```python
        from tools.approval import _hidden
        success_response(image=<url>)
        ```

        ```python
        success_response(image=<url>)
        ```
        """)

    assert scan(markdown) == (set(), [1])


def test_scanner_reads_long_and_tilde_fences_to_their_matching_close():
    markdown = textwrap.dedent("""\
        ````python
        from tools.approval import _four
        EXAMPLE = '''
        ```
        '''
        from tools.approval import _after_inner_backticks
        ````

        ~~~python
        from tools.approval import _tilde
        EXAMPLE = '''
        ```
        '''
        from tools.approval import _after_backticks_in_tilde
        ~~~

        ```text
        ~~~python
        from tools.approval import _inside_text_fence
        ~~~
        ```
        """)

    assert scan(markdown)[0] == {"tools.approval._four", "tools.approval._after_inner_backticks",
                                 "tools.approval._tilde", "tools.approval._after_backticks_in_tilde"}


def test_scanner_closes_fences_whose_indent_differs_from_the_opener():
    markdown = ("```python\nfrom tools.approval import _hidden\n ```\n\n"
                "  ```python\nfrom tools.approval import _opener_indented\n```\n\n"
                "1. step\n\n   ```python\n   from tools.approval import _in_list_item\n     ```\n\n"
                "```python\nEXAMPLE = '''\n    ```\n'''\nfrom tools.approval import _after_indented_code\n```\n")

    assert scan(markdown) == ({"tools.approval._hidden", "tools.approval._opener_indented",
                               "tools.approval._in_list_item", "tools.approval._after_indented_code"}, [])


def test_plugin_docs_import_only_the_published_plugin_api():
    scans = {doc.relative_to(_ROOT).as_posix(): scan(doc.read_text(encoding="utf-8")) for doc in _plugin_docs()}
    unparsable = [f"{page}:{line}" for page, (_, lines) in scans.items() for line in lines]
    found = {f"{page} {target}" for page, (targets, _) in scans.items() for target in targets}
    seams = set(json.loads(_SEAMS.read_text(encoding="utf-8"))["sites"])

    assert not unparsable, (
        "these plugin-docs Python blocks import something but do not parse, so their imports cannot be "
        f"checked; make the block valid Python (placeholders like `...` are fine): {unparsable}")
    assert not found - seams, (
        "plugin docs import Hermes names outside hermes_cli/data/plugin_api.json; rewrite the example "
        f"onto ctx.* or a listed name: {sorted(found - seams)}")
    assert not seams - found, f"no longer in the docs, delete from {_SEAMS.name}: {sorted(seams - found)}"
