"""Regression guard: the gateway core must not use bare read_text()/write_text().

On Windows (and any host where ``locale.getpreferredencoding()`` is not UTF-8 —
cp1252 on US locales, GBK/CP936 on Chinese locales, etc.), ``Path.read_text()``
and ``Path.write_text()`` without an explicit ``encoding=`` use the locale
codec. In the ``hermes update``-over-gateway flow this surfaces as:

* ``UnicodeEncodeError`` when writing the user's reply (emoji / CJK) to the
  ``.update_response`` file, and
* ``UnicodeDecodeError`` / mojibake when reading the update subprocess's
  captured UTF-8 stdout.

Both are ``UnicodeError`` subclasses, so the surrounding ``except OSError``
guards do NOT catch them — the gateway command handler / update-stream
coroutine crashes.

``ruff``'s PLW1514 and ``scripts/check-windows-footguns.py`` both miss these:
PLW1514 does not resolve ``.read_text``/``.write_text`` on a variable/attribute
receiver, and the footgun script scans builtin ``open()`` only.

The ``/update`` coordination files are written and read from *both* modules
below — the handlers were extracted into ``GatewaySlashCommandsMixin``, but the
prompt/response and output-streaming halves still live in ``run.py`` — so the
guard covers both rather than a single file.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

GUARDED_FILES = (
    PROJECT_ROOT / "gateway" / "run.py",
    PROJECT_ROOT / "gateway" / "slash_commands.py",
)

_ENCODING_SENSITIVE = {"read_text", "write_text"}


def _calls_without_encoding(filepath: Path) -> list[tuple[int, str]]:
    """Return (lineno, method) for read_text/write_text calls lacking encoding=."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _ENCODING_SENSITIVE:
            continue
        if any(kw.arg == "encoding" for kw in node.keywords):
            continue
        offenders.append((node.lineno, func.attr))
    return sorted(offenders)


@pytest.mark.parametrize("path", GUARDED_FILES, ids=lambda p: p.name)
def test_gateway_core_has_no_bare_read_write_text(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    offenders = _calls_without_encoding(path)
    assert not offenders, (
        f"{path.name} has read_text()/write_text() calls without an explicit "
        'encoding= (Windows cp1252/GBK footgun). Add encoding="utf-8":\n'
        + "\n".join(f"  {path.name}:{ln}  .{attr}()" for ln, attr in offenders)
    )
