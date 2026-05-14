"""Regression guard: no ``print()`` calls in platform adapter async hot paths.

``print()`` is synchronous, holds the GIL, and bypasses our log filtering /
sampling / structured-logging pipeline.  Hot-path async handlers (Discord +
Telegram message dispatch, attachment caching, send_* error branches) must
use ``logger`` instead.

This is a lightweight static scan rather than a runtime test — it just reads
each source file and asserts no line-leading ``print(`` token appears.

Scope is deliberately limited to the two adapters that the W2 audit flagged
(discord.py, telegram.py).  feishu.py has legitimate ``print()`` calls in its
interactive onboarding / QR-code setup helpers (not async hot paths), so it
is intentionally excluded from this guard.  If feishu later grows async
hot-path prints, extend this list.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Files that must never contain ``print(`` calls — async hot paths only.
GUARDED_FILES = [
    "gateway/platforms/discord.py",
    "gateway/platforms/telegram.py",
]

# Matches a function-call to ``print`` at the start of a (possibly indented)
# line.  This intentionally does NOT match string literals containing the
# substring "print" (e.g. docstrings, log message text), because those don't
# start with ``print(`` after optional whitespace.
_PRINT_CALL_RE = re.compile(r"^\s*print\(", re.MULTILINE)


@pytest.mark.parametrize("rel_path", GUARDED_FILES)
def test_no_print_calls_in_platform_adapters(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    assert path.exists(), f"Guarded file is missing: {rel_path}"
    text = path.read_text(encoding="utf-8")
    matches = _PRINT_CALL_RE.findall(text)
    assert not matches, (
        f"Found {len(matches)} print() call(s) in {rel_path}. "
        f"Async hot paths must use the module ``logger`` instead "
        f"(see W2-T22)."
    )
