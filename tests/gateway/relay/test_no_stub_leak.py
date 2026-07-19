"""CI guard: the test-only StubConnector must never leak into production paths.

The relay stub connector lives under tests/ and exists only to prove the
gateway side of the relay without the real (Node) connector. If it ever appears
under gateway/ or plugins/, that's a production leak — fail loudly.
"""

from __future__ import annotations

import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_FORBIDDEN_DIRS = ("gateway", "plugins")
# Match actual code leaks (imports / class definitions), not prose mentions in
# docstrings/comments. A production file that *imports* the stub or *defines*
# StubConnector is a real leak; a docstring that references the stub's path as
# documentation is not.
_LEAK_PATTERNS = (
    re.compile(r"^\s*(?:from|import)\s+.*stub_connector", re.MULTILINE),
    re.compile(r"^\s*(?:from|import)\s+.*\bStubConnector\b", re.MULTILINE),
    re.compile(r"^\s*class\s+StubConnector\b", re.MULTILINE),
)


def test_stub_connector_does_not_leak_into_production_paths():
    offenders: list[str] = []
    for top in _FORBIDDEN_DIRS:
        base = _REPO_ROOT / top
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover
                continue
            for pat in _LEAK_PATTERNS:
                if pat.search(text):
                    offenders.append(
                        f"{path.relative_to(_REPO_ROOT)} matches {pat.pattern!r}"
                    )
    assert not offenders, (
        "relay test stub leaked into production paths:\n  " + "\n  ".join(offenders)
    )
