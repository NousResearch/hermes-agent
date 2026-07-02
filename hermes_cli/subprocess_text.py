"""Lossy UTF-8 text subprocess helpers."""

from __future__ import annotations

import subprocess
from typing import Any


def text_subprocess_kwargs(**kwargs: Any) -> dict[str, Any]:
    merged = dict(kwargs)
    merged.setdefault("text", True)
    merged.setdefault("encoding", "utf-8")
    merged.setdefault("errors", "replace")
    return merged


def popen_text_kwargs(**kwargs: Any) -> dict[str, Any]:
    return text_subprocess_kwargs(**kwargs)


def run_text_subprocess(*popenargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(*popenargs, **text_subprocess_kwargs(**kwargs))
