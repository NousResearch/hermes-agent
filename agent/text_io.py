"""
Safe text file I/O for cross-platform encoding (especially Windows).

Use these helpers for any text file that may contain user content, tool output,
or config so we never hit UnicodeEncodeError/UnicodeDecodeError when the
system default encoding is CP1252 or the file has stray bytes.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional, Union

# Defaults that avoid encoding crashes on Windows (cp1252) and odd bytes in content.
READ_DEFAULTS: dict[str, Any] = {
    "encoding": "utf-8",
    "errors": "replace",
}
WRITE_DEFAULTS: dict[str, Any] = {
    "encoding": "utf-8",
    "errors": "replace",
    "newline": "",
}


def open_text(
    path: Union[Path, str],
    mode: str = "r",
    *,
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    **kwargs: Any,
) -> io.TextIOWrapper:
    """Open a text file with UTF-8 and replace errors by default.

    Use for config, transcripts, JSON, YAML, logs, or any text that might
    contain non-ASCII or come from another platform. Avoids UnicodeDecodeError
    when reading and UnicodeEncodeError when writing on Windows.

    mode: "r", "w", "a", "r+", etc.
    encoding: default "utf-8"
    errors: default "replace" (replace bad bytes instead of raising)
    newline: default "" for write/append (consistent line endings)
    """
    if encoding is None:
        encoding = READ_DEFAULTS["encoding"] if "r" in mode else WRITE_DEFAULTS["encoding"]
    if errors is None:
        errors = READ_DEFAULTS["errors"] if "r" in mode else WRITE_DEFAULTS["errors"]
    if newline is None and ("w" in mode or "a" in mode):
        newline = WRITE_DEFAULTS["newline"]
    return open(  # noqa: SIM115
        path,
        mode,
        encoding=encoding,
        errors=errors,
        newline=newline if newline is not None else "",
        **kwargs,
    )
