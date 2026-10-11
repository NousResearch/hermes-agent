from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP


mcp = FastMCP("__SERVER_NAME__")


def _read_text(path: str) -> str:
    file_path = Path(path).expanduser()
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {file_path}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8 text: {file_path}") from exc


@mcp.tool
def summarize_text_file(path: str, preview_chars: int = 1200) -> dict[str, int | str]:
    """Return basic metadata and a preview for a UTF-8 text file."""
    file_path = Path(path).expanduser()
    text = _read_text(path)
    return {
        "path": str(file_path),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "preview": text[:preview_chars],
    }


@mcp.tool
def search_text_file(path: str, needle: str, max_matches: int = 20) -> dict[str, Any]:
    """Find matching lines in a UTF-8 text file."""
    file_path = Path(path).expanduser()
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
        if needle.lower() in line.lower():
            matches.append({"line_number": line_number, "line": line})
            if len(matches) >= max_matches:
                break
    return {"path": str(file_path), "needle": needle, "matches": matches}


@mcp.resource("file://{path}")
def read_file_resource(path: str) -> str:
    """Expose a text file as a resource."""
    return _read_text(path)


if __name__ == "__main__":
    mcp.run()
