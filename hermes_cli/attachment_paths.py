"""Attachment path parsing shared by the CLI and served input handlers."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

_IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.gif', '.webp',
    '.bmp', '.tiff', '.tif', '.svg', '.ico',
})


def _split_path_input(raw: str) -> tuple[str, str]:
    r"""Split a leading path token (quoted or with ``\ `` escapes) from trailing free-form text."""
    raw = str(raw or "").strip()
    if not raw:
        return "", ""

    if raw[0] in {'"', "'"}:
        quote = raw[0]
        pos = 1
        while pos < len(raw):
            ch = raw[pos]
            if ch == '\\' and pos + 1 < len(raw):
                pos += 2
                continue
            if ch == quote:
                return raw[1:pos], raw[pos + 1 :].strip()
            pos += 1
        return raw[1:], ""

    pos = 0
    while pos < len(raw):
        ch = raw[pos]
        if ch == '\\' and pos + 1 < len(raw) and raw[pos + 1] == ' ':
            pos += 2
        elif ch == ' ':
            break
        else:
            pos += 1

    return raw[:pos].replace('\\ ', ' '), raw[pos:].strip()



def _resolve_attachment_path(raw_path: str) -> Path | None:
    """Resolve a user-supplied attachment path (quotes, ``~``, env vars, ``file://``; relative to TERMINAL_CWD).

    Returns ``None`` unless it resolves to an existing file.
    """
    token = str(raw_path or "").strip()
    if not token:
        return None

    if token[0] == token[-1] and token[0] in {'"', "'"}:
        token = token[1:-1].strip()
    token = token.replace('\\ ', ' ')
    if not token:
        return None

    expanded = token
    if token.startswith("file://"):
        try:
            parsed = urlparse(token)
            if parsed.scheme == "file":
                expanded = unquote(parsed.path or "")
                if parsed.netloc and os.name == "nt":
                    expanded = f"//{parsed.netloc}{expanded}"
                elif os.name == "nt" and len(expanded) >= 3 and expanded[0] == "/" and expanded[1].isalpha() and expanded[2] == ":":
                    # file:///C:/... parses to path "/C:/..." — drop the leading slash
                    # so it resolves as a drive-letter path.
                    expanded = expanded[1:]
        except Exception:
            expanded = token
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/" and normalized[0].isalpha():
            expanded = f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    path = Path(expanded)
    if not path.is_absolute():
        base_dir = Path(os.getenv("TERMINAL_CWD", os.getcwd()))
        path = base_dir / path

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    # ENAMETOOLONG for a pasted `/goal <long prose>` that passed the `/` prefilter
    # would otherwise reach process_loop and silently lose the input.
    try:
        if not resolved.exists() or not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved



def _file_drop_result(path: Path, remainder: str) -> dict:
    from cli import _IMAGE_EXTENSIONS
    return {"path": path, "is_image": path.suffix.lower() in _IMAGE_EXTENSIONS, "remainder": remainder}



def _detect_file_drop(user_input: str) -> dict | None:
    """Detect a dragged/pasted file path at the start of *user_input* -> ``{path, is_image, remainder}`` or None."""
    from cli import _file_drop_result, _resolve_attachment_path, _split_path_input
    if not isinstance(user_input, str):
        return None

    stripped = user_input.strip()
    if not stripped:
        return None

    # Optionally quoted; then /, ~, ./, ../, a Windows drive prefix, or (unquoted) file://.
    quoted = stripped[:1] in {"'", '"'}
    unquoted = stripped[1:] if quoted else stripped
    starts_like_path = (
        unquoted.startswith(("/", "~", "./", "../"))
        or (not quoted and unquoted.startswith("file://"))
        or (len(unquoted) >= 3 and unquoted[1] == ":" and unquoted[2] in {"\\", "/"} and unquoted[0].isalpha())
    )
    if not starts_like_path:
        return None

    direct_path = _resolve_attachment_path(stripped)
    if direct_path is not None:
        return _file_drop_result(direct_path, "")

    first_token, remainder = _split_path_input(stripped)
    drop_path = _resolve_attachment_path(first_token)
    if drop_path is None and " " in stripped and not quoted:
        for pos in reversed([idx for idx, ch in enumerate(stripped) if ch == " "]):
            drop_path = _resolve_attachment_path(stripped[:pos].rstrip())
            if drop_path is not None:
                remainder = stripped[pos + 1 :].strip()
                break
    if drop_path is None:
        return None
    return _file_drop_result(drop_path, remainder)
