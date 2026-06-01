"""Shared sanitization helpers for error messages and tracebacks.

Used by both the HTTP API server (gateway/platforms/api_server.py) and the
terminal tool (tools/terminal_tool.py) to strip filesystem paths from error
messages before they reach external clients or LLM context.
"""
import re

# Known filesystem root directory names.
_FS_ROOTS = (
    r'(?:home|Users|opt|var|tmp|etc|usr|root|srv|proc|sys|dev|mnt|'
    r'media|run|boot|lib|bin|sbin|snap|nix|private)'
)

# Match an absolute Unix path rooted at a known FS directory.
# - `(?<![\w/])` prevents matching inside longer paths or identifiers
#   (e.g. /api/v1/home/user stays unmatched because of the leading /api).
# - Requires at least 2 segments after the root so short fragments like
#   /home/user are not clobbered — only full paths like /home/user/file.
# - Negative lookahead `(?!/v\d/)` right after the root name prevents
#   matching API-style continuations like /var/lib/v1/chat.
_UNIX_PATH = re.compile(
    r'(?<![\w/])'                    # not preceded by word char or /
    r'/' + _FS_ROOTS +               # /<root>
    r'(?!/v\d/)'                     # not /root/v1/... (API path)
    r'(?:/[\w.-]+){2,}',             # at least 2 more segments
)

# Match an absolute Windows path (C:\Users\..., D:\..., etc.).
_WIN_PATH = re.compile(
    r'[A-Za-z]:\\(?:[\w.-]+\\)+[\w.-]+'
)


def sanitize_error_msg(exc: Exception, max_len: int = 200) -> str:
    """Return a sanitized error string safe for HTTP responses and LLM context.

    Truncates long messages and strips absolute paths to avoid leaking
    internal filesystem layout or stack details.
    """
    msg = str(exc)
    msg = _UNIX_PATH.sub('<path>', msg)
    msg = _WIN_PATH.sub('<path>', msg)
    if len(msg) > max_len:
        msg = msg[:max_len] + "..."
    return msg


def sanitize_traceback(tb_str: str, max_len: int = 2000) -> str:
    """Return a sanitized traceback string safe for LLM context.

    Strips absolute paths and truncates long tracebacks.
    """
    tb_str = _UNIX_PATH.sub('<path>', tb_str)
    tb_str = _WIN_PATH.sub('<path>', tb_str)
    if len(tb_str) > max_len:
        tb_str = tb_str[:max_len] + "\n... (truncated)"
    return tb_str
