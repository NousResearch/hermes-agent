"""Reading what an agent actually did: its logs, and the credentials it still needs.

Both are per-profile, both touch the runtime's directory layout, and so both live in the
adapter — the only package allowed to know it.

**Logs are tailed, never streamed whole.** A log is unbounded and an agent that has been
running for a month has a large one. Every read here is capped in bytes *before* it is
capped in lines, so a caller cannot ask for a megabyte of text by asking for a lot of
lines, and the file is read from the end rather than from the start.

**Logs are admin-only at the route, and that is load-bearing.** A log line can contain
anything the runtime chose to write — a prompt, a tool argument, part of a document. NOVA
does not attempt to sanitise that, because a sanitiser that misses one pattern is worse
than a clear statement of who may read. What it does do is refuse to read anything outside
the profile's own ``logs/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

#: Log files a profile may have. An explicit map, not a directory listing: the parameter
#: comes from a URL, and resolving a caller-supplied name against a directory is how path
#: traversal happens even when every individual check looks fine.
STREAMS: dict[str, str] = {
    "agent": "agent.log",
    "errors": "errors.log",
    "gateway": "gateway.log",
}

#: Hard ceiling per read, applied before line counting.
MAX_BYTES = 256 * 1024
MAX_LINES = 2000
DEFAULT_LINES = 200


def log_streams(profile_dir: Path) -> tuple[dict[str, Any], ...]:
    """Which log files this profile actually has, and how big they are."""
    out = []
    for key, filename in STREAMS.items():
        path = Path(profile_dir) / "logs" / filename
        try:
            size = path.stat().st_size if path.is_file() else None
        except OSError:
            size = None
        out.append({"stream": key, "filename": filename, "present": size is not None, "bytes": size})
    return tuple(out)


def tail(profile_dir: Path, stream: str, *, lines: int = DEFAULT_LINES) -> dict[str, Any]:
    """The last ``lines`` lines of one log, bounded.

    Returns ``truncated`` so the screen can say it is showing the end of a longer file
    rather than implying it is showing all of it.
    """
    filename = STREAMS.get(stream)
    if filename is None:
        from nova.errors import RuntimeAdapterError

        raise RuntimeAdapterError(
            f"{stream!r} is not a log this runtime keeps. Available: {', '.join(sorted(STREAMS))}"
        )
    lines = max(1, min(int(lines or DEFAULT_LINES), MAX_LINES))
    path = Path(profile_dir) / "logs" / filename

    if not path.is_file():
        return {"stream": stream, "filename": filename, "present": False,
                "lines": [], "truncated": False, "bytes": 0}

    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > MAX_BYTES:
                start = size - MAX_BYTES
                fh.seek(start)
                # The seek lands mid-line; drop the partial first line rather than show it.
                fh.readline()
                raw = fh.read()
                if not raw:
                    # A log with no newline in the last MAX_BYTES — one enormous line, which
                    # a crash dump or a stack trace written in one call really can be.
                    # Dropping the partial line ate everything, so take the raw tail instead
                    # and let `truncated` say it starts mid-line.
                    fh.seek(start)
                    raw = fh.read()
            else:
                raw = fh.read()
    except OSError as exc:
        return {"stream": stream, "filename": filename, "present": True,
                "lines": [], "truncated": False, "bytes": 0, "error": str(exc)}

    text = raw.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    shown = all_lines[-lines:]
    return {
        "stream": stream,
        "filename": filename,
        "present": True,
        "lines": shown,
        "truncated": size > MAX_BYTES or len(all_lines) > len(shown),
        "bytes": size,
    }


def credential_presence(profile_dir: Path, names: tuple[str, ...]) -> dict[str, bool]:
    """Which of ``names`` are defined in this profile's ``.env``.

    Presence only — :func:`nova._env.read_env_file` discards values as it parses, so there
    is no code path from here to a credential's content.
    """
    from nova._env import ENV_FILENAME, read_env_file

    defined = read_env_file(Path(profile_dir) / ENV_FILENAME)
    return {name: name in defined for name in names}


def write_credentials(profile_dir: Path, values: dict[str, Optional[str]]) -> tuple[str, ...]:
    """Set or clear credentials in this profile's ``.env``. Returns the names that changed.

    The caller is responsible for having checked the names against
    :func:`nova.credentials.check_writable` first; this function does not know what an
    agent is allowed to have, only how to write the file.
    """
    from nova._env import ENV_FILENAME, write_env_values

    profile = Path(profile_dir)
    if not profile.is_dir():
        from nova.errors import RuntimeAdapterError

        raise RuntimeAdapterError(
            f"{profile} does not exist. Apply the bundle first: a credential belongs to a "
            "profile the runtime has actually materialised"
        )
    return write_env_values(profile / ENV_FILENAME, values)
