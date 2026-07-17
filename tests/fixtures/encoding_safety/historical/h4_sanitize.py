"""Historical instance #4 — _sanitize_env_file_if_needed (pre-#66475).

utf-8-sig + errors=replace on a user .env, then rewrite when sanitize changes
lines. UTF-16 BOM (Notepad "Unicode") becomes U+FFFD U+FFFD KEY and is
permanently written back. See #66474.
"""
from pathlib import Path
import os
import tempfile


def atomic_replace(src, dst):
    os.replace(src, dst)


def _sanitize_env_lines(lines):
    return lines


def _sanitize_env_file_if_needed(path: Path) -> None:
    if not path.exists():
        return

    read_kw = {"encoding": "utf-8-sig", "errors": "replace"}
    try:
        with open(path, **read_kw) as f:
            original = f.readlines()
        stripped = [line.replace("\x00", "") for line in original]
        sanitized = _sanitize_env_lines(stripped)
        if sanitized != original:
            fd, tmp = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".env_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(sanitized)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    except Exception:
        pass
