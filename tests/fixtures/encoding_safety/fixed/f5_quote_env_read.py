"""Fixed form of H5 — save_env_value reads with utf-8-sig."""
from pathlib import Path
import os
import tempfile


def _quote_env_value(value: str) -> str:
    if value == "" or value == value.strip() and "#" not in value:
        return value
    return f'"{value}"'


def save_env_value(key: str, value: str, env_path: Path) -> None:
    # utf-8-sig strips BOM; errors=replace is for embedded NULs only and
    # the rewrite path is the intentional writer for this key — not a
    # sanitize-on-unknown-encoding path (see R3 / #66474).
    read_kw = {"encoding": "utf-8-sig", "errors": "replace"}
    write_kw = {"encoding": "utf-8"}

    lines = []
    if env_path.exists():
        with open(env_path, **read_kw) as f:  # encoding-safety: ok — intentional writer; utf-8-sig primary; not sanitize-unknown-encoding
            lines = f.readlines()

    lines.append(f"{key}={_quote_env_value(value)}\n")

    fd, tmp_path = tempfile.mkstemp(
        dir=str(env_path.parent), suffix=".tmp", prefix=".env_"
    )
    try:
        with os.fdopen(fd, "w", **write_kw) as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, env_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
