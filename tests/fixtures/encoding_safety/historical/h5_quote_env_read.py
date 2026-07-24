"""Historical instance #5 — quote_env_value / save_env_value read context.

Pre-utf-8-sig: save_env_value read ~/.hermes/.env with plain utf-8 before
rewriting a quoted KEY=value line. A BOM stuck to the first key; the
rewrite then persisted the mangled name. (Current main uses utf-8-sig;
this fixture freezes the pre-fix shape.)
"""
from pathlib import Path
import os
import tempfile


def _quote_env_value(value: str) -> str:
    if value == "" or value == value.strip() and "#" not in value:
        return value
    return f'"{value.replace(chr(34), chr(92) + chr(34))}"'


def save_env_value(key: str, value: str, env_path: Path) -> None:
    read_kw = {"encoding": "utf-8", "errors": "replace"}
    write_kw = {"encoding": "utf-8"}

    lines = []
    if env_path.exists():
        with open(env_path, **read_kw) as f:
            lines = f.readlines()

    serialized_value = _quote_env_value(value)
    lines.append(f"{key}={serialized_value}\n")

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
