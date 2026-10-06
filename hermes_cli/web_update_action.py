"""Durable dashboard update admission, separate from the child-owned output log."""
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

ACTION_RECORD_NAME = "hermes-update-action.json"


def write_update_admission(log_dir: Path, action_id: str) -> None:
    """Replace and sync the admitted identity before Popen; I/O failure refuses spawn."""
    if not isinstance(action_id, str) or re.fullmatch(r"[0-9a-f]{32}", action_id) is None:
        raise ValueError("Update admission requires an exact 32-hex action id")
    record = {"version": 1, "action_id": action_id, "admitted_at": time.time()}
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=log_dir,
                                         prefix=".hermes-update-action-", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(record, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        # Close first: Windows does not permit replacing an open source file.
        os.replace(temporary, log_dir / ACTION_RECORD_NAME)
        if os.name != "nt":
            fd = os.open(log_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_update_admission(log_dir: Path) -> tuple[dict | None, bool]:
    """Return valid admission and whether a record exists (invalid is a barrier too)."""
    try:
        record = json.loads((log_dir / ACTION_RECORD_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, False
    except (OSError, ValueError):
        return None, True
    if (not isinstance(record, dict) or set(record) != {"version", "action_id", "admitted_at"}
            or type(record["version"]) is not int or record["version"] != 1
            or not isinstance(record["action_id"], str)
            or re.fullmatch(r"[0-9a-f]{32}", record["action_id"]) is None
            or type(record["admitted_at"]) not in (int, float)):
        return None, True
    try:
        if not math.isfinite(record["admitted_at"]) or record["admitted_at"] < 0:
            return None, True
    except OverflowError:
        return None, True
    return record, True
