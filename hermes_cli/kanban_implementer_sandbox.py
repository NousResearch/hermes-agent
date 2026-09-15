"""macOS process sandbox profile for restricted implementer workers."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _sbpl_string(value: Path) -> str:
    """Encode a persisted path as one SBPL string literal, never profile syntax."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


def sandboxed_implementer_argv(
    command: list[str], *, workspace: str, hermes_home: str | None, board_db: str,
) -> list[str]:
    """Wrap a dispatcher-owned implementer in a fail-closed macOS write fence."""
    if sys.platform != "darwin" or not os.path.exists("/usr/bin/sandbox-exec"):
        raise RuntimeError("implementer workers require macOS /usr/bin/sandbox-exec")
    if not hermes_home:
        raise RuntimeError("implementer workers require a profile-scoped HERMES_HOME")
    home = Path(hermes_home).resolve()
    roots = [Path(workspace).resolve(), *(home / name for name in ("sessions", "logs", "cache", "checkpoints"))]
    files = [
        *(home / name for name in ("state.db", "state.db-wal", "state.db-shm", "state.db-journal")),
        *(Path(f"{Path(board_db).resolve()}{suffix}") for suffix in ("", "-wal", "-shm", "-journal")),
    ]
    write_rules = "\n".join((
        *(f'(allow file-write* (subpath "{_sbpl_string(root)}"))' for root in dict.fromkeys(roots)),
        *(f'(allow file-write* (literal "{_sbpl_string(path)}"))' for path in dict.fromkeys(files)),
    ))
    profile = "\n".join((
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow network*)",
        "(allow file-read*)",
        write_rules,
    ))
    return ["/usr/bin/sandbox-exec", "-p", profile, *command]
