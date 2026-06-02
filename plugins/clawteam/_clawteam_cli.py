"""Shared helper for running the `clawteam` CLI from this plugin.

Used by both the dashboard FastAPI handlers (plugin_api.py) and the
agent-facing tools (tools.py). Single source of truth for binary
discovery, name validation, timeout, and JSON parsing so the two
surfaces cannot drift.

No FastAPI imports here — handlers/tools wrap CliError into their own
error shapes (HTTPException for FastAPI, tool_error JSON for tools).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any

log = logging.getLogger(__name__)

CLAWTEAM_BIN = os.environ.get("CLAWTEAM_BIN") or shutil.which("clawteam") or "clawteam"

# Matches clawteam's own validate_identifier rules: "only letters, digits,
# '.', '_' and '-' are allowed". First char cannot be '-' so the CLI never
# parses a slipped-through name as an option flag. Earlier this regex also
# allowed '/' and ':' but clawteam rejects those — so they would pass our
# guard and surface as a 500 from the CLI (Codex PR #18 review finding).
_NAME_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.\-]*$")


class CliError(Exception):
    """Raised when the clawteam CLI fails or returns unusable output."""

    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def check_clawteam_available() -> bool:
    """True iff a clawteam binary is reachable. Cached by Hermes registry."""
    return bool(shutil.which(CLAWTEAM_BIN) or os.path.isfile(CLAWTEAM_BIN))


def require_str(value: Any, *, field: str) -> str:
    """Require a non-empty string; reject None/non-string/empty/whitespace.

    Use BEFORE validate_name on tool args — `args.get(k)` returns None when
    the LLM omits the key, and `str(None)` -> "None" passes validate_name
    (a 4-letter alpha name) but is obviously wrong. This guard short-circuits
    that class of bug (Codex PR #18 review finding).
    """
    if value is None:
        raise CliError(f"Missing required argument: {field}", status_code=400)
    if not isinstance(value, str):
        raise CliError(
            f"Invalid {field}: expected string, got {type(value).__name__}",
            status_code=400,
        )
    stripped = value.strip()
    if not stripped:
        raise CliError(f"Empty {field} not allowed", status_code=400)
    return stripped


def validate_name(name: str, *, field: str = "name") -> str:
    """Reject names clawteam itself would reject + leading dash for CLI safety."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise CliError(
            f"Invalid {field}: {name!r}. Allowed: letters, digits, '.', '_', '-' "
            f"(first char must not be '-').",
            status_code=400,
        )
    return name


def run_clawteam_json(*args: str, timeout: float = 15.0) -> Any:
    """Run `clawteam --json <args...>` and return parsed JSON.

    Argv as a list (no shell). Surfaces both stderr AND stdout on
    non-zero exit (clawteam sometimes emits structured JSON errors
    on stdout).
    """
    cmd = [CLAWTEAM_BIN, "--json", *args]
    log.debug("clawteam cmd: %s", cmd)
    try:
        proc = subprocess.run(  # noqa: S603 — argv list, no shell
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise CliError(
            f"clawteam binary not found at {CLAWTEAM_BIN!r}. "
            f"Install with apps/molt-factory/scripts/install-clawteam.sh "
            f"or set CLAWTEAM_BIN.",
            status_code=503,
        )
    except subprocess.TimeoutExpired:
        raise CliError(
            f"clawteam {' '.join(args)} timed out after {timeout}s",
            status_code=504,
        )

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        out = (proc.stdout or "").strip()
        joined = " | ".join(s for s in (err, out) if s) or "(no output)"
        raise CliError(
            f"clawteam {' '.join(args)} failed (exit {proc.returncode}): {joined}",
            status_code=500,
        )
    text = proc.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CliError(
            f"clawteam {' '.join(args)} returned non-JSON: {exc}",
            status_code=500,
        )
