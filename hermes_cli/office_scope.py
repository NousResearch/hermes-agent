"""Parse and record Agent Office scope-change requests.

Scope changes are intentionally explicit and machine-parseable. Free-form
"I only did X" caveats are not accepted as gate relaxations.
"""

from __future__ import annotations

import re
from typing import Any

_REQUIRED_FIELDS = (
    "requirement_ref",
    "requested_change",
    "reason",
    "attempted_evidence",
    "impact",
)

SCOPE_CHANGE_REQUEST_FORMAT = """SCOPE_CHANGE_REQUEST
requirement_ref: <original requirement id/text>
requested_change: <exact proposed reduction/substitution>
reason: <why original cannot be satisfied>
attempted_evidence: <commands/files/actions tried>
impact: <what success claim would no longer mean>
options:
  - <option 1>
  - <option 2>
END_SCOPE_CHANGE_REQUEST"""

_BLOCK_RE = re.compile(
    r"^SCOPE_CHANGE_REQUEST\s*\n(?P<body>.*?)\nEND_SCOPE_CHANGE_REQUEST\s*$",
    re.MULTILINE | re.DOTALL,
)


def scope_change_request_format() -> str:
    """Canonical machine-parseable scope-change block format."""
    return SCOPE_CHANGE_REQUEST_FORMAT


def parse_scope_change_requests(text: str | None) -> list[dict[str, Any]]:
    """Return valid SCOPE_CHANGE_REQUEST blocks from *text*.

    Required format:

    SCOPE_CHANGE_REQUEST
    requirement_ref: <original requirement id/text>
    requested_change: <exact proposed reduction/substitution>
    reason: <why original cannot be satisfied>
    attempted_evidence: <commands/files/actions tried>
    impact: <what success claim would no longer mean>
    options:
      - <option 1>
      - <option 2>
    END_SCOPE_CHANGE_REQUEST

    Blocks missing any required scalar field or at least one option are ignored;
    malformed caveats must not relax gates.
    """
    if not text:
        return []
    out: list[dict[str, Any]] = []
    for m in _BLOCK_RE.finditer(text):
        body = m.group("body")
        parsed: dict[str, Any] = {}
        current_key: str | None = None
        options: list[str] = []
        for raw in body.splitlines():
            line = raw.rstrip()
            if not line.strip():
                continue
            if line.lstrip().startswith("-") and current_key == "options":
                opt = line.lstrip()[1:].strip()
                if opt:
                    options.append(opt)
                continue
            if ":" not in line:
                current_key = None
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key == "options":
                current_key = "options"
                if val:
                    options.append(val)
                continue
            current_key = key
            parsed[key] = val
        if options:
            parsed["options"] = options
        if all(parsed.get(k) for k in _REQUIRED_FIELDS) and parsed.get("options"):
            parsed["raw_block"] = m.group(0)
            out.append(parsed)
    return out


def emit_scope_change_events(conn, task_id: str, text: str | None, *, source: str, run_id: int | None = None) -> int:
    """Emit ``office.scope_change_requested`` events for valid blocks.

    Returns the number of events emitted. The caller must already be inside the
    desired transaction.
    """
    requests = parse_scope_change_requests(text)
    if not requests:
        return 0
    import json
    import time

    for req in requests:
        payload = {k: v for k, v in req.items() if k != "raw_block"}
        payload["source"] = source
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, run_id, "office.scope_change_requested", json.dumps(payload, ensure_ascii=False), int(time.time())),
        )
    return len(requests)
