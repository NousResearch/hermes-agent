"""Which versions of the runtime this adapter was verified against.

The adapter calls ``kanban_db.create_task`` with fifteen keyword arguments, reads the
``tasks`` table by column name, copies a plugin onto the ``pre_tool_call`` hook and writes
config keys that specific call sites read. Every one of those was verified against Hermes
**0.21.1** — and nothing recorded that, so an upstream change surfaced on a customer's host
at task-submission time rather than at ``nova apply``.

The runtime already has the idea: a plugin manifest declares ``requires_hermes`` and the
loader skips cleanly on a mismatch rather than dying mid-register. This is the same idea
pointed the other way.

**A mismatch warns; it does not refuse.** Refusing would make a patch release of the runtime
an outage, and the adapter is written defensively enough that most upstream changes will
either work or fail loudly at the seam. What an operator needs is to be *told* they are
outside what was verified, before they discover it from a worker that will not start.
"""

from __future__ import annotations

import re
from typing import Optional

#: The version every call site in this adapter was verified against.
VERIFIED_AGAINST = "0.21.1"

#: Verified exactly at 0.21.x. A different minor is where the interfaces this adapter
#: reaches through — the kanban schema, the plugin contract, the config keys — are free to
#: move, so that is the boundary worth naming.
SUPPORTED_MAJOR_MINOR = (0, 21)

_VERSION = re.compile(r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?")


def parse_version(raw: str) -> Optional[tuple[int, int, int]]:
    """``(major, minor, patch)``, or None when the string is not a version.

    Non-numeric suffixes are ignored rather than rejected: a source checkout reporting
    ``0.21.1.dev3+g1a2b3c`` is the same 0.21.1 for compatibility purposes, and refusing to
    parse it would turn every developer install into a warning.
    """
    match = _VERSION.match(raw or "")
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def runtime_version() -> str:
    """The installed runtime's version, or ``""`` when it cannot be determined.

    Lazily imported, inside the function, for the reason every runtime import in this
    package is: ``nova`` must load and test with only the standard library and PyYAML.
    """
    try:
        from hermes_cli.plugins_manifest import running_hermes_version

        return str(running_hermes_version() or "")
    except Exception:  # noqa: BLE001 — an unreadable version is a fact, not a failure
        return ""


def check(version: Optional[str] = None) -> list[str]:
    """Warnings about running against a version this adapter was not verified on."""
    found = runtime_version() if version is None else version
    if not found:
        return [
            "could not determine the runtime version, so NOVA cannot tell whether this "
            f"adapter was verified against it (verified against {VERIFIED_AGAINST})"
        ]

    parsed = parse_version(found)
    if parsed is None:
        return [
            f"the runtime reports version {found!r}, which NOVA cannot parse; verified "
            f"against {VERIFIED_AGAINST}"
        ]

    major, minor, _patch = parsed
    if (major, minor) == SUPPORTED_MAJOR_MINOR:
        return []

    direction = "newer" if (major, minor) > SUPPORTED_MAJOR_MINOR else "older"
    return [
        f"runtime {found} is {direction} than the {VERIFIED_AGAINST} this adapter was "
        f"verified against. NOVA reaches into the work-store schema, the plugin hook "
        f"contract and specific config keys; any of those may have moved. Re-verify "
        f"before relying on policy enforcement or work submission"
    ]
