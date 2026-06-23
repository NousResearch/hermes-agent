"""Enforcement: no gateway-reachable module writes per-session state to the
process-global os.environ (the v3-latch bug class).

The gateway runs concurrent sessions in ONE process. A per-session
``os.environ["HERMES_SESSION_ID"|"HERMES_SESSION_KEY"|"HERMES_CRON_SESSION"] =``
write in a gateway-reachable module clobbers other concurrent sessions. v3 fixed
HERMES_CRON_SESSION; the gateway-session-env-leak PRD fixed SESSION_ID/KEY. This
test fails if a regression reintroduces such a write outside the sanctioned
single-process entrypoints (CLI / TUI worker / ACP server / oneshot), which are
single-session-per-process and where os.environ is the correct mechanism.
"""

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Per-session vars that must never be written to process-global os.environ from
# a gateway-reachable module.
_VARS = ("HERMES_SESSION_ID", "HERMES_SESSION_KEY", "HERMES_CRON_SESSION")

# Single-session-per-process entrypoints where os.environ writes are correct.
_ALLOWED_PREFIXES = (
    "cli.py",
    "hermes_cli/",
    "tui_gateway/",
    "acp_adapter/",
    "scripts/",
    "tests/",
)

# The ONLY sanctioned per-session os.environ writes are:
#   (a) the gated write in set_current_session_id — guarded by a
#       `if ... _HERMES_GATEWAY ... != "1"` check on a nearby preceding line, and
#   (b) the import-`except` fallbacks in agent init/compression — which only
#       fire if importing the (gateway-aware) set_current_session_id fails.
# We allowlist these PER-LINE (not per-file) so a future *ungated* write in any
# of these files — e.g. someone removing the `not _HERMES_GATEWAY` guard — is
# still caught (Greptile P2).
_GATEWAY_GUARD_RE = re.compile(r"_HERMES_GATEWAY")
_IMPORT_FALLBACK_FILES = (
    "agent/agent_init.py",
    "agent/conversation_compression.py",
)

_WRITE_RE = re.compile(
    r"""os\.environ\[\s*['"](HERMES_SESSION_ID|HERMES_SESSION_KEY|HERMES_CRON_SESSION)['"]\s*\]\s*="""
)


def _is_allowed(rel: str) -> bool:
    return any(rel == p or rel.startswith(p) for p in _ALLOWED_PREFIXES)


def _write_is_sanctioned(lines: list, idx: int, rel: str) -> bool:
    """A per-session os.environ write at 0-based line ``idx`` is sanctioned iff:
      (a) it is guarded by an ``_HERMES_GATEWAY`` check within the preceding 3
          lines (the gateway-aware setter), OR
      (b) it sits inside an ``except`` block in the import-fallback files.
    """
    window = "\n".join(lines[max(0, idx - 3): idx + 1])
    if _GATEWAY_GUARD_RE.search(window):
        return True
    if rel in _IMPORT_FALLBACK_FILES:
        # Allow only when an `except` appears in the preceding few lines (the
        # import-failure fallback), not an arbitrary unguarded write.
        preceding = "\n".join(lines[max(0, idx - 4): idx])
        if re.search(r"^\s*except\b", preceding, re.MULTILINE):
            return True
    return False


def test_no_gateway_session_env_writes():
    violations = []
    for path in REPO.rglob("*.py"):
        rel = str(path.relative_to(REPO))
        if _is_allowed(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.split("\n")
        for m in _WRITE_RE.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            if _write_is_sanctioned(lines, line_no - 1, rel):
                continue
            violations.append(f"{rel}:{line_no}: {m.group(0)}")
    assert not violations, (
        "Gateway-reachable per-session os.environ write(s) found (v3-latch bug "
        "class). Use the per-turn contextvar (set_session_vars/set_cron_session), "
        "or guard the write with `if ... _HERMES_GATEWAY ... != '1'` if it is a "
        "genuine CLI/cron/worker fallback:\n  " + "\n  ".join(violations)
    )
