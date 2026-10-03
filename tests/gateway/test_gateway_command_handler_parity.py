"""Invariant: every gateway-advertised slash command has a real dispatch handler.

Regression guard for the silent-orphan class: a CommandDef lands in
COMMAND_REGISTRY without cli_only=True, so it is auto-derived into
GATEWAY_KNOWN_COMMANDS and auto-printed by gateway_help_lines() -- but nobody
adds a `canonical == "<name>"` branch in gateway/run.py.

Failure mode is SILENT: the known-command gate in gateway/run.py suppresses the
"Unknown command" notice (the name IS known), so the text falls through to the
LLM as a plain user turn. The user sees the command advertised in /help, types
it, and gets a hallucinated conversational reply instead of the feature.
"""

from __future__ import annotations

import re
from pathlib import Path

from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command


def _gateway_run() -> Path:
    """Locate gateway/run.py relative to the imported hermes_cli package.

    Anchored on the package rather than a fixed parent depth so the test reads
    the same file whether it lives in tests/gateway/ or a scratch dir.
    """
    import hermes_cli

    pkg = getattr(hermes_cli, "__file__", None)
    assert pkg, "hermes_cli has no __file__; cannot locate the repo root"
    path = Path(pkg).resolve().parent.parent / "gateway" / "run.py"
    assert path.is_file(), f"gateway/run.py not found next to hermes_cli: {path}"
    return path


# Commands that legitimately have no `canonical == "x"` branch because they are
# dispatched by another documented mechanism. Add here ONLY with a reason.
_DISPATCHED_ELSEWHERE: dict[str, str] = {}


def _handled_canonicals(source: str) -> set[str]:
    """Canonical names gateway/run.py actually dispatches on.

    gateway/run.py routes slash commands through TWO mechanisms, and both must
    be recognized or this guard manufactures false orphans:

    1. ``canonical == "x"`` / ``canonical in (...)`` branches (the original
       shape), and
    2. a ``{"x": self._handle_x_command, ...}`` dispatch TABLE mapping the
       canonical name to a bound handler.

    Matching only (1) reported 21 false orphans (/status, /btw, /help, ...)
    once upstream migrated the bulk of dispatch to the table — which would
    drown the single real orphan this test exists to catch.
    """
    names = set(re.findall(r"""canonical\s*==\s*["']([a-z0-9_-]+)["']""", source))
    for group in re.findall(r"canonical\s+in\s*[\(\{]([^)\}]*)[\)\}]", source):
        for tok in re.findall(r"""["']([a-z0-9_-]+)["']""", group):
            names.add(tok)
    # Dispatch-table entries: "name": self._handle_...  (any bound callable).
    names |= set(
        re.findall(r"""["']([a-z0-9_-]+)["']\s*:\s*self\.[A-Za-z_][A-Za-z0-9_]*""", source)
    )
    return names


def test_every_gateway_known_command_has_a_dispatch_handler():
    source = _gateway_run().read_text(encoding="utf-8")
    handled = _handled_canonicals(source)

    orphans = []
    for name in sorted(GATEWAY_KNOWN_COMMANDS):
        cmd = resolve_command(name)
        if cmd is None or cmd.name != name:
            continue  # alias; canonical is checked on its own pass
        if cmd.name in _DISPATCHED_ELSEWHERE:
            continue
        if cmd.name not in handled:
            orphans.append(cmd.name)

    assert not orphans, (
        "Gateway-advertised commands with no dispatch handler in gateway/run.py: "
        + ", ".join("/" + o for o in orphans)
        + ". These pass the GATEWAY_KNOWN_COMMANDS gate (so the user gets NO "
        "'Unknown command' notice) and fall through to the LLM as raw text. "
        "Either add a `canonical == \"<name>\"` branch, mark the CommandDef "
        "cli_only=True, or register it in _DISPATCHED_ELSEWHERE with a reason."
    )


def test_harness_detects_a_synthetic_orphan():
    """Negative probe: prove the parser would actually catch an orphan."""
    handled = _handled_canonicals('if canonical == "model":\n    pass\n')
    assert "model" in handled
    assert "definitely-not-a-command" not in handled


def test_harness_recognizes_the_dispatch_table_mechanism():
    """Positive control for mechanism 2 — a table entry counts as handled.

    Without this the guard false-orphans every table-dispatched command.
    """
    handled = _handled_canonicals('{\n  "btw": self._handle_btw_command,\n}\n')
    assert "btw" in handled
    assert "not-in-the-table" not in handled
