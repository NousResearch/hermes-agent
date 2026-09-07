#!/usr/bin/env python3
"""Generate the packaged update scanner from its single source module.

``hermes_cli/_scan_venv_blockers.py`` is the source of truth.  The Desktop
ships ``apps/desktop/resources/update-scanner/scan-venv-blockers.py`` (the
"carrier") and runs it with ``python -I <carrier> --root <root>`` under the
target venv interpreter, so the scan bytes come from the candidate build and
never from the mutable checkout being updated.

The two files used to be hand-synced, which is how they drifted by 160 lines
and how the carrier ended up stubbing ``_redact_sensitive_cmdline`` to a
blanket ``"<redacted>"`` that silently disabled holder classification on the
only build that ships (#104687 B3/B5).  The carrier is now derived from the
module by the explicit, enumerated substitutions in ``SUBSTITUTIONS`` below.

Usage::

    python scripts/desktop-update/sync_scan_venv_blockers.py --check
    python scripts/desktop-update/sync_scan_venv_blockers.py --write

``--check`` exits non-zero when the committed carrier differs from the
generated output and prints the diff.  ``--write`` regenerates the carrier.
``tests/hermes_cli/test_scan_venv_blockers_parity.py`` runs the check
in-process so ordinary Python CI enforces parity with no extra workflow.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "hermes_cli" / "_scan_venv_blockers.py"
CARRIER_PATH = (
    REPO_ROOT / "apps" / "desktop" / "resources" / "update-scanner" / "scan-venv-blockers.py"
)


@dataclass(frozen=True)
class Substitution:
    """One deliberate difference between the module copy and the carrier."""

    name: str
    why: str
    old: str
    new: str


_MODULE_DOCSTRING = '''"""Strict, target-root process scanner for safe native-Windows updates.

``python -m hermes_cli._scan_venv_blockers --root <install>`` writes exactly
one JSON document (schema v2) to stdout. Valid clear and blocked scans exit
zero for backwards compatibility with the Desktop probe. Invalid roots and
probe failures exit one and are fail-closed.

Two copies of this scanner exist on purpose. The Desktop never runs this
module: it ships ``apps/desktop/resources/update-scanner/scan-venv-blockers.py``
(the "carrier") and runs it with ``python -I <carrier> --root <root>`` under
the target venv interpreter, so the scan bytes come from the candidate build
and never from the mutable checkout being updated.

This file is the single source. The carrier is *generated* from it by
``scripts/desktop-update/sync_scan_venv_blockers.py``, whose ``SUBSTITUTIONS``
list is the authoritative description of how the two copies differ; never
hand-edit the carrier. ``tests/hermes_cli/test_scan_venv_blockers_parity.py``
fails when the committed carrier is not what that generator produces.
"""
'''

_CARRIER_DOCSTRING = '''"""Candidate-owned target-root scanner for safe native-Windows updates.

GENERATED FILE -- do not edit. Regenerate with::

    python scripts/desktop-update/sync_scan_venv_blockers.py --write

Source: ``hermes_cli/_scan_venv_blockers.py``.  Parity is enforced by
``tests/hermes_cli/test_scan_venv_blockers_parity.py``.

This is ``hermes_cli/_scan_venv_blockers.py`` as shipped inside the Desktop
build (``extraResources``). The Desktop runs it with ``python -I <this file>
--root <install>`` under the target venv interpreter so the scan logic comes
from the candidate build, never from the mutable checkout being updated; it
therefore imports nothing from that checkout. Exactly six deliberate
substitutions separate it from the module copy:

* this docstring;
* the MCP argv classifier is inlined instead of imported from
  ``hermes_mcp_update_gate``;
* ``_validated_root``'s docstring drops the provenance wording;
* ``_validated_root`` does not require the scanner to live inside the target
  root (the carrier lives in the Desktop's resources);
* ``_updater_owned_backend_entry`` never consults the checkout's spawn ledger,
  so no serve/dashboard backend is deferred and ``deferred_backend_evidence``
  is always empty;
* ``venv_bin_dir`` is not imported from ``hermes_constants``; the Windows
  ``Scripts`` layout is spelled out instead.

Everything else is byte-identical to the module copy. In particular
``_redact_sensitive_cmdline`` is *not* substituted: it is self-contained in
both copies, so the packaged scanner reports the same diagnostic command line
the module does rather than a blanket ``"<redacted>"``.

The external interface always writes exactly one JSON document to stdout.
Valid clear and blocked scans exit zero. Invalid roots and probe failures exit
one and are fail-closed.
"""
'''

_MCP_IMPORT = "from hermes_mcp_update_gate import is_exact_mcp_module_argv\n"

_MCP_INLINE = '''MCP_MAIN_MODULE = "agent.transports.hermes_tools_mcp_server"


def is_exact_mcp_module_argv(argv: Sequence[str]) -> bool:
    """True only for an argument-free ``-m <MCP_MAIN_MODULE>`` launch."""
    parts = [str(part) for part in argv]
    indexes = [index for index, token in enumerate(parts) if token == "-m"]
    if not (
        parts
        and len(indexes) == 1
        and indexes[0] + 2 == len(parts)
        and parts[indexes[0] + 1] == MCP_MAIN_MODULE
    ):
        return False

    # ``-m`` must be the interpreter's operative action, not text after a
    # script name.  Accept Python's ordinary pre-module runtime switches, but
    # reject ``-c``, ``--``, unknown switches, and non-option operands.  This
    # is intentionally smaller than the complete Python CLI grammar: failure
    # to recognize an exotic launch only makes an updater refuse to kill it.
    before_module = parts[1 : indexes[0]]
    argument_options = {"-W", "-X", "--check-hash-based-pycs"}
    long_flags = {
        "--bytes-warning",
        "--debug",
        "--dont-write-bytecode",
        "--help-env",
        "--help-xoptions",
        "--ignore-environment",
        "--inspect",
        "--isolated",
        "--no-site",
        "--no-user-site",
        "--safe-path",
        "--unbuffered",
        "--verbose",
    }
    short_flag_chars = frozenset("bBdEIiOPqRsStuUvx")
    index = 0
    while index < len(before_module):
        token = before_module[index]
        if token in argument_options:
            index += 1
            if index >= len(before_module) or before_module[index].startswith("-"):
                return False
        elif token.startswith("-W") and token != "-W":
            pass
        elif token.startswith("-X") and token != "-X":
            pass
        elif token.startswith("--check-hash-based-pycs="):
            pass
        elif token in long_flags:
            pass
        elif token.startswith("-") and not token.startswith("--"):
            if not token[1:] or any(char not in short_flag_chars for char in token[1:]):
                return False
        else:
            return False
        index += 1
    return True
'''

_VALIDATED_ROOT_PROVENANCE = '''    code_root = Path(__file__).resolve().parents[1]
    if _canonical(code_root) != _canonical(root):
        raise ValueError("scanner was not imported from the target root")
'''

_LEDGER_BODY = '''    """Ledger entry for a deferred serve/dashboard backend, or ``None`` when it must block.

    Returning the entry lets ``main()`` emit sanitized decision evidence — structured identity
    fields only, never argv, which can carry tokens or private endpoints.

    See #98350.
    """
    try:
        from hermes_cli.update_cmd import _hermes_holder_subcommand  # noqa: PLC0415

        purpose = _hermes_holder_subcommand(cmdline)
    except Exception:
        return None
    if purpose not in _UPDATER_STOPPABLE_PURPOSES:
        return None
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead  # noqa: PLC0415

        entries = ledger_entries()
    except Exception:
        return None
    for entry in entries:
        if entry.get("pid") != pid:
            continue
        if entry.get("purpose") not in _UPDATER_STOPPABLE_PURPOSES:
            return None
        # Spawner dead, unrecorded, or unprovable-but-registered: the updater's ledger rungs own
        # this holder (reap or stop+relaunch).
        if spawner_is_dead(entry) is not False or _spawner_is_this_handoff_desktop(entry):
            return entry
        return None
    return None
'''

_LEDGER_STUB = '''    """Candidate scan cannot trust a mutable target checkout's ownership ledger."""
    return None
'''

_VENV_BIN_DIR_IMPORT = '''        from hermes_constants import venv_bin_dir
        expected_hermes = venv_bin_dir(venv_dir, windows=True) / "hermes.exe"
        expected_python = venv_bin_dir(venv_dir, windows=True) / "python.exe"
'''

_VENV_BIN_DIR_INLINE = '''        expected_hermes = venv_dir / "Scripts" / "hermes.exe"
        expected_python = venv_dir / "Scripts" / "python.exe"
'''


# Order is not significant (every ``old`` must appear exactly once), but this
# list is the reviewable contract: anything not listed here is byte-identical
# in both copies.
SUBSTITUTIONS: tuple[Substitution, ...] = (
    Substitution(
        name="module-docstring",
        why="the carrier documents its own provenance and generation, not the module's",
        old=_MODULE_DOCSTRING,
        new=_CARRIER_DOCSTRING,
    ),
    Substitution(
        name="inline-mcp-argv-classifier",
        why=(
            "the carrier must not import hermes_mcp_update_gate from the mutable "
            "target checkout, so the exact-argv classifier is inlined"
        ),
        old=_MCP_IMPORT,
        new=_MCP_INLINE,
    ),
    Substitution(
        name="validated-root-docstring",
        why="the carrier validates the target root only; it has no code-root provenance to check",
        old='    """Validate both the requested install and this scanner\'s provenance."""\n',
        new='    """Validate the explicit target root and target-venv provenance."""\n',
    ),
    Substitution(
        name="validated-root-provenance-check",
        why=(
            "the carrier lives in the Desktop's resources, not under the target root, "
            "so requiring it to be imported from the target root would always fail"
        ),
        old=_VALIDATED_ROOT_PROVENANCE,
        new="",
    ),
    Substitution(
        name="updater-owned-backend-entry",
        why=(
            "the candidate scan cannot trust the target checkout's spawn ledger, so no "
            "serve/dashboard backend is deferred and deferred_backend_evidence stays empty"
        ),
        old=_LEDGER_BODY,
        new=_LEDGER_STUB,
    ),
    Substitution(
        name="venv-bin-dir",
        why="the carrier must not import hermes_constants; Windows is the only carrier platform",
        old=_VENV_BIN_DIR_IMPORT,
        new=_VENV_BIN_DIR_INLINE,
    ),
)


def render_carrier(source_text: str) -> str:
    """Apply every substitution to *source_text*; each must match exactly once."""
    rendered = source_text

    for substitution in SUBSTITUTIONS:
        occurrences = rendered.count(substitution.old)

        if occurrences != 1:
            raise SystemExit(
                f"substitution {substitution.name!r} matched {occurrences} times in "
                f"{SOURCE_PATH.name} (expected exactly 1). The source moved; update "
                f"{Path(__file__).name}."
            )

        rendered = rendered.replace(substitution.old, substitution.new, 1)

    return rendered


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_text(stream, text: str) -> None:
    """Write *text* to *stream*, degrading unencodable characters instead of raising."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))


def carrier_drift() -> str:
    """Unified diff of committed carrier vs. generated carrier; empty when in sync."""
    expected = render_carrier(_read(SOURCE_PATH))
    actual = _read(CARRIER_PATH)

    if expected == actual:
        return ""

    return "".join(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"{CARRIER_PATH.name} (committed)",
            tofile=f"{CARRIER_PATH.name} (generated)",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="fail when the carrier is stale")
    group.add_argument("--write", action="store_true", help="regenerate the carrier")
    args = parser.parse_args(argv)

    if args.write:
        _write(CARRIER_PATH, render_carrier(_read(SOURCE_PATH)))
        print(f"wrote {CARRIER_PATH}")

        return 0

    drift = carrier_drift()

    if drift:
        # The scanner source contains non-cp1252 characters; a legacy Windows
        # console must not turn a stale-carrier report into a traceback.
        _write_text(sys.stdout, drift)
        print(
            f"\n{CARRIER_PATH} is stale. Run: "
            f"python scripts/desktop-update/sync_scan_venv_blockers.py --write",
            file=sys.stderr,
        )

        return 1

    print(f"{CARRIER_PATH.name} matches {SOURCE_PATH.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
