#!/usr/bin/env python3
"""Acceptance check for t_000bf106: a deliberate import error in the cron
dispatch path produces ONE clear diagnostic naming the module and traceback.

Reproduces the 2026-10-24 02:16Z incident shape: a REQUIRED field appended to
an existing NamedTuple in ``agent/auxiliary_client.py``, then asks the smoke
probe what the cron dispatch path would do.

Run it with the repo's own interpreter so the probe's subprocess uses the SAME
python the cron worker uses:

    venv/bin/python3 scripts/verify_cron_import_gate.py
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
# The managed environment's site-packages must be on sys.path before any
# first-party import — the same boot the cron worker does.
os.environ.pop("PYTHONHOME", None)
os.environ.pop("PYTHONPATH", None)
sys.path.insert(0, str(REPO))
try:
    import hermes_bootstrap  # noqa: F401
except Exception as exc:  # health: allow BLE001 -- bootstrap may raise anything; the probe reports the real reason
    # The probe child does the same thing; a bootstrap that fails here would
    # fail there too, and the probe reports the real reason. Say so rather
    # than swallow it: a silent pass hides the failure the report is about.
    print(
        f"warning: hermes_bootstrap did not load ({type(exc).__name__}: {exc}); "
        "the import probe will report the real reason",
        file=sys.stderr,
    )

from cron import import_smoke  # noqa: E402

# The first-party modules the chain needs from the repo root.
ROOT_MODULES = ("hermes_yaml.py", "hermes_constants.py", "hermes_logging.py")


def _make_broken_tree(dest: pathlib.Path) -> pathlib.Path:
    """Copy the tree, then break ``agent/auxiliary_client.py`` like the incident did.

    The incident's break was an IMPORT-TIME failure, so that is what this
    reproduces: the class-based ``_LadderRoute`` is rewritten into the
    FUNCTIONAL ``NamedTuple("Name", [...])`` form with a ``defaults=``
    argument. ``typing.NamedTuple`` rejects that combination at import time
    ("Either list of fields or keywords can be provided to NamedTuple, not
    both") — which is how one line took down 17 cron jobs at once.
    """
    for name in ("gateway", "agent", "cron", "hermes_cli"):
        shutil.copytree(
            REPO / name, dest / name, ignore=shutil.ignore_patterns("__pycache__"),
        )
    for name in ROOT_MODULES:
        source = REPO / name
        if source.exists():
            shutil.copy2(source, dest / name)
    target = dest / "agent" / "auxiliary_client.py"
    text = target.read_text(encoding="utf-8-sig")
    marker = "class _LadderRoute(NamedTuple):"
    if marker not in text:
        raise SystemExit("marker not found: _LadderRoute is not a class-based NamedTuple")
    # The functional form with `defaults=` — the incident's exact import-time
    # TypeError. The field list keeps the real fields so the module is a
    # faithful reproduction, not a synthetic stub. The fields are read with
    # the AST: line parsing trips over the comments between them.
    fields = _class_fields_by_ast(target)
    if not fields:
        raise SystemExit("could not read _LadderRoute's fields")
    field_literal = ", ".join(f'("{n}", {a})' for n, a in fields)
    functional = (
        f'_LadderRoute = NamedTuple("_LadderRoute", [{field_literal}], defaults=(None,))\n'
    )
    # Replace the class block with the functional form.
    class_start = text.index(marker)
    after_class = text.index("\n\n", class_start)
    target.write_text(
        text[:class_start] + functional + text[after_class:],
        encoding="utf-8",
    )
    return target


def _class_fields_by_ast(path: pathlib.Path) -> list[tuple[str, str]]:
    """``(name, annotation_source)`` for each field of ``_LadderRoute``."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_LadderRoute":
            out = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    annotation = ast.unparse(stmt.annotation)
                    out.append((stmt.target.id, annotation))
            return out
    return []


def main() -> int:
    failures: list[str] = []

    print("=" * 72)
    print("1. The REAL tree: the probe must report the chain healthy.")
    print("=" * 72)
    healthy = import_smoke.check_import_chain(force=True)
    print(f"   ok={healthy.ok} failing_module={healthy.failing_module}")
    if not healthy.ok:
        failures.append(
            f"the live tree failed its own probe: {healthy.failing_module} "
            f"{healthy.error_type}: {healthy.error_message}"
        )

    print()
    print("=" * 72)
    print("2. A BROKEN tree: the probe must name the module, the cause, and")
    print("   carry the traceback — ONE diagnostic, not N job failures.")
    print("=" * 72)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cron-import-gate-"))
    try:
        broken_file = _make_broken_tree(tmp)
        print(f"   broke: {broken_file}")
        result = import_smoke._run_probe(import_smoke.SMOKE_CHAIN, tmp)
        print(f"   ok={result.ok}")
        print(f"   failing_module={result.failing_module}")
        print(f"   error_type={result.error_type}")
        print(f"   error_message={result.error_message}")
        print(f"   traceback carried: {bool(result.traceback)}")
        print()
        print("   --- the ONE diagnostic an operator would see ---")
        print(f"   {result.diagnostic()}")
        print()
        if result.ok:
            failures.append("the probe did not detect the broken tree")
        else:
            if result.failing_module != "agent.auxiliary_client":
                failures.append(
                    f"wrong failing module: {result.failing_module!r} "
                    f"(expected 'agent.auxiliary_client')"
                )
            if "NamedTuple" not in (result.error_message or ""):
                failures.append(
                    f"the error message does not name the cause: {result.error_message!r}"
                )
            if result.error_type != "TypeError":
                failures.append(
                    f"wrong error type: {result.error_type!r} (expected 'TypeError')"
                )
            if not result.traceback:
                failures.append("the traceback was not carried")
            elif "auxiliary_client.py" not in result.traceback:
                failures.append("the traceback does not name the broken file")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASSED: a deliberate import error produces ONE diagnostic naming the")
    print("        module and the traceback, instead of N identical job failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
