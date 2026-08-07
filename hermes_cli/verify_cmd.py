"""``hermes verify`` — detect a project's run recipe and smoke-test it.

Scoped port of superagent-ai/grok-cli's verify subsystem entrypoint.
Statically detects the project kind (or loads the saved manifest at
``.hermes/environment.json``), then runs bootstrap/build/test phases and an
optional background start + readiness poll, printing an evidence summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run_verify_command(args) -> int:
    from agent.verify import (
        load_or_detect,
        manifest_path,
        run_verify,
        save_manifest,
    )

    root = Path(getattr(args, "path", None) or ".").resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    recipe, source = load_or_detect(root)
    if recipe is None:
        message = (
            f"No recognizable project found at {root}.\n"
            f"Create {manifest_path(root)} to define a recipe manually."
        )
        if args.json:
            print(json.dumps({"ok": False, "error": "no-recipe", "root": str(root)}))
        else:
            print(message, file=sys.stderr)
        return 1

    if args.port:
        recipe.port = args.port

    if args.save:
        path = save_manifest(root, recipe)
        if not args.json:
            print(f"Saved manifest: {path}")

    if args.detect_only:
        payload = {"source": source, "recipe": recipe.to_dict()}
        print(json.dumps(payload, indent=None if args.json else 2))
        return 0

    phases = None
    if args.phase:
        phases = tuple(args.phase)

    result = run_verify(
        root,
        recipe,
        phases=phases,
        phase_timeout=args.timeout,
        ready_timeout=args.ready_timeout,
        skip_start=args.skip_start,
        port_override=args.port,
    )

    if args.json:
        payload = result.to_dict()
        payload["source"] = source
        print(json.dumps(payload))
        return 0 if result.ok else 1

    _print_human_report(recipe, source, result)
    return 0 if result.ok else 1


def _print_human_report(recipe, source, result) -> None:
    print(f"Recipe: {recipe.name} ({recipe.kind}) — source: {source}")
    print()
    if result.phases:
        width = max(len(p.phase) for p in result.phases)
        for p in result.phases:
            status = "PASS" if p.ok else ("TIMEOUT" if p.timed_out else "FAIL")
            print(f"  {p.phase.ljust(width)}  {status:<7}  {p.duration:6.1f}s  {p.command}")
    else:
        print("  (no phases executed)")

    if result.readiness is not None:
        r = result.readiness
        status = f"ready (HTTP {r.status_code})" if r.ready else f"not ready ({r.error or 'timeout'})"
        print(f"  {'start'.ljust(max(5, max((len(p.phase) for p in result.phases), default=5)))}  "
              f"{'PASS' if r.ready else 'FAIL':<7}  {r.duration:6.1f}s  {recipe.start}")
        print()
        print(f"Readiness: {r.url} -> {status}")

    failed = [p for p in result.phases if not p.ok]
    print()
    print(f"Result: {'OK' if result.ok else 'FAILED'}")
    for p in failed:
        print(f"\n--- output tail: {p.phase} ({p.command}) ---")
        print(p.output_tail.rstrip())
    if result.readiness is not None and not result.readiness.ready and result.readiness.output_tail:
        print("\n--- output tail: start ---")
        print(result.readiness.output_tail.rstrip())
