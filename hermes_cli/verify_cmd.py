"""``hermes verify`` — detect a project's run recipe and smoke-test it."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def run_verify_command(args) -> int:
    from agent.verify import load_or_detect, manifest_path, run_verify, save_manifest

    root = Path(getattr(args, "path", None) or ".").resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    recipe, source = load_or_detect(root)
    if recipe is None:
        if args.json:
            print(json.dumps({"ok": False, "error": "no-recipe", "root": str(root)}))
        else:
            print(
                f"No recognizable project found at {root}.\n"
                f"Create {manifest_path(root)} to define a recipe manually.",
                file=sys.stderr,
            )
        return 1

    if source == "detected":
        _merge_project_facts_commands(root, recipe)

    if args.port:
        recipe.port = args.port

    if args.save:
        path = save_manifest(root, recipe)
        if not args.json:
            print(f"Saved manifest: {path}")

    if args.detect_only:
        print(json.dumps({"source": source, "recipe": recipe.to_dict()}, indent=None if args.json else 2))
        return 0

    phases = tuple(args.phase) if args.phase else None
    result = run_verify(
        root, recipe, phases=phases, phase_timeout=args.timeout, ready_timeout=args.ready_timeout,
        skip_start=args.skip_start, port_override=args.port,
    )

    _record_evidence(root, recipe, result, partial=bool(phases or args.skip_start))

    if args.json:
        payload = result.to_dict()
        payload["source"] = source
        print(json.dumps(payload))
        return 0 if result.ok else 1

    _print_human_report(recipe, source, result)
    return 0 if result.ok else 1


def _merge_project_facts_commands(root: Path, recipe) -> None:
    """Fold ``detect_project_facts`` verify commands into a detected recipe (best-effort union).

    Never applied to a saved manifest — the user-edited manifest is the source of truth.
    """
    try:
        from agent.coding_context import detect_project_facts

        facts_commands = list(detect_project_facts(root).verify_commands)
    except Exception:
        return
    existing = {c.strip() for c in (*recipe.bootstrap, *recipe.build, *recipe.test) if c}
    for command in facts_commands:
        command = command.strip()
        if command and command not in existing:
            recipe.test.append(command)
            existing.add(command)


def _record_evidence(root: Path, recipe, result, *, partial: bool) -> None:
    """Record the completed run into the verification evidence ledger.

    Fail-silent: a ledger problem must never change the CLI's exit code or output. ``partial``
    (a ``--phase`` subset or ``--skip-start``) downgrades the scope to ``targeted`` so a partial
    pass is never presented as a full workspace green.
    """
    try:
        from agent.verification_evidence import record_verify_run

        tails = [f"[{p.phase}] {p.command}\n{p.output_tail}" for p in result.phases if p.output_tail]
        if result.readiness is not None:
            tails.append(f"[start] {recipe.start} -> {_readiness_status(result.readiness)}")
        record_verify_run(
            root=root,
            session_id=os.environ.get("HERMES_SESSION_ID"),
            ok=result.ok,
            command="hermes verify",
            scope="targeted" if partial else "full",
            output="\n".join(tails),
        )
    except Exception:
        pass


def _readiness_status(r) -> str:
    return f"ready (HTTP {r.status_code})" if r.ready else f"not ready ({r.error or 'timeout'})"


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
        print(f"  {'start'.ljust(max(5, max((len(p.phase) for p in result.phases), default=5)))}  "
              f"{'PASS' if r.ready else 'FAIL':<7}  {r.duration:6.1f}s  {recipe.start}")
        print()
        print(f"Readiness: {r.url} -> {_readiness_status(r)}")

    failed = [p for p in result.phases if not p.ok]
    print()
    print(f"Result: {'OK' if result.ok else 'FAILED'}")
    for p in failed:
        print(f"\n--- output tail: {p.phase} ({p.command}) ---")
        print(p.output_tail.rstrip())
    if result.readiness is not None and not result.readiness.ready and result.readiness.output_tail:
        print("\n--- output tail: start ---")
        print(result.readiness.output_tail.rstrip())
