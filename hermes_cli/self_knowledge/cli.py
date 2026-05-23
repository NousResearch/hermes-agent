"""CLI plumbing for `hermes self-knowledge`."""

from __future__ import annotations

from pathlib import Path

from hermes_cli.self_knowledge.drift import ALLOWLIST_PATH, DOC_PATH, PROJECT_ROOT, check_drift
from hermes_cli.self_knowledge.renderer import refresh_self_knowledge, render_self_knowledge


def run_self_knowledge_command(
    *,
    render: bool,
    refresh: bool,
    check: bool,
    strict: bool,
    doc_path: Path = DOC_PATH,
    project_root: Path = PROJECT_ROOT,
    allowlist_path: Path = ALLOWLIST_PATH,
) -> int:
    """Run the requested self-knowledge operation and return a process code."""
    if render:
        print(render_self_knowledge(doc_path))
    if refresh:
        changed = refresh_self_knowledge(doc_path)
        print("self-knowledge refreshed" if changed else "self-knowledge already current")
    if check:
        findings = check_drift(
            doc_path,
            project_root=project_root,
            allowlist_path=allowlist_path,
        )
        if not findings:
            print("self-knowledge drift check passed")
        else:
            print(f"self-knowledge drift findings: {len(findings)}")
            for finding in findings:
                print(
                    f"- {finding.kind}: {finding.reference} at "
                    f"{finding.location_in_doc} — {finding.reason}"
                )
            if strict:
                return 1
    if not any((render, refresh, check)):
        print(render_self_knowledge(doc_path))
    return 0


def _cmd_self_knowledge(args) -> None:
    code = run_self_knowledge_command(
        render=bool(args.render),
        refresh=bool(args.refresh),
        check=bool(args.check),
        strict=bool(args.strict),
    )
    if code:
        raise SystemExit(code)


def configure_parser(parser) -> None:
    """Configure an argparse parser for the self-knowledge command."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--render", action="store_true", help="Render the self-knowledge doc to stdout")
    group.add_argument("--refresh", action="store_true", help="Refresh the self-knowledge doc on disk")
    group.add_argument("--check", action="store_true", help="Check hand-written sections for drift")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when --check finds drift")
    parser.set_defaults(func=_cmd_self_knowledge)
