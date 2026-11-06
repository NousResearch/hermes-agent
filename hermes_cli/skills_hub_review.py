"""Explicit local operator review for Skills Hub packages refused by scan policy."""

import sys
from pathlib import Path


def _confirm_review(console, path: Path, bundle, result, skip_confirm: bool) -> str | None:
    from tools.skills_guard import _content_digest

    if skip_confirm or not sys.stdin.isatty() or not console.is_terminal:
        console.print("Review requires a local interactive terminal without --yes.", markup=False)
        return None
    expected = "sha256:" + _content_digest(path)
    if expected != result.scan_provenance.get("bundle_hash"):
        console.print("Package changed after scanning; fetch and review it again.", markup=False)
        return None
    console.print("The scanner refused this package. Review all scripts and references before approving.", markup=False)
    console.print(f"Package: {bundle.identifier} ({bundle.trust_level})", markup=False)
    console.print(f"Review files at: {path}", markup=False)
    console.print(f"Exact package: {expected}", markup=False)
    console.print("Approval permits this installation only. It does not approve later updates or execute setup.", markup=False)
    try:
        answer = console.input("Type 'approve ' followed by the full package hash, or press Enter to cancel: ")
    except (EOFError, KeyboardInterrupt):
        return None
    return expected if answer.strip() == f"approve {expected}" else None


def scan_for_install(console, path, bundle, meta, identifier, *, force=False, review=False, skip_confirm=False):
    from hermes_cli import skills_hub as cli_hub
    from tools.skills_guard import should_allow_install

    result = cli_hub._scan_quarantined(console, path, bundle, meta, identifier)
    allowed, _reason = should_allow_install(result, force=force)
    if allowed:
        return result
    approved = _confirm_review(console, path, bundle, result, skip_confirm) if review else None
    if approved:
        result.scan_provenance = {
            **result.scan_provenance,
            "operator_review": {"bundle_hash": approved, "method": "local_terminal"},
        }
        return result
    message = ("Package review was not approved; nothing installed." if review
               else cli_hub._scan_block_message(result, identifier))
    cli_hub._install_blocked(console, bundle, message, result.verdict,
                            f"{len(result.findings)}_findings", q_path=path, lead="\n", label="Not installed:")
    return None


def verify_review(path, bundle, result) -> None:
    """A review cannot authorize bytes changed during either confirmation prompt."""
    from tools.skills_guard import _content_digest
    from tools.skills_hub import append_audit_log

    review = getattr(result, "scan_provenance", {}).get("operator_review")
    if review is None:
        return
    if review["bundle_hash"] != "sha256:" + _content_digest(path):
        raise ValueError("Package changed after operator review; fetch, scan and review it again.")
    append_audit_log("REVIEW_APPROVED", bundle.name, bundle.source, bundle.trust_level,
                     result.verdict, review["bundle_hash"])
