from __future__ import annotations

import os

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.calendar_audit import render_calendar_alignment_audit
from hermes_cli.signal_coo.calendar_sync import (
    calendar_alignment_sync_needs_attention,
    render_calendar_alignment_sync,
    sync_calendar_alignment_blocks,
)
from hermes_cli.signal_coo.google_evidence import collect_google_ea_evidence, write_json_artifact


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    dry_run = _truthy(os.getenv("TORBEN_CALENDAR_ALIGNMENT_DRY_RUN"))
    max_mutations = int(os.getenv("TORBEN_CALENDAR_ALIGNMENT_MAX_MUTATIONS", "60"))
    payload = collect_google_ea_evidence(
        config_path=home / "config" / "google_accounts.yaml",
        days=21,
        max_calendar_events=80,
        max_email_messages=0,
        max_calendar_block_candidates=25,
        include_secondary_calendars=False,
    )
    output_path = state_dir / "torben-calendar-alignment-audit-latest.json"
    report_path = state_dir / "torben-calendar-alignment-audit-latest.txt"
    write_json_artifact(payload, output_path)
    report = render_calendar_alignment_audit(payload)
    report_path.write_text(report, encoding="utf-8")
    candidates = list(((payload.get("ea") or {}).get("calendar_block_candidates") or []))
    if not candidates:
        return 0
    sync = sync_calendar_alignment_blocks(
        config_path=home / "config" / "google_accounts.yaml",
        candidates=candidates,
        dry_run=dry_run,
        max_mutations=max_mutations,
    )
    payload.setdefault("ea", {})["calendar_alignment_sync"] = sync
    sync_output_path = state_dir / "torben-calendar-alignment-sync-latest.json"
    sync_report_path = state_dir / "torben-calendar-alignment-sync-latest.txt"
    write_json_artifact(payload, sync_output_path)
    sync_report = render_calendar_alignment_sync(payload)
    sync_report_path.write_text(sync_report, encoding="utf-8")
    if not calendar_alignment_sync_needs_attention(sync):
        return 0
    print(sync_report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
