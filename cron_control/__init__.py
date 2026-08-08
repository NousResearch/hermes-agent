"""Hermes Cron control-plane helpers.

This package provides a read-only control-plane seam over the existing cron
job, execution, delivery, provider, heartbeat, and state-store artifacts.
"""

from .adapters import (
    collect_delivery_evidence,
    collect_dead_man_switch_evidence,
    collect_execution_evidence,
    collect_heartbeat_evidence,
    collect_job_registry_evidence,
    collect_provider_evidence,
    collect_state_store_evidence,
    read_execution_rows,
    read_job_rows,
)
from .normalizer import (
    build_evidence,
    canonical_json,
    evidence_id_for,
    normalize_dead_man_switch_snapshot,
    normalize_delivery_event,
    normalize_execution_row,
    normalize_heartbeat_snapshot,
    normalize_job_row,
    normalize_provider_assessment,
    normalize_state_store_probe,
)
from .shadow import collect_shadow_snapshot, main as shadow_main
from .report import build_shadow_diff_report, main as report_main
from .p5 import load_p5_allowlist, load_p5_dataset, run_p5_canaries
from .evaluator import evaluate_job_verdict, evaluate_snapshot, persist_verdicts
from .actions import execute_verdict_action, execute_verdict_actions, main as actions_main
from .runner import control_plane_settings, run_control_plane_cycle
from .store import (
    append_audit_event,
    default_control_plane_db_path,
    open_control_plane_db,
    acquire_lease,
    record_component_heartbeat,
    record_action,
    record_evidence,
    record_incident,
    record_verdict,
    release_lease,
    update_action,
)
