"""The backend's copies of main's code that carry a code-health waiver.

A waiver (``# health: allow <RULES> -- <reason>``) on a copied line waives a finding main's own line has. ``COPIES`` names every such copy: the backend module and function, the rules of its waivers in line order, and the main module and function it copies. ``waived_functions`` names the function holding each waiver of a module, and ``strip_moved_allows`` drops the trailing waiver comments from a module's text.
"""

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_WAIVER = re.compile(r"# health: allow (?P<rules>[A-Z0-9 ]+) -- ")
_TRAILING_WAIVER = re.compile(r"  # health: allow [A-Z0-9 ]+ -- .*$")


@dataclass(frozen=True)
class WaivedCopy:
    """``function`` of ``backend`` copies ``main_function`` of ``main``; ``rules`` are its waivers' rules in line
    order. Paths are relative to the tree root, functions dotted (``Class.method``)."""

    backend: str
    function: str
    rules: tuple[str, ...]
    main: str
    main_function: str


_HOLDER, _SCHEMA = "persistence/sqlite/_holder.py", "persistence/sqlite/_schema.py"
_ADMIN, _STORE = "persistence/sqlite/admin/", "persistence/sqlite/conversation/_store/"
_COORDINATION = "persistence/sqlite/authority/coordination.py"
_MAINTENANCE = f"{_ADMIN}_maintenance.py"

COPIES = (
    WaivedCopy(_HOLDER, "_ensure_test_isolation", ("BLE001", "BLE001"), "hermes_state.py", "_ensure_test_isolation"),
    WaivedCopy(_HOLDER, "StoreHolder._close_conn_logged", ("BLE001",), "hermes_state.py",
               "SessionDB._close_conn_logged"),
    WaivedCopy(_HOLDER, "StoreHolder.__init__", ("BLE001 S110",), "hermes_state.py", "SessionDB.__init__"),
    WaivedCopy(_HOLDER, "StoreHolder._execute_write", ("CC", "BLE001 S110"), "hermes_state.py",
               "SessionDB._execute_write"),
    WaivedCopy(_HOLDER, "StoreHolder._try_wal_checkpoint", ("BLE001",), "hermes_state.py",
               "SessionDB._try_wal_checkpoint"),
    WaivedCopy(_HOLDER, "StoreHolder.close", ("BLE001",), "hermes_state.py", "SessionDB.close"),
    WaivedCopy(_HOLDER, "StoreHolder.__del__", ("BLE001 S110",), "hermes_state.py", "SessionDB.__del__"),
    WaivedCopy(_HOLDER, "StoreHolder.queue_token_counts", ("HX012",), "hermes_state_usage.py",
               "SessionUsageMixin.queue_token_counts"),
    WaivedCopy(_HOLDER, "StoreHolder._apply_token_batch", ("BLE001", "BLE001"), "hermes_state_usage.py",
               "SessionUsageMixin._apply_token_batch"),
    WaivedCopy(_SCHEMA, "_run_data_migrations", ("BLE001",), "hermes_state_schema.py",
               "SessionSchemaMixin._run_data_migrations"),
    WaivedCopy(_SCHEMA, "_try_incremental_merge_fts", ("BLE001",), "hermes_state_search.py",
               "SessionSearchMixin._try_incremental_merge_fts"),
    WaivedCopy(_SCHEMA, "_reap_inactive_orphan_desktop_holders", ("BLE001", "BLE001", "BLE001", "BLE001 S110"),
               "hermes_state_gateway.py", "SessionGatewayMixin._reap_inactive_orphan_desktop_holders"),
    WaivedCopy(f"{_ADMIN}_backup.py", "_safe_copy_db", ("BLE001",), "hermes_cli/backup_sqlite.py", "_safe_copy_db"),
    WaivedCopy(f"{_ADMIN}_backup.py", "_query_ro_sqlite", ("BLE001",), "hermes_cli/backup.py", "_query_ro_sqlite"),
    WaivedCopy(f"{_ADMIN}_doctor.py", "DoctorAdmin.integrity_check", ("BLE001",), "gateway/lifecycle_ledger.py",
               "check_state_db_integrity"),
    WaivedCopy(f"{_ADMIN}_lost_and_found.py", "_cli_recover_attempts", ("HX006", "HX006"),
               "hermes_cli/session_lost_and_found.py", "_cli_recover_attempts"),
    WaivedCopy(_MAINTENANCE, "_try_checkpoint", ("BLE001",), "hermes_state_maintenance.py",
               "SessionMaintenanceMixin._try_checkpoint"),
    WaivedCopy(_MAINTENANCE, "_FtsStorageOptimizer._optimize_vacuum", ("BLE001",), "hermes_state_search.py",
               "SessionSearchMixin._optimize_vacuum"),
    WaivedCopy(_MAINTENANCE, "MaintenanceAdmin.vacuum", ("BLE001",), "hermes_state_maintenance.py",
               "SessionMaintenanceMixin.vacuum"),
    WaivedCopy(_MAINTENANCE, "MaintenanceAdmin._page_pragmas", ("BLE001",), "hermes_state_maintenance.py",
               "SessionMaintenanceMixin._page_pragmas"),
    WaivedCopy(_MAINTENANCE, "MaintenanceAdmin.fts_layout_drift", ("BLE001",), "hermes_cli/update_cmd_maint.py",
               "_print_fts_optimize_available_notice"),
    WaivedCopy(f"{_ADMIN}_reconcile.py", "_profile_label_for_home", ("BLE001",), "gateway/status.py",
               "_profile_label_for_home"),
    WaivedCopy(_COORDINATION, "SQLiteCoordinationRepository.get_handoff", ("BLE001",), "hermes_state_gateway.py",
               "SessionGatewayMixin.get_handoff_state"),
    WaivedCopy(_COORDINATION, "SQLiteCoordinationRepository.list_pending_handoffs", ("BLE001",),
               "hermes_state_gateway.py", "SessionGatewayMixin.list_pending_handoffs"),
    WaivedCopy(f"{_STORE}core.py", "_compression_lock_holder_process_is_dead", ("BLE001",), "hermes_state.py",
               "_compression_lock_holder_process_is_dead"),
    WaivedCopy(f"{_STORE}messages.py", "_carry_parent_timestamps", ("BLE001",), "hermes_state_messages.py",
               "SessionMessagesMixin._carry_parent_timestamps"),
    WaivedCopy(f"{_STORE}messages.py", "resolve_resume_session_id", ("BLE001 S110", "BLE001"),
               "hermes_state_messages.py", "SessionMessagesMixin.resolve_resume_session_id"),
    WaivedCopy(f"{_STORE}messages.py", "_rows_to_conversation", ("CC",), "hermes_state_messages.py",
               "SessionMessagesMixin._rows_to_conversation"),
    WaivedCopy(f"{_STORE}search.py", "_describe_search_path", ("BLE001",), "hermes_state_search.py",
               "SessionSearchMixin._describe_search_path"),
    WaivedCopy(f"{_STORE}search.py", "_finalize_search_matches", ("BLE001", "BLE001"), "hermes_state_search.py",
               "SessionSearchMixin._finalize_search_matches"),
    WaivedCopy(f"{_STORE}sessions.py", "_own_profile_name", ("BLE001",), "hermes_state_sessions.py",
               "SessionSessionsMixin._own_profile_name"),
    WaivedCopy(f"{_STORE}sessions.py", "promote_to_session_reset", ("BLE001",), "hermes_state_sessions.py",
               "SessionSessionsMixin.promote_to_session_reset"),
    WaivedCopy(f"{_STORE}sessions.py", "unarchive_recoverable_session", ("BLE001", "BLE001 S110"),
               "hermes_state_sessions.py", "SessionSessionsMixin.unarchive_recoverable_session"),
)


def strip_moved_allows(source: str) -> str:
    """*source* without its trailing waiver comments, which main's lines do not carry."""
    return "\n".join(_TRAILING_WAIVER.sub("", line) for line in source.split("\n"))


def _definitions(tree: ast.AST, prefix: str = ""):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield f"{prefix}{node.name}", node
            yield from _definitions(node, f"{prefix}{node.name}.")
        elif not isinstance(node, ast.expr):
            yield from _definitions(node, prefix)


def _waiver_lines(source: str) -> dict[int, str]:
    """Line number to the rules of each waiver comment in *source*."""
    return {tok.start[0]: m["rules"] for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.COMMENT and (m := _WAIVER.match(tok.string))}


def _start(node: ast.AST) -> int:
    return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))])


def _functions(tree: ast.AST):
    return [(name, node) for name, node in _definitions(tree) if not isinstance(node, ast.ClassDef)]


def waived_functions(path: str) -> set[str]:
    """The innermost function holding each waiver comment of the module at *path*."""
    source = (REPO / path).read_text(encoding="utf-8")
    functions = _functions(ast.parse(source))
    out = set()
    for line in _waiver_lines(source):
        holding = [(name, node) for name, node in functions if _start(node) <= line <= node.end_lineno]
        out.add(max(holding, key=lambda item: _start(item[1]))[0] if holding else "<module>")
    return out
