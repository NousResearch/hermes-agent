"""Session binding for native compressor cooldowns and transcript-shape guards."""


import logging

logger = logging.getLogger(__name__)


def _same_lineage(compressor, session_db, session_id, parent_session_id):
    if session_db is not compressor._session_db or not session_id:
        return False
    if session_id == compressor._session_id:
        return True
    if not parent_session_id or parent_session_id != compressor._session_id or session_db is None:
        return False
    try:
        row = session_db.get_session(session_id)
        return bool(row and row.get("parent_session_id") == parent_session_id)
    except Exception:
        logger.debug("Could not prove compression lineage; dropping structural backoff", exc_info=True)
        return False


def bind_session_state(compressor, session_db, session_id, *, parent_session_id=None):
    preserve_structural = _same_lineage(compressor, session_db, session_id, parent_session_id)
    compressor._session_db = session_db
    compressor._session_id = session_id or ""
    compressor._summary_failure_cooldown_until = 0.0
    compressor._cooldown_persist_failed = False
    compressor._last_summary_error = None
    compressor._consecutive_timeout_failures = compressor._consecutive_truncation_failures = compressor._fallback_compression_streak = 0
    compressor._consecutive_overload_aborts = 0
    compressor._ineffective_compression_count = compressor._prellm_skip_count = 0
    compressor._anti_thrash_recovery_deadline = compressor._fallback_probe_at = 0.0
    if not preserve_structural:
        compressor._structural_no_op_backoff_until = 0.0
        compressor._structural_no_op_len = 0
    compressor._reset_proactive_prune_rearm()
    compressor.get_active_compression_failure_cooldown()
    compressor._load_fallback_compression_streak()
    compressor._load_ineffective_compression_count()
    compressor._load_anti_thrash_recovery_deadline()
    compressor._load_fallback_probe_at()
    compressor._load_consecutive_overload_aborts()
    compressor._load_proactive_prune_rearm_tokens()

