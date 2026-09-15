"""After the 60/300/900 timeout ladder is exhausted, auto-compress must stay blocked.

Issue #107516: once ``_consecutive_timeout_failures`` reaches
``len(_TIMEOUT_COOLDOWN_LADDER)``, expiring the 900s cooldown must not
re-open automatic compression. Manual ``/compress`` (force / bypass) still
retries. Unrelated to ``compression.max_attempts``.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _TIMEOUT_COOLDOWN_LADDER


def _make_compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="main-model", quiet_mode=True)
    compressor.threshold_tokens = 1_000
    return compressor


def _expire_cooldown(compressor):
    compressor._summary_failure_cooldown_until = 0


def test_third_timeout_giveup_blocks_after_cooldown_expires():
    over_threshold = 50_000
    c = _make_compressor()
    for _ in range(len(_TIMEOUT_COOLDOWN_LADDER)):
        c.record_timeout_failure("aux timed out")
    _expire_cooldown(c)
    should, reason = c.should_compress_info(over_threshold)
    assert should is False
    assert reason and ("giveup" in reason or "timeout" in reason)
    assert "auxiliary.compression.timeout" in (reason or "")

    # CONTROL: one timeout + expired cooldown still allows auto-compress.
    c1 = _make_compressor()
    c1.record_timeout_failure("aux timed out")
    _expire_cooldown(c1)
    assert c1.should_compress_info(over_threshold)[0] is True

    # CONTROL: force=/compress bypass still allowed.
    assert c._automatic_compression_blocked_locally(ignore_cooldown=True) is False
    c._begin_compress_attempt(None, force=True)
    assert c.should_compress_info(over_threshold)[0] is True


def test_timeout_giveup_restored_from_durable_row_after_bind(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("GIVEUP", source="cli")
    c = _make_compressor()
    c.bind_session_state(db, "GIVEUP")
    for _ in range(len(_TIMEOUT_COOLDOWN_LADDER)):
        c.record_timeout_failure("aux timed out")
    row = db.get_compression_failure_cooldown_row("GIVEUP")
    assert (row.get("error") or "").startswith("giveup:timeout:")
    db.restore_compression_failure_cooldown_row(
        "GIVEUP",
        {"session_exists": True, "cooldown_until": 1.0, "error": row["error"]},
    )
    rebound = _make_compressor()
    rebound.bind_session_state(db, "GIVEUP")
    should, reason = rebound.should_compress_info(50_000)
    assert should is False
    assert "auxiliary.compression.timeout" in (reason or "")
