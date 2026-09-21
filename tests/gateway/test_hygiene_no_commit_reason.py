"""The hygiene "did not rotate or compact in place" warning names the real cause (#71097).

The gateway's terminal branch used to blame "no session_db on the hygiene agent" for every
route into it, including attempts that aborted before any commit boundary on a DB-backed agent.
Both tests drive the production adopt path (``_hmwa_hygiene_adopt_transcript``), the only
reader of the compressor's ``_last_compaction_in_place`` flag on the hygiene route.
"""

import logging
from types import SimpleNamespace

import pytest

from gateway.run_turn import GatewayTurnMixin, hygiene_no_commit_reason

_HISTORY = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": "y" * 400}] * 4
_COMPRESSED = [{"role": "user", "content": "summary"}, {"role": "assistant", "content": "ok"}]


def _agent(**signals):
    base = {
        "session_id": "sid-1",
        "_last_compaction_in_place": False,
        "_last_compression_attempt_recorded": True,
        "_last_compression_attempt_in_place": None,
        "_compression_skipped_due_to_lock": None,
        "_compression_blocked_transient": None,
        "_last_compression_timed_out": False,
        "_last_compression_summary_warning": None,
        "_session_db": object(),
    }
    base.update(signals)
    return SimpleNamespace(**base)


async def _adopt(agent, caplog):
    """Run the gateway adopt step for an unrotated hygiene attempt; returns (attempt, entry, result)."""
    runner = GatewayTurnMixin.__new__(GatewayTurnMixin)
    attempt = SimpleNamespace(agent=agent, history=_HISTORY)
    plan = SimpleNamespace(msg_count=len(_HISTORY), approx_tokens=4_000, warn_token_threshold=10**9)
    entry = SimpleNamespace(session_id="sid-1", last_prompt_tokens=4_000)
    with caplog.at_level(logging.WARNING, logger="gateway.run_turn"):
        result = await runner._hmwa_hygiene_adopt_transcript(
            attempt, _COMPRESSED, _HISTORY, plan, session_entry=entry, source=None, _quick_key=None, run_generation=0,
        )
    return attempt, entry, result


@pytest.mark.asyncio
async def test_aborted_attempt_on_db_backed_agent_is_not_blamed_on_missing_session_db(caplog):
    # codex_app_server thread interrupted / summary aborted: the compressor never reached the
    # commit boundary, so `_last_compression_attempt_in_place` stays None while `_session_db` is real.
    attempt, entry, (rotated, in_place, count, tokens) = await _adopt(_agent(), caplog)
    warnings = [r.getMessage() for r in caplog.records if "did not rotate or compact in place" in r.getMessage()]
    assert len(warnings) == 1
    assert "aborted before commit" in warnings[0] and "no session_db" not in warnings[0]
    assert (rotated, in_place, count, tokens) == (False, False, len(_HISTORY), 4_000)
    assert attempt.history is _HISTORY and entry.last_prompt_tokens == 4_000

    timed_out = hygiene_no_commit_reason(_agent(_last_compression_timed_out=True))
    assert "timed out" in timed_out and "no session_db" not in timed_out
    assert "lease" in hygiene_no_commit_reason(_agent(_compression_skipped_due_to_lock="other-sid"))
    assert "cooldown:59" in hygiene_no_commit_reason(_agent(_compression_blocked_transient="cooldown:59"))
    no_db = hygiene_no_commit_reason(_agent(_last_compression_attempt_in_place=False, _session_db=None))
    assert no_db == "no session_db on the hygiene agent"
    assert hygiene_no_commit_reason(_agent(_last_compression_attempt_recorded=False)) == "compression did not run"


@pytest.mark.asyncio
async def test_committed_in_place_compaction_is_adopted_not_warned_about(caplog):
    # #71097 reporter 1: split_status=in_place_committed. compress_context sets
    # `_last_compaction_in_place` from the same `compacted_in_place` variable that produces that
    # telemetry, so a committed in-place attempt reaches the adopt step flagged and is adopted.
    agent = _agent(_last_compaction_in_place=True, _last_compression_attempt_in_place=True)
    attempt, entry, (rotated, in_place, count, tokens) = await _adopt(agent, caplog)
    assert (rotated, in_place, count) == (False, True, len(_COMPRESSED))
    assert attempt.history is _COMPRESSED and entry.last_prompt_tokens == 0
    assert "did not rotate or compact in place" not in caplog.text
