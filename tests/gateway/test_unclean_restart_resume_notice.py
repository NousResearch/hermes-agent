"""After an UNCLEAN prior life, a boot-resumed session must be TOLD why.

Measured shape: the event loop is blocked long enough for the loop-liveness
watchdog to fire ``os._exit(75)`` from its own thread. The supervisor relaunches
the gateway minutes later and every session that was mid-turn is boot-resumed
(``reason=restart_interrupted``). Their replies then arrive minutes late with
**no explanation of the silence**.

The existing interrupt notice (``_INTERRUPT_REASON_GATEWAY_RESTART =
"Gateway restarting"``) is emitted from the GRACEFUL drain path only. An
``os._exit`` from a watchdog thread runs no drain, so no in-process notice is
possible — the only place that can still speak is the NEXT boot, on the resume
path, before the resumed turn runs.

Contract locked here:

(a) prior life ended UNCLEAN (watchdog exit 75 / SIGKILL / no exit path ran)
    => ONE short notice is posted to the session BEFORE the resumed turn runs,
    naming the time, the exit reason and (when known) the blocking site;
(b) a SIGKILL whose sender the ledger attributed => the notice names the killer;
(c) prior life exited CLEAN => no notice at all (the drain already told them);
(d) two resumes of the same session in one boot => exactly ONE notice;
(e) the send raising => the resume still runs, the error is logged.

Hermetic: pure predicates plus the real ``GatewayRunner`` methods driven
in-process against a temp home. No gateway process is ever spawned.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from gateway.restart_notice import (
    UNCLEAN_NOTICE_RESUME_REASONS,
    PriorLifeVerdict,
    claim_restart_notice,
    classify_prior_life,
    format_restart_notice,
    get_restart_notice_ledger_path,
    read_last_event_loop_blocked_site,
)
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

_KEY = "agent:main:telegram:dm:123456:u1"

# What this boot's sentinel looks like after lifecycle_ledger._claim_sentinel
# carried the previous life's verdict forward.
_WATCHDOG_SENTINEL = {
    "phase": "running",
    "pid": 4242,
    "started_at": "2026-09-20T23:04:16+00:00",
    "prior_phase": "exited",
    "prior_exit_code": 75,
    "prior_exit_reason": "loop_liveness_watchdog",
    "prior_exited_at": "2026-09-20T23:01:08+00:00",
}

_SIGKILL_SENTINEL = {
    "phase": "running",
    "pid": 4243,
    "started_at": "2026-09-20T23:04:16+00:00",
    "prior_unclean_exit": True,
    "prior_killer": "SIGKILL",
    "prior_kill_sender": "Python[35502]",
    "prior_kill_sender_label": "ai.hermes.gateway-watchdog",
    "prior_exited_at": "2026-09-20T23:01:08+00:00",
}

_CLEAN_SENTINEL = {
    "phase": "running",
    "pid": 4244,
    "started_at": "2026-09-20T23:04:16+00:00",
    "prior_phase": "exited",
    "prior_exit_code": 0,
    "prior_exit_reason": "graceful_shutdown",
    "prior_exited_at": "2026-09-20T23:01:08+00:00",
}


# --------------------------------------------------------------------------
# (1) Pure classification + formatting
# --------------------------------------------------------------------------


def test_watchdog_exit_75_is_unclean_and_names_reason_and_site():
    verdict = classify_prior_life(
        _WATCHDOG_SENTINEL, site="utils.py:230 atomic_replace"
    )
    assert verdict.unclean is True
    assert verdict.exit_code == 75
    assert verdict.exit_reason == "loop_liveness_watchdog"
    assert verdict.boot_id == "2026-09-20T23:04:16+00:00"

    text = format_restart_notice(verdict)
    assert text
    # the time of the DEATH, not of the boot
    assert "23:01" in text
    assert "exit 75" in text
    assert "event loop blocked" in text
    assert "utils.py:230 atomic_replace" in text
    assert "resuming" in text.lower()
    # one short message, not a wall
    assert text.count("\n") == 0
    assert len(text) < 320


def test_sigkill_prior_life_names_the_killer_and_sender_label():
    verdict = classify_prior_life(_SIGKILL_SENTINEL, site=None)
    assert verdict.unclean is True
    assert verdict.killer == "SIGKILL"

    text = format_restart_notice(verdict)
    assert text
    assert "SIGKILL" in text
    assert "ai.hermes.gateway-watchdog" in text
    assert "23:01" in text


def test_clean_prior_exit_produces_no_notice():
    verdict = classify_prior_life(_CLEAN_SENTINEL, site=None)
    assert verdict.unclean is False
    assert format_restart_notice(verdict) is None


def test_missing_or_empty_sentinel_is_not_treated_as_unclean():
    for sentinel in (None, {}, {"phase": "running"}):
        verdict = classify_prior_life(sentinel, site=None)
        assert verdict.unclean is False
        assert format_restart_notice(verdict) is None


def test_unattributed_unclean_exit_still_notifies_without_inventing_a_cause():
    verdict = classify_prior_life(
        {
            "phase": "running",
            "started_at": "2026-09-20T23:04:16+00:00",
            "prior_unclean_exit": True,
            "prior_killer": "unattributed",
            "prior_exited_at": "2026-09-20T23:01:08+00:00",
        },
        site=None,
    )
    assert verdict.unclean is True
    text = format_restart_notice(verdict)
    assert text
    assert "SIGKILL" not in text
    assert "exit 75" not in text


def test_restart_interrupted_is_the_gated_resume_reason():
    assert "restart_interrupted" in UNCLEAN_NOTICE_RESUME_REASONS
    assert "restart_consumed" not in UNCLEAN_NOTICE_RESUME_REASONS


def test_event_loop_blocked_site_is_read_from_the_log_tail(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "gateway.log").write_text(
        "noise\n"
        "PHASE=event_loop_blocked platform=discord seconds=30 "
        "site=utils.py:230 atomic_replace\n"
        "more noise\n",
        encoding="utf-8",
    )
    assert read_last_event_loop_blocked_site(tmp_path) == "utils.py:230 atomic_replace"


def test_event_loop_blocked_site_absent_is_none(tmp_path):
    assert read_last_event_loop_blocked_site(tmp_path) is None


# --------------------------------------------------------------------------
# (2) Idempotency ledger — once per session per boot
# --------------------------------------------------------------------------


def test_notice_ledger_claims_once_per_session_per_boot(tmp_path):
    assert claim_restart_notice("boot-A", _KEY, home=tmp_path) is True
    assert claim_restart_notice("boot-A", _KEY, home=tmp_path) is False
    # a different session in the same boot still gets its own notice
    assert claim_restart_notice("boot-A", "other:key", home=tmp_path) is True
    # a NEW boot resets the ledger (a crash loop must re-notify each boot)
    assert claim_restart_notice("boot-B", _KEY, home=tmp_path) is True
    assert claim_restart_notice("boot-B", _KEY, home=tmp_path) is False

    payload = json.loads(
        get_restart_notice_ledger_path(tmp_path).read_text(encoding="utf-8")
    )
    assert payload["boot_id"] == "boot-B"
    assert payload["notified"] == [_KEY]


def test_notice_ledger_survives_a_corrupt_file(tmp_path):
    path = get_restart_notice_ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    # fail-open: a corrupt ledger must not silence the notice
    assert claim_restart_notice("boot-A", _KEY, home=tmp_path) is True
    assert claim_restart_notice("boot-A", _KEY, home=tmp_path) is False


# --------------------------------------------------------------------------
# (3) Runner wiring — the notice lands BEFORE the resumed turn
# --------------------------------------------------------------------------


def _resume_runner(tmp_path, sentinel, site=None):
    runner, adapter = make_restart_runner()
    for name in (
        "_run_startup_resume_event",
        "_maybe_notify_unclean_restart",
        "_prior_life_verdict",
        "_session_state",
        "_peek_session_state",
        "_release_running_agent_state",
        "_thread_metadata_for_target",
    ):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner, GatewayRunner))
    runner._restart_notice_home = tmp_path
    runner._restart_notice_sentinel = sentinel
    runner._restart_notice_site = site
    return runner, adapter


def _resume_event():
    from gateway.platforms.base import MessageEvent, MessageType

    return MessageEvent(
        text="",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        internal=True,
    )


class _OrderRecordingAdapter:
    """Records the interleaving of notice sends and the resumed dispatch."""

    def __init__(self, inner):
        self._inner = inner
        self.order: list[str] = []
        self.sent: list[str] = []
        self._session_tasks: dict = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.order.append("send")
        self.sent.append(content)
        return await self._inner.send(
            chat_id, content, reply_to=reply_to, metadata=metadata
        )

    async def handle_message(self, event):
        self.order.append("handle_message")
        return None

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.asyncio
async def test_notice_is_posted_before_the_resumed_turn_runs(tmp_path):
    """(a) the incident shape: exit 75 => one notice, then the resume turn."""
    runner, base_adapter = _resume_runner(
        tmp_path, _WATCHDOG_SENTINEL, site="utils.py:230 atomic_replace"
    )
    adapter = _OrderRecordingAdapter(base_adapter)

    await runner._run_startup_resume_event(
        adapter, _resume_event(), _KEY, resume_reason="restart_interrupted"
    )

    assert adapter.order == ["send", "handle_message"], adapter.order
    assert len(adapter.sent) == 1
    notice = adapter.sent[0]
    assert "23:01" in notice
    assert "exit 75" in notice
    assert "utils.py:230 atomic_replace" in notice


@pytest.mark.asyncio
async def test_sigkill_notice_names_the_killer_through_the_runner(tmp_path):
    """(b) killer/sender attribution survives the wiring, not just the formatter."""
    runner, base_adapter = _resume_runner(tmp_path, _SIGKILL_SENTINEL)
    adapter = _OrderRecordingAdapter(base_adapter)

    await runner._run_startup_resume_event(
        adapter, _resume_event(), _KEY, resume_reason="restart_interrupted"
    )

    assert len(adapter.sent) == 1
    assert "SIGKILL" in adapter.sent[0]
    assert "ai.hermes.gateway-watchdog" in adapter.sent[0]


@pytest.mark.asyncio
async def test_clean_prior_exit_posts_no_notice(tmp_path):
    """(c) the graceful drain already said 'Gateway restarting' — stay quiet."""
    runner, base_adapter = _resume_runner(tmp_path, _CLEAN_SENTINEL)
    adapter = _OrderRecordingAdapter(base_adapter)

    await runner._run_startup_resume_event(
        adapter, _resume_event(), _KEY, resume_reason="restart_interrupted"
    )

    assert adapter.order == ["handle_message"]
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_non_restart_interrupted_reason_posts_no_notice(tmp_path):
    """A self-initiated clean restart resume is not this notice's business."""
    runner, base_adapter = _resume_runner(tmp_path, _WATCHDOG_SENTINEL)
    adapter = _OrderRecordingAdapter(base_adapter)

    await runner._run_startup_resume_event(
        adapter, _resume_event(), _KEY, resume_reason="restart_consumed"
    )

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_two_resumes_in_one_boot_produce_exactly_one_notice(tmp_path):
    """(d) idempotent: a re-scheduled resume (or a crash loop) must not spam."""
    runner, base_adapter = _resume_runner(tmp_path, _WATCHDOG_SENTINEL)
    adapter = _OrderRecordingAdapter(base_adapter)

    for _ in range(2):
        await runner._run_startup_resume_event(
            adapter, _resume_event(), _KEY, resume_reason="restart_interrupted"
        )

    assert adapter.order.count("send") == 1, adapter.order
    assert adapter.order.count("handle_message") == 2


@pytest.mark.asyncio
async def test_send_failure_does_not_block_the_resume(tmp_path, caplog):
    """(e) best-effort: the notice is never allowed to eat the resumed turn."""
    runner, base_adapter = _resume_runner(tmp_path, _WATCHDOG_SENTINEL)
    adapter = _OrderRecordingAdapter(base_adapter)

    async def _boom(*_a, **_kw):
        adapter.order.append("send_raised")
        raise RuntimeError("discord 503")

    adapter.send = _boom

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await runner._run_startup_resume_event(
            adapter, _resume_event(), _KEY, resume_reason="restart_interrupted"
        )

    assert adapter.order == ["send_raised", "handle_message"], adapter.order
    assert any(
        "unclean-restart notice" in r.getMessage().lower() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_notice_failure_inside_classification_is_swallowed(tmp_path):
    """A malformed sentinel must not take the resume down with it."""
    runner, base_adapter = _resume_runner(tmp_path, {"phase": ["not", "a", "string"]})
    adapter = _OrderRecordingAdapter(base_adapter)

    await runner._run_startup_resume_event(
        adapter, _resume_event(), _KEY, resume_reason="restart_interrupted"
    )

    assert "handle_message" in adapter.order


# --------------------------------------------------------------------------
# (4) The lifecycle ledger must carry the prior life's exit info forward
# --------------------------------------------------------------------------


def test_claim_sentinel_carries_prior_watchdog_exit_forward(tmp_path):
    """Without this, an exit-75 life is invisible to the next boot.

    ``shutdown_watchdog`` calls ``mark_exited(75, reason=...)`` before
    ``os._exit``, so the sentinel reads ``phase=exited`` and
    ``detect_unclean_exit`` correctly returns None — the death is *recorded*
    but the record is then overwritten by the next boot's claim.
    """
    from gateway.lifecycle_ledger import (
        get_lifecycle_sentinel_path,
        mark_exited,
        record_startup,
    )

    mark_exited(75, reason="loop_liveness_watchdog", home=tmp_path)
    record_startup(home=tmp_path)

    sentinel = json.loads(
        get_lifecycle_sentinel_path(tmp_path).read_text(encoding="utf-8")
    )
    assert sentinel["phase"] == "running"
    assert sentinel["prior_exit_code"] == 75
    assert sentinel["prior_exit_reason"] == "loop_liveness_watchdog"
    assert sentinel["prior_phase"] == "exited"

    verdict = classify_prior_life(sentinel, site=None)
    assert verdict.unclean is True
    assert "exit 75" in (format_restart_notice(verdict) or "")


def test_claim_sentinel_carries_a_graceful_exit_forward_as_clean(tmp_path):
    from gateway.lifecycle_ledger import (
        get_lifecycle_sentinel_path,
        mark_exited,
        record_startup,
    )

    mark_exited(0, reason="graceful_shutdown", home=tmp_path)
    record_startup(home=tmp_path)

    sentinel = json.loads(
        get_lifecycle_sentinel_path(tmp_path).read_text(encoding="utf-8")
    )
    assert sentinel["prior_exit_reason"] == "graceful_shutdown"
    assert classify_prior_life(sentinel, site=None).unclean is False
