"""A messaging ``/retry`` under a session authority completes the re-sent turn's review
ownership with the send that delivers it.

Under a session authority ``/retry`` only plans its transcript rewrite and lets the re-send's
own admission commit both (``GatewaySessionCommandsMixin._retry_in_admission``). The re-sent
turn is a synthetic event the handler built: the authority's drain executes it, and
``admit_message`` adopts the drained turn's review-ownership completion onto that synthetic
event (``session_ingress_review.adopt_review_completion``) for the lifecycle that sends the
reply. That lifecycle is the ``/retry`` COMMAND's -- the idle dispatch
(``_process_message_background``) and the inline one (``_dispatch_inline_reply``) read the
completion off the command event once the handler returns -- and neither reads the synthetic
turn. The handler therefore re-homes the completion onto the command event, as the direct
(no-authority) path does; without that the re-sent turn's token stays ``live_turn_active``
for the process lifetime, every later automatic review on the session is refused, and the
re-sent turn's own candidate never spawns.

Real GatewayRunner, adapter, SessionAuthority, drain and ``execute_admission`` over SQLite
(the harness of ``test_background_review_authority_ownership``). A refused re-send (a stale
active-ids pin, a drain beginning after the pre-check) admits nothing, rewrites nothing and
registers no token.
"""

import types

import pytest

from agent import review_admission
from agent.i18n import t
from gateway.platforms.event import MessageEvent
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_background_review_authority_ownership import (
    _install_turn,
    _last_index,
    _owner,
    _TimelineAdapter,
)
from tests.waiting import async_wait_until

REPLY = "reply to hello"


def _rows(authority):
    return [
        row["status"]
        for session_id in authority.sessions
        for row in list_session_admissions(
            authority.db, session_id=session_id, pending_only=False
        )
    ]


def _roles(db, sid):
    return [m["role"] for m in db.get_messages(sid)]


def _sends(timeline):
    return [item for item in timeline if item[0] == "sent"]


async def _first_turn(tmp_path, monkeypatch):
    """One delivered native turn, reviewed once, then the transcript rows the stand-in did
    not persist, so ``/retry`` has an exchange to drop."""
    timeline, turns = [], {}
    adapter = _TimelineAdapter(timeline)
    runner, authority = await _owner(tmp_path, monkeypatch, adapter)
    source = adapter.build_source(
        chat_id="retry-chat", chat_type="dm", user_id="fixture-user"
    )
    _install_turn(runner, monkeypatch, timeline, turns)
    await adapter.handle_message(
        MessageEvent(text="hello", source=source, message_id="m1")
    )
    await async_wait_until(
        lambda: _rows(authority) == ["terminal"] and not adapter._background_tasks,
        timeout=20,
    )
    sid, profile_key = turns["hello"]
    await async_wait_until(lambda: timeline.count(("spawn", sid)) == 1, timeout=10)
    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None
    db = authority.db
    db.append_message(sid, "user", "hello")
    db.append_message(sid, "assistant", REPLY)
    assert runner._retry_owned_store(sid) is db, "not the in-admission /retry path"
    return types.SimpleNamespace(
        runner=runner,
        authority=authority,
        adapter=adapter,
        source=source,
        timeline=timeline,
        sid=sid,
        profile_key=profile_key,
        db=db,
    )


async def _retry(fx):
    """``/retry`` on the idle session, and the command's own dispatch unwound."""
    await fx.adapter.handle_message(
        MessageEvent(text="/retry", source=fx.source, message_id="m2")
    )
    await async_wait_until(
        lambda: not fx.adapter._background_tasks and not fx.adapter._session_tasks,
        timeout=20,
    )


@pytest.mark.asyncio
async def test_resent_turn_releases_its_token_and_spawns_once_after_its_reply(
    tmp_path, monkeypatch
):
    """The rewrite commits in the re-send's own admission and the command's dispatch sends
    the drained reply: that send completes the re-sent turn's ownership, so the token is
    released and its candidate spawns exactly once, after the reply went out."""
    fx = await _first_turn(tmp_path, monkeypatch)
    await _retry(fx)
    await async_wait_until(
        lambda: _rows(fx.authority) == ["terminal", "terminal"], timeout=20
    )

    assert _sends(fx.timeline) == [("sent", REPLY), ("sent", REPLY)], fx.timeline
    # The dropped exchange is gone: the rewrite rode the re-send's admission write.
    assert _roles(fx.db, fx.sid) == []
    assert (
        review_admission.live_turn_block_reason(fx.sid, None, fx.profile_key) is None
    ), "the re-sent turn's review token outlived its delivered reply"
    await async_wait_until(
        lambda: fx.timeline.count(("spawn", fx.sid)) == 2, timeout=10
    )
    assert _last_index(fx.timeline, ("sent", REPLY)) < _last_index(
        fx.timeline, ("spawn", fx.sid)
    )
    assert getattr(fx.authority, "review_completions", {}) == {}


@pytest.mark.asyncio
async def test_a_refused_rewrite_leaves_no_row_no_edit_and_no_token(
    tmp_path, monkeypatch
):
    """The active-ids fence refuses inside the admission write (a stale pin stands in for a
    transcript that changed since the read): nothing is admitted, rewritten or registered,
    and the first turn's review stays the only one."""
    fx = await _first_turn(tmp_path, monkeypatch)
    real_ids = fx.db.get_active_message_ids(fx.sid)
    monkeypatch.setattr(fx.db, "get_active_message_ids", lambda _sid: real_ids[:-1])

    await _retry(fx)

    assert fx.timeline[-1] == ("sent", t("gateway.retry.failed_unchanged")), fx.timeline
    assert _rows(fx.authority) == ["terminal"]
    assert _roles(fx.db, fx.sid) == ["user", "assistant"]
    assert review_admission.live_turn_block_reason(fx.sid, None, fx.profile_key) is None
    assert fx.timeline.count(("spawn", fx.sid)) == 1
    assert getattr(fx.authority, "review_completions", {}) == {}


@pytest.mark.asyncio
async def test_a_drain_beginning_after_the_precheck_leaves_no_row_no_edit_and_no_token(
    tmp_path, monkeypatch
):
    """The authority starts retiring between the readmission pre-check and the admission
    write: ``runtime_draining`` with the rewrite unapplied answers ``drain_rejected``, with
    the transcript intact and no ownership registered."""
    from gateway import session_retry_admission

    fx = await _first_turn(tmp_path, monkeypatch)
    real_bind = session_retry_admission.bind

    def bind_then_retire(rewrite):
        token = real_bind(rewrite)
        fx.authority.retiring = True
        return token

    monkeypatch.setattr(session_retry_admission, "bind", bind_then_retire)

    await _retry(fx)
    fx.authority.retiring = False

    expected = t(
        "gateway.busy.drain_rejected", action=fx.runner._status_action_gerund()
    )
    assert fx.timeline[-1] == ("sent", expected), fx.timeline
    assert _rows(fx.authority) == ["terminal"]
    assert _roles(fx.db, fx.sid) == ["user", "assistant"]
    assert review_admission.live_turn_block_reason(fx.sid, None, fx.profile_key) is None
    assert fx.timeline.count(("spawn", fx.sid)) == 1
    assert getattr(fx.authority, "review_completions", {}) == {}
