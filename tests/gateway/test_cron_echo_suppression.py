"""Cron-echo suppression: a cron delivery is a machine ping, not a request.

A cron job delivers to a group topic as the owning profile's bot, so the receiving
gateway sees the post as an ordinary inbound message and the group agent answers it —
every time, even when the job's own state has not changed. The suppressor drops the
ping at ingress (no turn, no reply) and answers only when the payload CHANGED.

The escalation path is the load-bearing half of this contract: a blocked-card
escalation and a human message that replies to a cron post must both survive.
"""

import json
from types import SimpleNamespace

import pytest

from gateway.cron_echo import (
    CRON_ENVELOPE_HEADER,
    CronEchoSettings,
    CronEchoSuppressor,
    is_cron_envelope,
    load_settings,
    parse_cron_envelope,
    payload_fingerprint,
    settings_from_config,
)
from gateway.session import Platform, SessionSource

GROUP_CHAT = "-1004452024623"
TOPIC = "3560"
JOB_ID = "4155843881c1"


def _envelope(job: str = "mac-ssh-key-watchdog", job_id: str = JOB_ID, payload: str = "OK") -> str:
    """The exact shape the delivery layer writes (``cron/scheduler_delivery.py``)."""
    return (
        f"{CRON_ENVELOPE_HEADER} {job}\n"
        f"(job_id: {job_id})\n"
        f"-------------\n\n"
        f"{payload}\n\n"
        f'To stop or manage this job, send me a new message (e.g. "stop reminder {job}").'
    )


def _bot_source(user_name: str = "Ultron", chat_type: str = "group") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id=GROUP_CHAT, chat_type=chat_type,
        user_id="999111", user_name=user_name, thread_id=TOPIC, is_bot=True,
    )


def _human_source(user_name: str = "Bryan Wallace", chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="8943042074", chat_type=chat_type,
        user_id="100200300", user_name=user_name, is_bot=False,
    )


@pytest.fixture
def suppressor(tmp_path):
    return CronEchoSuppressor(state_dir=tmp_path / "cron-echo")


# --- envelope recognition ---------------------------------------------------


def test_the_delivery_layers_envelope_is_recognised():
    verdict = parse_cron_envelope(_envelope())
    assert verdict is not None
    assert verdict.is_envelope
    assert verdict.job_name == "mac-ssh-key-watchdog"
    assert verdict.job_id == JOB_ID
    assert verdict.job_key == JOB_ID


def test_a_human_message_that_replies_to_a_cron_post_is_not_an_envelope():
    """Bryan replying to a cron post is human text — it must keep the normal path."""
    body = 'Can you kill the VPN alert as it is giving a ERROR because there is no VPN setup.'
    assert parse_cron_envelope(body) is None
    assert not is_cron_envelope(body)


def test_a_human_message_that_quotes_a_cron_envelope_is_not_an_envelope():
    """A quote of the envelope inside human prose is not a delivery."""
    body = f"Vision the cron failed\n\n{CRON_ENVELOPE_HEADER} some-job (job_id: abc12345)"
    assert parse_cron_envelope(body) is None


def test_the_envelope_must_start_the_message():
    """Only the START of the body counts — leading whitespace is tolerated, a prefix is not."""
    assert parse_cron_envelope(f"  \n{CRON_ENVELOPE_HEADER} job (job_id: abc12345)") is not None
    assert parse_cron_envelope("prefix text\n" + _envelope()) is None


def test_a_plain_group_message_is_not_an_envelope():
    assert parse_cron_envelope("Can you check the status of the cards?") is None
    assert parse_cron_envelope("") is None
    assert parse_cron_envelope(None) is None


def test_an_envelope_carrying_an_escalation_is_flagged():
    payload = "I need help with **CODE: Track B mcp-servers stack** -- credentials are missing"
    verdict = parse_cron_envelope(_envelope(payload=payload))
    assert verdict is not None
    assert verdict.is_escalation


def test_a_ordinary_envelope_is_not_flagged_as_an_escalation():
    assert parse_cron_envelope(_envelope(payload="OK")).is_escalation is False


# --- the default: an envelope is a machine ping, not a request ---------------


def test_a_cron_envelope_is_suppressed_by_default(suppressor):
    """The default posture: a cron delivery to a group topic is answered by nobody."""
    verdict = suppressor.verdict(_envelope(payload="OK"))
    assert verdict.is_envelope
    assert verdict.suppress is True
    assert "machine ping" in verdict.reason


def test_the_default_says_nothing_even_for_the_first_ping(suppressor):
    """A steady-state job's first delivery is not news either."""
    assert suppressor.verdict(_envelope(payload="OK")).suppress is True
    assert suppressor.verdict(_envelope(payload="OK")).suppress is True


def test_a_cron_ping_never_creates_a_state_file(suppressor, tmp_path):
    """The default path records nothing — there is no answer to remember."""
    suppressor.verdict(_envelope(payload="OK"))
    state_dir = tmp_path / "cron-echo"
    assert not state_dir.exists() or not list(state_dir.glob("*.json"))


# --- the "reply only when something changed" rule ---------------------------


def _change_suppressor(tmp_path):
    """A suppressor in change-driven mode (``reply_on_change_only: true``)."""
    return CronEchoSuppressor(
        settings_provider=lambda: CronEchoSettings(reply_on_change_only=True),
        state_dir=tmp_path / "cron-echo",
    )


def test_the_first_ping_is_answered_and_the_same_ping_is_silent(tmp_path):
    suppressor = _change_suppressor(tmp_path)
    body = _envelope(payload="OK")

    first = suppressor.verdict(body)
    assert first.is_envelope
    assert first.suppress is False, "the first delivery of a payload is a change"

    second = suppressor.verdict(body)
    assert second.suppress is True
    assert "unchanged" in second.reason


def test_a_changed_payload_is_answered_again(tmp_path):
    suppressor = _change_suppressor(tmp_path)
    suppressor.verdict(_envelope(payload="OK"))

    changed = suppressor.verdict(_envelope(payload="WROTE: host unreachable"))
    assert changed.suppress is False


def test_whitespace_only_changes_do_not_count_as_a_change(tmp_path):
    suppressor = _change_suppressor(tmp_path)
    suppressor.verdict(_envelope(payload="OK"))
    assert suppressor.verdict(_envelope(payload="OK\n")).suppress is True


def test_each_job_keeps_its_own_fingerprint(tmp_path):
    suppressor = _change_suppressor(tmp_path)
    suppressor.verdict(_envelope(job="job-a", job_id="aaaa1111", payload="OK"))

    other = suppressor.verdict(_envelope(job="job-b", job_id="bbbb2222", payload="OK"))
    assert other.suppress is False, "a different job is a different state"


def test_the_fingerprint_state_is_persisted_and_readable(tmp_path):
    body = _envelope(payload="host unreachable")
    suppressor = _change_suppressor(tmp_path)
    suppressor.verdict(body)

    state_files = list((tmp_path / "cron-echo").glob("*.json"))
    assert len(state_files) == 1
    state = json.loads(state_files[0].read_text())
    assert state["fingerprint"] == payload_fingerprint(body)
    assert state["job_id"] == JOB_ID
    assert state["job_name"] == "mac-ssh-key-watchdog"


def test_a_restarted_gateway_stays_silent_on_an_unchanged_ping(tmp_path):
    """Without persisted state, every restart would re-answer the last ping of every job."""
    body = _envelope(payload="OK")
    first = _change_suppressor(tmp_path)
    assert first.verdict(body).suppress is False

    second = CronEchoSuppressor(
        settings_provider=lambda: CronEchoSettings(reply_on_change_only=True),
        state_dir=tmp_path / "cron-echo",
    )
    assert second.verdict(body).suppress is True


def test_disabled_suppressor_answers_every_ping(tmp_path):
    suppressor = CronEchoSuppressor(
        settings_provider=lambda: CronEchoSettings(enabled=False),
        state_dir=tmp_path / "cron-echo",
    )
    assert suppressor.verdict(_envelope()).suppress is False
    assert suppressor.verdict(_envelope()).suppress is False


def test_reply_on_change_only_off_suppresses_every_ping(tmp_path):
    """The default: every envelope is dropped, changed or not."""
    suppressor = CronEchoSuppressor(
        settings_provider=lambda: CronEchoSettings(reply_on_change_only=False),
        state_dir=tmp_path / "cron-echo",
    )
    assert suppressor.verdict(_envelope()).suppress is True
    assert suppressor.verdict(_envelope(payload="something else")).suppress is True


def test_an_escalation_is_never_silenced(suppressor):
    payload = "I need help with **CODE: Track B** -- missing credentials"
    verdict = suppressor.verdict(_envelope(payload=payload))
    assert verdict.is_escalation
    assert verdict.suppress is False


# --- the fingerprint tracks the situation, not the counters ------------------


def _watchdog_page(failures: int, error_class: str = "no_route_host", host: str = "mac-pro-bryan") -> str:
    """A page shaped like the ssh-probe watchdog's real stdout body."""
    return _envelope(payload=(
        f"⚠️  *SSH-KEY PROBE FAIL* — {host}\n"
        f"• host: {host} (10.0.100.40)\n"
        f"• consecutive failures: {failures} (threshold 3)\n"
        f"• streak started: 2026-10-22T14:59:13Z\n"
        f"• last success: never-recorded\n"
        f"• last error class: `{error_class}`\n"
        f"• last error: ssh: connect to host 10.0.100.40 port 22: No route to host\n"
        f"• identity tried: ultron (freeipa/ultron)"
    ))


def test_a_rising_failure_counter_is_not_a_change(tmp_path):
    """The watchdog's own counter climbs every tick; the SITUATION is unchanged.

    This is the exact failure the card describes: 14-18 replies a day to topic
    3560 for an outage that had not changed in 15 days.
    """
    suppressor = _change_suppressor(tmp_path)
    assert suppressor.verdict(_watchdog_page(3)).suppress is False
    assert suppressor.verdict(_watchdog_page(12)).suppress is True
    assert suppressor.verdict(_watchdog_page(18)).suppress is True
    assert suppressor.verdict(_watchdog_page(33)).suppress is True


def test_a_different_error_class_is_a_change(tmp_path):
    """A different failure class IS a change and must be answered."""
    suppressor = _change_suppressor(tmp_path)
    suppressor.verdict(_watchdog_page(3, error_class="no_route_host"))
    changed = suppressor.verdict(_watchdog_page(3, error_class="host_key_changed"))
    assert changed.suppress is False


def test_a_different_host_is_a_change(tmp_path):
    suppressor = _change_suppressor(tmp_path)
    suppressor.verdict(_watchdog_page(3, host="mac-pro-bryan"))
    changed = suppressor.verdict(_watchdog_page(3, host="denises-imac"))
    assert changed.suppress is False


def test_a_recovery_is_a_change(tmp_path):
    """The failure clearing is the most important change of all."""
    suppressor = _change_suppressor(tmp_path)
    suppressor.verdict(_watchdog_page(3))
    recovered = suppressor.verdict(_envelope(payload="mac-pro-bryan: connected=True identity=ultron"))
    assert recovered.suppress is False


def test_the_fingerprint_ignores_the_counter_but_not_the_identity(tmp_path):
    same = payload_fingerprint(_watchdog_page(3))
    assert same == payload_fingerprint(_watchdog_page(12))
    assert same != payload_fingerprint(_watchdog_page(3, error_class="host_key_changed"))


# --- the ingress seam -------------------------------------------------------


def _runner(suppressor):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._cron_echo_suppressor = suppressor
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def _passthrough_hook(event, source):
        return event

    runner._hm_pre_gateway_dispatch_hook = _passthrough_hook
    runner._is_user_authorized_for_source = lambda source, **kw: True
    return runner


@pytest.mark.asyncio
async def test_a_cron_ping_produces_no_turn_and_no_reply(suppressor):
    """The acceptance case: a synthetic cron ping must not reach the agent."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(text=_envelope(), message_id="m1", source=_bot_source())

    assert await runner._hm_admit_event(event) is None


@pytest.mark.asyncio
async def test_a_changed_cron_ping_still_reaches_the_agent(tmp_path):
    """In change-driven mode a changed state is worth a reply."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(_change_suppressor(tmp_path))
    first = MessageEvent(text=_envelope(payload="OK"), message_id="m1", source=_bot_source())
    assert (await runner._hm_admit_event(first))[0] is first

    changed = MessageEvent(
        text=_envelope(payload="WROTE: host unreachable"), message_id="m2", source=_bot_source(),
    )
    assert (await runner._hm_admit_event(changed))[0] is changed


@pytest.mark.asyncio
async def test_a_bryan_dm_still_gets_a_response(suppressor):
    """A human DM is never an envelope, so it keeps the normal path."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(
        text="Vision look at your responses on the WallaceArizona Services group.",
        message_id="m1", source=_human_source(),
    )
    assert (await runner._hm_admit_event(event))[0] is event


@pytest.mark.asyncio
async def test_a_blocked_card_escalation_still_gets_a_response(suppressor):
    """The escalation format is an agent message, not a cron envelope — it must survive."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(
        text="@ultron_wallace_bot I need help with **CODE: Track B mcp-servers stack** -- credentials missing",
        message_id="m1", source=_bot_source(user_name="F.R.I.D.A.Y"),
    )
    assert (await runner._hm_admit_event(event))[0] is event


@pytest.mark.asyncio
async def test_a_human_pasting_a_cron_envelope_still_gets_a_response(suppressor):
    """A cron delivery comes from a BOT. A human pasting the envelope is asking about it.

    Forwarding a cron log verbatim to ask "why did this fail?" is a real request.
    Silencing it would be a false negative on the escalation path.
    """
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(text=_envelope(payload="host unreachable"), message_id="m1", source=_human_source())
    assert (await runner._hm_admit_event(event))[0] is event


@pytest.mark.asyncio
async def test_a_human_replying_to_a_cron_post_still_gets_a_response(suppressor):
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(
        text="Vision the cron failed",
        message_id="m1", source=_human_source(user_name="Bryan Wallace", chat_type="group"),
    )
    assert (await runner._hm_admit_event(event))[0] is event


@pytest.mark.asyncio
async def test_the_busy_path_drops_a_cron_ping_before_it_can_steer(suppressor):
    """A cron ping arriving while a turn runs must not steer, queue, or redirect it."""
    from unittest.mock import AsyncMock

    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    runner._draining = False
    runner._effective_busy_input_mode = lambda source: "steer"
    runner._effective_busy_text_mode = lambda source: "steer"
    runner._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
    runner._delivery_adapter_for = lambda source: SimpleNamespace(_pending_messages={})
    runner._peek_session_state = lambda key: SimpleNamespace(
        turn=SimpleNamespace(agent=object(), busy_ack_ts=0)
    )
    steer = AsyncMock(return_value=SimpleNamespace(effective_mode="steer", redirected=False, steered=True))
    runner._resolve_busy_steer_or_redirect = steer

    event = MessageEvent(text=_envelope(), message_id="m1", source=_bot_source())
    assert await runner._handle_active_session_busy_message(event, "session") is True
    assert steer.await_count == 0, "a cron ping must not steer the running turn"


@pytest.mark.asyncio
async def test_an_escalation_envelope_still_reaches_the_agent(suppressor):
    """A cron payload that carries an escalation is a real request wearing an envelope."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(
        text=_envelope(payload="I need help with **CODE: Track B** -- credentials missing"),
        message_id="m1", source=_bot_source(),
    )
    assert (await runner._hm_admit_event(event))[0] is event


# --- settings ---------------------------------------------------------------


def test_settings_read_config_yaml():
    cfg = {"gateway": {"cron_echo": {"enabled": False, "reply_on_change_only": False}}}
    settings = settings_from_config(cfg)
    assert settings.enabled is False
    assert settings.reply_on_change_only is False


def test_unusable_settings_keep_the_defaults():
    settings = settings_from_config({"gateway": {"cron_echo": {"enabled": "maybe", "state_ttl_seconds": -5}}})
    assert settings.enabled is True
    assert settings.state_ttl_seconds == 0


def test_settings_default_when_the_block_is_absent():
    assert settings_from_config({}) == CronEchoSettings()
    assert settings_from_config({"gateway": {"cron_echo": "nope"}}) == CronEchoSettings()


def test_an_unreadable_config_keeps_the_defaults(monkeypatch):
    """The suppressor must never be the reason ingress fails on a config error."""
    import hermes_cli.config as hermes_config

    def boom(*_args, **_kwargs):
        raise RuntimeError("no yaml loader")

    monkeypatch.setattr(hermes_config, "cfg_get", boom)
    assert settings_from_config({"gateway": {"cron_echo": {"enabled": False}}}) == CronEchoSettings()


def test_load_settings_falls_back_to_the_defaults(monkeypatch):
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(hermes_config, "load_config_readonly", lambda: {"gateway": {"cron_echo": {"enabled": False}}})
    assert load_settings().enabled is False

    def boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(hermes_config, "load_config_readonly", boom)
    assert load_settings() == CronEchoSettings()


# ---------------------------------------------------------------------------
# The FLATTENED envelope: what actually reaches the gateway.
#
# The delivery layer writes a multi-line envelope, but the text that arrives at
# the receiving gateway has been flattened onto one line — Telegram cannot carry
# a raw newline in a caption. Verified against gateway.log:
#
#   inbound ... msg='Cronjob Response: mac-ssh-key-watchdog
#   (job_id: 4155843881c1) -------------  ⚠️ '
#
# A parser that only understands the multi-line form silently loses the job id,
# and the per-job state file falls back to a mangled name. These tests pin the
# flattened form.
# ---------------------------------------------------------------------------


def _flat_envelope(job: str = "mac-ssh-key-watchdog", job_id: str = JOB_ID, payload: str = "OK") -> str:
    """The single-line form the receiving gateway actually sees."""
    return (
        f"{CRON_ENVELOPE_HEADER} {job} (job_id: {job_id}) ------------- {payload} "
        'To stop or manage this job, send me a new message (e.g. "stop reminder").'
    )


def test_the_flattened_envelope_is_recognised():
    assert is_cron_envelope(_flat_envelope()) is True


def test_the_flattened_envelope_yields_the_job_id():
    verdict = parse_cron_envelope(_flat_envelope())
    assert verdict is not None
    assert verdict.job_id == JOB_ID


def test_the_flattened_envelope_yields_a_clean_job_name():
    """The name must not carry the "(job_id: ...)" tail or the divider run."""
    verdict = parse_cron_envelope(_flat_envelope())
    assert verdict is not None
    assert verdict.job_name == "mac-ssh-key-watchdog"


def test_the_flattened_envelope_keys_the_state_file_on_the_job_id():
    verdict = parse_cron_envelope(_flat_envelope())
    assert verdict is not None
    assert verdict.job_key == JOB_ID


def test_the_flattened_and_multiline_forms_agree_on_name_and_id():
    multi = parse_cron_envelope(_envelope())
    flat = parse_cron_envelope(_flat_envelope())
    assert multi is not None and flat is not None
    assert (multi.job_name, multi.job_id) == (flat.job_name, flat.job_id)


def test_the_flattened_and_multiline_forms_agree_on_the_fingerprint():
    """The same job must fingerprint the same in either form, or the state
    file reads one job as two and the hub answers it twice."""
    multi = payload_fingerprint(_envelope(payload="SSH key probe OK for mac-mini"))
    flat = payload_fingerprint(_flat_envelope(payload="SSH key probe OK for mac-mini"))
    assert multi == flat


def test_the_flattened_envelope_is_suppressed():
    suppressor = CronEchoSuppressor()
    verdict = suppressor.verdict(_flat_envelope())
    assert verdict.is_envelope is True
    assert verdict.suppress is True


def test_a_flattened_escalation_is_still_admitted():
    payload = "I need help with **Patroni disk usage** -- volume full"
    verdict = CronEchoSuppressor().verdict(_flat_envelope(payload=payload))
    assert verdict.is_escalation is True
    assert verdict.suppress is False


def test_the_flattened_envelope_detects_a_changed_payload(tmp_path):
    settings = CronEchoSettings(enabled=True, reply_on_change_only=True)
    suppressor = CronEchoSuppressor(settings_provider=lambda: settings, state_dir=tmp_path)
    first = suppressor.verdict(_flat_envelope(payload="probe OK for mac-mini"))
    assert first.suppress is False
    same = suppressor.verdict(_flat_envelope(payload="probe OK for mac-mini"))
    assert same.suppress is True
    changed = suppressor.verdict(_flat_envelope(payload="probe FAILED for mac-mini: host key mismatch"))
    assert changed.suppress is False


def test_a_state_dir_given_as_a_plain_string_path_works(tmp_path):
    """The constructor accepts a str path; it must not crash on the mkdir."""
    settings = CronEchoSettings(enabled=True, reply_on_change_only=True)
    suppressor = CronEchoSuppressor(
        settings_provider=lambda: settings,
        state_dir=str(tmp_path / "cron-echo"),
    )
    verdict = suppressor.verdict(_envelope())
    assert verdict.suppress is False
    assert (tmp_path / "cron-echo").exists()


# --- the acceptance case, in the form the gateway actually receives ---------
#
# The tests above use the multi-line envelope the delivery layer WRITES. These
# use the FLATTENED envelope the gateway RECEIVES (Telegram cannot carry a raw
# newline in a caption), driven through the same real `_hm_admit_event` seam.


@pytest.mark.asyncio
async def test_a_flattened_cron_ping_produces_no_turn_and_no_reply(suppressor):
    """The acceptance case, in production form: no turn, no reply."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(text=_flat_envelope(), message_id="m1", source=_bot_source())

    assert await runner._hm_admit_event(event) is None


@pytest.mark.asyncio
async def test_a_flattened_escalation_still_reaches_the_agent(suppressor):
    """A blocked-card escalation must survive in the production form too."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(
        text=_flat_envelope(payload="I need help with **Patroni disk usage** -- volume full"),
        message_id="m1", source=_bot_source(),
    )
    assert (await runner._hm_admit_event(event))[0] is event


@pytest.mark.asyncio
async def test_a_flattened_ping_still_honours_the_change_rule(tmp_path):
    """In change-driven mode the flattened form must detect a changed payload."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(_change_suppressor(tmp_path))
    first = MessageEvent(text=_flat_envelope(payload="probe OK for mac-mini"),
                         message_id="m1", source=_bot_source())
    assert (await runner._hm_admit_event(first))[0] is first

    same = MessageEvent(text=_flat_envelope(payload="probe OK for mac-mini"),
                         message_id="m2", source=_bot_source())
    assert await runner._hm_admit_event(same) is None

    changed = MessageEvent(
        text=_flat_envelope(payload="probe FAILED for mac-mini: host key mismatch"),
        message_id="m3", source=_bot_source(),
    )
    assert (await runner._hm_admit_event(changed))[0] is changed


@pytest.mark.asyncio
async def test_a_flattened_ping_from_a_human_still_reaches_the_agent(suppressor):
    """Only a bot sender is silenced, in the production form too."""
    from gateway.platforms.event import MessageEvent

    runner = _runner(suppressor)
    event = MessageEvent(
        text=_flat_envelope(payload="why is this firing every four hours?"),
        message_id="m1", source=_human_source(),
    )
    assert (await runner._hm_admit_event(event))[0] is event
