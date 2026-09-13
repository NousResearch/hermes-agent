"""Bot Chat cron delivery: deliver='bot-chat[:<profile>]' injects job output
into a local profile's canonical Bot Chat session as a real inbound turn.

Covers token parsing, target resolution (own profile / named / missing),
preflight exemption, create-time validation, the subprocess delivery lane,
and the delivery-targets listing used by UI pickers.
"""

import subprocess
from unittest import mock

import pytest

from cron import scheduler as sched
from cron import scheduler_delivery as sched_delivery
from cron.scheduler import _resolve_delivery_targets
from cron.scheduler_delivery import (
    BOT_CHAT_PLATFORM,
    _deliver_to_bot_chat,
    _resolve_bot_chat_target,
    parse_bot_chat_deliver_token,
)
from cron.scheduler_preflight import _preflight_check_delivery


# ── token parsing ────────────────────────────────────────────────────────────

def test_bare_token_targets_own_profile():
    assert parse_bot_chat_deliver_token("bot-chat") == ""
    assert parse_bot_chat_deliver_token("  Bot-Chat  ") == ""


def test_named_token_returns_profile():
    assert parse_bot_chat_deliver_token("bot-chat:research") == "research"
    assert parse_bot_chat_deliver_token("BOT-CHAT:Research") == "Research"


def test_non_bot_chat_tokens_pass_through():
    assert parse_bot_chat_deliver_token("telegram:-100:17") is None
    assert parse_bot_chat_deliver_token("origin") is None
    assert parse_bot_chat_deliver_token("local") is None
    assert parse_bot_chat_deliver_token("all") is None
    # A platform whose name merely CONTAINS bot-chat must not match.
    assert parse_bot_chat_deliver_token("bot-chatter") is None


# ── target resolution ────────────────────────────────────────────────────────

def test_own_profile_resolves_without_name():
    target = _resolve_bot_chat_target({"id": "j1"}, "")
    assert target == {"platform": BOT_CHAT_PLATFORM, "chat_id": "", "thread_id": None}


def test_named_profile_resolves_when_exists():
    with mock.patch("hermes_cli.profiles.profile_exists", return_value=True):
        target = _resolve_bot_chat_target({"id": "j1"}, "research")
    assert target is not None
    assert target["platform"] == BOT_CHAT_PLATFORM
    assert target["chat_id"] == "research"


def test_unknown_profile_resolves_to_none():
    with mock.patch("hermes_cli.profiles.profile_exists", return_value=False):
        assert _resolve_bot_chat_target({"id": "j1"}, "ghost") is None


def test_resolve_delivery_targets_combines_with_platform_targets():
    """bot-chat rides the same comma-separated deliver string as platforms."""
    job = {"id": "j1", "deliver": "bot-chat,telegram"}
    with mock.patch.object(sched_delivery, "_get_home_target_chat_id", return_value="-100123"), \
         mock.patch.object(sched_delivery, "_get_home_target_thread_id", return_value=None), \
         mock.patch.object(sched_delivery, "_is_known_delivery_platform", return_value=True), \
         mock.patch.object(sched_delivery, "_resolve_origin", return_value=None):
        targets = _resolve_delivery_targets(job)
    platforms = {t["platform"] for t in targets}
    assert BOT_CHAT_PLATFORM in platforms
    assert "telegram" in platforms


# ── preflight ────────────────────────────────────────────────────────────────

def test_preflight_ignores_bot_chat_targets():
    """bot-chat needs no gateway credentials — preflight must not block it."""
    assert _preflight_check_delivery({"id": "j1", "deliver": "bot-chat"}) is None
    assert _preflight_check_delivery({"id": "j1", "deliver": "bot-chat:research"}) is None


def test_preflight_still_blocks_unknown_platforms():
    with mock.patch.object(sched_delivery, "_is_known_delivery_platform", return_value=False):
        err = _preflight_check_delivery({"id": "j1", "deliver": "nonexistent-platform"})
    assert err is not None and "not a known" in err


# ── create-time validation ───────────────────────────────────────────────────

def test_create_validation_rejects_unknown_profile():
    from tools.cronjob_tools import _validate_bot_chat_deliver

    with mock.patch("hermes_cli.profiles.profile_exists", return_value=False):
        err = _validate_bot_chat_deliver("bot-chat:ghost")
    assert err is not None
    assert "machine-local" in err


def test_create_validation_accepts_bare_and_existing():
    from tools.cronjob_tools import _validate_bot_chat_deliver

    assert _validate_bot_chat_deliver("bot-chat") is None
    assert _validate_bot_chat_deliver(None) is None
    assert _validate_bot_chat_deliver("telegram:-100") is None
    with mock.patch("hermes_cli.profiles.profile_exists", return_value=True):
        assert _validate_bot_chat_deliver("bot-chat:research") is None


# ── delivery lane ────────────────────────────────────────────────────────────

def _completed(returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_deliver_runs_canonical_bot_chat_lane():
    """The subprocess must use the Bot Mode agent-to-agent chat lane:
    chat --in ~ -c "Bot Chat" --create-if-missing -Q --query-file <tmp>."""
    calls = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        return _completed()

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
         mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "Daily digest"}, "the output", "")

    assert err is None
    argv = calls["argv"]
    assert argv[0] == "/usr/bin/hermes"
    assert "-p" not in argv  # own profile: subprocess inherits HERMES_HOME
    assert "chat" in argv
    assert "Bot Chat" in argv
    assert "--create-if-missing" in argv
    assert "-Q" in argv
    assert "--query-file" in argv
    # Message rides a temp file, never inline argv (quote/expansion safety).
    assert not any("the output" in str(a) for a in argv)


def test_deliver_named_profile_uses_p_flag_and_clears_home():
    calls = {}

    def fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        return _completed()

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
         mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
         mock.patch.dict(sched.os.environ, {"HERMES_HOME": "/tmp/other-profile"}):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "research")

    assert err is None
    argv = calls["argv"]
    assert argv[1:3] == ["-p", "research"]
    # -p owns resolution; the scheduler's own HERMES_HOME must not leak in.
    assert "HERMES_HOME" not in calls["kwargs"]["env"]


def test_deliver_failure_returns_error_string():
    with mock.patch.object(
        sched.subprocess, "run", return_value=_completed(returncode=1, stderr="boom")
    ), mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")
    assert err is not None
    assert "boom" in err


def test_deliver_timeout_returns_error_string():
    with mock.patch.object(
        sched.subprocess, "run",
        side_effect=subprocess.TimeoutExpired(cmd="hermes", timeout=600),
    ), mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")
    assert err is not None
    assert "timed out" in err


def test_deliver_retries_when_recipient_session_is_busy():
    """A busy recipient is capacity, not failure: retry rather than drop the finding.

    Bot-chat is excluded from the durable delivery queue, so a refusal that is
    discarded loses the payload permanently. `SESSION_NOT_OWNED` is the typed
    "busy, come back later" refusal a live Bot Chat owner raises while its turn
    runs; treating it as terminal is what silently lost scheduled bot findings.
    """
    busy = _completed(
        returncode=1,
        stderr=(
            "hermes-refusal-reason: SESSION_NOT_OWNED\n"
            "Session 20260101_000000_abcdef already has a live owner (cli, pid 1)."
        ),
    )
    attempts = []

    def fake_run(argv, **kwargs):
        attempts.append(argv)
        return busy if len(attempts) == 1 else _completed()

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery, "_sleep_unless_interrupted", lambda *_a, **_k: False):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    assert len(attempts) == 2, "a busy recipient must be retried, not dropped"
    assert err is None


def test_deliver_does_not_retry_a_genuine_failure():
    """Only capacity refusals retry; a real error must stay terminal and loud."""
    attempts = []

    def fake_run(argv, **kwargs):
        attempts.append(argv)
        return _completed(returncode=1, stderr="boom")

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery, "_sleep_unless_interrupted", lambda *_a, **_k: False):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    assert len(attempts) == 1, "a non-capacity failure must not be retried"
    assert err is not None and "boom" in err


def test_deliver_reports_error_when_recipient_stays_busy():
    """Exhausted retries must surface the refusal, never report a phantom success."""
    busy = _completed(
        returncode=1,
        stderr="hermes-refusal-reason: SESSION_NOT_OWNED\nSession busy.",
    )
    attempts = []

    def fake_run(argv, **kwargs):
        attempts.append(argv)
        return busy

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery, "_sleep_unless_interrupted", lambda *_a, **_k: False):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    assert len(attempts) > 1, "a persistently busy recipient must be retried"
    assert err is not None
    assert "SESSION_NOT_OWNED" in err or "busy" in err.lower()


def test_payload_mentioning_a_refusal_reason_is_not_retried():
    """A reason NAME inside ordinary output must not be mistaken for a refusal.

    Substring-matching the child's output would retry any job whose payload or
    traceback merely mentions SESSION_NOT_OWNED. Only the typed
    `hermes-refusal-reason:` marker line means the CLI actually refused.
    """
    attempts = []

    def fake_run(argv, **kwargs):
        attempts.append(argv)
        return _completed(
            returncode=1,
            stderr="ValueError: audit found SESSION_NOT_OWNED in 3 log lines",
        )

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery, "_sleep_unless_interrupted", lambda *_a, **_k: False):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    assert len(attempts) == 1, "output merely naming a reason must not be retried"
    assert err is not None


def test_coordination_unavailable_is_not_retried():
    """Unprovable ownership is not capacity: retrying it is the fail-open hole."""
    attempts = []

    def fake_run(argv, **kwargs):
        attempts.append(argv)
        return _completed(
            returncode=1,
            stderr="hermes-refusal-reason: SESSION_COORDINATION_UNAVAILABLE\nregistry unreadable",
        )

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery, "_sleep_unless_interrupted", lambda *_a, **_k: False):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    assert len(attempts) == 1, "SESSION_COORDINATION_UNAVAILABLE must never retry"
    assert err is not None


def test_capacity_refusal_is_detected_behind_a_long_preamble():
    """Classification reads the whole output; only the DISPLAYED error is truncated.

    The refusal marker is printed before the CLI's long explanatory text, so
    truncating to the last 500 chars before matching loses it and drops the
    payload — the exact bug this retry exists to prevent.
    """
    stderr = (
        "hermes-refusal-reason: SESSION_NOT_OWNED\n"
        + "Session already has a live owner. " * 40
    )
    assert len(stderr) > 500
    attempts = []

    def fake_run(argv, **kwargs):
        attempts.append(argv)
        return busy if len(attempts) == 1 else _completed()

    busy = _completed(returncode=1, stderr=stderr)

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery, "_sleep_unless_interrupted", lambda *_a, **_k: False):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    assert len(attempts) == 2, "a refusal marker before 500 chars of text must still retry"
    assert err is None


def test_busy_retries_stop_at_the_wall_clock_budget():
    """Elapsed wall time must not exceed the budget -- the invariant, not attempt count.

    An earlier version of this test asserted only `attempts < 4`. It passed while
    the implementation overshot to 1230s against a 900s budget, because each new
    attempt was still handed a FULL per-attempt timeout regardless of time left.
    Assert the clock, and assert a later attempt is clamped to what remains, or
    the budget is advertised without being enforced.
    """
    busy = _completed(
        returncode=1,
        stderr="hermes-refusal-reason: SESSION_NOT_OWNED\nbusy",
    )
    timeouts = []
    clock = {"t": 0.0}

    def fake_run(argv, **kwargs):
        timeouts.append(kwargs["timeout"])
        # Every attempt hangs for exactly the timeout it was granted.
        clock["t"] += kwargs["timeout"]
        return busy

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery.time, "monotonic", lambda: clock["t"]), \
            mock.patch.object(sched_delivery, "_sleep_unless_interrupted",
                              lambda d, *_a, **_k: (clock.__setitem__("t", clock["t"] + d), False)[1]):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    budget = sched_delivery._get_bot_chat_busy_budget_seconds()
    assert clock["t"] <= budget, (
        f"elapsed {clock['t']}s exceeded the advertised budget {budget}s")
    assert timeouts[-1] < timeouts[0], (
        "a later attempt must be clamped to the remaining budget, not given a full timeout")
    assert err is not None
    assert "budget" in err.lower()


def test_budget_deadline_timeout_reports_budget_not_recipient_hang():
    """A timeout caused by OUR deadline must not be reported as the recipient hanging."""
    clock = {"t": 0.0}

    def fake_run(argv, **kwargs):
        clock["t"] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(cmd="hermes", timeout=kwargs["timeout"])

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery.time, "monotonic", lambda: clock["t"]), \
            mock.patch.object(sched_delivery, "_get_bot_chat_busy_budget_seconds", lambda: 100.0), \
            mock.patch.object(sched_delivery, "_get_bot_chat_delivery_timeout", lambda: 600), \
            mock.patch.object(sched_delivery, "_sleep_unless_interrupted",
                              lambda d, *_a, **_k: (clock.__setitem__("t", clock["t"] + d), False)[1]):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    assert err is not None
    assert "budget" in err.lower(), "a budget-clamped timeout must be reported as budget exhaustion"
    assert clock["t"] <= 100.0


def test_budget_does_not_interrupt_a_fast_retry_ladder():
    """The realistic path (refusals return in seconds) must still retry normally."""
    busy = _completed(
        returncode=1,
        stderr="hermes-refusal-reason: SESSION_NOT_OWNED\nbusy",
    )
    attempts = []
    clock = {"t": 0.0}

    def fake_run(argv, **kwargs):
        attempts.append(argv)
        clock["t"] += 3.4  # measured live refusal latency
        return busy if len(attempts) == 1 else _completed()

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"), \
            mock.patch.object(sched_delivery.time, "monotonic", lambda: clock["t"]), \
            mock.patch.object(sched_delivery.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")

    assert err is None
    assert len(attempts) == 2


def test_retry_wait_abandons_on_gateway_shutdown():
    """A retry sleep must yield to shutdown, not outlive the cron drain.

    Gateway stop marks in-flight executions interrupted and drains cron for
    agent.cron_drain_timeout (default 30s). The backoff ladder sleeps up to 210s,
    so a bare time.sleep is still waiting when the drain gives up and the worker
    is SIGKILLed -- a wedged job instead of a cleanly interrupted one.
    """
    busy = _completed(
        returncode=1,
        stderr="hermes-refusal-reason: SESSION_NOT_OWNED\nbusy",
    )
    attempts = []

    def fake_run(argv, **kwargs):
        attempts.append(argv)
        # Shutdown lands while the first attempt is in flight.
        sched._interrupted_job_ids.add("j-shutdown")
        return busy

    try:
        with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
            err = _deliver_to_bot_chat({"id": "j-shutdown", "name": "n"}, "out", "")
    finally:
        sched._interrupted_job_ids.discard("j-shutdown")

    assert len(attempts) == 1, "shutdown must stop the ladder, not run it to exhaustion"
    assert err is not None
    assert "shutdown" in err.lower()


def test_retry_wait_sleeps_normally_without_shutdown():
    """The interruptible sleep must still wait when nothing interrupts it."""
    slept = []

    with mock.patch.object(sched_delivery.time, "sleep", lambda s: slept.append(s)):
        interrupted = sched_delivery._sleep_unless_interrupted(2.5, "quiet-job")

    assert interrupted is False
    assert slept, "it must actually sleep when not interrupted"
    assert max(slept) <= 1.0, "sleep must be sliced so shutdown is noticed promptly"


def test_deliver_message_carries_cron_attribution(tmp_path):
    """The injected turn must self-identify as scheduled output, not the user."""
    captured = {}

    def fake_run(argv, **kwargs):
        qf = argv[argv.index("--query-file") + 1]
        with open(qf, encoding="utf-8") as fh:
            captured["message"] = fh.read()
        return _completed()

    with mock.patch.object(sched.subprocess, "run", side_effect=fake_run), \
         mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        _deliver_to_bot_chat({"id": "j1", "name": "Daily digest"}, "the payload", "")

    assert 'Cronjob "Daily digest" output' in captured["message"]
    assert "not the user" in captured["message"]
    assert "the payload" in captured["message"]


# ── delivery-targets listing (UI pickers) ────────────────────────────────────

def test_delivery_targets_include_local_profiles():
    with mock.patch("hermes_cli.profiles.list_profile_names",
                    return_value=["default", "research"]):
        targets = sched_delivery.cron_delivery_targets()
    ids = [t["id"] for t in targets]
    assert f"{BOT_CHAT_PLATFORM}:default" in ids
    assert f"{BOT_CHAT_PLATFORM}:research" in ids
    bot_chat_entries = [t for t in targets if t["id"].startswith(BOT_CHAT_PLATFORM)]
    # No gateway home channel needed for bot-chat targets.
    assert all(t["home_target_set"] for t in bot_chat_entries)
