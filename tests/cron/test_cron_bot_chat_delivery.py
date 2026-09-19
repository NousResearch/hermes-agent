"""Bot Chat cron delivery: deliver='bot-chat[:<profile>]' injects job output
into a local profile's canonical Bot Chat session as a real inbound turn.

Covers token parsing, target resolution (own profile / named / missing),
preflight exemption, create-time validation, the subprocess delivery lane,
and the delivery-targets listing used by UI pickers.
"""

import os
import subprocess
import sys
import textwrap
import time
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
from hermes_cli.quiet_single_query import TURN_REPORT_FILE_ENV, write_turn_report


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

def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_deliver_runs_canonical_bot_chat_lane():
    """The subprocess must use the Bot Mode agent-to-agent chat lane:
    chat --in ~ -c "Bot Chat" --create-if-missing -Q --query-file <tmp>."""
    calls = {}

    def fake_run(argv, env, report_path, timeout):
        calls["argv"], calls["env"], calls["report_path"] = argv, env, report_path
        return _completed()

    with mock.patch.object(sched_delivery, "_run_bot_chat_turn", side_effect=fake_run), \
         mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "Daily digest"}, "the output", "")

    assert err is None
    argv = calls["argv"]
    # The running install's interpreter, not whatever `hermes` PATH names (same order as /update).
    assert argv[:3] == [sys.executable, "-m", "hermes_cli.main"]
    assert argv[3:5] == ["-p", "default"]  # do not follow active_profile
    assert "chat" in argv
    assert "Bot Chat" in argv
    assert "--create-if-missing" in argv
    assert "-Q" in argv
    assert "--query-file" in argv
    # Message rides a temp file, never inline argv (quote/expansion safety).
    assert not any("the output" in str(a) for a in argv)
    # The child reports its turn outcome here so the cap bounds the turn, not the exit linger.
    assert calls["env"][TURN_REPORT_FILE_ENV] == calls["report_path"]


def test_deliver_failure_returns_error_string():
    with mock.patch.object(
        sched_delivery, "_run_bot_chat_turn", return_value=_completed(returncode=1, stderr="boom")
    ), mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")
    assert err is not None
    assert "boom" in err


def test_deliver_failure_reports_both_streams_labeled():
    """A failed turn must keep stderr AND stdout, labeled — ``stderr or
    stdout`` discarded half the signal (#104056)."""
    with mock.patch.object(
        sched_delivery, "_run_bot_chat_turn",
        return_value=_completed(returncode=1, stdout="banner out", stderr="boom-err"),
    ), mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")
    assert err is not None
    assert "stderr: boom-err" in err
    assert "stdout: banner out" in err


def test_deliver_failure_banner_only_stdout_names_exit_code_not_banner():
    """The reported shape: empty stderr, stdout holding only the resume
    banner — the recorded error must say what happened (exit code, banner-only
    stdout) instead of echoing the banner as if it were a reason (#104056)."""
    banner = ('↻ Resumed session 20260905_121420_8084c7 "Bot Chat" (1 user message, 1 total messages)'
              '\n\nsession_id: 20260905_121420_8084c7')
    with mock.patch.object(
        sched_delivery, "_run_bot_chat_turn",
        return_value=_completed(returncode=1, stdout=banner, stderr=""),
    ), mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")
    assert err is not None
    assert "exit code 1" in err
    assert "stdout was only the resume banner" in err
    assert "Resumed session" not in err
    assert "stderr:" not in err


def test_deliver_failure_persisted_stdout_tail_is_short_and_redacted():
    """``last_delivery_error`` lands in jobs.json / the ledger: the model's
    answer on stdout is capped to a short tail and secrets are scrubbed."""
    answer = "x" * 5000 + "\nToken: sk-ant-api03-" + "A" * 80 + " done"
    with mock.patch.object(
        sched_delivery, "_run_bot_chat_turn",
        return_value=_completed(returncode=1, stdout=answer, stderr="boom-err"),
    ), mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")
    assert err is not None
    stdout_part = err.split("stdout: ", 1)[1]
    assert len(stdout_part) <= 200
    assert "sk-ant-api03-" + "A" * 80 not in err


def test_deliver_timeout_returns_error_string():
    with mock.patch.object(
        sched_delivery, "_run_bot_chat_turn",
        side_effect=subprocess.TimeoutExpired(cmd="hermes", timeout=600),
    ), mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        err = _deliver_to_bot_chat({"id": "j1", "name": "n"}, "out", "")
    assert err is not None
    assert "timed out" in err


def test_deliver_message_carries_cron_attribution(tmp_path):
    """The injected turn must self-identify as scheduled output, not the user."""
    captured = {}

    def fake_run(argv, env, report_path, timeout):
        qf = argv[argv.index("--query-file") + 1]
        with open(qf, encoding="utf-8") as fh:
            captured["message"] = fh.read()
        return _completed()

    with mock.patch.object(sched_delivery, "_run_bot_chat_turn", side_effect=fake_run), \
         mock.patch.object(sched_delivery.shutil, "which", return_value="/usr/bin/hermes"):
        _deliver_to_bot_chat({"id": "j1", "name": "Daily digest"}, "the payload", "")

    assert 'Cronjob "Daily digest" output' in captured["message"]
    assert "not the user" in captured["message"]
    assert "the payload" in captured["message"]


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sched_delivery.__file__)))


def _child_env() -> dict:
    """The stand-in child imports ``hermes_cli`` from this checkout, like the real ``-m hermes_cli.main``."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (_REPO_ROOT, os.environ.get("PYTHONPATH")) if p)}


def test_turn_report_books_the_delivery_while_the_child_still_lingers(tmp_path):
    """The cap bounds the TURN: a child that reported its turn and then lingers for a nested
    notify_on_complete reply (bounded by oneshot_completion_wait_seconds, default == the cap) is
    booked from the report promptly and is NOT killed (#113608)."""
    report = tmp_path / "turn.json"
    child = textwrap.dedent("""
        import os, time
        from hermes_cli.quiet_single_query import TURN_REPORT_FILE_ENV, write_turn_report
        write_turn_report(os.environ.pop(TURN_REPORT_FILE_ENV), exit_code=0)
        time.sleep(30)
        """)
    procs, real_popen = [], subprocess.Popen

    def spy(*args, **kwargs):
        procs.append(real_popen(*args, **kwargs))
        return procs[-1]

    started = time.monotonic()
    try:
        with mock.patch.object(sched_delivery.subprocess, "Popen", side_effect=spy):
            result = sched_delivery._run_bot_chat_turn(
                [sys.executable, "-c", child], {**_child_env(), TURN_REPORT_FILE_ENV: str(report)}, str(report), timeout=10)
        elapsed = time.monotonic() - started
        assert result.returncode == 0
        assert elapsed < 8, elapsed
        assert procs[0].poll() is None, "the lingering child must survive the booking"
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=10)


def test_turn_that_never_ends_is_still_killed_at_the_cap(tmp_path):
    """Control: with no turn report the cap stays the guard it always was."""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        sched_delivery._run_bot_chat_turn(
            [sys.executable, "-c", "import time; time.sleep(30)"], _child_env(), str(tmp_path / "turn.json"), timeout=1)
    assert time.monotonic() - started < 8


@pytest.mark.linux_only
def test_bot_chat_turn_decodes_stray_stderr_bytes_lossily(tmp_path):
    """A stray non-UTF-8 byte on the delivery child's stderr (e.g. a grandchild
    sharing the pipe interleaving a partial multi-byte write) must not raise
    UnicodeDecodeError in the drain thread — the turn is booked and the stream
    tail survives with U+FFFD (#105582)."""
    child = "import os, sys; os.write(2, b'noise before \\x80 after\\n'); sys.exit(0)"
    result = sched_delivery._run_bot_chat_turn(
        [sys.executable, "-c", child], _child_env(), str(tmp_path / "turn.json"), timeout=15)

    assert result.returncode == 0
    assert result.stderr == "noise before \ufffd after\n"


@pytest.mark.linux_only
def test_bot_chat_turn_failure_tail_decodes_lossily(tmp_path):
    """The exit-1 tail is still surfaced (with U+FFFD for the bad byte) instead of
    vanishing when the drain thread dies at the first undecodable byte (#105582)."""
    child = "import os, sys; os.write(2, b'boom before \\x80 after\\n'); sys.exit(1)"
    result = sched_delivery._run_bot_chat_turn(
        [sys.executable, "-c", child], _child_env(), str(tmp_path / "turn.json"), timeout=15)

    assert result.returncode == 1
    assert result.stderr == "boom before \ufffd after\n"


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
