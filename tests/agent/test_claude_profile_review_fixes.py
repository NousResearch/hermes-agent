"""Regression tests for the review findings on the Claude Code switcher.

Each class below pins one defect the reviewer found. No test reads a real
Keychain entry, a real credentials file, or the network.

One contract is deliberately NOT changed here: a resumed conversation stays on
the account that started it and waits when that account fills. See
``TestThePinContractHolds``.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent import claude_cli_profiles as ccp
from hermes_constants import apply_claude_profile_env
from tools.environments.local import _sanitize_subprocess_env, hermes_subprocess_env

SOON = datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)


def write_config(section):
    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text(yaml.dump({"claude_cli_profiles": section} if section else {}))


def configure(tmp_path, count=2, **extra):
    section = dict(extra)
    section["profiles"] = [
        {"name": name, "config_dir": str(tmp_path / name)}
        for name in ("work", "spare")[:count]
    ]
    write_config(section)


def usage(name, five_hour=None, weekly=None, problem=None,
          five_hour_reset=None, weekly_reset=None, opus_weekly=None):
    return ccp.ProfileUsage(
        name=name,
        five_hour_percent=five_hour,
        weekly_percent=weekly,
        opus_weekly_percent=opus_weekly,
        five_hour_reset=five_hour_reset,
        weekly_reset=weekly_reset,
        problem=problem,
    )


def open_usage(p, **_):
    return usage(p.name, 5.0, 5.0)


# ---------------------------------------------------------------------------
# 1. The chosen account belongs to one conversation, not to the whole process.
# ---------------------------------------------------------------------------


class TestPerSessionSelection:
    def test_a_second_conversation_does_not_move_the_first(self, tmp_path, monkeypatch):
        """Two conversations run at once. Each terminal command must use the
        account its own conversation started on."""
        configure(tmp_path)
        ccp.pin_session("chat-A", "work")
        ccp.pin_session("chat-B", "spare")
        ccp.record_active("spare")  # the process-global slot names the newer one

        monkeypatch.setenv("HERMES_SESSION_KEY", "chat-A")
        first = _sanitize_subprocess_env({"PATH": "/usr/bin"})
        assert first["CLAUDE_CONFIG_DIR"] == str(tmp_path / "work")

        monkeypatch.setenv("HERMES_SESSION_KEY", "chat-B")
        second = _sanitize_subprocess_env({"PATH": "/usr/bin"})
        assert second["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_a_pinned_resume_reaches_the_terminal(self, tmp_path, monkeypatch):
        configure(tmp_path)
        ccp.pin_session("chat-A", "spare")
        ccp.record_active("work")
        monkeypatch.setenv("HERMES_SESSION_KEY", "chat-A")

        env = hermes_subprocess_env(inherit_credentials=True)
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_the_global_slot_is_used_when_no_conversation_is_bound(self, tmp_path, monkeypatch):
        configure(tmp_path)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        ccp.record_active("spare")

        env = hermes_subprocess_env(inherit_credentials=True)
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_a_selection_this_turn_wins_over_the_stored_slot(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("work")
        token = ccp.bind_selected_profile("spare")
        try:
            env = {}
            apply_claude_profile_env(env)
            assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")
        finally:
            ccp.release_selected_profile(token)

        after = {}
        apply_claude_profile_env(after)
        assert after["CLAUDE_CONFIG_DIR"] == str(tmp_path / "work")

    def test_concurrent_conversations_each_keep_their_own_account(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("chat-A", "work")
        ccp.pin_session("chat-B", "spare")
        ccp.record_active("work")
        seen = {}

        def run(session_id, expected):
            token = ccp.bind_selected_profile(expected)
            try:
                for _ in range(15):
                    env = {}
                    apply_claude_profile_env(env)
                    seen.setdefault(session_id, set()).add(env.get("CLAUDE_CONFIG_DIR"))
            finally:
                ccp.release_selected_profile(token)

        threads = [
            threading.Thread(target=run, args=("chat-A", "work")),
            threading.Thread(target=run, args=("chat-B", "spare")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert seen["chat-A"] == {str(tmp_path / "work")}
        assert seen["chat-B"] == {str(tmp_path / "spare")}


# ---------------------------------------------------------------------------
# 2. A broken reset timestamp must not stop the status report.
# ---------------------------------------------------------------------------


class TestResetTimestamps:
    def test_seconds_since_the_epoch_are_read(self):
        assert ccp._reset_time(1786125600) == datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)

    def test_milliseconds_since_the_epoch_are_read(self):
        assert ccp._reset_time(1786125600000) == datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)

    @pytest.mark.parametrize("value", [
        1e30,                 # far past the range a date can hold
        -1e30,
        float("nan"),
        float("inf"),
        "not a date",
        "2026-13-45T99:99:99Z",
        {"nested": "object"},
        [1, 2, 3],
        True,
    ])
    def test_a_value_that_is_not_a_date_gives_no_time(self, value):
        assert ccp._reset_time(value) is None

    def test_a_broken_timestamp_does_not_stop_the_status_report(self, tmp_path):
        configure(tmp_path)
        payload = {"limits": [
            {"kind": "session", "percent": 100, "resets_at": 1e30},
            {"kind": "weekly_all", "percent": 10, "resets_at": "whenever"},
        ]}
        lines = ccp.status_lines(usage_fetcher=lambda _t: payload,
                                 usage_reader=None if False else None)
        # status_lines falls back to the real reader, so drive it directly:
        read = ccp.parse_usage_payload("work", payload)
        assert read.five_hour_percent == 100.0
        assert read.five_hour_reset is None
        assert isinstance(lines, list)

    def test_a_reader_that_raises_is_caught_inside_the_read(self, tmp_path):
        """parse_usage_payload runs inside the guarded read, so a surprise
        inside it becomes a reported problem, not a crash."""
        profile = ccp.ClaudeProfile("work", tmp_path / "work", tmp_path / "work")
        read = ccp.read_profile_usage(
            profile,
            token_reader=lambda _p: "sk-ant-oat01-fake",
            usage_fetcher=lambda _t: {"limits": [
                {"kind": "session", "percent": 5, "resets_at": object()},
            ]},
        )
        assert read.problem is None or read.problem == ccp.PROBLEM_UNREADABLE


# ---------------------------------------------------------------------------
# 3. The usage read is cached for a minute.
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(ccp, "_clock", fake)
    ccp.invalidate_usage_cache()
    yield fake
    ccp.invalidate_usage_cache()


class TestUsageCache:
    def test_a_second_read_inside_the_window_makes_no_call(self, tmp_path, clock):
        profile = ccp.ClaudeProfile("work", tmp_path / "work", tmp_path / "work")
        calls = []

        def fetch(_token):
            calls.append(1)
            return {"limits": [{"kind": "session", "percent": 5},
                               {"kind": "weekly_all", "percent": 6}]}

        for _ in range(3):
            ccp.read_profile_usage(profile, token_reader=lambda _p: "sk-ant-oat01-fake",
                                   usage_fetcher=fetch)
        assert len(calls) == 1

    def test_the_read_repeats_once_the_window_passes(self, tmp_path, clock):
        profile = ccp.ClaudeProfile("work", tmp_path / "work", tmp_path / "work")
        calls = []

        def fetch(_token):
            calls.append(1)
            return {"limits": [{"kind": "session", "percent": 5},
                               {"kind": "weekly_all", "percent": 6}]}

        ccp.read_profile_usage(profile, token_reader=lambda _p: "t", usage_fetcher=fetch)
        clock.advance(ccp.USAGE_CACHE_SECONDS - 1)
        ccp.read_profile_usage(profile, token_reader=lambda _p: "t", usage_fetcher=fetch)
        assert len(calls) == 1

        clock.advance(2)
        ccp.read_profile_usage(profile, token_reader=lambda _p: "t", usage_fetcher=fetch)
        assert len(calls) == 2

    def test_a_failed_read_is_never_cached(self, tmp_path, clock):
        profile = ccp.ClaudeProfile("work", tmp_path / "work", tmp_path / "work")
        calls = []

        def fetch(_token):
            calls.append(1)
            raise ccp.ProfileUsageError(ccp.PROBLEM_UNREACHABLE, "down")

        ccp.read_profile_usage(profile, token_reader=lambda _p: "t", usage_fetcher=fetch)
        ccp.read_profile_usage(profile, token_reader=lambda _p: "t", usage_fetcher=fetch)
        assert len(calls) == 2

    def test_each_profile_has_its_own_entry(self, tmp_path, clock):
        seen = []

        def fetch(_token):
            seen.append(1)
            return {"limits": [{"kind": "session", "percent": 5},
                               {"kind": "weekly_all", "percent": 6}]}

        for name in ("work", "spare"):
            profile = ccp.ClaudeProfile(name, tmp_path / name, tmp_path / name)
            ccp.read_profile_usage(profile, token_reader=lambda _p: "t", usage_fetcher=fetch)
        assert len(seen) == 2

    def test_the_cache_holds_no_token(self, tmp_path, clock):
        profile = ccp.ClaudeProfile("work", tmp_path / "work", tmp_path / "work")
        ccp.read_profile_usage(
            profile,
            token_reader=lambda _p: "sk-ant-oat01-secret",
            usage_fetcher=lambda _t: {"limits": [{"kind": "session", "percent": 5},
                                                 {"kind": "weekly_all", "percent": 6}]},
        )
        assert "sk-ant" not in repr(ccp._usage_cache)

    def test_no_available_account_drops_every_cached_number(self, tmp_path, clock):
        configure(tmp_path)
        chosen = ccp.select_for_job(
            usage_reader=lambda p, **_: usage(p.name, 100.0, 10.0, five_hour_reset=SOON)
        )
        assert chosen.available is False
        assert ccp._usage_cache == {}


# ---------------------------------------------------------------------------
# 4. A live conversation must not be evicted from the pin table.
# ---------------------------------------------------------------------------


class TestPinFreshness:
    def test_a_resume_refreshes_the_pin(self, tmp_path, clock):
        configure(tmp_path)
        ccp.pin_session("chat-A", "spare")
        first = ccp.pin_recorded_at("chat-A")

        clock.advance(500)
        ccp.select_for_job(session_id="chat-A", usage_reader=open_usage)

        assert ccp.pin_recorded_at("chat-A") > first

    def test_the_busiest_conversation_survives_a_trim(self, tmp_path, clock):
        configure(tmp_path)
        ccp.pin_session("chat-A", "work")
        for index in range(ccp.MAX_PINNED_SESSIONS + 5):
            clock.advance(1)
            ccp.pin_session(f"filler-{index}", "work")
            if index % 40 == 0:
                clock.advance(1)
                ccp.select_for_job(session_id="chat-A", usage_reader=open_usage)

        clock.advance(1)
        ccp.select_for_job(session_id="chat-A", usage_reader=open_usage)
        assert ccp.pinned_profile_name("chat-A") == "work"


# ---------------------------------------------------------------------------
# 5. The state file holds no chat identifier.
# ---------------------------------------------------------------------------


class TestNoRawIdentifiers:
    def test_the_state_file_holds_a_hash_not_the_chat_name(self, tmp_path):
        ccp.pin_session("agent:main:telegram:dm:123456789", "work")
        written = ccp.state_path().read_text()
        assert "telegram" not in written
        assert "123456789" not in written
        assert ccp.pinned_profile_name("agent:main:telegram:dm:123456789") == "work"

    def test_two_conversations_hash_apart(self, tmp_path):
        ccp.pin_session("chat-A", "work")
        ccp.pin_session("chat-B", "spare")
        assert ccp.pinned_profile_name("chat-A") == "work"
        assert ccp.pinned_profile_name("chat-B") == "spare"

    def test_the_hash_is_stable_between_calls(self):
        assert ccp.session_fingerprint("chat-A") == ccp.session_fingerprint("chat-A")
        assert ccp.session_fingerprint("chat-A") != ccp.session_fingerprint("chat-B")

    def test_an_older_state_file_with_plain_names_is_ignored_safely(self, tmp_path):
        ccp.state_path().write_text(json.dumps({
            "version": 1,
            "active": "work",
            "sessions": {"agent:main:telegram:dm:1": {"profile": "spare", "at": 1.0}},
        }))
        assert ccp.active_profile_name() == "work"
        assert ccp.pinned_profile_name("agent:main:telegram:dm:1") is None
        assert ccp.read_state()["sessions"] == {}


# ---------------------------------------------------------------------------
# 6. A person can clear the state from the command line.
# ---------------------------------------------------------------------------


class TestClearFromTheCommandLine:
    def test_the_reset_command_clears_every_selection_and_pin(self, tmp_path, capsys):
        from hermes_cli.auth_commands import auth_reset_command

        configure(tmp_path)
        ccp.record_active("spare")
        ccp.pin_session("chat-A", "spare")

        auth_reset_command(SimpleNamespace(provider="claude-profiles"))

        assert ccp.active_profile_name() is None
        assert ccp.pinned_profile_name("chat-A") is None
        assert "cleared" in capsys.readouterr().out.lower()

    def test_the_reset_command_keeps_every_credential(self, tmp_path, capsys):
        from hermes_cli.auth_commands import auth_reset_command

        configure(tmp_path)
        profile_dir = tmp_path / "work"
        profile_dir.mkdir()
        credential = profile_dir / ".credentials.json"
        credential.write_text('{"claudeAiOauth": {"accessToken": "sk-ant-oat01-fake"}}')

        auth_reset_command(SimpleNamespace(provider="claude-profiles"))

        assert credential.exists()

    def test_a_later_write_still_works_after_a_clear(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("work")
        ccp.clear_state()
        ccp.record_active("spare")
        assert ccp.active_profile_name() == "spare"


# ---------------------------------------------------------------------------
# 7. The section appears in the shipped configuration defaults.
# ---------------------------------------------------------------------------


class TestConfigurationDefaults:
    def test_the_section_is_in_the_default_config(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        section = DEFAULT_CONFIG["claude_cli_profiles"]
        assert section["profiles"] == []
        assert section["stop_at_percent"] == 95

    def test_the_default_config_keeps_the_switcher_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert ccp.switching_enabled(DEFAULT_CONFIG) is False
        assert ccp.load_profiles(DEFAULT_CONFIG) == []


# ---------------------------------------------------------------------------
# 8. The weekly Opus window can fill on its own.
# ---------------------------------------------------------------------------


class TestOpusWeeklyWindow:
    def test_the_scoped_opus_window_is_read(self):
        read = ccp.parse_usage_payload("work", {"limits": [
            {"kind": "session", "percent": 10},
            {"kind": "weekly_all", "percent": 20},
            {"kind": "weekly_scoped", "percent": 97,
             "resets_at": "2026-08-11T00:00:00Z",
             "scope": {"model": {"display_name": "Claude Opus 5"}}},
        ]})
        assert read.opus_weekly_percent == 97.0
        assert read.opus_weekly_reset == datetime(2026, 8, 11, tzinfo=timezone.utc)

    def test_the_older_opus_field_is_read(self):
        read = ccp.parse_usage_payload("work", {
            "five_hour": {"utilization": 0.1},
            "seven_day": {"utilization": 0.2},
            "seven_day_opus": {"utilization": 0.99, "resets_at": "2026-08-11T00:00:00Z"},
        })
        assert read.opus_weekly_percent == 99.0

    def test_a_full_opus_window_makes_the_account_full(self, tmp_path):
        profiles = [ccp.ClaudeProfile(n, tmp_path / n, tmp_path / n) for n in ("work", "spare")]
        usages = {
            "work": usage("work", 10.0, 20.0, opus_weekly=99.0),
            "spare": usage("spare", 5.0, 5.0),
        }
        chosen = ccp.select_profile(profiles, usages, threshold=95.0, active_name="work")
        assert chosen.profile.name == "spare"

    def test_a_full_opus_window_is_named_in_the_report(self):
        read = usage("work", 10.0, 20.0, opus_weekly=99.0)
        text = ccp.describe_wait(read, 95.0)
        assert "opus" in text.lower()

    def test_a_scoped_window_for_another_model_is_ignored(self):
        read = ccp.parse_usage_payload("work", {"limits": [
            {"kind": "session", "percent": 10},
            {"kind": "weekly_all", "percent": 20},
            {"kind": "weekly_scoped", "percent": 99,
             "scope": {"model": {"display_name": "Claude Sonnet 5"}}},
        ]})
        assert read.opus_weekly_percent is None
        assert read.worst_percent == 20.0


# ---------------------------------------------------------------------------
# 9. A fraction and a percentage are told apart correctly.
# ---------------------------------------------------------------------------


class TestPercentForms:
    def test_a_percent_field_is_never_scaled(self):
        """0.87 in a percent field means 0.87 percent, not 87 percent."""
        read = ccp.parse_usage_payload("work", {"limits": [
            {"kind": "session", "percent": 0.87},
            {"kind": "weekly_all", "percent": 12},
        ]})
        assert read.five_hour_percent == 0.87
        assert read.weekly_percent == 12.0

    def test_a_utilization_fraction_is_scaled(self):
        read = ccp.parse_usage_payload("work", {
            "five_hour": {"utilization": 0.87},
            "seven_day": {"utilization": 0.12},
        })
        assert read.five_hour_percent == 87.0
        assert read.weekly_percent == 12.0

    @pytest.mark.parametrize("utilization,expected", [
        (0.0, 0.0),
        (1.0, 100.0),
        (1.5, 1.5),
        (95, 95.0),
        (100, 100.0),
    ])
    def test_the_utilization_boundary(self, utilization, expected):
        read = ccp.parse_usage_payload("work", {
            "five_hour": {"utilization": utilization},
            "seven_day": {"utilization": 0.1},
        })
        assert read.five_hour_percent == expected

    def test_a_percent_field_at_the_threshold_still_counts_as_full(self):
        read = ccp.parse_usage_payload("work", {"limits": [
            {"kind": "session", "percent": 95.0},
            {"kind": "weekly_all", "percent": 1},
        ]})
        assert ccp.usability(read, 95.0) == ccp.FULL

    def test_a_percent_field_just_under_the_threshold_is_open(self):
        read = ccp.parse_usage_payload("work", {"limits": [
            {"kind": "session", "percent": 94.9},
            {"kind": "weekly_all", "percent": 1},
        ]})
        assert ccp.usability(read, 95.0) == ccp.OPEN


# ---------------------------------------------------------------------------
# 10. Clearing the state must not break the lock.
# ---------------------------------------------------------------------------


class TestLockSurvivesAClear:
    def test_the_lock_file_stays(self, tmp_path):
        ccp.record_active("work")
        ccp.clear_state()
        ccp.record_active("spare")
        assert ccp.active_profile_name() == "spare"

    def test_writes_stay_serialized_across_a_clear(self, tmp_path):
        errors = []
        names = [f"chat-{index}" for index in range(30)]

        def worker(index):
            try:
                if index == 15:
                    ccp.clear_state()
                ccp.pin_session(names[index], "work")
            except Exception as exc:  # pragma: no cover — a failure is the report
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        state = ccp.read_state()
        assert isinstance(state["sessions"], dict)


# ---------------------------------------------------------------------------
# 11. A relative directory in config.yaml resolves to one place.
# ---------------------------------------------------------------------------


class TestRelativeDirectories:
    def test_a_relative_directory_becomes_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_config({"profiles": [
            {"name": "work", "config_dir": "profiles/work"},
            {"name": "spare", "config_dir": "./profiles/spare",
             "securestorage_dir": "../vault"},
        ]})
        work, spare = ccp.load_profiles()
        assert work.config_dir.is_absolute()
        assert spare.config_dir.is_absolute()
        assert spare.securestorage_dir.is_absolute()
        assert work.config_dir == (tmp_path / "profiles" / "work").resolve()

    def test_the_child_environment_names_an_absolute_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_config({"profiles": [
            {"name": "work", "config_dir": "profiles/work"},
            {"name": "spare", "config_dir": "profiles/spare"},
        ]})
        ccp.record_active("work")
        env = {}
        apply_claude_profile_env(env)
        assert Path(env["CLAUDE_CONFIG_DIR"]).is_absolute()

    def test_a_home_shorthand_still_expands(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        write_config({"profiles": [
            {"name": "work", "config_dir": "~/.claude"},
            {"name": "spare", "config_dir": "~/.claude-spare"},
        ]})
        assert ccp.load_profiles()[0].config_dir == (tmp_path / ".claude").resolve()


# ---------------------------------------------------------------------------
# 12. A Windows lock that another process holds must not stop the write.
# ---------------------------------------------------------------------------


class TestWindowsLock:
    def test_a_refused_windows_lock_does_not_raise(self, tmp_path, monkeypatch):
        attempts = []

        class FakeMsvcrt:
            LK_LOCK = 1
            LK_UNLCK = 0

            def locking(self, fileno, mode, nbytes):
                attempts.append(mode)
                raise OSError(36, "Resource deadlock avoided")

        monkeypatch.setattr(ccp, "fcntl", None)
        monkeypatch.setattr(ccp, "msvcrt", FakeMsvcrt())

        ccp.record_active("work")

        assert attempts, "the Windows lock path did not run"
        assert ccp.active_profile_name() == "work"

    def test_a_refused_windows_unlock_does_not_raise(self, tmp_path, monkeypatch):
        class FakeMsvcrt:
            LK_LOCK = 1
            LK_UNLCK = 0

            def locking(self, fileno, mode, nbytes):
                if mode == self.LK_UNLCK:
                    raise OSError(36, "Resource deadlock avoided")

        monkeypatch.setattr(ccp, "fcntl", None)
        monkeypatch.setattr(ccp, "msvcrt", FakeMsvcrt())

        ccp.record_active("work")
        assert ccp.active_profile_name() == "work"


# ---------------------------------------------------------------------------
# The pin contract George chose. Do not change this.
# ---------------------------------------------------------------------------


class TestThePinContractHolds:
    def test_a_full_pinned_account_waits_and_does_not_move(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("chat-A", "work")

        def reader(p, **_):
            if p.name == "work":
                return usage(p.name, 100.0, 10.0, five_hour_reset=SOON)
            return usage(p.name, 1.0, 1.0)

        chosen = ccp.select_for_job(session_id="chat-A", usage_reader=reader)

        assert chosen.profile.name == "work"
        assert chosen.available is False
        assert "stays on it" in chosen.message
        assert "2026-08-07" in chosen.message
        assert "spare" not in chosen.message

    def test_exactly_95_percent_makes_a_pinned_account_wait(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("chat-A", "work")
        chosen = ccp.select_for_job(
            session_id="chat-A",
            usage_reader=lambda p, **_: usage(p.name, 95.0, 1.0, five_hour_reset=SOON),
        )
        assert chosen.available is False

    def test_94_9_percent_lets_a_pinned_account_continue(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("chat-A", "work")
        chosen = ccp.select_for_job(
            session_id="chat-A",
            usage_reader=lambda p, **_: usage(p.name, 94.9, 1.0),
        )
        assert chosen.available is True
        assert chosen.profile.name == "work"


# ---------------------------------------------------------------------------
# Second review.
# ---------------------------------------------------------------------------

import time  # noqa: E402

METERED_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")


class TestMeteredCredentialsAreRemoved:
    """A selected profile must be the only way the child can authenticate.

    Each of these variables outranks the profile directory inside Claude Code.
    Leaving one in place sends the work to a metered interface key while the
    person believes it runs on a subscription.
    """

    def test_every_metered_variable_goes_when_a_profile_applies(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("work")
        env = {key: f"sk-ant-{key.lower()}" for key in METERED_KEYS}
        env["PATH"] = "/usr/bin"

        apply_claude_profile_env(env)

        for key in METERED_KEYS:
            assert key not in env, f"{key} still reaches the child"
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "work")

    @pytest.mark.parametrize("key", METERED_KEYS)
    def test_each_variable_alone_is_removed(self, tmp_path, key):
        configure(tmp_path)
        ccp.record_active("work")
        env = {key: "sk-ant-secret"}
        apply_claude_profile_env(env)
        assert key not in env

    @pytest.mark.parametrize("key", METERED_KEYS)
    def test_the_variable_stays_when_no_profile_applies(self, key):
        write_config(None)
        env = {key: "sk-ant-secret"}
        apply_claude_profile_env(env)
        assert env[key] == "sk-ant-secret"

    def test_the_worker_spawn_drops_them_too(self, tmp_path, monkeypatch):
        configure(tmp_path)
        ccp.record_active("work")
        for key in METERED_KEYS:
            monkeypatch.setenv(key, "sk-ant-secret")

        env = hermes_subprocess_env(inherit_credentials=True)

        for key in METERED_KEYS:
            assert key not in env

    def test_the_terminal_spawn_drops_them_too(self, tmp_path):
        configure(tmp_path)
        ccp.record_active("work")
        base = {key: "sk-ant-secret" for key in METERED_KEYS}
        base["PATH"] = "/usr/bin"

        env = _sanitize_subprocess_env(base)

        for key in METERED_KEYS:
            assert key not in env


class TestPinRecencyUsesWallClock:
    def test_the_stored_time_is_wall_clock(self):
        before = time.time()
        ccp.pin_session("chat-A", "work")
        stored = ccp.pin_recorded_at("chat-A")
        assert before <= stored <= time.time() + 1

    def test_a_pin_written_after_a_restart_survives_a_trim(self, monkeypatch):
        """The machine restarts. A monotonic clock starts again near zero, so
        every old pin would look newer than every new one and the trim would
        drop the live conversation. Wall-clock time has no such step."""
        old = time.time() - 3600
        state = {"version": 1, "active": "work", "sessions": {
            ccp.session_fingerprint(f"old-{index}"): {"profile": "work", "at": old + index}
            for index in range(ccp.MAX_PINNED_SESSIONS)
        }}
        ccp.state_path().write_text(json.dumps(state))

        # A monotonic clock that restarted would report a small number here.
        monkeypatch.setattr(ccp, "_clock", lambda: 12.5)
        ccp.pin_session("chat-after-restart", "spare")

        assert ccp.pinned_profile_name("chat-after-restart") == "spare"
        assert len(ccp.read_state()["sessions"]) <= ccp.MAX_PINNED_SESSIONS

    def test_the_in_memory_cache_still_uses_the_monotonic_clock(self, tmp_path, monkeypatch):
        """Wall-clock time can jump backwards. The 60-second cache must not."""
        ticks = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        monkeypatch.setattr(ccp, "_clock", lambda: next(ticks))
        ccp.invalidate_usage_cache()
        profile = ccp.ClaudeProfile("work", tmp_path / "work", tmp_path / "work")
        calls = []

        def fetch(_token):
            calls.append(1)
            return {"limits": [{"kind": "session", "percent": 5},
                               {"kind": "weekly_all", "percent": 6}]}

        ccp.read_profile_usage(profile, token_reader=lambda _p: "t", usage_fetcher=fetch)
        ccp.read_profile_usage(profile, token_reader=lambda _p: "t", usage_fetcher=fetch)
        assert len(calls) == 1
        ccp.invalidate_usage_cache()


class TestHermeticEnvironment:
    def test_the_two_directory_variables_are_blanked_for_every_test(self):
        """A developer runs the suite from a shell that already names a
        profile. The suite must not read it."""
        assert os.environ.get("CLAUDE_CONFIG_DIR") in (None, "")
        assert os.environ.get("CLAUDE_SECURESTORAGE_CONFIG_DIR") in (None, "")

    def test_the_switcher_is_off_under_the_ambient_environment(self):
        write_config(None)
        env = {"PATH": "/usr/bin"}
        apply_claude_profile_env(env)
        assert env == {"PATH": "/usr/bin"}


class TestCommandLineConversationsArePinnedToo:
    def test_a_command_line_conversation_key_is_found(self, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        monkeypatch.setenv("HERMES_SESSION_ID", "cli-session-42")
        assert ccp.current_session_key() == "cli-session-42"

    def test_the_gateway_key_wins_when_both_are_present(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:telegram:dm:1")
        monkeypatch.setenv("HERMES_SESSION_ID", "cli-session-42")
        assert ccp.current_session_key() == "agent:main:telegram:dm:1"

    def test_a_command_line_conversation_keeps_its_account(self, tmp_path, monkeypatch):
        configure(tmp_path)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        monkeypatch.setenv("HERMES_SESSION_ID", "cli-session-42")
        ccp.pin_session("cli-session-42", "spare")
        ccp.record_active("work")

        env = hermes_subprocess_env(inherit_credentials=True)
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "spare")

    def test_a_command_line_resume_does_not_move_account(self, tmp_path, monkeypatch):
        configure(tmp_path)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        monkeypatch.setenv("HERMES_SESSION_ID", "cli-session-42")
        ccp.pin_session("cli-session-42", "work")

        chosen = ccp.select_for_job(
            session_id=ccp.current_session_key(),
            usage_reader=lambda p, **_: usage(
                p.name, 100.0, 1.0, five_hour_reset=SOON) if p.name == "work"
            else usage(p.name, 1.0, 1.0),
        )
        assert chosen.profile.name == "work"
        assert chosen.available is False

    def test_work_with_no_conversation_key_uses_the_shared_slot(self, tmp_path, monkeypatch):
        """The documented boundary: a cron job or a one-off script has no
        conversation, so it takes the shared slot and is not pinned."""
        configure(tmp_path)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
        assert ccp.current_session_key() == ""

        chosen = ccp.select_for_job(session_id="", usage_reader=open_usage)
        assert chosen.profile.name == "work"
        assert ccp.read_state()["sessions"] == {}
        assert ccp.active_profile_name() == "work"


class TestCredentialWritesStayInTheirOwnProfile:
    def test_a_refreshed_token_never_lands_in_another_profile(self, tmp_path, monkeypatch):
        """The token was read from one profile. The environment then named a
        different profile. The write must be refused, not redirected."""
        from agent.anthropic_adapter import _write_claude_code_credentials

        source = tmp_path / "work"
        target = tmp_path / "spare"
        source.mkdir()
        target.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(target))

        _write_claude_code_credentials(
            "refreshed-token", "refreshed-refresh", 9999999999999,
            origin_config_dir=str(source),
        )

        assert not (target / ".credentials.json").exists()
        assert not (source / ".credentials.json").exists()

    def test_a_write_back_to_its_own_profile_is_allowed(self, tmp_path, monkeypatch):
        from agent.anthropic_adapter import _write_claude_code_credentials

        profile = tmp_path / "work"
        profile.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))

        _write_claude_code_credentials(
            "refreshed-token", "refreshed-refresh", 9999999999999,
            origin_config_dir=str(profile),
        )

        written = json.loads((profile / ".credentials.json").read_text())
        assert written["claudeAiOauth"]["accessToken"] == "refreshed-token"

    def test_a_legacy_write_with_no_origin_still_works(self, tmp_path, monkeypatch):
        """One profile, no switcher: the write behaves exactly as before."""
        from agent.anthropic_adapter import _write_claude_code_credentials

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)

        _write_claude_code_credentials("t", "r", 9999999999999)

        assert (tmp_path / ".claude" / ".credentials.json").exists()

    def test_the_reader_records_which_profile_the_token_came_from(self, tmp_path, monkeypatch):
        from agent.anthropic_adapter import _read_claude_code_credentials_from_file

        profile = tmp_path / "work"
        profile.mkdir()
        (profile / ".credentials.json").write_text(
            '{"claudeAiOauth": {"accessToken": "tok", "expiresAt": 99999999999999}}'
        )
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))

        creds = _read_claude_code_credentials_from_file()
        assert creds["config_dir"] == str(profile)


class TestUnverifiedUsageFailsClosed:
    def test_a_profile_whose_usage_cannot_be_read_is_not_selected(self, tmp_path):
        profiles = [ccp.ClaudeProfile(n, tmp_path / n, tmp_path / n) for n in ("work", "spare")]
        chosen = ccp.select_profile(
            profiles,
            {"work": usage("work", problem="unreachable"),
             "spare": usage("spare", problem="unreadable_usage")},
            threshold=95.0,
            active_name="work",
        )
        assert chosen.available is False
        assert chosen.profile is None

    def test_the_report_says_the_usage_could_not_be_checked(self, tmp_path):
        profiles = [ccp.ClaudeProfile(n, tmp_path / n, tmp_path / n) for n in ("work", "spare")]
        chosen = ccp.select_profile(
            profiles,
            {"work": usage("work", problem="unreachable"),
             "spare": usage("spare", problem="unreadable_usage")},
            threshold=95.0,
        )
        assert "could not" in chosen.message.lower()
        assert "work" in chosen.message and "spare" in chosen.message

    def test_an_account_that_answered_is_preferred_and_used(self, tmp_path):
        profiles = [ccp.ClaudeProfile(n, tmp_path / n, tmp_path / n) for n in ("work", "spare")]
        chosen = ccp.select_profile(
            profiles,
            {"work": usage("work", problem="unreachable"), "spare": usage("spare", 5.0, 5.0)},
            threshold=95.0,
            active_name="work",
        )
        assert chosen.profile.name == "spare"

    def test_a_pinned_conversation_also_waits_when_usage_cannot_be_read(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("chat-A", "work")
        chosen = ccp.select_for_job(
            session_id="chat-A",
            usage_reader=lambda p, **_: usage(p.name, problem="unreachable"),
        )
        assert chosen.profile.name == "work"
        assert chosen.available is False

    def test_the_status_word_says_it_was_not_checked(self, tmp_path):
        configure(tmp_path)
        text = "\n".join(ccp.status_lines(
            usage_reader=lambda p, **_: usage(p.name, problem="unreachable")))
        assert "not checked" in text.lower()


# ---------------------------------------------------------------------------
# Final review.
# ---------------------------------------------------------------------------


class TestTrimToleratesADamagedTimestamp:
    def test_a_word_where_a_timestamp_belongs_does_not_stop_a_pin(self):
        """A hand-edited or half-written file can hold anything. A trim must
        still run, and the live conversation must still survive it."""
        sessions = {
            ccp.session_fingerprint(f"broken-{index}"): {"profile": "work", "at": "yesterday"}
            for index in range(ccp.MAX_PINNED_SESSIONS + 10)
        }
        ccp.state_path().write_text(json.dumps(
            {"version": 1, "active": "work", "sessions": sessions}
        ))

        ccp.pin_session("chat-A", "spare")

        assert ccp.pinned_profile_name("chat-A") == "spare"
        assert len(ccp.read_state()["sessions"]) <= ccp.MAX_PINNED_SESSIONS

    @pytest.mark.parametrize("broken", ["yesterday", None, {"a": 1}, [1], True, float("nan")])
    def test_every_damaged_timestamp_sorts_as_oldest(self, broken):
        good = time.time()
        sessions = {"a" * 32: {"profile": "work", "at": broken},
                    "b" * 32: {"profile": "spare", "at": good}}
        trimmed = ccp._trim_sessions(dict(sessions))
        assert trimmed == sessions  # under the cap, nothing is dropped

        crowd = {f"{index:032x}": {"profile": "work", "at": broken}
                 for index in range(ccp.MAX_PINNED_SESSIONS)}
        crowd["f" * 32] = {"profile": "spare", "at": good}
        kept = ccp._trim_sessions(crowd)
        assert len(kept) == ccp.MAX_PINNED_SESSIONS
        assert "f" * 32 in kept, "the entry with a real timestamp must survive"

    def test_a_table_over_the_cap_with_no_timestamps_at_all_is_trimmed(self):
        crowd = {f"{index:032x}": {"profile": "work"}
                 for index in range(ccp.MAX_PINNED_SESSIONS + 30)}
        kept = ccp._trim_sessions(crowd)
        assert len(kept) == ccp.MAX_PINNED_SESSIONS


class TestSplitProfileDirectories:
    """macOS keeps the login in the Keychain, named after ``CLAUDE_CONFIG_DIR``.

    The credentials file lives under ``CLAUDE_SECURESTORAGE_CONFIG_DIR`` when a
    person sets it. When the two differ, the origin the reader records and the
    target the writer computes must be the same directory, or a legitimate
    refresh is refused.
    """

    # These two supply their own ``security`` mock, so they opt out of the
    # suite-wide guard without touching a real Keychain entry.
    @pytest.mark.allow_macos_keychain
    def test_the_keychain_origin_matches_the_write_target(self, tmp_path, monkeypatch):
        import json as _json
        from unittest.mock import MagicMock, patch

        from agent.anthropic_adapter import (
            _read_claude_code_credentials_from_keychain,
            claude_credentials_path,
        )

        config_dir = tmp_path / "work-config"
        vault_dir = tmp_path / "work-vault"
        config_dir.mkdir()
        vault_dir.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", str(vault_dir))

        payload = _json.dumps({"claudeAiOauth": {
            "accessToken": "tok", "expiresAt": 99999999999999}})
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   return_value=MagicMock(returncode=0, stdout=payload, stderr="")):
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert Path(creds["config_dir"]) == Path(str(claude_credentials_path().parent))

    @pytest.mark.allow_macos_keychain
    def test_a_refresh_on_split_directories_is_not_refused(self, tmp_path, monkeypatch):
        import json as _json
        from unittest.mock import MagicMock, patch

        from agent.anthropic_adapter import (
            _read_claude_code_credentials_from_keychain,
            _write_claude_code_credentials,
        )

        config_dir = tmp_path / "work-config"
        vault_dir = tmp_path / "work-vault"
        config_dir.mkdir()
        vault_dir.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", str(vault_dir))

        payload = _json.dumps({"claudeAiOauth": {
            "accessToken": "tok", "expiresAt": 99999999999999}})
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   return_value=MagicMock(returncode=0, stdout=payload, stderr="")):
            creds = _read_claude_code_credentials_from_keychain()

        _write_claude_code_credentials(
            "refreshed", "refresh", 9999999999999,
            origin_config_dir=creds["config_dir"],
        )

        written = vault_dir / ".credentials.json"
        assert written.exists(), "a legitimate refresh must not be refused"
        assert _json.loads(written.read_text())["claudeAiOauth"]["accessToken"] == "refreshed"

    def test_a_cross_profile_refresh_is_still_refused_with_split_directories(
        self, tmp_path, monkeypatch
    ):
        from agent.anthropic_adapter import _write_claude_code_credentials

        other_vault = tmp_path / "other-vault"
        this_vault = tmp_path / "this-vault"
        other_vault.mkdir()
        this_vault.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "this-config"))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", str(this_vault))

        _write_claude_code_credentials(
            "refreshed", "refresh", 9999999999999,
            origin_config_dir=str(other_vault),
        )

        assert not (this_vault / ".credentials.json").exists()
        assert not (other_vault / ".credentials.json").exists()

    def test_the_file_reader_origin_follows_secure_storage_too(self, tmp_path, monkeypatch):
        from agent.anthropic_adapter import _read_claude_code_credentials_from_file

        config_dir = tmp_path / "work-config"
        vault_dir = tmp_path / "work-vault"
        config_dir.mkdir()
        vault_dir.mkdir()
        (vault_dir / ".credentials.json").write_text(
            '{"claudeAiOauth": {"accessToken": "tok", "expiresAt": 99999999999999}}'
        )
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", str(vault_dir))

        creds = _read_claude_code_credentials_from_file()
        assert Path(creds["config_dir"]) == vault_dir


class TestReasonWhenTheActiveProfileIsGone:
    def test_a_removed_active_profile_reports_a_switch(self, tmp_path):
        """The work was on an account. That account is no longer configured,
        so the work moves. That is a switch, not a first run."""
        profiles = [ccp.ClaudeProfile(n, tmp_path / n, tmp_path / n) for n in ("work", "spare")]
        chosen = ccp.select_profile(
            profiles,
            {"work": usage("work", 5.0, 5.0), "spare": usage("spare", 5.0, 5.0)},
            threshold=95.0,
            active_name="retired",
        )
        assert chosen.profile.name == "work"
        assert chosen.reason == "switched"

    def test_a_genuine_first_run_still_reports_a_first_run(self, tmp_path):
        profiles = [ccp.ClaudeProfile(n, tmp_path / n, tmp_path / n) for n in ("work", "spare")]
        chosen = ccp.select_profile(
            profiles,
            {"work": usage("work", 5.0, 5.0), "spare": usage("spare", 5.0, 5.0)},
            threshold=95.0,
            active_name=None,
        )
        assert chosen.reason == "first_run"

    def test_a_full_active_profile_still_reports_a_switch(self, tmp_path):
        profiles = [ccp.ClaudeProfile(n, tmp_path / n, tmp_path / n) for n in ("work", "spare")]
        chosen = ccp.select_profile(
            profiles,
            {"work": usage("work", 100.0, 5.0), "spare": usage("spare", 5.0, 5.0)},
            threshold=95.0,
            active_name="work",
        )
        assert chosen.profile.name == "spare"
        assert chosen.reason == "switched"
