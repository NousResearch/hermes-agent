"""Configuration and the two-profile gate for the Claude Code profile switcher.

The switcher is off until a person configures two or more profiles. One
profile, or none, must behave exactly as Hermes behaves today.
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from agent import claude_cli_profiles as ccp


def write_config(section):
    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text(yaml.dump({"claude_cli_profiles": section} if section else {}))
    return path


class TestProfileConfiguration:
    def test_no_configuration_gives_no_profiles(self):
        write_config(None)
        assert ccp.load_profiles() == []
        assert ccp.switching_enabled() is False

    def test_one_profile_does_not_enable_switching(self):
        write_config({"profiles": [{"name": "work", "config_dir": "~/.claude"}]})
        assert len(ccp.load_profiles()) == 1
        assert ccp.switching_enabled() is False

    def test_two_profiles_enable_switching(self):
        write_config({"profiles": [
            {"name": "work", "config_dir": "~/.claude"},
            {"name": "spare", "config_dir": "~/.claude-spare"},
        ]})
        assert [p.name for p in ccp.load_profiles()] == ["work", "spare"]
        assert ccp.switching_enabled() is True

    def test_secure_storage_directory_defaults_to_the_config_directory(self, tmp_path):
        write_config({"profiles": [
            {"name": "work", "config_dir": str(tmp_path / "a")},
            {"name": "spare", "config_dir": str(tmp_path / "b"),
             "securestorage_dir": str(tmp_path / "vault")},
        ]})
        work, spare = ccp.load_profiles()
        assert work.securestorage_dir == tmp_path / "a"
        assert spare.securestorage_dir == tmp_path / "vault"

    def test_home_shorthand_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        write_config({"profiles": [
            {"name": "work", "config_dir": "~/.claude"},
            {"name": "spare", "config_dir": "~/.claude-spare"},
        ]})
        assert ccp.load_profiles()[0].config_dir == tmp_path / ".claude"

    def test_a_profile_without_a_name_or_directory_is_dropped(self):
        write_config({"profiles": [
            {"name": "work", "config_dir": "~/.claude"},
            {"config_dir": "~/.nameless"},
            {"name": "no-dir"},
            "not-a-mapping",
        ]})
        assert [p.name for p in ccp.load_profiles()] == ["work"]

    def test_duplicate_names_keep_the_first_entry(self, tmp_path):
        write_config({"profiles": [
            {"name": "work", "config_dir": str(tmp_path / "a")},
            {"name": "work", "config_dir": str(tmp_path / "b")},
        ]})
        profiles = ccp.load_profiles()
        assert len(profiles) == 1
        assert profiles[0].config_dir == tmp_path / "a"


class TestStopThreshold:
    def test_default_is_95_percent(self):
        write_config(None)
        assert ccp.stop_at_percent() == 95.0

    def test_a_person_can_change_it(self):
        write_config({"stop_at_percent": 80})
        assert ccp.stop_at_percent() == 80.0

    def test_a_bad_value_falls_back_to_the_default(self):
        write_config({"stop_at_percent": "soon"})
        assert ccp.stop_at_percent() == 95.0

    def test_the_value_is_held_between_1_and_100(self):
        write_config({"stop_at_percent": 250})
        assert ccp.stop_at_percent() == 100.0


# ---------------------------------------------------------------------------
# Reading how much of a plan an account has used.
#
# The read asks one usage endpoint for one number set. It starts no model and
# it spends no tokens. Every test here injects its own reader, so no test
# touches a real Keychain entry or the network.
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

FAKE_TOKEN = "sk-ant-oat01-not-a-real-token"


def profile(tmp_path, name="work"):
    return ccp.ClaudeProfile(
        name=name,
        config_dir=tmp_path / name,
        securestorage_dir=tmp_path / name,
    )


def payload_with(session_percent, weekly_percent, *, session_reset=None, weekly_reset=None):
    return {
        "limits": [
            {"kind": "session", "percent": session_percent, "resets_at": session_reset},
            {"kind": "weekly_all", "percent": weekly_percent, "resets_at": weekly_reset},
        ]
    }


class TestReadProfileUsage:
    def test_reads_the_five_hour_and_weekly_windows(self, tmp_path):
        reset = "2026-08-07T18:00:00Z"
        usage = ccp.read_profile_usage(
            profile(tmp_path),
            token_reader=lambda _p: FAKE_TOKEN,
            usage_fetcher=lambda _t: payload_with(31, 44, session_reset=reset),
        )
        assert usage.five_hour_percent == 31.0
        assert usage.weekly_percent == 44.0
        assert usage.five_hour_reset == datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)
        assert usage.problem is None

    def test_reads_the_older_utilization_fields_when_no_limits_list_is_present(self, tmp_path):
        usage = ccp.read_profile_usage(
            profile(tmp_path),
            token_reader=lambda _p: FAKE_TOKEN,
            usage_fetcher=lambda _t: {
                "five_hour": {"utilization": 0.62, "resets_at": "2026-08-07T18:00:00Z"},
                "seven_day": {"utilization": 71, "resets_at": "2026-08-11T00:00:00Z"},
            },
        )
        assert usage.five_hour_percent == 62.0
        assert usage.weekly_percent == 71.0

    def test_a_profile_with_no_stored_login_is_reported(self, tmp_path):
        usage = ccp.read_profile_usage(
            profile(tmp_path),
            token_reader=lambda _p: None,
            usage_fetcher=lambda _t: pytest.fail("must not ask for usage without a login"),
        )
        assert usage.problem == "no_login"
        assert usage.five_hour_percent is None

    def test_a_rejected_login_is_reported_and_gives_no_reset_time(self, tmp_path):
        def reject(_token):
            raise ccp.ProfileUsageError("login_rejected", "HTTP 401")

        usage = ccp.read_profile_usage(
            profile(tmp_path),
            token_reader=lambda _p: FAKE_TOKEN,
            usage_fetcher=reject,
        )
        assert usage.problem == "login_rejected"
        assert usage.five_hour_reset is None
        assert usage.weekly_reset is None

    def test_malformed_usage_data_is_not_read_as_an_empty_account(self, tmp_path):
        usage = ccp.read_profile_usage(
            profile(tmp_path),
            token_reader=lambda _p: FAKE_TOKEN,
            usage_fetcher=lambda _t: {"limits": "this is not a list"},
        )
        assert usage.problem == "unreadable_usage"
        assert usage.five_hour_percent is None
        assert usage.weekly_percent is None

    def test_a_window_with_a_word_where_a_number_belongs_is_unreadable(self, tmp_path):
        usage = ccp.read_profile_usage(
            profile(tmp_path),
            token_reader=lambda _p: FAKE_TOKEN,
            usage_fetcher=lambda _t: payload_with("almost full", 12),
        )
        assert usage.problem == "unreadable_usage"
        assert usage.five_hour_percent is None

    def test_a_network_failure_is_reported_as_unreachable(self, tmp_path):
        def fail(_token):
            raise ccp.ProfileUsageError("unreachable", "connection reset")

        usage = ccp.read_profile_usage(
            profile(tmp_path), token_reader=lambda _p: FAKE_TOKEN, usage_fetcher=fail
        )
        assert usage.problem == "unreachable"

    def test_an_expired_stored_login_is_not_called_a_rejection(self, tmp_path):
        """Claude Code refreshes its own token when it starts. An expired
        stored token means "ask again later", not "this account is dead"."""
        yesterday = int(
            (datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000
        )
        usage = ccp.read_profile_usage(
            profile(tmp_path),
            token_reader=lambda _p: ccp.StoredLogin(token=FAKE_TOKEN, expires_at_ms=yesterday),
            usage_fetcher=lambda _t: pytest.fail("must not spend a call on a stale token"),
        )
        assert usage.problem == "stale_login"

    def test_the_usage_record_holds_no_token(self, tmp_path):
        usage = ccp.read_profile_usage(
            profile(tmp_path),
            token_reader=lambda _p: FAKE_TOKEN,
            usage_fetcher=lambda _t: payload_with(10, 20),
        )
        assert FAKE_TOKEN not in repr(usage)
        assert "sk-ant" not in repr(usage)


class TestUsageEndpoint:
    def test_the_default_fetcher_gets_the_oauth_usage_endpoint_only(self, monkeypatch):
        calls = []

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return payload_with(1, 2)

        class FakeClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url, headers=None):
                calls.append(("GET", url, headers or {}))
                return FakeResponse()

            def post(self, *args, **kwargs):
                raise AssertionError("the usage read must never post to a model")

        monkeypatch.setattr(ccp.httpx, "Client", FakeClient)
        ccp.fetch_usage(FAKE_TOKEN)

        assert len(calls) == 1
        method, url, headers = calls[0]
        assert method == "GET"
        assert url == "https://api.anthropic.com/api/oauth/usage"
        assert headers["Authorization"] == "Bearer " + FAKE_TOKEN


# ---------------------------------------------------------------------------
# Choosing a profile.
#
# Hermes keeps the account it is on until a window reaches the stop
# percentage. Then it moves to another account that is still open. When no
# account is open it stops and reports when each one reopens. It never spends
# paid usage past the plan.
# ---------------------------------------------------------------------------

SOON = datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 8, 11, 0, 0, tzinfo=timezone.utc)


def two_profiles(tmp_path):
    return [profile(tmp_path, "work"), profile(tmp_path, "spare")]


def usage(name, five_hour=None, weekly=None, problem=None,
          five_hour_reset=None, weekly_reset=None):
    return ccp.ProfileUsage(
        name=name,
        five_hour_percent=five_hour,
        weekly_percent=weekly,
        five_hour_reset=five_hour_reset,
        weekly_reset=weekly_reset,
        problem=problem,
    )


def choose(tmp_path, usages, active=None, threshold=95.0):
    return ccp.select_profile(
        two_profiles(tmp_path),
        {u.name: u for u in usages},
        threshold=threshold,
        active_name=active,
    )


class TestTheStopThreshold:
    def test_94_9_percent_is_still_usable(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", 94.9, 10.0), usage("spare", 1.0, 1.0)],
            active="work",
        )
        assert chosen.available is True
        assert chosen.profile.name == "work"
        assert chosen.reason == "kept_active"

    def test_exactly_95_percent_is_full(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", 95.0, 10.0), usage("spare", 1.0, 1.0)],
            active="work",
        )
        assert chosen.profile.name == "spare"
        assert chosen.reason == "switched"

    def test_a_person_can_lower_the_threshold(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", 50.0, 10.0), usage("spare", 1.0, 1.0)],
            active="work",
            threshold=40.0,
        )
        assert chosen.profile.name == "spare"


class TestExhaustion:
    def test_a_full_five_hour_window_moves_to_the_other_profile(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", 100.0, 12.0), usage("spare", 5.0, 5.0)],
            active="work",
        )
        assert chosen.profile.name == "spare"

    def test_a_full_weekly_window_moves_to_the_other_profile(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", 3.0, 99.0), usage("spare", 5.0, 5.0)],
            active="work",
        )
        assert chosen.profile.name == "spare"

    def test_both_profiles_full_gives_no_profile(self, tmp_path):
        chosen = choose(
            tmp_path,
            [
                usage("work", 100.0, 20.0, five_hour_reset=SOON),
                usage("spare", 10.0, 100.0, weekly_reset=LATER),
            ],
            active="work",
        )
        assert chosen.available is False
        assert chosen.profile is None
        assert chosen.reason == "none_available"

    def test_the_report_names_every_reset_time(self, tmp_path):
        chosen = choose(
            tmp_path,
            [
                usage("work", 100.0, 20.0, five_hour_reset=SOON),
                usage("spare", 10.0, 100.0, weekly_reset=LATER),
            ],
            active="work",
        )
        assert "work" in chosen.message and "spare" in chosen.message
        assert "2026-08-07" in chosen.message
        assert "2026-08-11" in chosen.message

    def test_the_report_says_a_person_must_sign_in(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", problem="no_login"), usage("spare", problem="login_rejected")],
        )
        assert chosen.available is False
        assert "sign in" in chosen.message.lower()


class TestStickiness:
    def test_the_active_profile_is_kept_while_it_is_open(self, tmp_path):
        """A lower number on another account is not a reason to move."""
        chosen = choose(
            tmp_path,
            [usage("work", 80.0, 80.0), usage("spare", 1.0, 1.0)],
            active="work",
        )
        assert chosen.profile.name == "work"
        assert chosen.reason == "kept_active"

    def test_the_first_run_takes_the_first_configured_profile(self, tmp_path):
        chosen = choose(tmp_path, [usage("work", 10.0, 10.0), usage("spare", 1.0, 1.0)])
        assert chosen.profile.name == "work"
        assert chosen.reason == "first_run"

    def test_an_active_name_that_is_no_longer_configured_is_ignored(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", 10.0, 10.0), usage("spare", 1.0, 1.0)],
            active="deleted-profile",
        )
        assert chosen.profile.name == "work"


class TestProfilesAPersonMustFix:
    def test_a_profile_with_no_login_is_never_selected(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", 100.0, 10.0), usage("spare", problem="no_login")],
            active="work",
        )
        assert chosen.available is False

    def test_a_rejected_login_is_never_selected(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", 100.0, 10.0), usage("spare", problem="login_rejected")],
            active="work",
        )
        assert chosen.available is False


class TestUnreadableUsage:
    def test_an_unreadable_profile_is_a_last_resort(self, tmp_path):
        chosen = choose(
            tmp_path,
            [usage("work", problem="unreadable_usage"), usage("spare", 5.0, 5.0)],
        )
        assert chosen.profile.name == "spare"

    def test_an_unreadable_profile_is_never_used_as_a_last_resort(self, tmp_path):
        """Fail closed. The usage read is also the check that the account
        answers as itself. Without it, running the account could bill a
        subscription the person did not choose."""
        chosen = choose(
            tmp_path,
            [usage("work", 100.0, 10.0), usage("spare", problem="unreachable")],
            active="work",
        )
        assert chosen.available is False
        assert chosen.profile is None
        assert "could not be checked" in chosen.message

    def test_the_active_profile_is_dropped_when_its_usage_could_not_be_read(self, tmp_path):
        """A failed read does not keep the account in use. Another account
        that did answer takes the work instead."""
        chosen = choose(
            tmp_path,
            [usage("work", problem="unreachable"), usage("spare", 5.0, 5.0)],
            active="work",
        )
        assert chosen.profile.name == "spare"
        assert chosen.reason == "switched"


class TestPaidExtraUsage:
    def test_paid_extra_usage_never_makes_a_full_account_usable(self, tmp_path):
        """The account can pay past its plan. Hermes must not spend that."""
        payload = payload_with(100, 100)
        payload["extra_usage"] = {"is_enabled": True, "used_credits": 0, "monthly_limit": 500}
        read = ccp.parse_usage_payload("work", payload)
        assert read.five_hour_percent == 100.0

        chosen = choose(
            tmp_path,
            [read, usage("spare", 100.0, 100.0)],
            active="work",
        )
        assert chosen.available is False

    def test_the_usage_record_carries_no_paid_usage_allowance(self):
        payload = payload_with(10, 10)
        payload["extra_usage"] = {"is_enabled": True, "monthly_limit": 500}
        read = ccp.parse_usage_payload("work", payload)
        assert "500" not in repr(read)


# ---------------------------------------------------------------------------
# The state file.
#
# It records which account the work is on and which account each conversation
# started on. It holds no token, no address, and no account number.
# ---------------------------------------------------------------------------

import threading  # noqa: E402


class TestStateFile:
    def test_it_lives_under_the_hermes_home(self):
        assert ccp.state_path().parent == Path(os.environ["HERMES_HOME"])
        assert ccp.state_path().name == "claude_cli_profiles.json"

    def test_a_missing_file_reads_as_empty(self):
        assert ccp.read_state() == {"version": 1, "active": None, "sessions": {}}

    def test_a_recorded_active_profile_survives_a_reread(self):
        ccp.record_active("spare")
        assert ccp.active_profile_name() == "spare"

    def test_a_corrupt_file_reads_as_empty_instead_of_failing(self):
        ccp.state_path().write_text("{ this is not json")
        assert ccp.active_profile_name() is None
        ccp.record_active("work")
        assert ccp.active_profile_name() == "work"

    def test_the_file_is_readable_by_its_owner_only(self):
        ccp.record_active("work")
        assert ccp.state_path().stat().st_mode & 0o077 == 0

    def test_clearing_the_state_removes_the_file(self):
        ccp.record_active("work")
        ccp.clear_state()
        assert not ccp.state_path().exists()
        assert ccp.active_profile_name() is None


class TestSessionPinning:
    def test_a_session_remembers_the_profile_that_created_it(self):
        ccp.pin_session("session-1", "spare")
        assert ccp.pinned_profile_name("session-1") == "spare"

    def test_an_unknown_session_has_no_pin(self):
        assert ccp.pinned_profile_name("never-seen") is None

    def test_an_empty_session_identifier_is_not_pinned(self):
        ccp.pin_session("", "spare")
        assert ccp.read_state()["sessions"] == {}

    def test_pinning_keeps_the_active_profile(self):
        ccp.record_active("work")
        ccp.pin_session("session-1", "spare")
        assert ccp.active_profile_name() == "work"
        assert ccp.pinned_profile_name("session-1") == "spare"

    def test_the_session_map_does_not_grow_without_bound(self):
        for index in range(ccp.MAX_PINNED_SESSIONS + 25):
            ccp.pin_session(f"session-{index}", "work")
        assert len(ccp.read_state()["sessions"]) <= ccp.MAX_PINNED_SESSIONS
        # The newest pin always survives the trim.
        assert ccp.pinned_profile_name(f"session-{ccp.MAX_PINNED_SESSIONS + 24}") == "work"


class TestConcurrentWrites:
    def test_every_concurrent_pin_survives(self):
        names = [f"session-{index}" for index in range(40)]
        errors = []

        def pin(session_id):
            try:
                ccp.pin_session(session_id, "work")
            except Exception as exc:  # pragma: no cover — a failure is the report
                errors.append(exc)

        threads = [threading.Thread(target=pin, args=(name,)) for name in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        stored = ccp.read_state()["sessions"]
        # The file holds a fingerprint per conversation, never the chat name.
        assert sorted(stored) == sorted(ccp.session_fingerprint(n) for n in names)
        assert all(ccp.pinned_profile_name(n) == "work" for n in names)

    def test_concurrent_selection_gives_every_caller_a_profile(self, tmp_path):
        write_config({"profiles": [
            {"name": "work", "config_dir": str(tmp_path / "work")},
            {"name": "spare", "config_dir": str(tmp_path / "spare")},
        ]})
        results = []

        def pick(index):
            results.append(
                ccp.select_for_job(
                    session_id=f"session-{index}",
                    usage_reader=lambda p: usage(p.name, 10.0, 10.0),
                )
            )

        threads = [threading.Thread(target=pick, args=(index,)) for index in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 12
        assert all(r.available and r.profile is not None for r in results)
        assert len(ccp.read_state()["sessions"]) == 12


# ---------------------------------------------------------------------------
# The one call a job makes, end to end.
# ---------------------------------------------------------------------------


def configure(tmp_path, count=2):
    entries = [
        {"name": name, "config_dir": str(tmp_path / name)}
        for name in ("work", "spare")[:count]
    ]
    write_config({"profiles": entries})
    return entries


class TestResumePinning:
    def test_a_resumed_session_stays_on_the_profile_that_created_it(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("session-1", "spare")
        ccp.record_active("work")

        chosen = ccp.select_for_job(
            session_id="session-1", usage_reader=lambda p: usage(p.name, 5.0, 5.0)
        )

        assert chosen.profile.name == "spare"
        assert chosen.reason == "pinned"

    def test_a_resumed_session_is_not_moved_when_its_own_profile_is_full(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("session-1", "spare")

        def reader(p):
            if p.name == "spare":
                return usage(p.name, 100.0, 20.0, five_hour_reset=SOON)
            return usage(p.name, 1.0, 1.0)

        chosen = ccp.select_for_job(session_id="session-1", usage_reader=reader)

        assert chosen.profile.name == "spare"
        assert chosen.available is False
        assert "2026-08-07" in chosen.message
        assert "work" not in chosen.message

    def test_a_pinned_resume_reads_only_its_own_profile(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("session-1", "spare")
        asked = []

        def reader(p):
            asked.append(p.name)
            return usage(p.name, 5.0, 5.0)

        ccp.select_for_job(session_id="session-1", usage_reader=reader)
        assert asked == ["spare"]

    def test_a_pin_to_a_profile_a_person_removed_falls_back_to_a_fresh_choice(self, tmp_path):
        configure(tmp_path)
        ccp.pin_session("session-1", "retired")

        chosen = ccp.select_for_job(
            session_id="session-1", usage_reader=lambda p: usage(p.name, 5.0, 5.0)
        )

        assert chosen.profile.name == "work"
        assert chosen.reason == "first_run"
        assert ccp.pinned_profile_name("session-1") == "work"

    def test_a_new_session_is_pinned_to_the_profile_it_selected(self, tmp_path):
        configure(tmp_path)
        chosen = ccp.select_for_job(
            session_id="session-9", usage_reader=lambda p: usage(p.name, 5.0, 5.0)
        )
        assert ccp.pinned_profile_name("session-9") == chosen.profile.name


class TestUnchangedSingleProfileBehaviour:
    def test_no_profiles_configured_changes_nothing(self, tmp_path):
        write_config(None)
        chosen = ccp.select_for_job(session_id="session-1")
        assert chosen.reason == "disabled"
        assert chosen.available is True
        assert chosen.env() == {}
        assert not ccp.state_path().exists()

    def test_one_profile_configured_changes_nothing(self, tmp_path):
        configure(tmp_path, count=1)
        chosen = ccp.select_for_job(
            session_id="session-1", usage_reader=lambda p: pytest.fail("no usage read")
        )
        assert chosen.reason == "disabled"
        assert chosen.env() == {}
        assert not ccp.state_path().exists()


class TestSelectedEnvironment:
    def test_the_environment_names_both_profile_directories(self, tmp_path):
        configure(tmp_path)
        chosen = ccp.select_for_job(usage_reader=lambda p: usage(p.name, 5.0, 5.0))
        assert chosen.env() == {
            "CLAUDE_CONFIG_DIR": str(tmp_path / "work"),
            "CLAUDE_SECURESTORAGE_CONFIG_DIR": str(tmp_path / "work"),
        }

    def test_no_profile_gives_no_environment_change(self, tmp_path):
        configure(tmp_path)
        chosen = ccp.select_for_job(
            usage_reader=lambda p: usage(p.name, 100.0, 100.0, five_hour_reset=SOON)
        )
        assert chosen.available is False
        assert chosen.env() == {}


class TestSecretsNeverLand:
    def test_a_full_selection_writes_no_token_to_the_state_file(self, tmp_path):
        """The whole path runs against a throwaway credentials file, so the
        real secret store is never opened."""
        configure(tmp_path)
        for name in ("work", "spare"):
            directory = tmp_path / name
            directory.mkdir()
            (directory / ".credentials.json").write_text(json.dumps({
                "claudeAiOauth": {
                    "accessToken": f"sk-ant-oat01-fake-{name}",
                    "refreshToken": f"sk-ant-ort01-fake-{name}",
                    "expiresAt": 99999999999999,
                }
            }))

        seen_tokens = []

        def fetcher(token):
            seen_tokens.append(token)
            return payload_with(12, 13)

        chosen = ccp.select_for_job(
            session_id="session-1",
            usage_reader=lambda p: ccp.read_profile_usage(p, usage_fetcher=fetcher),
        )

        # A new job reads every configured profile before it picks one.
        assert chosen.profile.name == "work"
        assert sorted(seen_tokens) == [
            "sk-ant-oat01-fake-spare",
            "sk-ant-oat01-fake-work",
        ]
        written = ccp.state_path().read_text()
        assert "sk-ant" not in written
        assert "accessToken" not in written
        assert "@" not in written

    def test_the_report_names_no_address(self, tmp_path):
        configure(tmp_path)
        chosen = ccp.select_for_job(
            usage_reader=lambda p: usage(p.name, problem="login_rejected")
        )
        assert chosen.available is False
        assert "@" not in chosen.message
        assert "sk-ant" not in chosen.message
