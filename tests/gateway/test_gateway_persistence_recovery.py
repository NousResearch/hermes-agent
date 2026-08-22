"""Every persistence-failure cause gets guidance that matches the cause.

``_normalize_empty_agent_response`` renders the message a Discord/Telegram
user actually reads when a turn is failed closed because the SessionDB write
did not land. It used to special-case ``:disk`` and send everything else to
one generic string:

    "Session storage was temporarily unavailable ... Your message should
     already be saved -- please send it again in a moment."

For a structurally corrupt state.db all three clauses are wrong: corruption
is not temporary, nothing was saved, and the retry fails identically forever.
The ``:corrupt`` bucket was added to PERSISTENCE_ERROR_CAUSES after this
branch was written (#87853), and the branch was never taught about it -- the
exact desynchronization that tuple's own comment exists to prevent.

The completeness test at the bottom is the part that keeps this fixed: it
iterates PERSISTENCE_ERROR_CAUSES, so a future bucket cannot silently fall
back to the transient wording again.
"""

import pytest

from gateway.run import (
    _CORRUPT_RECOVERY_STEPS,
    _PERSISTENCE_RECOVERY_MESSAGES,
    _normalize_empty_agent_response,
    _with_state_db_paths,
)
from hermes_state import PERSISTENCE_ERROR_CAUSES

# The generic reassurance that must not be given for a permanent failure.
TRANSIENT_CLAIMS = ("temporarily unavailable", "should already be saved", "in a moment")

# Causes where the write genuinely may have landed and a retry is reasonable.
TRANSIENT_CAUSES = {"locked", "compression", "unknown"}

# ...but only two of them can honestly say the write landed.  ``unknown``
# is the bucket the classifier reached when it could not identify the
# failure, so it is transient (a retry is reasonable) without being
# reassuring (the outcome of the write is exactly what we failed to
# determine).
CAUSES_THAT_MAY_CLAIM_THE_WRITE_LANDED = TRANSIENT_CAUSES - {"unknown"}


def _failed(cause, error="session storage could not be written"):
    return {
        "final_response": "",
        "failed": True,
        "failure_reason": f"session_persistence_failed:{cause}",
        "error": error,
        "api_calls": 1,
    }


def _render(agent_result):
    return _normalize_empty_agent_response(agent_result, "", history_len=10)


class TestCorruptIsNotReportedAsTransient:
    def test_does_not_call_permanent_damage_temporary(self):
        response = _render(_failed("corrupt"))

        for claim in TRANSIENT_CLAIMS:
            assert claim not in response.lower(), (
                f"a corrupt state.db was described as {claim!r}; corruption is "
                "permanent until restored and the retry cannot succeed"
            )

    def test_names_corruption_and_gives_recovery_steps(self):
        response = _render(_failed("corrupt"))

        assert "corruption" in response.lower()
        assert "hermes doctor" in response
        assert ".recover" in response
        assert "backups" in response

    def test_says_disk_space_will_not_help(self):
        """The #77386 misdiagnosis, stated outright rather than implied."""
        assert "disk space will not help" in _render(_failed("corrupt")).lower()

    def test_never_suggests_reset(self):
        assert "/reset" not in _render(_failed("corrupt"))


class TestCausesThatLostDataSaySo:
    def test_turn_lease_does_not_claim_the_message_was_saved(self):
        """run_agent's explainer says "Your reply was not saved" for this
        cause; the gateway used to say the opposite for the same turn."""
        response = _render(_failed("turn_lease"))

        assert "not saved" in response.lower()
        assert "should already be saved" not in response.lower()

    def test_compression_closed_asks_for_a_refresh_not_a_retry(self):
        """The session id changed. "Send it again in a moment" would resend
        against the id that no longer exists."""
        response = _render(_failed("compression_closed")).lower()

        assert "refresh" in response
        assert "session id" in response
        assert "in a moment" not in response


class TestTransientCausesAreUnchanged:
    @pytest.mark.parametrize("cause", sorted(TRANSIENT_CAUSES))
    def test_still_reassures_and_asks_for_a_resend(self, cause):
        response = _render(_failed(cause))

        assert "send it again" in response.lower()
        assert "/reset" not in response

    def test_locked_names_the_contention(self):
        assert "busy" in _render(_failed("locked")).lower()

    def test_disk_still_mentions_disk_space(self):
        assert "disk space" in _render(_failed("disk")).lower()


class TestUnstructuredErrorsUseTheCanonicalClassifier:
    """No ``failure_reason`` means the branch was reached through the
    "session storage" error-string fallback. The old code then tested
    ``"disk" in error_str``, which "database disk image is malformed"
    satisfies -- the substring steal documented on #77386."""

    def test_corruption_text_is_not_routed_to_the_disk_bucket(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": "session storage could not be written: database disk image is malformed",
            "api_calls": 1,
        }

        assert _render(agent_result) == _with_state_db_paths(
            _PERSISTENCE_RECOVERY_MESSAGES["corrupt"]
        )

    def test_real_disk_exhaustion_still_reaches_the_disk_bucket(self):
        agent_result = {
            "final_response": "",
            "failed": True,
            "error": "session storage could not be written: database or disk is full",
            "api_calls": 1,
        }

        assert _render(agent_result) == _with_state_db_paths(
            _PERSISTENCE_RECOVERY_MESSAGES["disk"]
        )


class TestEveryBucketIsCovered:
    """The guard that stops the next bucket from desynchronizing."""

    @pytest.mark.parametrize("cause", PERSISTENCE_ERROR_CAUSES)
    def test_bucket_has_its_own_message(self, cause):
        assert cause in _PERSISTENCE_RECOVERY_MESSAGES, (
            f"{cause!r} is in PERSISTENCE_ERROR_CAUSES but has no gateway "
            "recovery message, so it would silently render as the generic "
            "transient text -- the failure this test exists to prevent"
        )

    @pytest.mark.parametrize("cause", PERSISTENCE_ERROR_CAUSES)
    def test_permanent_causes_do_not_promise_a_successful_retry(self, cause):
        if cause in CAUSES_THAT_MAY_CLAIM_THE_WRITE_LANDED:
            pytest.skip("the write genuinely may have landed for this cause")

        response = _PERSISTENCE_RECOVERY_MESSAGES[cause].lower()
        assert "should already be saved" not in response


class TestRecoveryStepsNameThisInstallsPaths:
    """A salvage command has to point at the file that actually broke.

    ``HERMES_HOME`` is configurable, every non-default profile keeps its own
    database under ``<root>/profiles/<name>/``, and a native-Windows install
    lives under ``%LOCALAPPDATA%``.  A literal ``~/.hermes/state.db`` in the
    recovery steps is therefore wrong for three ordinary setups -- and wrong
    in the worst direction, since an operator who follows it either salvages
    nothing or salvages a different, healthy database while the corrupt one
    stays corrupt.
    """

    def test_the_salvage_step_names_the_configured_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        response = _render(_failed("corrupt"))

        assert str(tmp_path / "state.db") in response
        assert str(tmp_path / "backups") in response

    def test_a_relocated_home_is_never_reported_as_the_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        assert "~/.hermes" not in _render(_failed("corrupt"))

    def test_the_home_is_resolved_per_call_not_at_import(self, tmp_path, monkeypatch):
        """Two profiles in one process must not be told the same path."""
        first = tmp_path / "profile-a"
        second = tmp_path / "profile-b"

        monkeypatch.setenv("HERMES_HOME", str(first))
        rendered_first = _render(_failed("corrupt"))
        monkeypatch.setenv("HERMES_HOME", str(second))
        rendered_second = _render(_failed("corrupt"))

        assert str(first / "state.db") in rendered_first
        assert str(second / "state.db") in rendered_second

    def test_the_salvage_path_is_quoted_for_paths_with_spaces(self):
        """``%LOCALAPPDATA%`` sits under ``C:\\Users\\<name>``, which may contain
        a space; an unquoted path would split into two sqlite3 arguments."""
        assert '"{state_db}"' in _CORRUPT_RECOVERY_STEPS

    def test_the_startup_warning_shares_the_same_steps(self):
        """The gateway-startup home-channel warning is where this wording came
        from.  Leaving its own copy hardcoded is how the cause table drifted
        out of sync with PERSISTENCE_ERROR_CAUSES in the first place, so pin
        that gateway/run.py holds exactly one set of recovery steps and no
        literal home directory anywhere."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[2] / "gateway" / "run.py").read_text(
            encoding="utf-8"
        )

        assert "sqlite3 ~/.hermes" not in source
        assert "backup in ~/.hermes" not in source
        assert source.count("1. Run `hermes doctor --fix`") == 1

    def test_no_recovery_message_hardcodes_a_home_directory(self):
        for cause, message in _PERSISTENCE_RECOVERY_MESSAGES.items():
            assert "~/.hermes" not in message, (
                f"{cause!r} hardcodes ~/.hermes; use the {{state_db}}/{{backups}} "
                "placeholders so the message names this install's real paths"
            )


class TestUnknownDoesNotPromiseTheWriteLanded:
    """The generic bucket is reached when classification failed outright.

    Telling that user their message "should already be saved" is a claim
    about the one thing we just failed to determine.  A resend after a
    message that did land costs a duplicate; trusting a false "saved" costs
    the message.
    """

    def test_does_not_claim_the_message_was_saved(self):
        assert "should already be saved" not in _render(_failed("unknown"))

    def test_still_asks_for_a_resend(self):
        response = _render(_failed("unknown")).lower()

        assert "send it" in response
        assert "/reset" not in response

    def test_the_named_transient_causes_still_reassure(self):
        """The hedge is scoped to ``unknown`` -- ``locked`` and ``compression``
        identified the failure, so they keep the reassurance they earned."""
        for cause in sorted(CAUSES_THAT_MAY_CLAIM_THE_WRITE_LANDED):
            assert "should already be saved" in _render(_failed(cause))
