"""Regression tests for the resume-safety guard's fail-open contract.

`_resume_history_limit_error` documents that "Generic guard failures fail OPEN
(resume proceeds) — only a genuine over-limit result blocks." Its
`except Exception` handler called a bare `logger`, which is not bound at module
level in this mixin, so an unexpected failure raised NameError out of the
handler and the guard failed CLOSED instead.
"""

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


def _mixin(session_db):
    obj = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
    obj._session_db = session_db
    obj.session_id = "20260819_120000_abc123"
    return obj


class _RaisingDB:
    """A SessionDB whose safety probes blow up the way a locked DB would."""

    def assert_resume_safe(self, session_id):
        raise RuntimeError("sqlite3.OperationalError: database is locked")

    def assert_export_safe(self, session_id, max_messages=None):
        raise RuntimeError("sqlite3.OperationalError: database is locked")


class TestResumeGuardFailsOpen:
    def test_unexpected_failure_returns_none(self):
        """A generic probe failure must let the resume proceed, not crash."""
        assert _mixin(_RaisingDB())._resume_history_limit_error() is None

    def test_unexpected_failure_returns_none_tip_only(self):
        assert _mixin(_RaisingDB())._resume_history_limit_error(tip_only=True) is None

    def test_unexpected_failure_does_not_raise_nameerror(self):
        """Guard against a regression that re-breaks the handler itself."""
        try:
            _mixin(_RaisingDB())._resume_history_limit_error()
        except NameError as exc:  # pragma: no cover - the bug being fixed
            pytest.fail(f"resume guard handler raised NameError: {exc}")

    def test_no_session_db_is_a_noop(self):
        assert _mixin(None)._resume_history_limit_error() is None

    def test_over_limit_still_blocks(self):
        """The fail-open path must not swallow a genuine over-limit verdict."""
        from hermes_state import SessionResumeTooLargeError

        class _TooLargeDB:
            def assert_resume_safe(self, session_id):
                raise SessionResumeTooLargeError(50_000, 10_000)

        result = _mixin(_TooLargeDB())._resume_history_limit_error()
        assert result is not None
        assert isinstance(result, str)
