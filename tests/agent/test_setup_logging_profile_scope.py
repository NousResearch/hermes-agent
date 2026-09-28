"""#125974 — ``agent.agent_init._setup_logging`` must honor the active profile scope.

A Desktop serve backend (tui_gateway/compute_host.py) builds agents for several profiles
inside ``set_hermes_home_override(profile_home)``. ``_setup_logging`` used to pass the
import-time ``run_agent._hermes_home`` freeze, so the second profile's ``setup_logging``
call saw a home it already served, never adopted it, and every profile's records kept
landing in the launch profile's agent.log.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_logging
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_logging import RotatingFileHandler, _ProfileRoutingFileHandler


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset hermes_logging module globals so each test starts unconfigured."""
    hermes_logging._logging_initialized = False
    hermes_logging._reset_queued_handlers()
    yield
    hermes_logging._reset_queued_handlers()
    hermes_logging._logging_initialized = False
    hermes_logging.clear_session_context()


def _agent_stub() -> SimpleNamespace:
    # _setup_logging only reads verbose_logging; quiet keeps console clean.
    return SimpleNamespace(verbose_logging=False, quiet_mode=True)


def _witness(name: str, message: str) -> None:
    logging.getLogger(name).info(message)


class TestSetupLoggingProfileScope:
    def test_routes_agent_records_to_profile_scope(self, tmp_path: Path) -> None:
        """L1: an agent built under override B logs into B's agent.log, not A's."""
        from agent.agent_init import _setup_logging

        home_a = tmp_path / "home-a"
        home_b = tmp_path / "home-b"
        home_a.mkdir()
        home_b.mkdir()

        token = set_hermes_home_override(str(home_a))
        try:
            _setup_logging(_agent_stub())
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(str(home_b))
        try:
            _setup_logging(_agent_stub())
            _witness("agent.test_profile_scope_125974", "scoped-witness-b")
        finally:
            reset_hermes_home_override(token)
        hermes_logging.flush_log_queue()

        log_b = home_b / "logs" / "agent.log"
        assert log_b.is_file(), "the profile-scoped agent's records must reach its own logs/"
        assert "scoped-witness-b" in log_b.read_text(encoding="utf-8-sig")
        log_a = home_a / "logs" / "agent.log"
        text_a = log_a.read_text(encoding="utf-8-sig") if log_a.is_file() else ""
        assert "scoped-witness-b" not in text_a, "record must not leak into the launch home"

    def test_without_override_uses_active_home(self, tmp_path: Path, monkeypatch) -> None:
        """L2: with no override, the active (env) home wins over the import-time freeze."""
        from agent.agent_init import _setup_logging

        home_a = tmp_path / "home-a"
        home_a.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home_a))

        _setup_logging(_agent_stub())
        _witness("agent.test_active_home_125974", "active-home-witness")
        hermes_logging.flush_log_queue()

        log_a = home_a / "logs" / "agent.log"
        assert log_a.is_file(), "records must follow the active home, not the import freeze"
        assert "active-home-witness" in log_a.read_text(encoding="utf-8-sig")

    def test_missing_named_profile_falls_back_to_launch_home(self, tmp_path: Path) -> None:
        """L4: a named-profile override that does not exist must not break agent setup.

        mkdir_under_hermes_home() refuses to materialize a missing/tombstoned named
        profile on purpose; _setup_logging must fall back to the launch home instead of
        taking agent construction down.
        """
        from agent.agent_init import _setup_logging

        home_a = tmp_path / "home-a"
        home_a.mkdir()

        token = set_hermes_home_override(str(tmp_path / ".hermes" / "profiles" / "ghost"))
        try:
            _setup_logging(_agent_stub())  # must not raise
            _witness("agent.test_ghost_profile_125974", "ghost-fallback-witness")
        finally:
            reset_hermes_home_override(token)
        hermes_logging.flush_log_queue()

        # The ghost profile dir must NOT have been materialized by logging setup.
        assert not (tmp_path / ".hermes" / "profiles" / "ghost").exists()
        # The record fell back to the process home (the isolated HERMES_HOME).
        from hermes_constants import get_process_hermes_home

        fallback_log = get_process_hermes_home() / "logs" / "agent.log"
        assert "ghost-fallback-witness" in fallback_log.read_text(encoding="utf-8-sig")

    def test_no_live_home_skips_file_logging_without_raising(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """L4b: when not even the process home can host logs, construction still succeeds.

        The env home itself may point at a missing named profile (the scenario behind
        test_skip_background_review.py::test_persistence_failure_error_fallback...):
        no home is materialized, nothing raises, and the user is told file logging is off.
        """
        from agent.agent_init import _setup_logging

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "research"))

        token = set_hermes_home_override(str(tmp_path / ".hermes" / "profiles" / "ghost"))
        try:
            _setup_logging(_agent_stub())  # must not raise
        finally:
            reset_hermes_home_override(token)

        assert not (tmp_path / ".hermes" / "profiles" / "research").exists()
        assert not (tmp_path / ".hermes" / "profiles" / "ghost").exists()
        assert "logging to files is disabled" in capsys.readouterr().err

    def test_under_routing_adds_no_bare_handler(self, tmp_path: Path) -> None:
        """L3 guard: with routing already live, a scoped _setup_logging adds no bare handler.

        A bare RotatingFileHandler beside the profile routers would take every home's
        records — exactly the leak this fix closes. Passing on both sides of the fix is
        the point: it pins the blast radius.
        """
        from agent.agent_init import _setup_logging

        home_a = tmp_path / "home-a"
        home_b = tmp_path / "home-b"
        home_a.mkdir()
        home_b.mkdir()

        hermes_logging.setup_logging(hermes_home=home_a)
        assert hermes_logging.enable_profile_log_routing([home_a, home_b]) is True

        token = set_hermes_home_override(str(home_b))
        try:
            _setup_logging(_agent_stub())
            _witness("agent.test_routing_guard_125974", "routing-guard-witness-b")
        finally:
            reset_hermes_home_override(token)
        hermes_logging.flush_log_queue()

        bare = [
            h for h in hermes_logging._queued_file_handlers
            if isinstance(h, RotatingFileHandler)
        ]
        assert not bare, f"bare file handlers must not ride beside the routers: {bare}"
        assert "routing-guard-witness-b" in (
            home_b / "logs" / "agent.log"
        ).read_text(encoding="utf-8-sig")
        # And the routers themselves must still be routing, not replaced.
        assert any(
            isinstance(h, _ProfileRoutingFileHandler)
            for h in hermes_logging._queued_file_handlers
        )
