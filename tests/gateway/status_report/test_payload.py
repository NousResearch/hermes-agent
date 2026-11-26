"""The report body: contract v1, and a ``status`` that holds only the allow-listed fields."""

from __future__ import annotations

import json

import pytest

from gateway import status as gateway_status
from gateway.status_report import payload
from gateway.status_report.payload import allowlisted_status, build_report, read_status_source
from tests.gateway.status_report.support import assert_matches_contract, fixture

ERROR_TEXT = "401 from https://api.platform.example/bot<token>/getMe for owner@example.com"


def _full_status_body() -> dict:
    """An ``/api/status``-like body: the graded fields plus everything else that endpoint carries."""
    return {
        "version": "9.9.9", "release_date": "2026-01-01", "config_version": 30,
        "gateway_running": True, "gateway_state": "running",
        "gateway_exit_reason": "Shutdown after signal SIGTERM from pid 4242",
        "gateway_updated_at": "2026-01-01T00:00:00+00:00", "gateway_pid": 4242,
        "gateway_platforms": {
            "telegram": {"state": "connected", "updated_at": "2026-01-01T00:00:00+00:00", "writer_pid": 4242},
            "discord": {"state": "fatal", "error_code": "discord_login_failed", "error_message": ERROR_TEXT,
                        "needs_attention": True, "metrics": {"sent": 12}},
            "work:slack": {"state": "retrying", "error_code": "Socket closed: " + ERROR_TEXT,
                           "retrying_since": "2026-01-01T00:00:00+00:00"},
            "matrix": {"state": "exploded " + ERROR_TEXT},
            "not a platform key": {"state": "connected"},
            "signal": "connected",
        },
        "nous_session_valid": "valid",
        "install_id": "install-0001", "profiles": ["default", "work"], "parked_profiles": ["old"],
        "active_sessions": 3, "active_agents": 1, "auth_required": True, "auth_providers": ["password"],
        "memory": {"pressure": "ok"}, "disk": {"pressure": "ok"}, "hermes_home": "/srv/hermes",
        "multiplex_standalone_reason": "profiles default and work share " + ERROR_TEXT,
    }


def test_status_keeps_only_the_allow_listed_fields():
    status = allowlisted_status(_full_status_body())

    assert status == {
        "gateway_running": True,
        "gateway_state": "running",
        "gateway_exit_reason": "signal",
        "gateway_platforms": {
            "telegram": {"state": "connected"},
            "discord": {"state": "fatal", "error_code": "discord_login_failed"},
            "work:slack": {"state": "retrying"},
            "matrix": {"state": "unknown"},
        },
        "nous_session_valid": "valid",
    }


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"gateway_running": "yes", "gateway_state": "Running since " + ERROR_TEXT, "gateway_platforms": ERROR_TEXT,
         "gateway_exit_reason": None, "nous_session_valid": ERROR_TEXT},
    ],
)
def test_a_source_with_missing_or_misshapen_fields_gives_the_minimal_status(source):
    assert allowlisted_status(source) == {
        "gateway_running": False, "gateway_state": None, "gateway_exit_reason": None, "gateway_platforms": {}}


@pytest.mark.parametrize("kind", ["beat", "goodbye"])
def test_the_report_body_matches_the_v1_fixture(kind):
    raw = build_report(kind, seq=3, boot_id="0" * 32, source=_full_status_body())

    body = json.loads(raw)
    assert_matches_contract(body, kind)
    assert body["seq"] == 3 and body["boot_id"] == "0" * 32
    assert ERROR_TEXT not in raw.decode("utf-8") and "install-0001" not in raw.decode("utf-8")


@pytest.mark.parametrize("name", ["beat.json", "beat-minimal.json", "goodbye.json"])
def test_the_fixtures_are_what_the_sender_would_send(name):
    """A fixture the sender itself would reduce further is not a body it can produce."""
    example = fixture(name)

    assert_matches_contract(example, example["kind"])
    assert allowlisted_status(example["status"]) == example["status"]


def test_the_status_comes_from_this_gateways_runtime_record():
    """The gateway's own ``gateway_state.json`` is the source ``/api/status`` reads too."""
    assert gateway_status.write_runtime_status(
        gateway_state="running", exit_reason=None, active_agents=2, served_profiles=["default", "work"],
        platform="telegram", platform_state="fatal", error_code="telegram_polling_conflict",
        error_message=ERROR_TEXT, wait_timeout=10.0)
    assert gateway_status.write_runtime_status(platform="work:slack", platform_state="connected", wait_timeout=10.0)

    status = allowlisted_status(read_status_source())

    assert status == {
        "gateway_running": True,
        "gateway_state": "running",
        "gateway_exit_reason": None,
        "gateway_platforms": {
            "telegram": {"state": "fatal", "error_code": "telegram_polling_conflict"},
            "work:slack": {"state": "connected"},
        },
        "nous_session_valid": "unknown",  # no sign-in in this home
    }


def test_a_platform_entry_left_by_an_earlier_gateway_process_is_not_reported(monkeypatch):
    """A restart keeps the old process's entries in the record: a platform removed since must not
    be reported as failing for good."""
    assert gateway_status.write_runtime_status(
        gateway_state="running", platform="telegram", platform_state="connected", wait_timeout=10.0)
    record = gateway_status.read_runtime_status()
    ours = record["platforms"]["telegram"]
    record["platforms"]["discord"] = {**ours, "state": "fatal", "writer_pid": ours["writer_pid"] + 1}
    monkeypatch.setattr(payload, "read_runtime_status", lambda: record)

    assert allowlisted_status(read_status_source())["gateway_platforms"] == {"telegram": {"state": "connected"}}


def test_an_unreadable_sign_in_state_leaves_the_field_out(monkeypatch):
    def unreadable():
        raise OSError("auth store unreadable")

    monkeypatch.setattr(payload, "get_nous_session_validity", unreadable)

    status = allowlisted_status(read_status_source())

    assert "nous_session_valid" not in status and status["gateway_running"] is True
