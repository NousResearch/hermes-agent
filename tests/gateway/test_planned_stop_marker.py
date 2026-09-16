"""Offline contract tests for the systemd planned-stop marker entry point."""

import os
import subprocess
from unittest.mock import Mock

from agent import secret_scope
from gateway import planned_stop_marker
from gateway import status
from hermes_cli import auth as hermes_auth


def test_no_main_pid_is_inert_success(monkeypatch):
    writer = Mock()
    monkeypatch.setattr(status, "write_planned_stop_marker", writer)

    assert planned_stop_marker.main([]) == 0
    writer.assert_not_called()


def test_valid_pid_is_forwarded_to_existing_marker_writer(monkeypatch):
    writer = Mock(return_value=True)
    monkeypatch.setattr(status, "write_planned_stop_marker", writer)

    assert planned_stop_marker.main(["4242"]) == 0
    writer.assert_called_once_with(4242)


def test_invalid_pid_or_failed_marker_write_is_nonzero(monkeypatch):
    writer = Mock(return_value=False)
    monkeypatch.setattr(status, "write_planned_stop_marker", writer)

    for argv in (["0"], ["-1"], ["not-a-pid"], ["1", "2"]):
        assert planned_stop_marker.main(argv) == 1
    writer.assert_not_called()

    assert planned_stop_marker.main(["4242"]) == 1
    writer.assert_called_once_with(4242)


def test_marker_writer_exception_is_nonzero(monkeypatch):
    def explode(_target_pid):
        raise OSError("marker storage unavailable")

    monkeypatch.setattr(status, "write_planned_stop_marker", explode)

    assert planned_stop_marker.main(["4242"]) == 1


def test_helper_never_signals_shells_out_or_reads_credentials(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("forbidden helper side effect"))
    writer = Mock(return_value=True)
    monkeypatch.setattr(status, "write_planned_stop_marker", writer)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(secret_scope, "load_env_file", forbidden)
    monkeypatch.setattr(hermes_auth, "read_credential_pool", forbidden)

    assert planned_stop_marker.main([]) == 0
    assert planned_stop_marker.main(["invalid"]) == 1
    assert planned_stop_marker.main(["4242"]) == 0
    writer.assert_called_once_with(4242)
    forbidden.assert_not_called()
