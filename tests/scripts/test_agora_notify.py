"""Unit tests for scripts/agora_notify.py wake-up helpers.

These tests cover the prepare-only design of ``_wake_tech_lead`` and the
``wake_on`` state filtering. They explicitly verify that the default path
never invokes ``hermes send`` or any other real external delivery.
"""

from pathlib import Path
from typing import Any

import pytest

import scripts.agora_notify as notify


class TestWakeTechLead:
    """Behavioral coverage for ``_wake_tech_lead``."""

    def test_no_target_returns_untriggered(self):
        """Without a wake target the function is a no-op."""
        result = notify._wake_tech_lead(
            target=None,
            message="help",
            state="error",
            wake_on={"blocked", "error"},
            enabled=False,
        )
        assert result == {"target": None, "triggered": False}

    def test_state_not_in_wake_on_is_skipped(self):
        """Only states listed in ``wake_on`` may trigger a ping."""
        result = notify._wake_tech_lead(
            target="telegram",
            message="help",
            state="working",
            wake_on={"blocked", "error"},
            enabled=False,
        )
        assert result["target"] == "telegram"
        assert result["triggered"] is False
        assert "working" in result["note"]

    def test_default_is_prepare_only(self, monkeypatch):
        """By default the payload is prepared but no real send happens."""
        calls = []

        def fake_run(target, message):
            calls.append((target, message))
            return {"success": True}

        monkeypatch.setattr(notify, "_run_hermes_send", fake_run)

        result = notify._wake_tech_lead(
            target="telegram",
            message="blocked on review",
            state="blocked",
            wake_on={"blocked", "error"},
            enabled=False,  # default / AGORA_WAKE_ENABLED unset
        )

        assert result["triggered"] is True
        assert result["action"] == "would-send"
        assert result["message"] == "blocked on review"
        assert result["target"] == "telegram"
        assert result["note"] == (
            "wake disabled; set AGORA_WAKE_ENABLED=1 or --wake-enabled to deliver"
        )
        assert "delivery" not in result
        assert calls == []  # no external delivery attempted

    def test_wake_on_filtering_blocks_non_matching_states(self, monkeypatch):
        """An allowed state triggers; a non-matching state does not."""
        calls = []
        monkeypatch.setattr(
            notify, "_run_hermes_send", lambda t, m: calls.append((t, m)) or {"success": True}
        )

        blocked = notify._wake_tech_lead(
            target="telegram",
            message="blocked",
            state="blocked",
            wake_on={"blocked"},
            enabled=True,
        )
        assert blocked["triggered"] is True
        assert blocked["action"] == "send"
        assert len(calls) == 1

        working = notify._wake_tech_lead(
            target="telegram",
            message="working",
            state="working",
            wake_on={"blocked"},
            enabled=True,
        )
        assert working["triggered"] is False
        assert len(calls) == 1  # no second send

    def test_enabled_uses_hermes_send_and_surfaces_errors(self, monkeypatch):
        """When enabled, ``_run_hermes_send`` is invoked and errors are surfaced."""

        def fake_run(target, message):
            return {"success": False, "error": f"cannot reach {target}"}

        monkeypatch.setattr(notify, "_run_hermes_send", fake_run)

        result = notify._wake_tech_lead(
            target="ntfy:alerts",
            message="agent error",
            state="error",
            wake_on={"error"},
            enabled=True,
        )

        assert result["triggered"] is True
        assert result["action"] == "send"
        assert result["delivery"] == {"success": False, "error": "cannot reach ntfy:alerts"}
        assert result["warning"] == "cannot reach ntfy:alerts"

    def test_default_message_fallback_when_enabled(self, monkeypatch):
        """When no explicit message is passed, a short status summary is built."""
        captured = {}

        def fake_run(target, message):
            captured["message"] = message
            return {"success": True}

        monkeypatch.setattr(notify, "_run_hermes_send", fake_run)

        notify._wake_tech_lead(
            target="telegram",
            message=None,
            state="error",
            wake_on={"error"},
            enabled=True,
        )

        assert captured["message"] == "Ágora wake-up ping"


class TestMainWakeDefaults:
    """Integration-style checks for ``main`` argparse wiring."""

    def test_wake_disabled_by_default(self):
        """``AGORA_WAKE_ENABLED`` is false by default and argparse respects it."""
        args = notify._build_arg_parser().parse_args(
            ["--profile", "tester", "--state", "error"]
        )
        assert args.wake_enabled is False
        assert args.wake_on == "blocked,error"
        assert args.wake_target is None

    def test_wake_enabled_flag_overrides_default(self):
        """``--wake-enabled`` flips delivery on."""
        args = notify._build_arg_parser().parse_args(
            ["--profile", "tester", "--state", "error", "--wake-enabled"]
        )
        assert args.wake_enabled is True

    def test_wake_on_env_override_parsed(self, monkeypatch):
        """``AGORA_WAKE_ON`` env is split into wake_on_set later; parser passes it."""
        monkeypatch.setenv("AGORA_WAKE_ON", "error,waiting-human")
        args = notify._build_arg_parser().parse_args(["--profile", "tester"])
        assert args.wake_on == "error,waiting-human"


class TestDashboardTokenPath:
    def test_token_file_uses_shared_root_when_running_inside_profile(self, monkeypatch, tmp_path):
        root = tmp_path / ".hermes"
        profile_home = root / "profiles" / "agent-techlead"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        assert notify._token_file() == root / ".dashboard-token"


class TestResolveProfilePid:
    def test_prefers_chat_process_over_gateway(self, monkeypatch):
        """Given both chat and gateway processes, the chat PID wins."""
        import psutil

        # Pane lookup always runs first; make it return no active pane so the
        # psutil path is exercised for the ordering assertion.
        monkeypatch.setattr(
            notify.shutil,
            "which",
            lambda name: None if name == "tmux" else "/usr/bin/pgrep",
        )

        fake_processes = [
            type("P", (), {"info": {"pid": 100, "cmdline": ["hermes", "-p", "agent-techlead", "gateway", "run"]}})(),
            type("P", (), {"info": {"pid": 200, "cmdline": ["hermes", "-p", "agent-techlead", "chat", "--cli"]}})(),
        ]
        monkeypatch.setattr(psutil, "process_iter", lambda attrs: fake_processes)
        assert notify._resolve_profile_pid("agent-techlead") == 200

    def test_returns_none_when_no_process_found(self, monkeypatch):
        import psutil

        monkeypatch.setattr(psutil, "process_iter", lambda attrs: [])
        monkeypatch.setattr(
            notify.shutil,
            "which",
            lambda name: None if name == "tmux" else "/usr/bin/pgrep",
        )
        monkeypatch.setattr(
            notify.subprocess,
            "run",
            lambda *args, **kwargs: type("R", (), {"returncode": 1, "stdout": ""})(),
        )
        assert notify._resolve_profile_pid("missing-profile") is None


class TestMainPidDiscovery:
    def test_main_uses_discovered_pid_when_pid_flag_missing(self, monkeypatch):
        """Without --pid, main() resolves the profile's process PID."""
        captured: dict[str, Any] = {}

        def fake_request(method, path, payload, *, base_url, token):
            captured["payload"] = payload
            return {"agent": {**payload, "last_heartbeat_at": 1}}

        monkeypatch.setattr(notify, "_request", fake_request)
        monkeypatch.setattr(notify, "_resolve_profile_pid", lambda profile: 424242)
        monkeypatch.setattr(notify, "_read_token", lambda token: "fake-token")

        notify.main(["--profile", "agent-techlead", "--state", "working"])

        assert captured["payload"]["pid"] == 424242
