"""_spawn_hermes_action's Windows detach: a job object that forbids breakaway must not
kill the action spawn (it must retry without CREATE_BREAKAWAY_FROM_JOB)."""

import sys
from unittest.mock import MagicMock

import hermes_cli._subprocess_compat as _subprocess_compat
import hermes_cli.web_server_gateway as _web_server_gateway


class TestSpawnHermesActionBreakawayFallback:
    """The contract in _subprocess_compat.windows_detach_flags: a job without
    JOB_OBJECT_LIMIT_BREAKAWAY_OK rejects CREATE_BREAKAWAY_FROM_JOB with ERROR_ACCESS_DENIED
    from Popen; callers catch OSError and fall back to windows_detach_flags_without_breakaway
    (#135179). gateway.py / gateway_windows.py / main_desktop.py follow it; this file must too."""

    @staticmethod
    def _prepare(monkeypatch, tmp_path):
        seen = []

        def fake_popen(cmd, **kwargs):
            seen.append(kwargs)
            if kwargs.get("creationflags") == 0x101:
                raise PermissionError(5, "Access is denied.")
            return MagicMock()

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", tmp_path)
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_FILES", {"update": "update.log"})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_RESULTS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_COMMANDS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_PROCS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_IDS", {})
        monkeypatch.setattr(_web_server_gateway, "_profile_action_environment", lambda sub, ov: {})
        monkeypatch.setattr(_web_server_gateway, "_action_targets_system_gateway", lambda sub: False)
        # Sentinels: the real helpers both return 0 on non-Windows, so stand in for the
        # flag values the way the win32 build would see them (with / without breakaway).
        monkeypatch.setattr(_web_server_gateway, "windows_detach_flags", lambda: 0x101)
        monkeypatch.setattr(_subprocess_compat, "windows_detach_flags_without_breakaway", lambda: 0x100)
        monkeypatch.setattr(_web_server_gateway.subprocess, "Popen", fake_popen)
        return seen

    def test_breakaway_denied_falls_back_without_breakaway(self, monkeypatch, tmp_path):
        seen = self._prepare(monkeypatch, tmp_path)

        proc = _web_server_gateway._spawn_hermes_action(["update"], "update")

        assert [kwargs.get("creationflags") for kwargs in seen] == [0x101, 0x100]
        assert _web_server_gateway._ACTION_PROCS["update"] is proc

    def test_breakaway_allowed_spawns_once(self, monkeypatch, tmp_path):
        seen = self._prepare(monkeypatch, tmp_path)
        # Job object permits breakaway: the first Popen already succeeds.
        monkeypatch.setattr(
            _web_server_gateway.subprocess,
            "Popen",
            lambda cmd, **kwargs: seen.append(kwargs) or MagicMock(),
        )

        _web_server_gateway._spawn_hermes_action(["update"], "update")

        assert len(seen) == 1
        assert seen[0].get("creationflags") == 0x101

    def test_posix_spawn_uses_start_new_session(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", tmp_path)
        monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_FILES", {"update": "update.log"})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_RESULTS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_COMMANDS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_PROCS", {})
        monkeypatch.setattr(_web_server_gateway, "_ACTION_IDS", {})
        monkeypatch.setattr(_web_server_gateway, "_profile_action_environment", lambda sub, ov: {})
        monkeypatch.setattr(_web_server_gateway, "_action_targets_system_gateway", lambda sub: False)
        seen = []
        monkeypatch.setattr(
            _web_server_gateway.subprocess,
            "Popen",
            lambda cmd, **kwargs: seen.append(kwargs) or MagicMock(),
        )

        _web_server_gateway._spawn_hermes_action(["update"], "update")

        assert len(seen) == 1
        assert seen[0].get("start_new_session") is True
        assert "creationflags" not in seen[0]
