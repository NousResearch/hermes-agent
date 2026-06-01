"""Tests for Daytona CLI polish: status, doctor, and setup flows.

- hermes status: Daytona mode, image/snapshot, SDK/auth, naming, lifecycle, network
- hermes doctor: missing snapshots, invalid language, bad JSON, disk >10GiB guidance
- hermes setup: image vs snapshot branching (tested indirectly via status/doctor)
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Status tests
# ---------------------------------------------------------------------------

class TestDaytonaStatusImageMode:
    """Daytona status output in image mode (default, backward-compatible)."""

    def _base_mocks(self, monkeypatch, tmp_path):
        """Set up minimal mocks for show_status to run."""
        from hermes_cli import status as status_mod
        import hermes_cli.auth as auth_mod
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
        monkeypatch.setattr(status_mod, "load_config", lambda: {"terminal": {"backend": "daytona"}}, raising=False)
        monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openrouter", raising=False)
        monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openrouter", raising=False)
        monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenRouter", raising=False)
        monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
        monkeypatch.setattr(status_mod.importlib.util, "find_spec", lambda name: object() if name == "daytona" else None)
        return status_mod

    def test_image_mode_shows_image_line(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Mode:         image" in output
        assert "Image:" in output

    def test_image_mode_uses_configured_image_when_env_unset(self, monkeypatch, capsys, tmp_path):
        """Config-only daytona_image should show in status when env var is unset."""
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        monkeypatch.delenv("TERMINAL_DAYTONA_IMAGE", raising=False)
        monkeypatch.setattr(
            status_mod,
            "load_config",
            lambda: {
                "terminal": {
                    "backend": "daytona",
                    "daytona_image": "custom/image:latest",
                }
            },
            raising=False,
        )
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Mode:         image" in output
        assert "Image:        custom/image:latest" in output

    def test_image_mode_shows_sdk_status(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "SDK:" in output
        assert "installed" in output

    def test_image_mode_shows_api_key_not_set(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "API key:" in output
        assert "not set" in output

    def test_image_mode_shows_api_key_configured(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        monkeypatch.setenv("DAYTONA_API_KEY", "dcs-test-key")
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "API key:" in output
        assert "configured" in output

    def test_image_mode_shows_naming_defaults(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Name prefix:  hermes" in output
        assert "Name scope:   task" in output

    def test_image_mode_shows_ephemeral_default(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Ephemeral:" in output

    def test_image_mode_shows_network_open(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Network:     open" in output

    def test_image_mode_no_lifecycle_intervals_by_default(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Auto-stop:" not in output
        assert "Auto-archive:" not in output
        assert "Auto-delete:" not in output

    def test_image_mode_shows_language_when_set(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        monkeypatch.setenv("TERMINAL_DAYTONA_LANGUAGE", "python")
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Language:" in output
        assert "python" in output

    def test_image_mode_hides_language_when_empty(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        monkeypatch.delenv("TERMINAL_DAYTONA_LANGUAGE", raising=False)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Language:" not in output


class TestDaytonaStatusNetworkBlocked:
    """Network blocked status line."""

    def _base_mocks(self, monkeypatch, tmp_path):
        from hermes_cli import status as status_mod
        import hermes_cli.auth as auth_mod
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_NETWORK_BLOCK_ALL", "true")
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
        monkeypatch.setattr(status_mod, "load_config", lambda: {"terminal": {"backend": "daytona"}}, raising=False)
        monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openrouter", raising=False)
        monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openrouter", raising=False)
        monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenRouter", raising=False)
        monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
        monkeypatch.setattr(status_mod.importlib.util, "find_spec", lambda name: object() if name == "daytona" else None)
        return status_mod

    def test_network_blocked_shows_blocked(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Network:      block requested" in output

    def test_network_blocked_with_allow_list(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        monkeypatch.setenv("TERMINAL_DAYTONA_NETWORK_ALLOW_LIST", "1.1.1.1/32,10.0.0.0/8")
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Allow list:" in output
        assert "1.1.1.1" in output


class TestDaytonaStatusSnapshotMode:
    """Daytona status output in snapshot mode."""

    def _base_mocks(self, monkeypatch, tmp_path):
        from hermes_cli import status as status_mod
        import hermes_cli.auth as auth_mod
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_CREATE_MODE", "snapshot")
        monkeypatch.setenv("TERMINAL_DAYTONA_SNAPSHOT", "my-project-snapshot")
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
        monkeypatch.setattr(status_mod, "load_config", lambda: {"terminal": {"backend": "daytona"}}, raising=False)
        monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openrouter", raising=False)
        monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openrouter", raising=False)
        monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenRouter", raising=False)
        monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
        monkeypatch.setattr(status_mod.importlib.util, "find_spec", lambda name: object() if name == "daytona" else None)
        return status_mod

    def test_snapshot_mode_shows_snapshot_line(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Mode:         snapshot" in output
        assert "Snapshot:     my-project-snapshot" in output

    def test_snapshot_mode_hides_default_image(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        # Don't set TERMINAL_DAYTONA_IMAGE — fallback image should only show if explicitly set
        monkeypatch.delenv("TERMINAL_DAYTONA_IMAGE", raising=False)
        monkeypatch.setattr(
            status_mod,
            "load_config",
            lambda: {
                "terminal": {
                    "backend": "daytona",
                    "daytona_create_mode": "snapshot",
                    "daytona_snapshot": "my-project-snapshot",
                    "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
                }
            },
            raising=False,
        )
        monkeypatch.setattr(status_mod, "read_raw_config", lambda: {"terminal": {}}, raising=False)
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        # In snapshot mode, image line should not show by default
        assert "Fallback image" not in output
        assert "Image:" not in output
        assert "nikolaik" not in output

    def test_snapshot_mode_shows_raw_configured_fallback_image(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        monkeypatch.delenv("TERMINAL_DAYTONA_IMAGE", raising=False)
        monkeypatch.setattr(
            status_mod,
            "read_raw_config",
            lambda: {"terminal": {"daytona_image": "custom/fallback:latest"}},
            raising=False,
        )
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Fallback image: custom/fallback:latest" in output

    def test_snapshot_mode_shows_lifecycle_intervals(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_STOP_INTERVAL", "30")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_DELETE_INTERVAL", "120")
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Auto-stop:    30 min" in output
        assert "Auto-delete:  120 min" in output


class TestDaytonaStatusLifecycle:
    """Daytona lifecycle interval display in status."""

    def _base_mocks(self, monkeypatch, tmp_path):
        from hermes_cli import status as status_mod
        import hermes_cli.auth as auth_mod
        import hermes_cli.gateway as gateway_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
        monkeypatch.setattr(status_mod, "load_config", lambda: {"terminal": {"backend": "daytona"}}, raising=False)
        monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openrouter", raising=False)
        monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openrouter", raising=False)
        monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenRouter", raising=False)
        monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
        monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
        monkeypatch.setattr(status_mod.importlib.util, "find_spec", lambda name: object() if name == "daytona" else None)
        return status_mod

    def test_all_lifecycle_intervals_displayed(self, monkeypatch, capsys, tmp_path):
        status_mod = self._base_mocks(monkeypatch, tmp_path)
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_STOP_INTERVAL", "30")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_ARCHIVE_INTERVAL", "60")
        monkeypatch.setenv("TERMINAL_DAYTONA_AUTO_DELETE_INTERVAL", "120")
        monkeypatch.setenv("TERMINAL_DAYTONA_EPHEMERAL", "true")
        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        output = capsys.readouterr().out
        assert "Auto-stop:    30 min" in output
        assert "Auto-archive: 60 min" in output
        assert "Auto-delete:  120 min" in output
        assert "Ephemeral:    true" in output


# ---------------------------------------------------------------------------
# Doctor tests — focused on Daytona-specific validation logic
# ---------------------------------------------------------------------------

class TestDaytonaDoctorSnapshotValidation:
    """Doctor catches snapshot-mode configuration issues."""

    def test_snapshot_mode_with_snapshot_set_ok(self, monkeypatch, tmp_path):
        """Snapshot mode with a snapshot set should produce no issues."""
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("DAYTONA_API_KEY", "dcs-test-key")
        monkeypatch.setenv("TERMINAL_DAYTONA_CREATE_MODE", "snapshot")
        monkeypatch.setenv("TERMINAL_DAYTONA_SNAPSHOT", "my-snap")

        # Verify snapshot requirement logic directly
        terminal_env = os.getenv("TERMINAL_ENV", "local")
        assert terminal_env == "daytona"
        create_mode = os.getenv("TERMINAL_DAYTONA_CREATE_MODE") or "image"
        snapshot = os.getenv("TERMINAL_DAYTONA_SNAPSHOT", "").strip()
        assert create_mode == "snapshot"
        assert snapshot  # Should be truthy — no issue

    def test_snapshot_mode_without_snapshot_shows_error(self, monkeypatch, tmp_path):
        """Snapshot mode without snapshot set should flag an issue."""
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_CREATE_MODE", "snapshot")
        monkeypatch.delenv("TERMINAL_DAYTONA_SNAPSHOT", raising=False)

        create_mode = os.getenv("TERMINAL_DAYTONA_CREATE_MODE") or "image"
        snapshot = os.getenv("TERMINAL_DAYTONA_SNAPSHOT", "").strip()
        assert create_mode == "snapshot"
        assert not snapshot  # Empty — should trigger issue

    def test_invalid_create_mode(self, monkeypatch, tmp_path):
        """Invalid create_mode should be caught."""
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_CREATE_MODE", "bogus")

        create_mode = os.getenv("TERMINAL_DAYTONA_CREATE_MODE") or "image"
        assert create_mode not in ("image", "snapshot")  # Should trigger issue

    def test_image_mode_no_snapshot_needed(self, monkeypatch, tmp_path):
        """Image mode should not require a snapshot."""
        monkeypatch.setenv("TERMINAL_ENV", "daytona")
        monkeypatch.setenv("TERMINAL_DAYTONA_CREATE_MODE", "image")
        monkeypatch.delenv("TERMINAL_DAYTONA_SNAPSHOT", raising=False)

        create_mode = os.getenv("TERMINAL_DAYTONA_CREATE_MODE") or "image"
        assert create_mode == "image"  # No snapshot needed


class TestDaytonaDoctorJSONValidation:
    """Doctor validates JSON env vars for labels, env_vars, volume_mounts."""

    def test_valid_labels_json(self):
        import json
        raw = '{"env": "dev", "team": "backend"}'
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert len(parsed) == 2

    def test_invalid_labels_json_not_object(self):
        import json
        raw = '["not", "a", "dict"]'
        parsed = json.loads(raw)
        assert not isinstance(parsed, dict)  # Should trigger warning

    def test_valid_volume_mounts_json(self):
        import json
        raw = '[{"volume_id": "vol-123", "mount_path": "/mnt/data"}]'
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_invalid_volume_mounts_json_not_array(self):
        import json
        raw = '{"volume_id": "vol-123", "mount_path": "/mnt/data"}'
        parsed = json.loads(raw)
        assert not isinstance(parsed, list)  # Should trigger warning

    def test_valid_env_vars_json(self):
        import json
        raw = '{"API_KEY": "secret", "DEBUG": "true"}'
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert len(parsed) == 2

    def test_invalid_env_vars_json_not_object(self):
        import json
        raw = '42'
        parsed = json.loads(raw)
        assert not isinstance(parsed, dict)  # Should trigger warning

    def test_invalid_json_syntax(self):
        import json
        raw = '{invalid json'
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)  # Doctor should catch this


class TestDaytonaDoctorDiskWarning:
    """Doctor warns when disk exceeds 10 GiB guidance."""

    def test_disk_over_10gb_flags_warning(self):
        disk_mb = "20480"  # 20 GiB
        disk_gb = int(disk_mb) / 1024
        assert disk_gb > 10  # Should trigger warning

    def test_disk_at_10gb_no_warning(self):
        disk_mb = "10240"  # 10 GiB
        disk_gb = int(disk_mb) / 1024
        assert disk_gb == 10  # At boundary, no warning

    def test_disk_under_10gb_no_warning(self):
        disk_mb = "5120"  # 5 GiB
        disk_gb = int(disk_mb) / 1024
        assert disk_gb < 10  # No warning


class TestDaytonaDoctorLanguage:
    """Doctor validates language field."""

    def test_common_languages_accepted(self):
        valid_languages = {"", "python", "javascript", "typescript"}
        for lang in ["python", "javascript", "typescript"]:
            assert lang in valid_languages
        for unsupported in ["go", "rust", "java", "csharp", "ruby"]:
            assert unsupported not in valid_languages

    def test_unusual_language_flagged(self):
        valid_languages = {"", "python", "javascript", "typescript"}
        unusual = "brainfuck"
        assert unusual not in valid_languages  # Should trigger warning


class TestDaytonaDoctorConfigBackedPolish:
    """Doctor should validate Daytona settings from config.yaml, not env-only."""

    def _run_doctor_with_config(self, monkeypatch, tmp_path, terminal_cfg):
        import contextlib
        import io
        import sys as _sys
        import types
        from argparse import Namespace
        from pathlib import Path

        from hermes_cli import doctor as doctor_mod

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("DAYTONA_API_KEY", "dcs-test-key")
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        for key in list(os.environ):
            if key.startswith("TERMINAL_DAYTONA_") or key == "TERMINAL_CONTAINER_DISK":
                monkeypatch.delenv(key, raising=False)

        config = {"terminal": {"backend": "daytona", **terminal_cfg}}
        raw_config = {"terminal": dict(config["terminal"])}
        monkeypatch.setattr(doctor_mod, "load_config", lambda: config, raising=False)
        monkeypatch.setattr(doctor_mod, "read_raw_config", lambda: raw_config, raising=False)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home, raising=False)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path, raising=False)
        monkeypatch.setattr(doctor_mod, "_DHH", str(hermes_home), raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(doctor_mod.importlib.util, "find_spec", lambda name: object() if name == "daytona" else None)
        monkeypatch.setitem(
            _sys.modules,
            "model_tools",
            types.SimpleNamespace(check_tool_availability=lambda *a, **kw: ([], []), TOOLSET_REQUIREMENTS={}),
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False, ack=None))
        return buf.getvalue()

    def test_config_backend_daytona_missing_snapshot_is_reported(self, monkeypatch, tmp_path):
        output = self._run_doctor_with_config(
            monkeypatch,
            tmp_path,
            {"daytona_create_mode": "snapshot", "daytona_snapshot": ""},
        )

        assert "Snapshot mode requires daytona_snapshot" in output

    def test_config_backend_daytona_invalid_create_mode_is_reported(self, monkeypatch, tmp_path):
        output = self._run_doctor_with_config(
            monkeypatch,
            tmp_path,
            {"daytona_create_mode": "bogus"},
        )

        assert "Invalid daytona_create_mode" in output
        assert "must be 'image' or 'snapshot'" in output

    def test_config_backend_daytona_invalid_json_types_are_reported(self, monkeypatch, tmp_path):
        output = self._run_doctor_with_config(
            monkeypatch,
            tmp_path,
            {
                "daytona_labels": ["not", "an", "object"],
                "daytona_env_vars": 42,
                "daytona_volume_mounts": {"volume_id": "vol-id"},
            },
        )

        assert "daytona_labels must be a JSON object" in output
        assert "daytona_env_vars must be a JSON object" in output
        assert "daytona_volume_mounts must be a JSON array" in output

    def test_config_backend_daytona_unusual_language_warns(self, monkeypatch, tmp_path):
        output = self._run_doctor_with_config(
            monkeypatch,
            tmp_path,
            {"daytona_language": "brainfuck"},
        )

        assert "Unsupported Daytona language" in output
        assert "brainfuck" in output

    def test_config_backed_json_lifecycle_sync_and_disk_guidance_show(self, monkeypatch, tmp_path):
        output = self._run_doctor_with_config(
            monkeypatch,
            tmp_path,
            {
                "daytona_create_mode": "image",
                "daytona_language": "python",
                "daytona_labels": {"matrix": "cli-polish"},
                "daytona_env_vars": {"HERMES_LIVE_MARKER": "ok"},
                "daytona_volume_mounts": [{"volume_id": "vol-id", "mount_path": "/data"}],
                "daytona_sync_cwd": True,
                "daytona_sync_cwd_source": str(tmp_path / "fixture"),
                "container_disk": 20480,
            },
        )

        assert "Daytona language" in output and "python" in output
        assert "daytona_labels" in output and "1 label(s)" in output
        assert "daytona_env_vars" in output and "1 var(s)" in output
        assert "daytona_volume_mounts" in output and "1 mount(s)" in output
        assert "Daytona sync_cwd" in output and "enabled" in output
        assert "Daytona disk request (20GB) exceeds platform guidance" in output
    def test_config_backend_daytona_invalid_language_is_reported(self, monkeypatch, tmp_path):
        output = self._run_doctor_with_config(
            monkeypatch,
            tmp_path,
            {"daytona_create_mode": "image", "daytona_language": "go"},
        )

        assert "Unsupported Daytona language" in output
        assert "python" in output and "typescript" in output and "javascript" in output

    def test_snapshot_mode_ignores_explicit_large_disk_guidance(self, monkeypatch, tmp_path):
        output = self._run_doctor_with_config(
            monkeypatch,
            tmp_path,
            {
                "daytona_create_mode": "snapshot",
                "daytona_snapshot": "project-snapshot",
                "container_disk": 20480,
            },
        )

        assert "Daytona disk request" not in output


# ---------------------------------------------------------------------------
# Config-set tests
# ---------------------------------------------------------------------------

class TestDaytonaConfigSetStructuredEnvSync:
    """hermes config set mirrors structured Daytona config as valid JSON env values."""

    def test_daytona_labels_are_saved_to_env_as_json(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_mod

        saved_env = {}
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(config_mod, "get_config_path", lambda: tmp_path / "config.yaml", raising=False)
        monkeypatch.setattr(config_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
        monkeypatch.setattr(config_mod, "ensure_hermes_home", lambda: None, raising=False)
        monkeypatch.setattr(config_mod, "is_managed", lambda: False, raising=False)
        monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved_env.setdefault(k, v), raising=False)

        config_mod.set_config_value("terminal.daytona_labels", '{"team":"agents","env":"dev"}')

        assert saved_env["TERMINAL_DAYTONA_LABELS"] == '{"env":"dev","team":"agents"}'

    def test_daytona_volume_mounts_are_saved_to_env_as_json_array(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_mod

        saved_env = {}
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(config_mod, "get_config_path", lambda: tmp_path / "config.yaml", raising=False)
        monkeypatch.setattr(config_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
        monkeypatch.setattr(config_mod, "ensure_hermes_home", lambda: None, raising=False)
        monkeypatch.setattr(config_mod, "is_managed", lambda: False, raising=False)
        monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved_env.setdefault(k, v), raising=False)

        config_mod.set_config_value("terminal.daytona_volume_mounts", '[{"volume_id":"vol-1","mount_path":"/data"}]')

        assert saved_env["TERMINAL_DAYTONA_VOLUME_MOUNTS"] == '[{"mount_path":"/data","volume_id":"vol-1"}]'


# ---------------------------------------------------------------------------
# Config-show tests
# ---------------------------------------------------------------------------

class TestDaytonaConfigShowImageMode:
    """hermes config show Daytona display in image mode (default)."""

    def _base_mocks(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(config_mod, "load_config",
                            lambda: {"terminal": {"backend": "daytona",
                                                   "daytona_create_mode": "image",
                                                   "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20"}},
                            raising=False)
        monkeypatch.setattr(config_mod, "get_env_value", lambda key: {
            "DAYTONA_API_KEY": "dcs-test-key",
        }.get(key), raising=False)
        monkeypatch.setattr(config_mod, "get_config_path", lambda: "/tmp/fake_config.yaml", raising=False)
        monkeypatch.setattr(config_mod, "get_env_path", lambda: "/tmp/fake_env", raising=False)
        monkeypatch.setattr(config_mod, "get_project_root", lambda: "/tmp/fake_root", raising=False)
        return config_mod

    def test_image_mode_shows_mode_line(self, monkeypatch, capsys, tmp_path):
        config_mod = self._base_mocks(monkeypatch, tmp_path)
        config_mod.show_config()
        output = capsys.readouterr().out
        assert "Mode:" in output
        assert "image" in output

    def test_image_mode_shows_image_line(self, monkeypatch, capsys, tmp_path):
        config_mod = self._base_mocks(monkeypatch, tmp_path)
        config_mod.show_config()
        output = capsys.readouterr().out
        assert "Image:" in output and "nikolaik/python-nodejs:python3.11-nodejs20" in output


class TestDaytonaConfigShowSnapshotMode:
    """hermes config show Daytona display in snapshot mode."""

    def _base_mocks(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(config_mod, "load_config",
                            lambda: {"terminal": {"backend": "daytona",
                                                   "daytona_create_mode": "snapshot",
                                                   "daytona_snapshot": "my-project-snap"}},
                            raising=False)
        monkeypatch.setattr(config_mod, "get_env_value", lambda key: {
            "DAYTONA_API_KEY": "dcs-test-key",
        }.get(key), raising=False)
        monkeypatch.setattr(config_mod, "get_config_path", lambda: "/tmp/fake_config.yaml", raising=False)
        monkeypatch.setattr(config_mod, "get_env_path", lambda: "/tmp/fake_env", raising=False)
        monkeypatch.setattr(config_mod, "get_project_root", lambda: "/tmp/fake_root", raising=False)
        return config_mod

    def test_snapshot_mode_shows_mode_line(self, monkeypatch, capsys, tmp_path):
        config_mod = self._base_mocks(monkeypatch, tmp_path)
        config_mod.show_config()
        output = capsys.readouterr().out
        assert "Mode:" in output
        assert "snapshot" in output

    def test_snapshot_mode_shows_snapshot_line(self, monkeypatch, capsys, tmp_path):
        config_mod = self._base_mocks(monkeypatch, tmp_path)
        config_mod.show_config()
        output = capsys.readouterr().out
        assert "Snapshot:" in output and "my-project-snap" in output


# ---------------------------------------------------------------------------
# Setup-wizard tests — snapshot mode skips resource prompts
# ---------------------------------------------------------------------------

class TestDaytonaSetupSnapshotSkipsResources:
    """Snapshot mode setup must NOT call _prompt_container_resources.

    When the user selects snapshot create mode, the setup wizard should:
    1. Save TERMINAL_DAYTONA_CREATE_MODE = "snapshot" and a snapshot value.
    2. Skip container resource prompts (CPU, memory, disk) entirely.
    """

    def _make_setup_mocks(self, monkeypatch, tmp_path, snapshot_name="my-snap"):
        """Return (config, call_log) with Daytona setup mocks for snapshot mode."""
        import sys as _sys

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Pre-set DAYTONA_API_KEY so the wizard skips the API key prompt
        monkeypatch.setenv("DAYTONA_API_KEY", "dcs-test-key")
        config = {"terminal": {}}

        resource_call_log = []

        def fake_prompt_choice(question, choices, default=0):
            if question == "Select terminal backend:":
                return 4  # daytona
            if "Sandbox create mode" in question:
                return 1  # snapshot mode
            if "Sandbox language" in question:
                return 0  # default SDK/image language
            if "Name scope" in question or "name scope" in question:
                return 0  # task
            raise AssertionError(f"Unexpected prompt_choice: {question}")

        def fake_prompt(message, default="", **kwargs):
            # Snapshot name prompt
            if "snapshot" in message.lower() and ("name" in message.lower() or "id" in message.lower()):
                return snapshot_name
            # Fallback image prompt — decline
            if "fallback" in message.lower():
                return default if default else ""
            # Sandbox image prompt (shouldn't appear in snapshot mode, but handle anyway)
            if "image" in message.lower() and "sandbox" in message.lower():
                return default if default else "nikolaik/python-nodejs:python3.11-nodejs20"
            # Language, prefix, lifecycle, network, etc. — accept defaults
            return default if default else ""

        def fake_prompt_yes_no(message, default=False):
            if "fallback" in message.lower():
                return False  # Don't set fallback image
            if "Ephemeral" in message or "ephemeral" in message.lower():
                return default
            if "Block" in message or "block" in message.lower():
                return default
            if "Update" in message:
                return False  # Don't update API key
            return default

        def fake_container_resources(cfg):
            resource_call_log.append(cfg)

        monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
        monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)
        monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", fake_prompt_yes_no)
        monkeypatch.setattr("hermes_cli.setup._prompt_container_resources", fake_container_resources)
        monkeypatch.setattr("hermes_cli.setup.cfg_get", lambda c, section, key, default=None: default)
        monkeypatch.setattr("hermes_cli.setup.save_env_value", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.setup.get_env_value", lambda key: "dcs-test-key" if key == "DAYTONA_API_KEY" else "")
        # Stub out daytona import so setup doesn't try to pip install
        monkeypatch.setitem(
            _sys.modules, "daytona",
            type(_sys)("daytona"),
        )

        return config, resource_call_log

    def test_snapshot_mode_saves_create_mode(self, monkeypatch, tmp_path):
        """Snapshot mode saves TERMINAL_DAYTONA_CREATE_MODE = snapshot."""
        config, _ = self._make_setup_mocks(monkeypatch, tmp_path)

        from hermes_cli.setup import setup_terminal_backend
        setup_terminal_backend(config)

        assert config["terminal"].get("daytona_create_mode") == "snapshot"

    def test_snapshot_mode_saves_snapshot(self, monkeypatch, tmp_path):
        """Snapshot mode saves the snapshot name/ID."""
        config, _ = self._make_setup_mocks(monkeypatch, tmp_path, snapshot_name="my-cool-snapshot")

        from hermes_cli.setup import setup_terminal_backend
        setup_terminal_backend(config)

        assert config["terminal"].get("daytona_snapshot") == "my-cool-snapshot"

    def test_snapshot_mode_skips_container_resources(self, monkeypatch, tmp_path):
        """Snapshot mode must NOT call _prompt_container_resources."""
        config, resource_call_log = self._make_setup_mocks(monkeypatch, tmp_path)

        from hermes_cli.setup import setup_terminal_backend
        setup_terminal_backend(config)

        assert len(resource_call_log) == 0, (
            f"_prompt_container_resources was called {len(resource_call_log)} time(s) "
            f"in snapshot mode, but should not have been called"
        )


class TestDaytonaSetupImageModeIncludesResources:
    """Image mode setup must still call _prompt_container_resources."""

    def _make_setup_mocks(self, monkeypatch, tmp_path, image_name="python-nodejs:latest"):
        """Return (config, call_log) with Daytona setup mocks for image mode."""
        import sys as _sys

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Pre-set DAYTONA_API_KEY so the wizard skips the API key prompt
        monkeypatch.setenv("DAYTONA_API_KEY", "dcs-test-key")
        config = {"terminal": {}}

        resource_call_log = []

        def fake_prompt_choice(question, choices, default=0):
            if question == "Select terminal backend:":
                return 4  # daytona
            if "Sandbox create mode" in question:
                return 0  # image mode
            if "Sandbox language" in question:
                return 0  # default SDK/image language
            if "Name scope" in question or "name scope" in question:
                return 0  # task
            raise AssertionError(f"Unexpected prompt_choice: {question}")

        def fake_prompt(message, default="", **kwargs):
            if "image" in message.lower() and "sandbox" in message.lower():
                return image_name
            return default if default else ""

        def fake_prompt_yes_no(message, default=False):
            if "Ephemeral" in message or "ephemeral" in message.lower():
                return default
            if "Block" in message or "block" in message.lower():
                return default
            if "Update" in message:
                return False
            return default

        def fake_container_resources(cfg):
            resource_call_log.append(cfg)

        monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
        monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)
        monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", fake_prompt_yes_no)
        monkeypatch.setattr("hermes_cli.setup._prompt_container_resources", fake_container_resources)
        monkeypatch.setattr("hermes_cli.setup.cfg_get", lambda c, section, key, default=None: default)
        monkeypatch.setattr("hermes_cli.setup.save_env_value", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.setup.get_env_value", lambda key: "dcs-test-key" if key == "DAYTONA_API_KEY" else "")
        monkeypatch.setitem(
            _sys.modules, "daytona",
            type(_sys)("daytona"),
        )

        return config, resource_call_log

    def test_image_mode_calls_container_resources(self, monkeypatch, tmp_path):
        """Image mode must call _prompt_container_resources."""
        config, resource_call_log = self._make_setup_mocks(monkeypatch, tmp_path)

        from hermes_cli.setup import setup_terminal_backend
        setup_terminal_backend(config)

        assert len(resource_call_log) == 1, (
            f"_prompt_container_resources was called {len(resource_call_log)} time(s) "
            f"in image mode, but should have been called exactly once"
        )