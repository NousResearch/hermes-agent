"""Regression tests for #137529: disabled toolsets must not appear in `hermes doctor`.

`hermes doctor` previously walked the full ``unavailable`` list and emitted a
`⚠ image_gen` (or similar) row for every toolset the operator had explicitly
turned off via ``known_builtin_toolsets.cli`` / ``platform_toolsets.cli``.
The hint text is actionable only when the toolset is on; surfacing a phantom
warning for a deliberately-disabled install sends the operator to fix
something that is not broken and inflates the ⚠ count on a clean install.
The summary filter (``_missing_api_key_toolsets_for_summary``) already
excluded disabled toolsets from the "run hermes setup" issue line; the
per-row Tool Availability section must do the same.
"""
from __future__ import annotations

import contextlib
import io
import sys
import types

import pytest

from hermes_cli import doctor_tools


class TestDoctorDisabledToolsetRows:
    def _patch_model_tools(self, monkeypatch, unavailable):
        """Install a fake ``model_tools`` so ``_check_tool_availability`` does not
        hit the real tool registry."""
        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda: ([], unavailable),
            TOOLSET_REQUIREMENTS={
                item["name"]: {"name": item["name"]} for item in unavailable
            },
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
        monkeypatch.setattr(doctor_tools, "_apply_doctor_tool_availability_overrides",
                            lambda a, u: (a, u))
        monkeypatch.setattr(doctor_tools, "_doctor_web_capability_rows", list)

    def test_disabled_toolset_emits_no_warning_row(self, monkeypatch):
        """`image_gen` removed from the enabled set must not show in the table."""
        unavailable = [
            {"name": "image_gen", "env_vars": [], "tools": ["image_generate"]},
            {"name": "tts", "missing_vars": ["ELEVENLABS_API_KEY"], "tools": []},
        ]
        self._patch_model_tools(monkeypatch, unavailable)
        # Only ``tts`` is enabled for the CLI; ``image_gen`` was turned off.
        monkeypatch.setattr(doctor_tools, "_enabled_cli_toolsets_for_doctor",
                            lambda: {"tts"})

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_tools._check_tool_availability(False)
        out = buf.getvalue()

        assert "image_gen" not in out, (
            f"image_gen was disabled for the CLI but still appears in the doctor "
            f"Tool Availability rows (#137529). Output:\n{out}"
        )
        # tts must still surface a row — it is enabled and missing its key.
        assert "tts" in out
        assert "ELEVENLABS_API_KEY" in out

    def test_all_disabled_toolsets_emit_no_rows(self, monkeypatch):
        """A clean install with no enabled toolsets produces an empty Tool Availability table."""
        unavailable = [
            {"name": "image_gen", "env_vars": [], "tools": ["image_generate"]},
            {"name": "tts", "missing_vars": ["ELEVENLABS_API_KEY"], "tools": []},
            {"name": "rl", "missing_vars": ["TINKER_API_KEY"], "tools": []},
        ]
        self._patch_model_tools(monkeypatch, unavailable)
        monkeypatch.setattr(doctor_tools, "_enabled_cli_toolsets_for_doctor", lambda: set())

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_tools._check_tool_availability(False)
        out = buf.getvalue()

        for name in ("image_gen", "tts", "rl"):
            assert name not in out, (
                f"{name} is disabled for the CLI but still appears in the doctor "
                f"Tool Availability rows (#137529). Output:\n{out}"
            )

    def test_config_resolution_failure_falls_back_to_previous_behaviour(self, monkeypatch):
        """If the enabled-set lookup raises, doctor still shows every unavailable row.

        ``_enabled_cli_toolsets_for_doctor`` already swallows the exception and
        returns ``None``; we verify the fallback here so a config-load regression
        does not silently strip every warning.
        """
        unavailable = [
            {"name": "image_gen", "env_vars": [], "tools": ["image_generate"]},
        ]
        self._patch_model_tools(monkeypatch, unavailable)
        monkeypatch.setattr(doctor_tools, "_enabled_cli_toolsets_for_doctor", lambda: None)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_tools._check_tool_availability(False)
        out = buf.getvalue()

        assert "image_gen" in out

    def test_enabled_set_with_extra_toolsets_keeps_warnings_for_enabled_only(self, monkeypatch):
        """Mixed scenario: only the actually-enabled toolsets warn."""
        unavailable = [
            {"name": "image_gen", "env_vars": [], "tools": ["image_generate"]},
            {"name": "tts", "missing_vars": ["ELEVENLABS_API_KEY"], "tools": []},
            {"name": "rl", "missing_vars": ["TINKER_API_KEY"], "tools": []},
        ]
        self._patch_model_tools(monkeypatch, unavailable)
        # ``rl`` and ``tts`` are enabled; ``image_gen`` is not.
        monkeypatch.setattr(doctor_tools, "_enabled_cli_toolsets_for_doctor",
                            lambda: {"rl", "tts"})

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_tools._check_tool_availability(False)
        out = buf.getvalue()

        assert "image_gen" not in out
        assert "tts" in out
        assert "rl" in out
