"""Tests for the `todo_progress` opt-in gate.

The feature: surface the `todo` tool's plan/status card in gateway progress
WITHOUT enabling general `tool_progress`. Mirrors the existing
`delegate_task_args` opt-in — a single tool's card renders even when
`tool_progress` is "off" (the Telegram default), so the multi-step plan +
per-item status/timing shows up without the rest of the tool noise.

These cover the two wiring seams a regression would hit:
  1. `_tool_progress_pipeline_enabled` must keep the queue/consumer alive when
     ONLY `todo_progress` is on (otherwise the card is built but never flushed).
  2. `display_config` must resolve + normalise the flag and expose it as an
     overrideable per-platform key.
"""

from gateway.run import _tool_progress_pipeline_enabled
from gateway.display_config import OVERRIDEABLE_KEYS, resolve_display_setting


# ── pipeline gate: todo_progress must keep the queue/consumer alive ──────────
class TestTodoProgressPipelineGate:
    def test_todo_keeps_pipeline_alive_with_everything_else_off(self):
        """The bug this prevents: tool_progress="off" tears down the progress
        queue, so a todo card would be built but never rendered. The pipeline
        MUST stay alive when todo_progress is enabled even with every other
        progress source off."""
        assert _tool_progress_pipeline_enabled(
            is_webhook=False,
            progress_mode="off",
            tool_completion_durations_enabled=False,
            subagent_progress_enabled=False,
            delegate_task_args_enabled=False,
            subagent_roster_enabled=False,
            todo_progress_enabled=True,
        ) is True

    def test_everything_off_is_disabled(self):
        assert _tool_progress_pipeline_enabled(
            is_webhook=False,
            progress_mode="off",
            tool_completion_durations_enabled=False,
            subagent_progress_enabled=False,
            delegate_task_args_enabled=False,
            subagent_roster_enabled=False,
            todo_progress_enabled=False,
        ) is False

    def test_webhook_always_disabled_even_with_todo(self):
        """Webhooks have no message editing — never spin up the pipeline."""
        assert _tool_progress_pipeline_enabled(
            is_webhook=True,
            progress_mode="off",
            tool_completion_durations_enabled=False,
            subagent_progress_enabled=False,
            delegate_task_args_enabled=False,
            subagent_roster_enabled=False,
            todo_progress_enabled=True,
        ) is False

    def test_default_arg_is_back_compat_false(self):
        """Existing callers/tests that omit todo_progress_enabled keep the old
        behavior (pipeline off when everything else is off)."""
        assert _tool_progress_pipeline_enabled(
            is_webhook=False,
            progress_mode="off",
            tool_completion_durations_enabled=False,
            subagent_progress_enabled=False,
            delegate_task_args_enabled=False,
        ) is False


# ── display_config: resolution + normalisation ──────────────────────────────
class TestTodoProgressDisplayConfig:
    def test_flag_is_in_overrideable_keys(self):
        assert "todo_progress" in OVERRIDEABLE_KEYS

    def test_default_is_off(self):
        assert resolve_display_setting({}, "telegram", "todo_progress") == "off"

    def test_per_platform_on_string(self):
        cfg = {"display": {"platforms": {"telegram": {"todo_progress": "on"}}}}
        assert resolve_display_setting(cfg, "telegram", "todo_progress") == "on"

    def test_yaml_bool_true_normalises_to_on(self):
        # YAML `todo_progress: true` parses to Python True.
        cfg = {"display": {"platforms": {"telegram": {"todo_progress": True}}}}
        assert resolve_display_setting(cfg, "telegram", "todo_progress") == "on"

    def test_yaml_bare_off_normalises_to_off(self):
        # YAML 1.1 bare `off` parses to Python False.
        cfg = {"display": {"platforms": {"telegram": {"todo_progress": False}}}}
        assert resolve_display_setting(cfg, "telegram", "todo_progress") == "off"

    def test_unknown_string_fails_safe_to_off(self):
        cfg = {"display": {"platforms": {"telegram": {"todo_progress": "loud"}}}}
        assert resolve_display_setting(cfg, "telegram", "todo_progress") == "off"

    def test_global_setting_applies_when_no_platform_override(self):
        cfg = {"display": {"todo_progress": "on"}}
        assert resolve_display_setting(cfg, "telegram", "todo_progress") == "on"

    def test_platform_override_beats_global(self):
        cfg = {
            "display": {
                "todo_progress": "on",
                "platforms": {"telegram": {"todo_progress": "off"}},
            }
        }
        assert resolve_display_setting(cfg, "telegram", "todo_progress") == "off"
