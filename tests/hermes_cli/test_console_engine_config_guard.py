# -*- coding: utf-8 -*-
"""
Test that the console_engine _config_set path (alternate CLI entrypoint)
also refuses security-sensitive keys — the guard lives inside
set_config_value itself, not just in config_command's pre-check.

This pins the finding from review on #81108: an alternate supported
entrypoint (console_engine's `config set`) reaches set_config_value
without going through config_command's CLI-level sensitive-key check.
The in-set_config_value guard (#81101) must independently refuse.
"""
import sys
import io
import pytest


class TestConsoleEngineSensitiveKeyGuard:
    """Pin that the console_engine path (which calls set_config_value
    directly without config_command's pre-check) is still guarded."""

    @pytest.mark.parametrize("key", [
        "approvals.mode",
        "approvals",
        "security.foo",
        "security",
        "command_allowlist",
    ])
    def test_console_engine_refuses_sensitive_keys(self, key, monkeypatch):
        """set_config_value(key, value) without approval_override must
        refuse security-sensitive keys regardless of entrypoint."""
        from hermes_cli.config import set_config_value

        err = io.StringIO()
        monkeypatch.setattr(sys, "stderr", err)

        with pytest.raises(SystemExit) as exc_info:
            set_config_value(key, "off")

        assert exc_info.value.code == 1
        assert "security policy" in err.getvalue()

    def test_non_sensitive_keys_not_refused(self):
        """Non-sensitive keys are classified correctly."""
        from hermes_cli.config import _is_sensitive_config_key

        assert not _is_sensitive_config_key("model.default")
        assert not _is_sensitive_config_key("provider.openai.api_key")
        assert not _is_sensitive_config_key("ui.theme")

    def test_approval_override_skips_guard(self, monkeypatch):
        """approval_override=True (sanctioned /approvals command) must
        NOT trigger the sensitive-key refusal — it proceeds past the guard."""
        import hermes_cli.config as cfg

        refuse_called = False

        def spy_refuse(*args, **kwargs):
            nonlocal refuse_called
            refuse_called = True
            raise SystemExit(1)

        monkeypatch.setattr(cfg, "_refuse_sensitive_config_key", spy_refuse)
        # Short-circuit after the guard: is_managed exits before any write
        monkeypatch.setattr(cfg, "is_managed", lambda: True)

        err = io.StringIO()
        monkeypatch.setattr(sys, "stderr", err)

        try:
            cfg.set_config_value("approvals.mode", "off", approval_override=True)
        except SystemExit:
            pass  # is_managed exits — that's AFTER the guard, which is what we test

        assert not refuse_called, \
            "approval_override=True must NOT trigger the sensitive-key refusal"
