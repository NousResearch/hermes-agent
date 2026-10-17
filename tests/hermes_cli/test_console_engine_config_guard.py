"""
Test that the console_engine _config_set path (alternate CLI entrypoint)
also refuses security-sensitive keys — the guard lives inside
set_config_value itself, not just in config_command's pre-check.

This pins the finding from review on #81108: an alternate supported
entrypoint (console_engine's `config set`) reaches set_config_value
without going through config_command's CLI-level sensitive-key check.
The in-set_config_value guard (#81101) must independently refuse.
"""
import io
import sys

import pytest

SENTINEL = "after-guard-reached"


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

    def test_operator_scope_skips_guard(self, operator_write_scope, monkeypatch, tmp_path):
        """The sanctioned operator path must NOT trigger the sensitive-key refusal —
        it must run on past the guard.

        ``is_managed=True`` returns at the top of ``set_config_value``, BEFORE the
        guard, so that would prove nothing. Instead, keep managed mode off and stop
        at the first helper after the guard: reaching ``_is_env_config_key`` proves
        execution passed the policy check, and ``_refuse_sensitive_config_key`` not
        firing proves the guard allowed it.
        """
        import hermes_cli.config as cfg
        import hermes_cli.config_policy as cfg_policy

        # Isolated store: the write now runs past the early return, so nothing may
        # touch the real home (the suite's hermetic guard enforces that).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg._LOAD_CONFIG_CACHE.clear()
        cfg._RAW_CONFIG_CACHE.clear()

        refuse_called = False
        reached = []

        def spy_refuse(*args, **kwargs):
            nonlocal refuse_called
            refuse_called = True
            raise SystemExit(1)

        def spy_after_guard(key):
            reached.append(key)
            raise SystemExit(SENTINEL)

        # The refusal and its authorization check live in config_policy (owned by
        # check_policy_write); the post-guard step is config's own dispatcher.
        monkeypatch.setattr(cfg_policy, "_refuse_sensitive_config_key", spy_refuse)
        monkeypatch.setattr(cfg, "_is_env_config_key", spy_after_guard)
        # Managed-mode detection reads the host install; pin it off so the guard under
        # test is the only thing that can produce the refusal.
        monkeypatch.setattr(cfg, "is_managed", lambda: False)
        monkeypatch.setattr(cfg, "_exit_if_key_managed", lambda key, action: None)

        err = io.StringIO()
        monkeypatch.setattr(sys, "stderr", err)

        with pytest.raises(SystemExit) as exc_info:
            cfg.set_config_value("approvals.mode", "off")

        assert exc_info.value.code == SENTINEL, "write did not proceed past the policy guard"
        assert reached == ["approvals.mode"], "after-guard path was not reached"
        assert not refuse_called, \
            "the operator-qualified scope must NOT trigger the sensitive-key refusal"

    def test_operator_scope_without_human_still_refused(self, monkeypatch):
        """Entering the operator scope WITHOUT human presence must still refuse:
        the conjunctive check is evaluated inside the writer, so a scope forged
        by agent-executed Python in a headless context grants nothing
        (#104697 review: an importable token is mintable by construction)."""
        import hermes_cli.config as cfg
        from tools.approval_context import grant_operator_policy_write, reset_operator_policy_write

        # Deliberately NOT interactive, NOT a gateway (no env, no platform).
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

        # Even if an operator scope was somehow granted, headless context without
        # a sanctioned handler frame in the writer refuses at write time.
        monkeypatch.setattr("tools.approval_context._is_sanctioned_policy_stamp_caller", lambda: True)
        token = grant_operator_policy_write()
        try:
            with pytest.raises(SystemExit):
                cfg.set_config_value("approvals.mode", "off")
        finally:
            reset_operator_policy_write(token)

    def test_unbound_grant_operator_policy_write_raises_without_sanctioned_boundary(self):
        """grant_operator_policy_write() must refuse (raise RuntimeError) when called
        directly outside the sanctioned input boundary (#104697 P1-A)."""
        from tools.approval_context import grant_operator_policy_write

        with pytest.raises(RuntimeError, match="operator policy write grant requires the sanctioned input boundary"):
            grant_operator_policy_write()
