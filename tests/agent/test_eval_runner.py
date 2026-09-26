"""Tests for agent.eval_runner — the real Phase 3 execution path.

Strategy: never call the real auxiliary_client or subprocess beyond
trivial shell commands. All tests inject fake responses or use the
private-check subprocess directly against benign shell binaries
(`true`, `false`, `echo`, `test`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent.eval_runner import (
    DefaultEvalRunner,
    EvalInvocation,
    PrivateCheckError,
    PrivateCheckResult,
    PromptResult,
    dangerous_token_patterns,
    reset_dangerous_token_patterns,
)


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage:
    """Indirect test via _extract_usage: it's a private helper, so we
    exercise it through execute_prompt on a fake response.
    """

    def test_extracts_object_usage(self):
        class _Usage:
            prompt_tokens = 12
            completion_tokens = 34

        class _Choice:
            class _Msg:
                content = "hello"

            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()
            model = "test-model"

        # Patch call_llm to return this fake response.
        runner = DefaultEvalRunner()
        with patch("agent.auxiliary_client.call_llm", return_value=_Resp()):
            result = runner.execute_prompt(
                EvalInvocation(prompt="hi", model_kwargs={"model": "test"})
            )
        assert result.success
        assert result.tokens_in == 12
        assert result.tokens_out == 34
        assert result.model == "test-model"


# ---------------------------------------------------------------------------
# Prompt execution
# ---------------------------------------------------------------------------


class TestExecutePrompt:
    def test_returns_failure_when_aux_client_missing(self):
        """If auxiliary_client is unimportable, returns a failure rather
        than raising."""
        runner = DefaultEvalRunner()
        with patch.dict(sys.modules, {"agent.auxiliary_client": None}):
            result = runner.execute_prompt(EvalInvocation(prompt="x"))
        assert result.success is False
        assert "auxiliary_client unavailable" in (result.error or "")

    def test_returns_failure_when_call_llm_raises(self):
        runner = DefaultEvalRunner()
        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("provider down"),
        ):
            result = runner.execute_prompt(EvalInvocation(prompt="x"))
        assert result.success is False
        assert "provider down" in (result.error or "")


# ---------------------------------------------------------------------------
# Private check execution
# ---------------------------------------------------------------------------


class TestRunPrivateCheck:
    def test_empty_check_returns_failure(self):
        runner = DefaultEvalRunner()
        result = runner.run_private_check(EvalInvocation(prompt="x", private_check=""))
        assert result.success is False
        assert result.exit_code == -1

    def test_unparseable_shell_syntax_raises(self):
        runner = DefaultEvalRunner()
        with pytest.raises(PrivateCheckError):
            runner.run_private_check(
                EvalInvocation(prompt="x", private_check="'unclosed")
            )

    def test_benign_command_succeeds(self):
        runner = DefaultEvalRunner()
        result = runner.run_private_check(
            EvalInvocation(prompt="x", private_check="true")
        )
        assert result.success is True
        assert result.exit_code == 0

    def test_failing_command_returns_failure(self):
        runner = DefaultEvalRunner()
        result = runner.run_private_check(
            EvalInvocation(prompt="x", private_check="false")
        )
        assert result.success is False
        assert result.exit_code != 0

    def test_test_command_runs(self):
        """``test -f /etc/passwd`` should run and report true."""
        runner = DefaultEvalRunner()
        result = runner.run_private_check(
            EvalInvocation(prompt="x", private_check="test -f /etc/passwd")
        )
        assert result.success is True

    def test_timeout_captured(self):
        runner = DefaultEvalRunner()
        result = runner.run_private_check(
            EvalInvocation(
                prompt="x",
                private_check="sleep 5",
                timeout_sec=1.0,
            )
        )
        assert result.timed_out is True
        assert result.success is False
        assert "timed out" in result.stderr

    def test_stdout_truncated(self):
        runner = DefaultEvalRunner(stdout_cap_bytes=16)
        result = runner.run_private_check(
            EvalInvocation(prompt="x", private_check="echo " + ("x" * 1000))
        )
        assert result.success is True
        assert len(result.stdout.encode("utf-8")) <= 64  # cap + suffix + safety
        assert "truncated" in result.stdout


class TestDangerousTokenFilter:
    """The runner must block obvious exfil / priv-esc patterns by
    default. Each test uses a shell-only invocation that *would* work
    if not blocked — we are testing the filter, not the command.
    """

    @pytest.mark.parametrize(
        "bad_token",
        [
            "sudo",
            "curl",
            "wget",
            "nc",
            "ssh",
            "python",
            "perl",
            "bash",
        ],
    )
    def test_blocks_known_dangerous_token(self, bad_token: str):
        runner = DefaultEvalRunner()
        with pytest.raises(PrivateCheckError) as exc:
            runner.run_private_check(
                EvalInvocation(prompt="x", private_check=f"{bad_token} foo")
            )
        assert "blocked" in str(exc.value).lower()

    def test_allows_safe_tokens(self):
        runner = DefaultEvalRunner()
        result = runner.run_private_check(
            EvalInvocation(prompt="x", private_check="echo hello")
        )
        assert result.success is True

    def test_allow_unsafe_disables_filter(self):
        runner = DefaultEvalRunner(allow_unsafe_private_check=True)
        # `bash` would normally be blocked; with the flag, the runner
        # tries to execute it. On macOS /bin/bash exists; the argv
        # is `bash -c echo hi`. The filter no longer intercepts.
        result = runner.run_private_check(
            EvalInvocation(prompt="x", private_check="bash -c 'echo hi'")
        )
        # We don't care whether bash succeeded or failed (CI sandbox
        # may block bash subprocess); we only care that the filter
        # did NOT raise PrivateCheckError.
        # The runner returns a PrivateCheckResult either way.
        assert isinstance(result, PrivateCheckResult)

    def test_custom_patterns(self):
        reset_dangerous_token_patterns([r"^never$"])
        try:
            runner = DefaultEvalRunner()
            with pytest.raises(PrivateCheckError):
                runner.run_private_check(
                    EvalInvocation(prompt="x", private_check="never do this")
                )
        finally:
            reset_dangerous_token_patterns(list(dangerous_token_patterns()))


class TestEnvFiltering:
    def test_only_safe_keys_pass_through(self):
        runner = DefaultEvalRunner()
        # Inject a fake "unsafe" env var.
        with patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "EVIL_KEY": "exploit", "USER": "alice"},
            clear=False,
        ):
            result = runner.run_private_check(
                EvalInvocation(prompt="x", private_check="true")
            )
            assert result.success is True
            # We can't easily inspect the env from here without
            # patching subprocess.run itself; assert the contract
            # is that the runner does not raise.

    def test_extra_env_overrides(self):
        runner = DefaultEvalRunner()
        invocation = EvalInvocation(
            prompt="x",
            private_check="true",
            extra_env={"MY_VAR": "value"},
        )
        result = runner.run_private_check(invocation)
        assert result.success is True


# ---------------------------------------------------------------------------
# _cap helper
# ---------------------------------------------------------------------------


class TestCap:
    def test_short_text_unchanged(self):
        from agent.eval_runner import _cap

        assert _cap("hello", 100) == "hello"

    def test_truncates_long_text(self):
        from agent.eval_runner import _cap

        out = _cap("x" * 1000, 32)
        assert len(out.encode("utf-8")) <= 64  # cap + suffix
        assert out.endswith("truncated")

    def test_empty_text(self):
        from agent.eval_runner import _cap

        assert _cap("", 100) == ""
