"""Tests for the claude-p delegation backend (PR4).

Covers config validation, argv construction, environment scrubbing, the
subscription auth-status projection, result normalization, process failure
handling, route selection/cooldown, and mixed native+Claude batches.

Every subprocess and auth call is mocked: these tests never invoke the real
``claude`` CLI, read credentials or account data, touch the network, or edit
user config.
"""

import asyncio
import json
import signal
import threading
import time

import pytest

from agent import claude_p_backend as cb
from agent import delegation_routing as dr
from agent import delegation_usage_cache as duc


CLAUDE_ROUTE = {
    "id": "claude-sub",
    "backend": "claude-p",
    "provider": "claude-p",
    "model": "claude-opus-5",
    "model_class": "frontier",
    "task_difficulties": ["complex", "frontier"],
    "capabilities": ["reasoning", "review"],
    "priority": 10,
}

CODEX_ROUTE = {
    "id": "codex-standard",
    "backend": "native",
    "provider": "openai-codex",
    "model": "gpt-5.6-sol",
    "model_class": "advanced",
    "task_difficulties": ["standard", "complex"],
    "capabilities": ["coding", "reasoning", "tool_use"],
    "priority": 20,
}


@pytest.fixture(autouse=True)
def _clean_module_state():
    cb.reset_cooldowns()
    cb.reset_workdir_locks()
    yield
    cb.reset_cooldowns()
    cb.reset_workdir_locks()


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


class TestClaudePRouteConfig:
    def test_claude_p_route_parses_with_defaults(self):
        catalog = dr.load_route_catalog(
            {"routing": {"enabled": True}, "routes": [CLAUDE_ROUTE]}
        )
        route = catalog.routes[0]
        assert route.backend == "claude-p"
        assert route.is_claude_p is True
        # Default is least privilege; coding must be explicit.
        assert route.tool_profile == "read_only"
        assert route.write_capable is False
        assert route.max_turns == dr.DEFAULT_CLAUDE_P_MAX_TURNS
        assert route.timeout_seconds == dr.DEFAULT_CLAUDE_P_TIMEOUT_SECONDS

    def test_coding_profile_is_explicit_opt_in_and_write_capable(self):
        catalog = dr.load_route_catalog(
            {
                "routing": {"enabled": True},
                "routes": [
                    {
                        **CLAUDE_ROUTE,
                        "tool_profile": "coding",
                        "capabilities": ["coding", "reasoning", "review"],
                    }
                ],
            }
        )
        assert catalog.routes[0].tool_profile == "coding"
        assert catalog.routes[0].write_capable is True

    def test_non_coding_profile_cannot_claim_coding_capability(self):
        with pytest.raises(dr.RouteConfigError, match="requires tool_profile 'coding'"):
            dr.load_route_catalog(
                {
                    "routing": {"enabled": True},
                    "routes": [
                        {
                            **CLAUDE_ROUTE,
                            "tool_profile": "review",
                            "capabilities": ["coding", "review"],
                        }
                    ],
                }
            )

    def test_unknown_tool_profile_is_a_loud_error(self):
        with pytest.raises(dr.RouteConfigError, match="tool_profile"):
            dr.load_route_catalog(
                {
                    "routing": {"enabled": True},
                    "routes": [{**CLAUDE_ROUTE, "tool_profile": "root"}],
                }
            )

    def test_claude_p_backend_rejects_api_key_provider_substitution(self):
        with pytest.raises(dr.RouteConfigError, match="requires provider 'claude-p'"):
            dr.load_route_catalog(
                {
                    "routing": {"enabled": True},
                    "routes": [{**CLAUDE_ROUTE, "provider": "anthropic"}],
                }
            )

    @pytest.mark.parametrize("backend", ["agy-p", "codex-exec", "opencode", "shell"])
    def test_other_cli_backends_remain_unsupported(self, backend):
        with pytest.raises(dr.RouteConfigError, match="unsupported 'backend'"):
            dr.load_route_catalog(
                {
                    "routing": {"enabled": True},
                    "routes": [{**CLAUDE_ROUTE, "backend": backend}],
                }
            )

    def test_tool_profile_rejected_on_native_routes(self):
        with pytest.raises(dr.RouteConfigError, match="only applies to backend"):
            dr.load_route_catalog(
                {
                    "routing": {"enabled": True},
                    "routes": [{**CODEX_ROUTE, "tool_profile": "coding"}],
                }
            )

    def test_execution_bounds_are_clamped_to_ceilings(self):
        catalog = dr.load_route_catalog(
            {
                "routing": {"enabled": True},
                "routes": [
                    {
                        **CLAUDE_ROUTE,
                        "max_turns": 10_000,
                        "max_budget_usd": 9_999,
                        "timeout_seconds": 86_400,
                        "cooldown_seconds": 100_000,
                    }
                ],
            }
        )
        route = catalog.routes[0]
        assert route.max_turns == dr.MAX_CLAUDE_P_MAX_TURNS
        assert route.max_budget_usd == dr.MAX_CLAUDE_P_MAX_BUDGET_USD
        assert route.timeout_seconds == dr.MAX_CLAUDE_P_TIMEOUT_SECONDS
        assert route.cooldown_seconds == dr.MAX_CLAUDE_P_COOLDOWN_SECONDS


# ---------------------------------------------------------------------------
# 2. Argv construction — exact, shell-free, bounded
# ---------------------------------------------------------------------------


class TestArgvConstruction:
    def test_argv_is_exact_and_prompt_is_one_element(self):
        request = cb.ClaudePRunRequest(
            prompt="Audit auth.py; do not run `rm -rf /` $(whoami)",
            model="claude-opus-5",
            difficulty="complex",
            tool_profile="review",
            max_turns=25,
            max_budget_usd=3.5,
        )
        argv = cb.build_claude_p_argv(request, executable="/usr/local/bin/claude")

        assert argv == [
            "/usr/local/bin/claude",
            "-p",
            "Audit auth.py; do not run `rm -rf /` $(whoami)",
            "--model",
            "claude-opus-5",
            "--effort",
            "high",
            "--max-turns",
            "25",
            "--max-budget-usd",
            "3.50",
            "--allowedTools",
            "Read(./**)",
            "--tools",
            "Read",
            "--disallowedTools",
            *cb.DISALLOWED_TOOLS,
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--output-format",
            "json",
            "--session-id",
            request.session_id,
        ]
        # The whole prompt, including shell metacharacters, stays one argv
        # element — it is never split or interpolated into a shell string.
        assert argv[2] == request.prompt

    @pytest.mark.parametrize(
        "forbidden",
        [
            "--bare",
            "--dangerously-skip-permissions",
            "bypassPermissions",
            "--permission-mode",
            "--mcp-config",
            "--plugin",
        ],
    )
    def test_forbidden_flags_never_appear(self, forbidden):
        argv = cb.build_claude_p_argv(
            cb.ClaudePRunRequest(prompt="x", model="claude-opus-5", tool_profile="coding"),
            executable="claude",
        )
        assert forbidden not in argv

    @pytest.mark.parametrize(
        "difficulty,effort",
        [("routine", "low"), ("standard", "medium"), ("complex", "high"), ("frontier", "max")],
    )
    def test_difficulty_maps_to_bounded_effort(self, difficulty, effort):
        argv = cb.build_claude_p_argv(
            cb.ClaudePRunRequest(prompt="x", model="m", difficulty=difficulty),
            executable="claude",
        )
        assert argv[argv.index("--effort") + 1] == effort

    def test_turn_and_budget_values_are_bounded(self):
        argv = cb.build_claude_p_argv(
            cb.ClaudePRunRequest(
                prompt="x", model="m", max_turns=10**9, max_budget_usd=10**9
            ),
            executable="claude",
        )
        assert int(argv[argv.index("--max-turns") + 1]) <= cb.MAX_TURNS_CEILING
        assert float(argv[argv.index("--max-budget-usd") + 1]) <= cb.MAX_BUDGET_USD_CEILING

    def test_tool_profiles_are_a_fixed_allowlist(self):
        assert cb.TOOL_PROFILES["read_only"] == "Read(./**)"
        assert "Write" not in cb.TOOL_PROFILES["read_only"]
        assert "Write" not in cb.TOOL_PROFILES["review"]
        assert "Write" in cb.TOOL_PROFILES["coding"]
        assert all("Bash" not in tools for tools in cb.PROFILE_TOOL_SETS.values())
        assert all("Read(./**)" in rules for rules in cb.TOOL_PROFILES.values())
        assert all("Read" != rules for rules in cb.TOOL_PROFILES.values())

    def test_unknown_profile_raises_rather_than_downgrading_silently(self):
        with pytest.raises(ValueError, match="unknown claude-p tool profile"):
            cb.resolve_tool_profile("everything")


# ---------------------------------------------------------------------------
# 3. Environment scrubbing
# ---------------------------------------------------------------------------


class TestEnvironmentScrub:
    def test_secrets_are_removed_and_essentials_preserved(self):
        base = {
            "HOME": "/home/dev",
            "PATH": "/usr/bin:/bin",
            "LANG": "en_US.UTF-8",
            "GIT_TERMINAL_PROMPT": "0",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "OPENAI_API_KEY": "sk-openai-secret",
            "GOOGLE_API_KEY": "goog-secret",
            "AWS_ACCESS_KEY_ID": "AKIA",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "AWS_SESSION_TOKEN": "aws-token",
            "GITHUB_TOKEN": "ghp_secret",
            "GH_TOKEN": "gh-secret",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/sa.json",
            "SOME_VENDOR_TOKEN": "vendor",
            "MY_SERVICE_SECRET": "hunter2",
            "DB_PASSWORD": "hunter2",
            "PROJECT_NAME": "hermes",
        }
        env = cb.build_scrubbed_environment(base)

        assert env["HOME"] == "/home/dev"
        assert env["PATH"] == "/usr/bin:/bin"
        assert env["LANG"] == "en_US.UTF-8"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert "PROJECT_NAME" not in env
        assert env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "1"

        for leaked in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "SOME_VENDOR_TOKEN",
            "MY_SERVICE_SECRET",
            "DB_PASSWORD",
        ):
            assert leaked not in env

        # No secret VALUE survives anywhere in the environment either.
        joined = "\n".join(f"{k}={v}" for k, v in env.items())
        for secret in ("sk-ant-secret", "sk-openai-secret", "ghp_secret", "hunter2", "AKIA"):
            assert secret not in joined

    @pytest.mark.parametrize(
        "name",
        [
            "AWS_PROFILE",
            "GOOGLE_CLOUD_PROJECT",
            "OPENAI_ORG_ID",
            "ANTHROPIC_CUSTOM_HEADERS",
            "GIT_ASKPASS",
            "SSH_ASKPASS",
            "PROJECT_NAME",
        ],
    )
    def test_unrelated_or_credential_context_is_not_inherited(self, name):
        env = cb.build_scrubbed_environment(
            {"HOME": "/h", "PATH": "/b", name: "identity-context"}
        )
        assert name not in env

    def test_native_provider_credentials_never_reach_the_claude_child(self):
        env = cb.build_scrubbed_environment(
            {
                "HOME": "/h",
                "PATH": "/b",
                "CODEX_API_KEY": "codex-secret",
                "ANTIGRAVITY_ACCESS_TOKEN": "agy-secret",
            }
        )
        assert "CODEX_API_KEY" not in env
        assert "ANTIGRAVITY_ACCESS_TOKEN" not in env


# ---------------------------------------------------------------------------
# 4. Auth status projection
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestAuthProjection:
    def _patch_probe(self, monkeypatch, result, *, executable="/usr/bin/claude"):
        monkeypatch.setattr(cb, "resolve_claude_executable", lambda: executable)
        import hermes_cli._subprocess_compat as sc

        monkeypatch.setattr(
            sc,
            "bounded_probe_run",
            lambda argv, timeout, env=None: result,
        )

    def test_projection_omits_identity_and_raw_fields(self, monkeypatch):
        raw = {
            "authenticated": True,
            "authMethod": "subscription",
            "email": "dev@example.com",
            "organization": "Acme Inc",
            "accountId": "acct_123",
            "accessToken": "oauth-token-value",
            "credentialPath": "/home/dev/.claude/creds.json",
        }
        self._patch_probe(monkeypatch, _FakeCompleted(0, json.dumps(raw)))

        availability = cb.check_claude_availability()

        assert availability.installed is True
        assert availability.authenticated is True
        assert availability.auth_method_class == "subscription"
        assert availability.available is True

        # Only booleans / the auth-method class survive the projection.
        blob = repr(availability)
        for identity in (
            "dev@example.com",
            "Acme Inc",
            "acct_123",
            "oauth-token-value",
            "/home/dev/.claude/creds.json",
        ):
            assert identity not in blob
        assert not hasattr(availability, "email")
        assert not hasattr(availability, "raw")

    def test_missing_cli_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(cb, "resolve_claude_executable", lambda: None)
        availability = cb.check_claude_availability()
        assert availability.installed is False
        assert availability.available is False

    def test_probe_failure_is_unauthenticated_not_a_raise(self, monkeypatch):
        self._patch_probe(monkeypatch, None)
        assert cb.check_claude_availability().available is False

    def test_malformed_json_is_unauthenticated(self, monkeypatch):
        self._patch_probe(monkeypatch, _FakeCompleted(0, "not json at all"))
        assert cb.check_claude_availability().available is False

    def test_no_remaining_quota_is_ever_derived(self, monkeypatch):
        raw = {"authenticated": True, "authMethod": "subscription", "usage": {"used": 42}}
        self._patch_probe(monkeypatch, _FakeCompleted(0, json.dumps(raw)))
        availability = cb.check_claude_availability()
        # The projection has no remaining/quota surface at all.
        for field in ("remaining", "quota", "allowance", "usage"):
            assert not hasattr(availability, field)

    @pytest.mark.parametrize(
        "raw",
        [
            {"authenticated": True, "authMethod": "apiKey", "apiProvider": "firstParty"},
            {"authenticated": True, "authMethod": "oauth", "apiProvider": "api"},
            {"authenticated": True, "authMethod": "oauth"},
            {"authenticated": True, "authMethod": "oauth", "apiProvider": "proxy"},
            {"authenticated": True, "authMethod": "oauth", "apiProvider": "production"},
            {"authenticated": True, "authMethod": "unknown"},
        ],
    )
    def test_api_key_or_unknown_auth_is_not_subscription_available(
        self, monkeypatch, raw
    ):
        self._patch_probe(monkeypatch, _FakeCompleted(0, json.dumps(raw)))
        assert cb.check_claude_availability().available is False

    def test_claude_ai_auth_is_subscription_available(self, monkeypatch):
        raw = {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
        }
        self._patch_probe(monkeypatch, _FakeCompleted(0, json.dumps(raw)))
        assert cb.check_claude_availability().available is True

    def test_explicit_pro_subscription_type_is_available(self, monkeypatch):
        raw = {
            "loggedIn": True,
            "authMethod": "oauth",
            "subscriptionType": "pro",
            "apiProvider": "firstParty",
        }
        self._patch_probe(monkeypatch, _FakeCompleted(0, json.dumps(raw)))
        assert cb.check_claude_availability().available is True


# ---------------------------------------------------------------------------
# 5 & 6. Result normalization and failure handling
# ---------------------------------------------------------------------------


class TestResultNormalization:
    def test_success_is_normalized(self):
        payload = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "Reviewed auth.py; found 2 issues.",
                "session_id": "12345678-1234-4234-8234-123456789abc",
                "num_turns": 7,
                "total_cost_usd": 0.1234,
                "usage": {"input_tokens": 1200, "output_tokens": 340},
                "modelUsage": {"claude-opus-5": {"inputTokens": 1200}},
            }
        )
        result = cb.normalize_claude_p_output(
            payload, session_id="local-uuid", duration_seconds=12.5, fallback_model="claude-opus-5"
        )
        assert result.status == "completed"
        assert result.summary == "Reviewed auth.py; found 2 issues."
        assert result.claude_session_id == "12345678-1234-4234-8234-123456789abc"
        assert result.num_turns == 7
        assert result.cost_usd == pytest.approx(0.1234)
        assert result.tokens == {"input": 1200, "output": 340}
        assert result.model_usage == {"claude-opus-5": {"inputTokens": 1200}}
        assert result.error is None

    def test_only_the_final_json_object_is_parsed(self):
        noisy = 'warning: something\n{"subtype":"partial"}\n' + json.dumps(
            {"subtype": "success", "is_error": False, "result": "final answer", "num_turns": 1}
        )
        result = cb.normalize_claude_p_output(noisy, session_id="s", duration_seconds=1.0)
        assert result.summary == "final answer"

    def test_valid_object_followed_by_malformed_tail_is_rejected(self):
        raw = json.dumps({"subtype": "success", "result": "do not trust"}) + "\n{partial"
        result = cb.normalize_claude_p_output(raw, session_id="s", duration_seconds=1.0)
        assert result.exit_reason == "malformed_output"
        assert result.summary is None

    @pytest.mark.parametrize("subtype", ["error_max_turns", "error_budget", "error_during_execution"])
    def test_error_subtypes_fail_safely(self, subtype):
        payload = json.dumps(
            {"subtype": subtype, "is_error": True, "result": "secret account text"}
        )
        result = cb.normalize_claude_p_output(payload, session_id="s", duration_seconds=2.0)
        assert result.status == "error"
        assert result.exit_reason == subtype
        assert result.error
        assert "secret account text" not in result.error
        assert result.summary is None

    def test_model_usage_is_numeric_allowlisted_and_identity_free(self):
        payload = json.dumps(
            {
                "subtype": "success",
                "result": "ok",
                "modelUsage": {
                    "claude-opus-5": {
                        "inputTokens": 10,
                        "email": "dev@example.com",
                        "accountId": "acct-secret",
                    },
                    "identity@example.com": {"inputTokens": 999},
                },
            }
        )
        result = cb.normalize_claude_p_output(
            payload,
            session_id="s",
            duration_seconds=1.0,
            fallback_model="claude-opus-5",
        )
        assert result.model_usage == {"claude-opus-5": {"inputTokens": 10}}
        assert "example.com" not in repr(result.model_usage)
        assert "acct-secret" not in repr(result.model_usage)

    def test_malformed_output_fails_safely_without_raw_text(self):
        raw = "Traceback (most recent call last): SECRET-TOKEN-VALUE leaked here"
        result = cb.normalize_claude_p_output(raw, session_id="s", duration_seconds=1.0)
        assert result.status == "error"
        assert result.exit_reason == "malformed_output"
        assert "SECRET-TOKEN-VALUE" not in (result.error or "")
        assert result.summary is None

    def test_oversized_output_is_capped(self):
        assert cb.MAX_STDOUT_BYTES <= 8 * 1024 * 1024
        assert cb.MAX_STDERR_BYTES <= 1024 * 1024


class TestProcessFailures:
    def test_windows_tree_kill_happens_before_direct_wait(self, monkeypatch):
        events = []

        class _Process:
            pid = 5151

            async def wait(self):
                events.append("wait")
                return 0

        import hermes_cli._subprocess_compat as sc

        monkeypatch.setattr(cb.os, "name", "nt")
        monkeypatch.setattr(sc, "kill_process_tree", lambda proc: events.append("tree-kill"))

        asyncio.run(cb._cancel_process(_Process()))  # type: ignore[arg-type]
        assert events == ["tree-kill", "wait"]

    def test_cancellation_during_process_wait_cleans_reader_tasks(self, monkeypatch):
        wait_started = asyncio.Event()
        readers_started = asyncio.Event()
        reader_count = 0

        class _RunningProcess:
            pid = 4241
            stdout = object()
            stderr = object()
            returncode = None
            terminated = False

            async def wait(self):
                wait_started.set()
                if self.terminated:
                    self.returncode = 0
                    return 0
                await asyncio.Future()

        proc = _RunningProcess()
        signals = []

        async def _fake_create(*args, **kwargs):
            return proc

        async def _held_pipe(stream, cap):
            nonlocal reader_count
            reader_count += 1
            if reader_count == 2:
                readers_started.set()
            await asyncio.Future()

        def _fake_signal(group_id, sig):
            signals.append(sig)
            if sig == signal.SIGTERM:
                proc.terminated = True

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)
        monkeypatch.setattr(cb, "_read_capped", _held_pipe)
        monkeypatch.setattr(cb, "_signal_process_group", _fake_signal)

        async def _scenario():
            task = asyncio.create_task(
                cb._run_claude_p_async(
                    ["claude", "-p", "prompt"],
                    env={},
                    workdir=".",
                    timeout_seconds=10,
                )
            )
            await asyncio.wait_for(wait_started.wait(), timeout=1)
            await asyncio.wait_for(readers_started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert asyncio.all_tasks() == {asyncio.current_task()}

        asyncio.run(_scenario())
        assert signal.SIGTERM in signals
        assert signal.SIGKILL in signals

    def test_cancellation_during_pipe_drain_cleans_all_tasks(self, monkeypatch):
        class _ExitedProcess:
            pid = 4343
            stdout = object()
            stderr = object()
            returncode = 0

            async def wait(self):
                return 0

        proc = _ExitedProcess()
        signals = []
        readers_started = asyncio.Event()
        reader_count = 0

        async def _fake_create(*args, **kwargs):
            return proc

        async def _held_pipe(stream, cap):
            nonlocal reader_count
            reader_count += 1
            if reader_count == 2:
                readers_started.set()
            await asyncio.Future()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)
        monkeypatch.setattr(cb, "_read_capped", _held_pipe)
        monkeypatch.setattr(
            cb,
            "_signal_process_group",
            lambda group_id, sig: signals.append(sig),
        )

        async def _scenario():
            task = asyncio.create_task(
                cb._run_claude_p_async(
                    ["claude", "-p", "prompt"],
                    env={},
                    workdir=".",
                    timeout_seconds=1,
                )
            )
            await asyncio.wait_for(readers_started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert asyncio.all_tasks() == {asyncio.current_task()}

        asyncio.run(_scenario())
        assert signal.SIGTERM in signals
        assert signal.SIGKILL in signals

    def test_parent_exit_with_descendant_held_pipes_is_bounded(self, monkeypatch):
        class _FakeProcess:
            pid = 4242
            stdout = object()
            stderr = object()
            returncode = None
            terminated = False

            async def wait(self):
                if self.terminated:
                    self.returncode = 0
                    return 0
                await asyncio.Future()

        proc = _FakeProcess()
        signals = []

        async def _fake_create(*args, **kwargs):
            return proc

        async def _held_pipe(stream, cap):
            await asyncio.Future()

        def _fake_signal(group_id, sig):
            assert group_id == proc.pid
            signals.append(sig)
            if sig == signal.SIGTERM:
                proc.terminated = True

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)
        monkeypatch.setattr(cb, "_read_capped", _held_pipe)
        monkeypatch.setattr(cb, "_signal_process_group", _fake_signal)
        monkeypatch.setattr(cb, "PROCESS_TERMINATE_GRACE_SECONDS", 0.01)
        monkeypatch.setattr(cb, "PROCESS_KILL_GRACE_SECONDS", 0.01)
        monkeypatch.setattr(cb, "PIPE_DRAIN_GRACE_SECONDS", 0.01)

        started = time.monotonic()
        result = asyncio.run(
            cb._run_claude_p_async(
                ["claude", "-p", "prompt"],
                env={},
                workdir=".",
                timeout_seconds=0.01,
            )
        )

        assert time.monotonic() - started < 1.0
        assert result[3] is True
        assert result[4] is True
        assert signal.SIGKILL in signals

    def _run_with(self, monkeypatch, *, returncode, stdout=b"", timed_out=False):
        monkeypatch.setattr(cb, "resolve_claude_executable", lambda: "/usr/bin/claude")

        async def _fake(argv, *, env, workdir, timeout_seconds):
            return returncode, stdout, b"stderr with SECRET-VALUE inside", timed_out, False

        monkeypatch.setattr(cb, "_run_claude_p_async", _fake)
        return cb.run_claude_p_task(
            cb.ClaudePRunRequest(prompt="p", model="claude-opus-5"), write_capable=False
        )

    def test_nonzero_exit_never_returns_raw_stderr(self, monkeypatch):
        result = self._run_with(monkeypatch, returncode=2)
        assert result.status == "error"
        assert result.exit_reason == "nonzero_exit"
        assert "SECRET-VALUE" not in (result.error or "")
        assert "stderr" not in (result.error or "").lower()

    def test_timeout_is_reported_without_raw_output(self, monkeypatch):
        result = self._run_with(monkeypatch, returncode=-1, timed_out=True)
        assert result.status == "timeout"
        assert result.exit_reason == "timeout"
        assert "SECRET-VALUE" not in (result.error or "")

    def test_missing_cli_fails_safely(self, monkeypatch):
        monkeypatch.setattr(cb, "resolve_claude_executable", lambda: None)
        result = cb.run_claude_p_task(
            cb.ClaudePRunRequest(prompt="p", model="m"), write_capable=False
        )
        assert result.status == "error"
        assert result.exit_reason == "unavailable"

    def test_spawn_error_fails_safely(self, monkeypatch):
        monkeypatch.setattr(cb, "resolve_claude_executable", lambda: "/usr/bin/claude")

        async def _boom(argv, *, env, workdir, timeout_seconds):
            raise OSError("permission denied opening /home/dev/.claude/creds.json")

        monkeypatch.setattr(cb, "_run_claude_p_async", _boom)
        result = cb.run_claude_p_task(
            cb.ClaudePRunRequest(prompt="p", model="m"), write_capable=False
        )
        assert result.status == "error"
        assert result.exit_reason == "spawn_error"
        assert "creds.json" not in (result.error or "")

    def test_success_path_normalizes(self, monkeypatch):
        payload = json.dumps(
            {"subtype": "success", "is_error": False, "result": "done", "num_turns": 3}
        ).encode()
        result = self._run_with(monkeypatch, returncode=0, stdout=payload)
        assert result.status == "completed"
        assert result.summary == "done"


# ---------------------------------------------------------------------------
# 7. Per-workdir serialization
# ---------------------------------------------------------------------------


class TestWorkdirSerialization:
    def _runner(self, monkeypatch, hold_seconds, observed, lock_for_writes):
        monkeypatch.setattr(cb, "resolve_claude_executable", lambda: "/usr/bin/claude")
        active = {"count": 0, "max": 0}
        guard = threading.Lock()

        async def _fake(argv, *, env, workdir, timeout_seconds):
            with guard:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(hold_seconds)
            with guard:
                active["count"] -= 1
            return 0, json.dumps({"subtype": "success", "result": "ok"}).encode(), b"", False, False

        monkeypatch.setattr(cb, "_run_claude_p_async", _fake)
        observed["active"] = active
        return active

    def test_write_capable_tasks_serialize_on_one_workdir(self, monkeypatch, tmp_path):
        observed = {}
        active = self._runner(monkeypatch, 0.15, observed, True)

        def _go():
            cb.run_claude_p_task(
                cb.ClaudePRunRequest(prompt="p", model="m", workdir=str(tmp_path)),
                write_capable=True,
            )

        threads = [threading.Thread(target=_go) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert active["max"] == 1, "write-capable tasks must not overlap on one workdir"

    def test_read_only_tasks_may_overlap(self, monkeypatch, tmp_path):
        observed = {}
        active = self._runner(monkeypatch, 0.2, observed, False)

        def _go():
            cb.run_claude_p_task(
                cb.ClaudePRunRequest(prompt="p", model="m", workdir=str(tmp_path)),
                write_capable=False,
            )

        threads = [threading.Thread(target=_go) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert active["max"] > 1, "read-only tasks should be allowed to run in parallel"

    def test_different_workdirs_do_not_block_each_other(self, tmp_path):
        a = cb.workdir_lock_for(str(tmp_path / "a"))
        b = cb.workdir_lock_for(str(tmp_path / "b"))
        assert a is not b
        assert cb.workdir_lock_for(str(tmp_path / "a")) is a


# ---------------------------------------------------------------------------
# 8. Cooldown + selection
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_failure_places_route_in_bounded_cooldown(self):
        assert cb.is_route_in_cooldown("claude-sub") is False
        cb.note_route_failure("claude-sub", retry_after_seconds=30)
        assert cb.is_route_in_cooldown("claude-sub") is True

    def test_cooldown_expires(self):
        cb.note_route_failure("claude-sub", retry_after_seconds=0.01)
        time.sleep(0.05)
        assert cb.is_route_in_cooldown("claude-sub") is False

    def test_cooldown_is_bounded_to_one_hour(self):
        cb.note_route_failure("claude-sub", retry_after_seconds=10**9)
        with cb._cooldown_lock:
            until = cb._cooldown_until["claude-sub"]
        assert until - time.time() <= 3601


class TestRouteSelection:
    def _catalog(self, routes):
        return dr.load_route_catalog({"routing": {"enabled": True}, "routes": routes})

    def test_explicit_claude_route_works_with_unknown_usage(self):
        catalog = self._catalog([CLAUDE_ROUTE, CODEX_ROUTE])
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX, route_id="claude-sub"),
            usage=dr.UsageView({}),
            available_providers=frozenset({"claude-p", "openai-codex"}),
        )
        assert decision.selected is True
        assert decision.route_id == "claude-sub"
        assert decision.provider == "claude-p"
        # No cached reading is supplied here (empty UsageView): usage stays
        # unknown and is never fabricated.
        assert decision.usage_freshness == "unknown"
        assert decision.usage_remaining_percent is None

    def test_claude_usage_stays_unknown_and_schedules_one_background_refresh(
        self, monkeypatch, tmp_path
    ):
        """With no cached reading yet, claude-p reports unknown (never fabricated)
        but — like every other provider — schedules exactly one out-of-band
        refresh rather than being permanently frozen at unknown."""
        catalog = self._catalog([CLAUDE_ROUTE])
        monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
        refreshes = []
        monkeypatch.setattr(duc, "_spawn_refresh", lambda provider: refreshes.append(provider))
        view = duc.build_route_usage_view(
            catalog.routes,
            ttl_seconds=300,
            stale_seconds=1800,
        )
        assert view.for_route(catalog.routes[0]).freshness == "unknown"
        assert refreshes == ["claude-p"]
        # Scheduling a refresh must not itself write anything inline — the
        # selection/read path stays non-blocking.
        assert not (tmp_path / "usage.json").exists()

    def test_auto_routing_uses_fixed_priority_when_usage_unknown(self):
        catalog = self._catalog([CODEX_ROUTE, CLAUDE_ROUTE])
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX),
            usage=dr.UsageView({}),
            available_providers=frozenset({"claude-p", "openai-codex"}),
        )
        # claude-sub has priority 10 vs codex 20 → lower number wins.
        assert decision.route_id == "claude-sub"

    def test_unavailable_claude_falls_through_to_native(self):
        catalog = self._catalog([CLAUDE_ROUTE, CODEX_ROUTE])
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX),
            usage=dr.UsageView({}),
            available_providers=frozenset({"openai-codex"}),
        )
        assert decision.selected is True
        assert decision.route_id == "codex-standard"
        assert any("claude-p" in x for x in decision.exclusions)

    def test_cooldown_chooses_the_next_native_route(self):
        catalog = self._catalog([CLAUDE_ROUTE, CODEX_ROUTE])
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(difficulty=dr.TaskDifficulty.COMPLEX),
            usage=dr.UsageView({}),
            available_providers=frozenset({"claude-p", "openai-codex"}),
            cooling_down_route_ids=frozenset({"claude-sub"}),
        )
        assert decision.route_id == "codex-standard"
        assert any("cooldown" in x for x in decision.exclusions)

    def test_capabilities_still_filter_claude_routes(self):
        catalog = self._catalog([CLAUDE_ROUTE])
        decision = dr.select_route(
            catalog,
            dr.RouteRequest(
                difficulty=dr.TaskDifficulty.COMPLEX,
                required_capabilities=frozenset({"vision"}),
            ),
            usage=dr.UsageView({}),
            available_providers=frozenset({"claude-p"}),
        )
        assert decision.selected is False
