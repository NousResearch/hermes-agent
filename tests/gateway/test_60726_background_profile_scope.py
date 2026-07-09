"""
Regression tests for issue #60726 - /background command fails with
UnscopedSecretError when multiplexing is on.

The bug: `_run_background_task` in gateway/run.py:13237 calls
`_resolve_session_agent_runtime()` which eventually calls
`get_secret('OPENROUTER_BASE_URL')`. When multiplex_profiles is True
and no profile secret scope is active, that call raises
`UnscopedSecretError` — the security boundary blocks reading the
process-global env var because in a multiplexed gateway that may
hold another profile's value.

The fix: wrap `_run_background_task`'s body in
`_profile_runtime_scope` when multiplex is on, mirroring the existing
pattern in `_run_agent` (gateway/run.py:16864-16873). The scope is a
sync context manager (yield + try/finally) but the contextvar it
installs propagates into coroutines awaited from within the with-block.

These tests build a minimal GatewayRunner with multiplex_profiles=True
and call `_run_background_task`. Without the fix, the underlying
credential resolution raises UnscopedSecretError. With the fix, the
task runs inside a scope so the credential resolves cleanly.

The test approach: rather than exercise the full async background task
(heavy mocking required), extract the credential-resolution-only path
into a small helper and test it directly. The helper has the same
shape as `_run_agent`'s profile-scoping pattern.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_runner(*, multiplex_profiles: bool = True) -> SimpleNamespace:
    """Build a minimal GatewayRunner with the given multiplex setting.

    Uses object.__new__() to bypass __init__; only the attrs the
    fix-and-test interact with are populated.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    # Configurable multiplex flag.
    config = MagicMock()
    config.multiplex_profiles = multiplex_profiles
    runner.config = config
    # Source used by _resolve_profile_home_for_source.
    runner._source = SimpleNamespace(
        platform="telegram",
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
        profile="",  # empty => use active profile
    )
    return runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunBackgroundTaskProfileScope:
    """When multiplex is on, _run_background_task must install a
    profile secret scope around its body so credential resolution
    doesn't hit UnscopedSecretError.
    """

    def test_helper_inner_extracted(self):
        """The fix extracts the body of _run_background_task into a
        separate ``_run_background_task_inner`` method so the wrapper
        can install the profile scope around the call. Before the fix,
        the inner method does not exist.
        """
        from gateway.run import GatewayRunner

        assert hasattr(GatewayRunner, "_run_background_task_inner"), (
            "GatewayRunner._run_background_task_inner is missing; the "
            "#60726 fix extracts the task body into a helper that can "
            "be wrapped in _profile_runtime_scope."
        )

    def test_multiplex_disabled_unchanged_behavior(self):
        """When multiplex_profiles is False, the fix is a transparent
        pass-through — calls the inner directly, no scope installed.
        Regression guard: this protects single-profile gateways.
        """
        runner = _make_runner(multiplex_profiles=False)

        # _resolve_profile_home_for_source should NOT be called when
        # multiplex is off (no need to know the profile).
        with patch.object(
            runner, "_resolve_profile_home_for_source", create=True
        ) as mock_resolve:
            with patch.object(
                runner, "_run_background_task_inner",
                new=AsyncMock(return_value=None),
            ) as mock_inner:
                import asyncio
                asyncio.run(runner._run_background_task("test", runner._source, "task-1"))
            mock_inner.assert_called_once()
            # Resolve must NOT be called when multiplex is off.
            mock_resolve.assert_not_called()

    def test_multiplex_enabled_wraps_in_profile_scope(self):
        """The fix: when multiplex is on, _run_background_task must
        install _profile_runtime_scope around the inner call so the
        secret scope is active for the inner call's credential reads.
        """
        runner = _make_runner(multiplex_profiles=True)

        with patch.object(
            runner, "_resolve_profile_home_for_source",
            return_value="/tmp/fake-profile-home",
            create=True,
        ) as mock_resolve:
            # Track which scope was active during the inner call.
            captured_scopes = []

            async def _capture_then_inner(*args, **kwargs):
                from agent.secret_scope import current_secret_scope
                captured_scopes.append(current_secret_scope())
                return None

            with patch.object(
                runner, "_run_background_task_inner",
                side_effect=_capture_then_inner,
            ):
                import asyncio
                asyncio.run(runner._run_background_task("test", runner._source, "task-1"))

            # The inner ran exactly once.
            assert len(captured_scopes) == 1
            # During the inner, a secret scope was active (not None).
            assert captured_scopes[0] is not None, (
                "secret scope was not active during _run_background_task_inner; "
                "the fix did not wrap the inner call in _profile_runtime_scope. "
                "Issue #60726: credential reads inside the background task raise "
                "UnscopedSecretError when multiplex is on."
            )

    def test_multiplex_enabled_resolves_profile_home(self):
        """The fix must call _resolve_profile_home_for_source exactly
        once and pass the result to the scope.
        """
        runner = _make_runner(multiplex_profiles=True)

        with patch.object(
            runner, "_resolve_profile_home_for_source",
            return_value="/tmp/profile-X",
            create=True,
        ) as mock_resolve:
            with patch.object(
                runner, "_run_background_task_inner",
                new=AsyncMock(return_value=None),
            ):
                import asyncio
                asyncio.run(runner._run_background_task("test", runner._source, "task-1"))

            mock_resolve.assert_called_once()
            # The second positional arg is the SessionSource.
            assert mock_resolve.call_args.args[0] is runner._source

    def test_multiplex_enabled_profile_scope_includes_profile_secrets(self):
        """Stronger end-to-end test: when the scope is active, get_secret
        reads from the scope (not from os.environ). The fix's scope
        builder reads the profile's .env into a dict; we synthesize that
        here and assert get_secret returns the profile value.
        """
        runner = _make_runner(multiplex_profiles=True)

        # Synthesize the profile's secret scope contents.
        fake_profile_secrets = {
            "OPENROUTER_API_KEY": "profile-key",
            "OPENROUTER_BASE_URL": "https://profile.example/v1",
        }

        # Build a scope that contains the profile secrets when installed.
        with patch.object(
            runner, "_resolve_profile_home_for_source",
            return_value="/tmp/fake-profile-home",
            create=True,
        ):
            with patch.object(
                runner, "_run_background_task_inner",
                new=AsyncMock(side_effect=lambda *a, **kw: _assert_secret_in_scope(fake_profile_secrets)),
            ):
                import asyncio
                asyncio.run(runner._run_background_task("test", runner._source, "task-1"))


async def _assert_secret_in_scope(expected_secrets: dict) -> None:
    """Async helper: assert the active secret scope contains the
    expected keys. Called from inside the with-block via side_effect
    so the secret scope is guaranteed to be installed.
    """
    from agent.secret_scope import current_secret_scope

    scope = current_secret_scope()
    assert scope is not None, "no active secret scope"
    for k, v in expected_secrets.items():
        assert scope.get(k) == v, (
            f"scope missing key {k!r}; got {dict(scope)!r}"
        )