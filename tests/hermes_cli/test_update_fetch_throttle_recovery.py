"""Repo-scoped HTTP 429 recovery for `hermes update` (#105857).

GitHub's repo-scoped 429 throttles git's pack negotiation — the fetch dies
mid-transfer with "RPC failed; HTTP 429 / expected 'packfile'" — while small
requests (ls-remote) still succeed, and the throttle can outlast the "try
again in 5 minutes" cadence the message suggests. #90696 classified the error
but the updater still exited after one attempt, leaving a healthy install
permanently behind upstream. The fetch now retries with the installer's
bounded backoff (#99480 parity) and, when the throttle persists, falls back
to the guarded ZIP path instead of exiting.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import update_cmd


RATE_LIMIT_STDERR = (
    "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429\n"
    "fatal: expected flush after ref listing"
)
DNS_STDERR = (
    "fatal: unable to access 'https://github.com/NousResearch/hermes-agent.git/':"
    " Could not resolve host: github.com"
)


def _fetch_result(returncode=0, stderr=""):
    return SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")


class TestIsRateLimited:
    def test_rpc_429_signature(self):
        assert update_cmd._is_rate_limited(RATE_LIMIT_STDERR) is True

    def test_rate_limit_phrase_without_code(self):
        assert update_cmd._is_rate_limited("fatal: GitHub rate limit exceeded") is True

    def test_dns_failure_is_not_rate_limit(self):
        assert update_cmd._is_rate_limited(DNS_STDERR) is False

    def test_empty_and_none_are_safe(self):
        assert update_cmd._is_rate_limited("") is False
        assert update_cmd._is_rate_limited(None) is False


class TestFetchWithThrottleRecovery:
    def _run(self, results):
        calls = []

        def fake_git_run(git_cmd, args, **kwargs):
            calls.append(list(args))
            return results[len(calls) - 1]

        with (
            patch.object(update_cmd, "_git_run", side_effect=fake_git_run),
            patch.object(update_cmd._time, "sleep") as sleeps,
        ):
            result = update_cmd._fetch_with_throttle_recovery(["git"], "main")
        return result, calls, sleeps

    def test_success_first_try_does_not_retry(self):
        result, calls, sleeps = self._run([_fetch_result(0)])
        assert result.returncode == 0
        assert calls == [["fetch", "origin", "main"]]
        sleeps.assert_not_called()

    def test_429_then_success_retries_with_backoff(self):
        result, calls, sleeps = self._run([
            _fetch_result(1, RATE_LIMIT_STDERR),
            _fetch_result(1, RATE_LIMIT_STDERR),
            _fetch_result(0),
        ])
        assert result.returncode == 0
        assert len(calls) == 3
        # Backoff matches the installer (#99480): 5s before attempt 2, 10s before attempt 3.
        assert [c.args[0] for c in sleeps.call_args_list] == [5.0, 10.0]

    def test_persistent_429_exhausts_attempts_and_keeps_last_result(self):
        result, calls, sleeps = self._run([_fetch_result(1, RATE_LIMIT_STDERR)] * 4)
        assert result.returncode == 1
        assert result.stderr == RATE_LIMIT_STDERR
        assert len(calls) == 4
        assert [c.args[0] for c in sleeps.call_args_list] == [5.0, 10.0, 15.0]

    def test_non_rate_failure_returns_without_retry(self):
        result, calls, sleeps = self._run([_fetch_result(1, DNS_STDERR)])
        assert result.returncode == 1
        assert calls == [["fetch", "origin", "main"]]
        sleeps.assert_not_called()

    def test_429_flipping_to_other_error_stops_retrying(self):
        result, calls, sleeps = self._run([
            _fetch_result(1, RATE_LIMIT_STDERR),
            _fetch_result(1, DNS_STDERR),
        ])
        assert result.returncode == 1
        assert result.stderr == DNS_STDERR
        assert len(calls) == 2
        assert [c.args[0] for c in sleeps.call_args_list] == [5.0]


class TestZipFallbackAfterThrottledFetch:
    def _run_fallback(self, *, gateway_mode, zip_return=True, zip_side_effect=None):
        resumes, exit_codes = [], []
        with (
            patch.object(
                update_cmd,
                "_update_via_zip",
                return_value=zip_return,
                side_effect=zip_side_effect,
            ) as zip_call,
            patch.object(
                update_cmd._m(),
                "_resume_windows_gateways_after_update",
                side_effect=lambda token: resumes.append(token),
            ),
            patch.object(
                update_cmd,
                "_write_gateway_update_exit_code",
                side_effect=lambda ok: exit_codes.append(ok),
            ),
        ):
            update_cmd._fallback_to_zip_after_throttled_fetch(
                "args-sentinel",
                had_desktop_app_before_update=False,
                gateway_mode=gateway_mode,
                _windows_gateway_resume="resume-token",
            )
        return zip_call, resumes, exit_codes

    def test_gateway_mode_calls_zip_resumes_gateways_writes_exit_code(self):
        zip_call, resumes, exit_codes = self._run_fallback(gateway_mode=True)
        zip_call.assert_called_once_with(
            "args-sentinel", had_desktop_app_before_update=False
        )
        assert resumes == ["resume-token"]
        assert exit_codes == [True]

    def test_interactive_mode_skips_gateway_exit_code(self):
        _, resumes, exit_codes = self._run_fallback(gateway_mode=False)
        assert resumes == ["resume-token"]
        assert exit_codes == []

    def test_paused_gateways_resume_even_when_zip_refuses(self, capsys):
        # _update_via_zip exits (dirty checkout / non-main branch) — the paused
        # Windows gateways must still be resumed (finally contract).
        def _refuse(_args, **_kwargs):
            raise SystemExit(1)

        resumes = []
        with (
            patch.object(update_cmd, "_update_via_zip", side_effect=_refuse),
            patch.object(
                update_cmd._m(),
                "_resume_windows_gateways_after_update",
                side_effect=lambda token: resumes.append(token),
            ),
            pytest.raises(SystemExit),
        ):
            update_cmd._fallback_to_zip_after_throttled_fetch(
                "args",
                had_desktop_app_before_update=False,
                gateway_mode=True,
                _windows_gateway_resume="token",
            )
        assert resumes == ["token"]
        assert "falling back to the guarded ZIP update" in capsys.readouterr().out


class TestCmdUpdateWiring:
    def test_fetch_failure_branch_routes_429_to_zip_and_exits_otherwise(self):
        import inspect

        src = inspect.getsource(update_cmd._cmd_update_impl)
        fetch_idx = src.index("→ Fetching updates...")
        assert "_fetch_with_throttle_recovery(" in src[fetch_idx:]
        tail = src[src.index("_print_fetch_failure", fetch_idx) :]
        assert "_fallback_to_zip_after_throttled_fetch(" in tail
        assert "sys.exit(1)" in tail

    def test_fallback_only_fires_for_rate_limit(self):
        import inspect

        src = inspect.getsource(update_cmd._cmd_update_impl)
        tail = src[
            src.index("_print_fetch_failure", src.index("→ Fetching updates...")) :
        ]
        guarded = tail[: tail.index("sys.exit(1)")]
        assert "_is_rate_limited(" in guarded
