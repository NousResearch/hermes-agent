"""Fail-closed corrupt-config guards on gateway, serve, and cron surfaces.

Companion to test_noninteractive_config_guard.py (PR #81988): issue #81952
extended to every non-interactive startup surface.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    yield
    os.environ.pop("HERMES_IGNORE_USER_CONFIG", None)


def _write_corrupt_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model: [unterminated\n", encoding="utf-8")
    return path


class TestGatewayGuard:
    def test_gateway_refuses_corrupt_config(self, tmp_path, capsys):
        from gateway.run import _guard_corrupt_user_config

        _write_corrupt_config(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            _guard_corrupt_user_config()

        assert exc_info.value.code == 2
        assert "Refusing non-interactive startup" in capsys.readouterr().err

    def test_gateway_allows_valid_config(self, tmp_path):
        from gateway.run import _guard_corrupt_user_config

        (tmp_path / "config.yaml").write_text("model:\n  default: local/test\n")
        _guard_corrupt_user_config()  # must not raise

    def test_gateway_allows_missing_config(self, tmp_path):
        from gateway.run import _guard_corrupt_user_config

        _guard_corrupt_user_config()  # first-run state: must not raise

    def test_gateway_escape_hatch(self, monkeypatch, tmp_path):
        from gateway.run import _guard_corrupt_user_config

        _write_corrupt_config(tmp_path)
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
        _guard_corrupt_user_config()  # must not raise


class TestCronRunJobGuard:
    def _job(self, **overrides):
        job = {"id": "job-test-1", "name": "guard test", "prompt": "hi"}
        job.update(overrides)
        return job

    def test_run_job_fails_closed_on_corrupt_config(self, tmp_path):
        from cron.scheduler import run_job

        _write_corrupt_config(tmp_path)

        success, output_doc, final_response, error = run_job(self._job())

        assert success is False
        assert error is not None
        assert "Refusing non-interactive startup" in error
        assert "config.yaml" in error
        assert final_response == ""

    def test_run_job_escape_hatch(self, monkeypatch, tmp_path):
        from cron.scheduler import run_job

        _write_corrupt_config(tmp_path)
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")

        # With the escape hatch active the guard must not trip. The job then
        # proceeds into normal execution; a missing provider/model in the
        # empty temp HERMES_HOME may fail later, but never with the guard's
        # refusal message.
        success, output_doc, final_response, error = run_job(
            self._job(no_agent=True, script="true", deliver="none")
        )
        assert "Refusing non-interactive startup" not in (error or "")

    def test_run_job_no_agent_exempt(self, tmp_path):
        from cron.scheduler import run_job

        _write_corrupt_config(tmp_path)

        success, output_doc, final_response, error = run_job(
            self._job(no_agent=True, script="true", deliver="none")
        )
        assert "Refusing non-interactive startup" not in (error or "")


class TestAcpGuard:
    """`hermes acp` / `hermes-acp`: driven by an IDE over stdio JSON-RPC, no
    human present to repair a corrupt config.yaml. Same policy as gateway/
    serve/cron (issue #81952)."""

    def test_acp_refuses_corrupt_config(self, tmp_path, capsys):
        from acp_adapter.entry import _guard_corrupt_user_config

        _write_corrupt_config(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            _guard_corrupt_user_config()

        assert exc_info.value.code == 2
        assert "Refusing non-interactive startup" in capsys.readouterr().err

    def test_acp_allows_valid_config(self, tmp_path):
        from acp_adapter.entry import _guard_corrupt_user_config

        (tmp_path / "config.yaml").write_text("model:\n  default: local/test\n")
        _guard_corrupt_user_config()  # must not raise

    def test_acp_allows_missing_config(self, tmp_path):
        from acp_adapter.entry import _guard_corrupt_user_config

        _guard_corrupt_user_config()  # first-run state: must not raise

    def test_acp_escape_hatch(self, monkeypatch, tmp_path):
        from acp_adapter.entry import _guard_corrupt_user_config

        _write_corrupt_config(tmp_path)
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
        _guard_corrupt_user_config()  # must not raise

    def test_acp_main_exits_before_agent_server_starts(self, tmp_path, monkeypatch, capsys):
        """`main()` must fail closed before touching HermesACPAgent/asyncio."""
        from acp_adapter import entry as entry_mod

        _write_corrupt_config(tmp_path)
        monkeypatch.setattr(entry_mod, "_setup_logging", lambda: None)
        monkeypatch.setattr(entry_mod, "_load_env", lambda: None)

        sentinel = RuntimeError("reached agent construction")

        def _fail_if_reached(*args, **kwargs):
            raise sentinel

        monkeypatch.setattr(entry_mod.asyncio, "run", _fail_if_reached)

        with pytest.raises(SystemExit) as exc_info:
            entry_mod.main([])

        assert exc_info.value.code == 2
        assert "Refusing non-interactive startup" in capsys.readouterr().err


class TestServeGuard:
    def test_serve_headless_refuses_corrupt_config(self, tmp_path, capsys):
        """The `hermes serve` headless path fails closed before startup."""
        from argparse import Namespace

        from hermes_cli import main as main_mod

        _write_corrupt_config(tmp_path)
        args = Namespace(
            headless_backend=True,
            ignore_user_config=False,
            ssh_session_token_file=None,
            ssh_owner_nonce=None,
            status=False,
            stop=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            main_mod.cmd_dashboard(args)

        assert exc_info.value.code == 2
        assert "Refusing non-interactive startup" in capsys.readouterr().err

    def test_serve_escape_hatch_passes_guard(self, tmp_path, monkeypatch):
        """--ignore-user-config lets serve get past the corrupt-config guard."""
        from argparse import Namespace

        from hermes_cli import main as main_mod

        _write_corrupt_config(tmp_path)
        args = Namespace(
            headless_backend=True,
            ignore_user_config=True,
            ssh_session_token_file=None,
            ssh_owner_nonce=None,
            status=False,
            stop=False,
        )

        # Stop execution right after the guard: the next thing cmd_dashboard
        # touches on the headless path is the nonce regex via `re`.
        sentinel = RuntimeError("passed-guard")

        class _ReStop:
            def fullmatch(self, *a, **k):
                raise sentinel

        monkeypatch.setattr(main_mod, "re", _ReStop())
        args.ssh_owner_nonce = "0123456789abcdef"

        with pytest.raises(RuntimeError, match="passed-guard"):
            main_mod.cmd_dashboard(args)
