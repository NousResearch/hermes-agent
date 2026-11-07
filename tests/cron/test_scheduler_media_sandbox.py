"""Cron media delivery resolves the job's own Docker sandbox, not the shared default one (#64889)."""

from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

from cron.scheduler_delivery import _send_media_via_adapter


def _run_with_loop(adapter, chat_id, media_files, metadata, job):
    """Run _send_media_via_adapter with immediate scheduling (the send coroutine is only scheduled)."""
    def fake_run_coro(coro, _loop):
        coro.close()
        completed = Future()
        completed.set_result(MagicMock(success=True))
        return completed

    with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        _send_media_via_adapter(adapter, chat_id, media_files, metadata, MagicMock(), job)


def test_job_id_resolves_own_isolated_docker_sandbox_not_default(monkeypatch):
    from tools.environments.base import get_sandbox_dir
    from tools.environments.path_utils import sanitize_task_id_for_path

    # terminal_tool bridges config into env once per process; let it re-evaluate under this env.
    monkeypatch.setattr("tools.terminal_tool._terminal_config_bridge_attempted", False)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    job_id = "isolated-job-42"
    workspace = get_sandbox_dir() / "docker" / sanitize_task_id_for_path(f"session:{job_id}") / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    produced = workspace / "chart.png"
    produced.write_bytes(b"png")
    # The shared default sandbox has no such file: without the job id, resolution would fail there.
    default_ws = get_sandbox_dir() / "docker" / "default" / "workspace"
    default_ws.mkdir(parents=True, exist_ok=True)
    assert not (default_ws / "chart.png").exists()

    adapter = MagicMock()
    adapter.send_image_file = AsyncMock()
    _run_with_loop(adapter, "123", [("/workspace/chart.png", False)], None, {"id": job_id})

    adapter.send_image_file.assert_called_once()
    # The send coroutine is closed unawaited, so the path is in call_args, not await_args.
    call_kwargs = adapter.send_image_file.call_args.kwargs
    assert str(produced.resolve()) in str(call_kwargs.get("image_path", call_kwargs))
