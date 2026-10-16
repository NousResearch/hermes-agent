"""Webex dependency installs retain process ownership across gateway retries."""

import asyncio
import os
import sys

import psutil
import pytest

from plugins.platforms.webex import adapter as webex
from plugins.platforms.webex import listener_runtime
from tests.waiting import async_wait_until


def _use_local_installer(tmp_path, monkeypatch, code):
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "installer.py"
    script.write_text(code, encoding="utf-8")
    create_subprocess = asyncio.create_subprocess_exec

    async def start_installer(npm, *args, **kwargs):
        assert npm == "local-npm"
        assert args == ("ci", "--omit=optional", "--ignore-scripts")
        return await create_subprocess(sys.executable, str(script), **kwargs)

    monkeypatch.setattr(
        listener_runtime, "find_node_executable", lambda _name: "local-npm"
    )
    monkeypatch.setattr(
        listener_runtime, "with_hermes_node_path", lambda: dict(os.environ)
    )
    monkeypatch.setattr(
        listener_runtime.asyncio, "create_subprocess_exec", start_installer
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "timeout"])
async def test_stopped_install_is_reaped_before_retry(tmp_path, monkeypatch, stop):
    if stop == "timeout":
        monkeypatch.setattr(listener_runtime, "_INSTALL_TIMEOUT_SECONDS", 2)
    _use_local_installer(
        tmp_path,
        monkeypatch,
        """
import os
from pathlib import Path
import threading
import json
import psutil

started = Path("started")
if not started.exists():
    for name, version in DEPENDENCY_VERSIONS.items():
        manifest = Path("node_modules") / name / "package.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"version": version}))
    started.write_text(str(os.getpid()))
    threading.Event().wait()
else:
    if psutil.pid_exists(int(started.read_text())):
        raise SystemExit(2)
    Path("retry-completed").touch()
""".replace("DEPENDENCY_VERSIONS", repr(webex._LISTENER_DEPENDENCIES)),
    )
    monkeypatch.setattr(webex, "_PLUGIN_DIR", tmp_path / "bundled")
    monkeypatch.setattr(webex, "_listener_runtime_dir", lambda: tmp_path)
    install = asyncio.create_task(
        webex.ensure_websocket_listener_requirements()
    )
    retry = None
    pid = None
    try:
        started = tmp_path / "started"
        pid = int(
            await async_wait_until(
                lambda: started.read_text() if started.exists() else ""
            )
        )
        assert webex._listener_dependencies_available(tmp_path)
        assert not listener_runtime.listener_installation_complete(tmp_path)
        if stop == "cancel":
            install.cancel()
        retry = asyncio.create_task(
            webex.ensure_websocket_listener_requirements()
        )
        if stop == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(install, timeout=10)
        else:
            assert await asyncio.wait_for(install, timeout=10) is None
        assert await asyncio.wait_for(retry, timeout=10) == tmp_path
        assert not psutil.pid_exists(pid)
        assert (tmp_path / "retry-completed").exists()
        assert listener_runtime.listener_installation_complete(tmp_path)
    finally:
        install.cancel()
        if retry is not None:
            retry.cancel()
        if pid is not None and psutil.pid_exists(pid):
            process = psutil.Process(pid)
            process.kill()
            process.wait(timeout=10)
        await asyncio.gather(
            install,
            *([retry] if retry is not None else []),
            return_exceptions=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("returncode", [0, 3])
async def test_installer_exit_status_is_reported(tmp_path, monkeypatch, returncode):
    _use_local_installer(tmp_path, monkeypatch, f"raise SystemExit({returncode})")

    assert await listener_runtime.install_listener_dependencies(tmp_path) is (
        returncode == 0
    )
    assert listener_runtime.listener_installation_complete(tmp_path) is (
        returncode == 0
    )


@pytest.mark.asyncio
async def test_lockfile_change_and_failed_reinstall_invalidate_completion(
    tmp_path, monkeypatch
):
    _use_local_installer(tmp_path, monkeypatch, "raise SystemExit(0)")
    assert await listener_runtime.install_listener_dependencies(tmp_path)
    assert listener_runtime.listener_installation_complete(tmp_path)

    await asyncio.to_thread(
        (tmp_path / "package-lock.json").write_text,
        '{"updated":true}',
        encoding="utf-8",
    )
    assert not listener_runtime.listener_installation_complete(tmp_path)

    assert await listener_runtime.install_listener_dependencies(tmp_path)
    assert listener_runtime.listener_installation_complete(tmp_path)
    await asyncio.to_thread(
        (tmp_path / "installer.py").write_text,
        "raise SystemExit(3)",
        encoding="utf-8",
    )
    assert not await listener_runtime.install_listener_dependencies(tmp_path)
    assert not listener_runtime.listener_installation_complete(tmp_path)
