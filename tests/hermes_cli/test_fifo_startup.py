"""Full startup regressions: no real credentials or installation mutations."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


# --help exits before dispatch; this checkout and HOME are disposable.
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("platform", [pytest.param("linux", marks=pytest.mark.linux_only), pytest.param("darwin", marks=pytest.mark.macos_only)])
@pytest.mark.parametrize("arguments", [["skills", "install", "official/creative/kanban-video-orchestrator", "--help"], ["update", "--help"]])
def test_writerless_fifo_cli_startup_is_bounded(tmp_path, platform, arguments):
    home = tmp_path / "home"
    profile = home / ".hermes"
    profile.mkdir(parents=True)
    fifo = profile / ".env"
    os.mkfifo(fifo, 0o600)
    before = fifo.stat()
    env = {"HOME": str(home), "HERMES_HOME": str(profile), "PATH": os.defpath,
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)}
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import faulthandler; faulthandler.dump_traceback_later(8, exit=True); from hermes_cli.main import main; main()", *arguments],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=12,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("CLI startup exceeded watchdog")
    assert "Timeout (0:00:08)" not in result.stderr, result.stderr
    assert result.returncode != 0
    assert "credential FIFO" in result.stderr
    assert "provider" in result.stderr
    assert time.monotonic() - started < 12
    after = fifo.stat()
    assert (before.st_ino, before.st_mode) == (after.st_ino, after.st_mode)


@pytest.mark.parametrize("platform", [pytest.param("linux", marks=pytest.mark.linux_only), pytest.param("darwin", marks=pytest.mark.macos_only)])
@pytest.mark.parametrize("location", ["user", "project", "op", "managed"])
def test_single_writer_supplies_complete_load(tmp_path, monkeypatch, platform, location):
    import threading
    from hermes_cli import env_loader, managed_scope

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-synthetic")
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "inherited-route")
    fifo = tmp_path / (".op.env" if location == "op" else ".env")
    project = None
    if location == "project":
        fifo = tmp_path / "project.env"
        project = fifo
    if location == "managed":
        directory = tmp_path / "managed"
        directory.mkdir()
        fifo = directory / ".env"
        monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: directory)
    else:
        monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    os.mkfifo(fifo, 0o600)
    before = fifo.stat()
    completed = threading.Event()

    def writer():
        with fifo.open("wb") as stream:
            stream.write(b"\xef\xbb\xbfOPENAI_API_KEY=writer-synthetic\nHERMES_ACP_AUTH_METHOD=synthetic-route\n")
        completed.set()

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    # Kill a blocked reread without leaving the test runner stuck.
    import signal
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("independent FIFO reread")))
    signal.alarm(10)
    try:
        env_loader.load_hermes_dotenv(hermes_home=tmp_path, project_env=project, load_external_secrets=False)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert completed.wait(2)
    thread.join(2)
    assert os.environ["OPENAI_API_KEY"] == ("inherited-synthetic" if location == "op" else "writer-synthetic")
    assert os.environ["HERMES_ACP_AUTH_METHOD"] == ("inherited-route" if location == "op" else "synthetic-route")
    after = fifo.stat()
    assert (before.st_ino, before.st_mode) == (after.st_ino, after.st_mode)


@pytest.mark.parametrize("platform", [pytest.param("linux", marks=pytest.mark.linux_only), pytest.param("darwin", marks=pytest.mark.macos_only)])
def test_unfinished_provider_does_not_apply_partial_credentials(tmp_path, monkeypatch, platform):
    import threading
    from hermes_cli.env_loader import _load_dotenv_with_fallback

    fifo = tmp_path / "synthetic.env"
    os.mkfifo(fifo, 0o600)
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-synthetic")
    release = threading.Event()

    def writer():
        with fifo.open("wb", buffering=0) as stream:
            stream.write(b"OPENAI_API_KEY=partial-synthetic\n")
            release.wait(10)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        with pytest.raises(TimeoutError, match="credential FIFO provider"):
            _load_dotenv_with_fallback(fifo, override=True)
        assert os.environ["OPENAI_API_KEY"] == "inherited-synthetic"
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
