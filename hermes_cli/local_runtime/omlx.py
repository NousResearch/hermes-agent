"""Private, loopback-only oMLX bundle; never adopts a user's oMLX installation."""
from __future__ import annotations

import atexit
from concurrent.futures import CancelledError
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

import psutil

from hermes_cli.local_runtime.binaries import runtimes_root
from hermes_cli.local_runtime.processes import server_child_env
from hermes_cli.local_runtime.recovery import recorded_process, _owner_is_dead
from hermes_platform.host.facts import macos_version, native_arch, os_family
from utils import atomic_write_text

VERSION = "0.7.2"
# The self-contained release avoids oMLX's Python <=3.13 requirement conflicting with Hermes.
ARCHIVE = f"https://github.com/jundot/omlx/releases/download/v{VERSION}/oMLX-{VERSION}-macos15-sequoia.dmg"
SHA256 = "86e50d13ecbdf8531a5e39684619b0b0a609ec06e1c1fab866daa74e0e47b32e"
BASE_URL = "http://127.0.0.1:18081/v1"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Unexpected oMLX redirect")


def root() -> Path:
    return runtimes_root().parent / "omlx"


def models_dir() -> Path:
    from hermes_cli.local_runtime.bootstrap import models_dir as shared_models
    return shared_models() / "omlx"


def cli() -> Path:
    return root() / VERSION / "oMLX.app/Contents/MacOS/omlx-cli"


def supported() -> bool:
    return os_family() == "darwin" and native_arch() == "arm64" and macos_version() >= (15,)


def require_supported() -> None:
    if not supported():
        raise ValueError("oMLX requires macOS 15 or newer on Apple Silicon (arm64)")


@contextmanager
def operation_lock(name: str = "operation"):
    """Fail closed on contention: two servers must never race the ownership record."""
    import fcntl
    root().mkdir(parents=True, exist_ok=True)
    with (root() / f"{name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another oMLX operation is in progress; try again shortly") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_state() -> dict:
    try:
        value = json.loads((root() / "server.json").read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _process(state: dict):
    proc = recorded_process(state)
    if proc is None or state.get("base_url") != BASE_URL:
        return None
    try:
        # Upstream setproctitle rewrites argv; birth identity + private executable are authority.
        if not Path(proc.exe()).is_relative_to(root() / VERSION):
            return None
        return proc
    except (OSError, ValueError, IndexError, psutil.Error):
        return None


def request(path: str, payload: dict | None = None, *, timeout: int = 5):
    state = read_state()
    if _process(state) is None:
        raise RuntimeError("Managed oMLX is not running")
    headers = {"Authorization": "Bearer " + state["api_key"]}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    # Never send the local bearer token through a configured HTTP proxy or redirect.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        if response.url != req.full_url:
            raise RuntimeError("Unexpected oMLX redirect")
        return json.load(response)


def is_configured(config: dict) -> bool:
    model = (config or {}).get("model") or {}
    return (isinstance(model, dict) and model.get("provider") == "custom"
            and str(model.get("base_url", "")).rstrip("/") == BASE_URL)


def local_models() -> list[str]:
    return sorted(p.name for p in models_dir().glob("*") if not p.name.startswith(".")
                  and p.is_dir() and (p / "config.json").is_file() and any(p.glob("*.safetensors")))


def delete_model(model_id: str, config: dict) -> None:
    require_supported()
    with operation_lock("download"), operation_lock():
        path = models_dir() / model_id
        if (model_id not in local_models() or path.is_symlink()
                or path.resolve().parent != models_dir().resolve()):
            raise ValueError("Model is not a managed oMLX snapshot")
        if is_configured(config) and (config.get("model") or {}).get("default") == model_id:
            raise ValueError("Select a different model before deleting the active model")
        if _process(read_state()) is not None:
            raise RuntimeError("Turn off oMLX before deleting a model")
        shutil.rmtree(path)


def ensure_for_runtime(runtime: dict, config: dict) -> None:
    """CLI, gateway and background turns share the same explicit lifecycle preference."""
    if (is_configured(config) and (config.get("local_runtime") or {}).get("omlx_enabled")
            and str(runtime.get("base_url", "")).rstrip("/") == BASE_URL):
        start()


def status(config: dict) -> dict:
    error = None
    running = False
    if supported() and _process(read_state()) is not None:
        try:
            request("/models")
            running = True
        except (OSError, ValueError, RuntimeError) as exc:
            error = str(exc)
    model = config.get("model") or {}
    active = model.get("default") if is_configured(config) else None
    return {"supported": supported(), "installed": cli().is_file(), "running": running,
            "selected": (config.get("local_runtime") or {}).get("engine", "llamacpp") == "omlx",
            "model_dir": str(models_dir()), "base_url": BASE_URL,
            "version": VERSION if cli().is_file() else None,
            "models": [{"id": mid, "active": mid == active,
                        "size_bytes": sum(p.stat().st_size for p in (models_dir() / mid).glob("*.safetensors"))}
                       for mid in local_models()],
            "error": error if supported() else "oMLX requires macOS 15 or newer on Apple Silicon (arm64)"}


def install(progress=None) -> None:
    """PM downloader verifies the release; publication leaves any previous bundle intact."""
    from pm.downloader import Download, Source
    require_supported()
    with operation_lock():
        if cli().is_file():
            return
        archive = root() / f"omlx-{VERSION}.dmg"
        Download([Source(ARCHIVE, archive, sha256=SHA256)]).run(progress=progress)
        with tempfile.TemporaryDirectory(prefix=".install-", dir=root()) as stage:
            mount = Path(stage) / "mount"
            mount.mkdir()
            subprocess.run(["/usr/bin/hdiutil", "attach", str(archive), "-readonly", "-nobrowse",
                            "-mountpoint", str(mount)], check=True, capture_output=True, timeout=120)
            try:
                app = Path(stage) / "oMLX.app"
                subprocess.run(["/usr/bin/ditto", str(mount / "oMLX.app"), str(app)],
                               check=True, capture_output=True, timeout=300)
                subprocess.run([str(app / "Contents/MacOS/omlx-cli"), "--version"],
                               check=True, capture_output=True, timeout=60, env=server_child_env(os.environ))
            finally:
                subprocess.run(["/usr/bin/hdiutil", "detach", str(mount)],
                               check=True, capture_output=True, timeout=60)
            destination = root() / VERSION
            if destination.exists():
                raise RuntimeError(f"Incomplete oMLX install at {destination}; remove it before retrying")
            candidate = Path(stage) / VERSION
            candidate.mkdir()
            app.rename(candidate / "oMLX.app")
            candidate.rename(destination)
        archive.unlink(missing_ok=True)


def _stop_process(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except psutil.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def start() -> None:
    require_supported()
    if not cli().is_file():
        raise RuntimeError("Install the managed oMLX runtime first")
    with operation_lock():
        state = read_state()
        if _process(state) is not None:
            request("/models")
            return
        # Do not adopt or signal anything that happens to listen on our port.
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", 18081))
            except OSError as exc:
                raise RuntimeError("Port 18081 is occupied by an unowned server; stop it yourself or use another runtime") from exc
        models_dir().mkdir(parents=True, exist_ok=True)
        key_path = root() / "api-key"
        if key_path.exists():
            key = key_path.read_text().strip()
            if not key:
                raise RuntimeError("Managed oMLX API key is empty")
        else:
            key = secrets.token_urlsafe(32)
            atomic_write_text(key_path, key, mode=0o600)
        cmd = [str(cli()), "serve", "--host", "127.0.0.1", "--port", "18081",
               "--base-path", str(root() / "data"), "--model-dir", str(models_dir()), "--api-key", key]
        with (root() / "server.log").open("ab") as log:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, env=server_child_env(os.environ))
        try:
            owner = psutil.Process()
            # The shell wrapper execs Python; wait until the recorded executable is the actual child.
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"oMLX exited ({child.returncode}); see {root() / 'server.log'}")
                proc = psutil.Process(child.pid)  # exe() caches the shell path across exec.
                if Path(proc.exe()).is_relative_to(root() / VERSION):
                    state = {"pid": child.pid, "create_time": proc.create_time(), "executable": proc.exe(),
                             "owner_pid": owner.pid, "owner_create_time": owner.create_time(),
                             "api_key": key, "base_url": BASE_URL}
                    atomic_write_text(root() / "server.json", json.dumps(state), mode=0o600)
                    try:
                        request("/models")
                        atexit.register(shutdown_owned)
                        return
                    except (OSError, ValueError, RuntimeError):
                        pass
                time.sleep(0.25)
            raise RuntimeError(f"oMLX did not become ready; see {root() / 'server.log'}")
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            (root() / "server.json").unlink(missing_ok=True)
            raise


def stop() -> None:
    require_supported()
    with operation_lock():
        state = read_state()
        proc = _process(state)
        if proc is None:
            if state:
                raise RuntimeError("oMLX ownership could not be verified; refusing to stop it")
            return
        owner = psutil.Process()
        ours = state["owner_pid"] == owner.pid and state["owner_create_time"] == owner.create_time()
        if not ours and not _owner_is_dead(state):
            raise RuntimeError("Another live Hermes process owns oMLX; stop it from that process")
        _stop_process(proc)
        (root() / "server.json").unlink(missing_ok=True)


def shutdown_owned() -> None:
    state = read_state()
    owner = psutil.Process()
    if state.get("owner_pid") == owner.pid and state.get("owner_create_time") == owner.create_time():
        stop()


def validate_model(model_id: str) -> None:
    if model_id not in local_models():
        raise ValueError("Model is not downloaded in the managed oMLX model directory")
    start()
    if model_id not in {m.get("id") for m in request("/models").get("data", [])}:
        stop()
        start()
    response = request("/chat/completions", {"model": model_id, "messages": [
        {"role": "user", "content": "Reply with OK."}], "max_tokens": 8, "stream": False}, timeout=300)
    if not response.get("choices") or not any(c.get("message", {}).get("content")
                                               or c.get("message", {}).get("reasoning_content")
                                               for c in response["choices"]):
        raise RuntimeError("oMLX generation validation returned no text; model was not selected")


def _report_download_line(line: bytes, progress, transfer_progress) -> None:
    if line.startswith(b"HERMES_OMLX_PROGRESS "):
        done, total = json.loads(line.removeprefix(b"HERMES_OMLX_PROGRESS "))
        if type(done) is not int or type(total) is not int or not 0 <= done <= total:
            raise RuntimeError("Invalid oMLX download byte counts")
        if progress is not None:
            progress(done, total, {})
    elif line.startswith(b"HERMES_OMLX_TRANSFER "):
        received = json.loads(line.removeprefix(b"HERMES_OMLX_TRANSFER "))
        if type(received) is not int or received < 0:
            raise RuntimeError("Invalid oMLX transfer byte count")
        if transfer_progress is not None:
            transfer_progress(received)


def download(repo: str, *, progress=None, transfer_progress=None, cancel_event=None, before_publish=None) -> str:
    require_supported()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise ValueError("Use a Hugging Face owner/model repository ID")
    python = cli().with_name("omlx-cluster-python")
    if not python.is_file():
        raise RuntimeError("Install the managed oMLX runtime first")
    model_id = repo.replace("/", "--")
    # Hidden staging prevents discovery of partial snapshots; HF resumes a failed transfer.
    with operation_lock("download"):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Download cancelled")
        stage = models_dir() / ("." + model_id)
        destination = models_dir() / model_id
        if destination.exists():
            return model_id
        stage.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).with_name("omlx_download.py")
        args = [str(python), str(script), repo, str(stage)]
        with tempfile.TemporaryFile(dir=stage) as errors, subprocess.Popen(
            args, env=server_child_env(os.environ), stdout=subprocess.PIPE, stderr=errors,
        ) as child:
            assert child.stdout is not None
            deadline = time.monotonic() + 14400
            pending = b""
            try:
                with selectors.DefaultSelector() as reader:
                    reader.register(child.stdout, selectors.EVENT_READ)
                    while reader.get_map():
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError("Download cancelled")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(args, 14400)
                        for key, _ in reader.select(min(1, remaining)):
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                reader.unregister(key.fd)
                                continue
                            lines = (pending + chunk).split(b"\n")
                            pending = lines.pop()
                            if len(pending) > 65536:
                                raise RuntimeError("Invalid oMLX download progress output")
                            for line in lines:
                                _report_download_line(line, progress, transfer_progress)
                child.wait(timeout=max(0, deadline - time.monotonic()))
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("Download cancelled")
            if child.returncode:
                errors.seek(max(0, errors.seek(0, os.SEEK_END) - 2000))
                raise RuntimeError(f"Hugging Face download failed (exit {child.returncode}): "
                                   + errors.read().decode(errors="replace"))
        if not (stage / "config.json").is_file() or not any(stage.glob("*.safetensors")):
            raise ValueError("Repository is not a safetensors MLX model (config.json and weights required)")
        if before_publish is not None:
            before_publish()
        stage.rename(destination)
        # Upstream scans at startup. A running owned server is refreshed on next activation.
    return model_id
