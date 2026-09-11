"""Synchronous PM mutations in an isolated interpreter, never the app's imports."""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid

from pm import paths
from pm.package import InstallError, Runner, StatePackage
from pm.runtime import is_runtime, runtime_command, runtime_environment


def _members(value):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {"sources": [[str(Path(key).absolute()), str(Path(source).absolute())]
                            for key, source in value.items()]}
    return {"paths": [str(Path(path).absolute()) for path in value]}


def _missing_or_refuse(name):
    from pm.ensure import _refuse_lazy, is_installed, lazy_installs_allowed
    from pm.registry import walk

    missing = [package.name for package in walk([name]) if not is_installed(package.name)]
    if missing and not lazy_installs_allowed():
        raise _refuse_lazy(name, ", ".join(missing))
    return missing


def _request(operation, arguments, *, callbacks=None, pause_event=None):
    from pm import receipt
    from pm.ensure import lazy_installs_allowed
    from pm.registry import get_package, package_definitions

    request_id = uuid.uuid4().hex
    update_id = receipt._ambient_update_id()
    callbacks = callbacks or {}
    names = ([arguments["name"]] if operation in ("ensure", "stage_only") else
             {"uv": ["uv"], "sync_venv": ["venv"]}.get(operation, []))
    message = {
        "id": request_id, "operation": operation, "arguments": arguments,
        "update_id": update_id,
        "callbacks": list(callbacks),
        "packages": package_definitions(names),
        "context": {"repo": str(paths.repo_root()), "lockfile": str(paths.lockfile_path())},
    }
    worker = Path(__file__).with_name("worker.py").resolve()
    environment = runtime_environment()
    state_sync = operation == "sync_venv" or (
        operation == "ensure" and isinstance(get_package(arguments["name"]), StatePackage))
    if (state_sync and not arguments.get("explicit") and not arguments.get("repair")
            and not lazy_installs_allowed()):
        # A ready PM still decides no-op/refusal under its install lock. A cold
        # PM is itself a missing prerequisite, not permission to bootstrap tools.
        try:
            command = runtime_command(worker, bootstrap=False)
        except InstallError as exc:
            token = receipt.begin("sync")
            try:
                receipt.record_refusal("lazy-install", str(exc))
                receipt.record_step("dependency-sync", False, f"{type(exc).__name__}: {exc}")
            finally:
                receipt.finalize("failed", 1, token=token)
            raise
        environment["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    else:
        command = runtime_command(worker)
    callback_error = None
    stopped = threading.Event()
    write_lock = threading.Lock()
    monitor = None
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", env=environment) as process:
        assert process.stdin is not None and process.stdout is not None
        writer = process.stdin

        def send(data):
            with write_lock:
                cancelled = pause_event is not None and pause_event.is_set()
                writer.write(json.dumps({"id": request_id, "cancel": cancelled, **data}) + "\n")
                writer.flush()

        def watch_pause():
            assert pause_event is not None
            while not stopped.wait(0.05):
                if pause_event.is_set():
                    try:
                        send({"type": "cancel"})
                    except (OSError, ValueError):
                        return  # The final response may already be on its way.
                    return

        try:
            send(message)
            if pause_event is not None:
                monitor = threading.Thread(target=watch_pause, daemon=True)
                monitor.start()
            while True:
                line = process.stdout.readline()
                if not line:
                    raise InstallError("pm", "worker exited without a result", "check the worker diagnostics on stderr")
                response = json.loads(line)
                if response["id"] != request_id:
                    raise InstallError("pm", "worker returned a different request id")
                if response["type"] == "result":
                    break
                name = response["callback"]
                try:
                    value = callbacks[name](*response.get("args", []))
                    if name not in ("plugin_dirs", "before_publish"):
                        value = None
                    send({"type": "callback_result", "call": response["call"], "result": value})
                except BaseException as exc:
                    if callback_error is None:
                        callback_error = exc
                    send({"type": "callback_result", "call": response["call"],
                          "error": f"{type(exc).__name__}: {exc}"})
            stopped.set()
            if monitor is not None:
                monitor.join()
            writer.close()
            if process.wait(timeout=5):
                raise InstallError("pm", "worker exited unsuccessfully")
            receipt.accept_worker_receipt(response.get("receipt"), update_id)
            if callback_error is not None:
                raise callback_error
            if "error" in response:
                error = response["error"]
                if "package" in error:
                    from pm.workspace import ResolutionConflict
                    kind = ResolutionConflict if error["type"] == "ResolutionConflict" else InstallError
                    raise kind(error["package"], error["cause"], error["remedy"])
                if error["type"] == "DownloadPaused":
                    from pm.downloader import DownloadPaused
                    raise DownloadPaused(error["message"])
                kind = {"ValueError": ValueError, "TypeError": TypeError,
                        "KeyError": KeyError, "OSError": OSError}.get(error["type"], RuntimeError)
                raise kind(error["message"])
            return response["result"]
        finally:
            stopped.set()
            if monitor is not None:
                monitor.join()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def ensure(name, *, base_env=None, explicit=False, progress=None, pause_event=None, download_progress=None) -> Runner:
    from pm.ensure import env_for
    from pm.registry import get_package

    if is_runtime():
        from pm.ensure import ensure as direct
        return direct(name, base_env=base_env, explicit=explicit, progress=progress,
                      pause_event=pause_event, download_progress=download_progress)
    if not explicit and not isinstance(get_package(name), StatePackage):
        if not _missing_or_refuse(name):
            return Runner(name, env_for(name, base_env=base_env))
    if pause_event is not None and pause_event.is_set():
        from pm.downloader import DownloadPaused
        raise DownloadPaused("install paused")
    callbacks = {}
    if progress is not None:
        callbacks["progress"] = progress
    if download_progress is not None:
        callbacks["download_progress"] = lambda done, total, ranges: download_progress(
            done, total, {key: [tuple(row) for row in rows] for key, rows in ranges.items()})
    _request("ensure", {"name": name, "explicit": explicit}, callbacks=callbacks, pause_event=pause_event)
    return Runner(name, env_for(name, base_env=base_env))


def sync_venv(extras=None, *, explicit=False, plugin_dirs=None, before_publish=None, repair=False) -> None:
    if is_runtime():
        from pm.ensure import sync_venv as direct
        return direct(extras, explicit=explicit, plugin_dirs=plugin_dirs,
                      before_publish=before_publish, repair=repair)
    callbacks = {}
    if callable(plugin_dirs):
        callbacks["plugin_dirs"] = lambda: _members(plugin_dirs())
        members = None
    else:
        members = _members(plugin_dirs)
    if before_publish is not None:
        def publish():
            publication = before_publish()
            if publication is not None:
                callbacks["undo"] = publication
            if hasattr(publication, "finish"):
                callbacks["finish"] = publication.finish
            return {"undo": publication is not None, "finish": hasattr(publication, "finish")}
        callbacks["before_publish"] = publish
    _request("sync_venv", {"extras": extras, "explicit": explicit, "repair": repair,
                          "plugin_dirs": members}, callbacks=callbacks)


def stage_only(name, target, *, progress=None) -> Path:
    if is_runtime():
        from pm.ensure import stage_only as direct
        return direct(name, target, progress=progress)
    callbacks = {"progress": progress} if progress is not None else {}
    return Path(_request("stage_only", {"name": name, "target": target}, callbacks=callbacks))


def uv(command="uv", *, venv=None, realize=True, explicit=False, base_env=None):
    if command not in ("uv", "uvx"):
        raise ValueError(f"unknown uv executable: {command}")
    if not realize or is_runtime():
        from pm.ensure import uv as direct
        return direct(command, venv=venv, realize=realize, explicit=explicit, base_env=base_env)
    if not explicit:
        _missing_or_refuse("uv")
    binary, environment = _request("uv", {
        "command": command, "venv": str(venv) if venv is not None else None,
        "realize": realize, "explicit": explicit,
        "base_env": dict(os.environ if base_env is None else base_env),
    })
    return binary, environment
