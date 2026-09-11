"""One stdlib JSON-line PM request per isolated process."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import queue
import sys
import threading


def _members(value):
    if value is None:
        return None
    if "sources" in value:
        return {Path(identity): Path(source) for identity, source in value["sources"]}
    return [Path(path) for path in value["paths"]]


def _read_controls(messages, pause):
    # Raw reads avoid a daemon thread holding sys.stdin's buffered lock at exit.
    pending = b""
    request_id = None
    try:
        while block := os.read(0, 65536):
            pending += block
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                message = json.loads(line)
                if request_id is None:
                    request_id = message["id"]
                if message["id"] != request_id:
                    raise ValueError("unexpected PM control request id")
                if message.get("cancel"):
                    pause.set()
                if message.get("type") != "cancel":
                    messages.put(message)
    except (OSError, ValueError, KeyError) as exc:
        messages.put(exc)
    finally:
        pause.set()
        messages.put(None)


def main():
    import truststore

    truststore.inject_into_ssl()
    # Capture the protocol FD before redirecting even native/subprocess stdout.
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    messages = queue.Queue()
    pause = threading.Event()
    threading.Thread(target=_read_controls, args=(messages, pause), daemon=True).start()

    def receive():
        message = messages.get()
        if message is None:
            raise RuntimeError("PM client disconnected")
        if isinstance(message, Exception):
            raise message
        return message

    request = receive()
    from pm import paths, receipt
    from pm.package import InstallError
    from pm.registry import load_package_definitions
    context = request["context"]
    paths.repo_root = lambda: Path(context["repo"])
    paths.lockfile_path = lambda: Path(context["lockfile"])
    engine = importlib.import_module("pm.ensure")
    call = 0
    callback_lock = threading.Lock()

    def send(data):
        wire.write(json.dumps({"id": request["id"], **data}) + "\n")

    def callback(name, *args):
        with callback_lock:
            return exchange(name, args)

    def exchange(name, args):
        nonlocal call
        call += 1
        send({"type": "callback", "callback": name, "call": call, "args": args})
        reply = receive()
        if reply["id"] != request["id"] or reply["call"] != call:
            raise RuntimeError("unexpected PM callback response")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def sync_venv(**arguments):
        if "plugin_dirs" in request["callbacks"]:
            arguments["plugin_dirs"] = lambda: _members(callback("plugin_dirs"))
        else:
            arguments["plugin_dirs"] = _members(arguments["plugin_dirs"])
        if "before_publish" in request["callbacks"]:
            def publish():
                hooks = callback("before_publish")
                if not any(hooks.values()):
                    return None
                def undo():
                    if hooks["undo"]:
                        callback("undo")
                if hooks["finish"]:
                    undo.finish = lambda: callback("finish")
                return undo
            arguments["before_publish"] = publish
        return engine.sync_venv(**arguments)

    with receipt.worker_context(request.get("update_id")):
        try:
            load_package_definitions(request.get("packages", []))
            operations = {"ensure": engine.ensure, "sync_venv": sync_venv,
                          "stage_only": engine.stage_only, "uv": engine.uv}
            arguments = request["arguments"]
            if request["operation"] == "ensure":
                arguments["pause_event"] = pause
            for name in ("progress", "download_progress"):
                if name in request["callbacks"]:
                    arguments[name] = lambda *args, name=name: callback(name, *args)
            result = operations[request["operation"]](**arguments)
            if request["operation"] == "ensure":
                result = None  # Runner is reconstructed from the caller's base env.
            if isinstance(result, Path):
                result = str(result)
            response = {"result": result}
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, InstallError):
                error.update(package=exc.package, cause=exc.cause, remedy=exc.remedy)
            response = {"error": error}
        response["receipt"] = receipt.last_completed()
    send({"type": "result", **response})


if __name__ == "__main__":
    main()
