"""Synchronous PM mutations in an isolated interpreter, never the app's imports."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from pm import paths, plugin_inputs
from pm.package import InstallError, Runner, StatePackage
from pm.plugin_inputs import Candidates, Members, PluginInput, Selection
from pm.runtime import is_runtime, runtime_command, runtime_environment
from pm.worker_operations import OPERATIONS


def _missing_or_refuse(name):
    from pm.install import _refuse_lazy, is_installed, lazy_installs_allowed
    from pm.registry import walk

    missing = [package.name for package in walk([name]) if not is_installed(package.name)]
    if missing and not lazy_installs_allowed():
        raise _refuse_lazy(name, ", ".join(missing))
    return missing


def _refuse_cold_runtime(cold: InstallError, arguments) -> InstallError:
    """Record a lazy sync refused because PM's own runtime is missing; return the error to raise."""
    from pm import receipt

    exc = cold
    if arguments.get("extras"):
        # The user asked for an extra, not for PM's own runtime: name
        # the command that provisions both.
        from pm.extras import install_hint

        exc = InstallError(cold.package, f"{cold.cause} while enabling {list(arguments['extras'])}",
                           "run `" + "`, `".join(install_hint(extra) for extra in arguments["extras"]) + "`")
    token = receipt.begin("sync")
    try:
        receipt.record_refusal("lazy-install", str(exc))
        receipt.record_step("dependency-sync", False, f"{type(exc).__name__}: {exc}")
    finally:
        receipt.finalize("failed", 1, token=token)
    return exc


def _worker_command(spec, arguments, worker: Path, environment: dict) -> list[str]:
    """How to start the worker; may disable lazy installs in *environment*."""
    from pm.install import lazy_installs_allowed
    from pm.registry import get_package

    # Bootstrap precedes dispatch and must share the operation's selected cache.
    cache = Path(arguments["cache"]) if arguments.get("cache") is not None else None
    state_sync = spec.bootstrap == "policy" or (
        spec.bootstrap == "state" and isinstance(get_package(arguments["name"]), StatePackage))
    if (state_sync and not arguments.get("explicit") and not arguments.get("repair")
            and not lazy_installs_allowed()):
        # A ready PM still decides no-op/refusal under its install lock. A cold
        # PM is itself a missing prerequisite, not permission to bootstrap tools.
        try:
            command = runtime_command(worker, bootstrap=False, cache=cache)
        except InstallError as cold:
            raise _refuse_cold_runtime(cold, arguments) from None
        environment["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
        return command
    if spec.bootstrap == "never":
        return runtime_command(worker, bootstrap=False, cache=cache)
    return runtime_command(worker, cache=cache)


_WORKER_ERRORS = {"ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
                  "OSError": OSError, "FileExistsError": FileExistsError,
                  "FileNotFoundError": FileNotFoundError, "PermissionError": PermissionError}


def _raise_worker_error(error: dict):
    """Re-raise a worker failure as the caller-side exception it names."""
    if "package" in error:
        from pm.workspace import ResolutionConflict
        kind = ResolutionConflict if error["type"] == "ResolutionConflict" else InstallError
        raise kind(error["package"], error["cause"], error["remedy"])
    if error["type"] == "DownloadPaused":
        from pm.downloader import DownloadPaused
        raise DownloadPaused(error["message"])
    raise _WORKER_ERRORS.get(error["type"], RuntimeError)(error["message"])


def _request(operation, arguments, *, callbacks=None, pause_event=None, project_root=None):
    from pm import receipt
    from pm.registry import package_definitions

    request_id = uuid.uuid4().hex
    update_id = receipt._ambient_update_id()
    callbacks = callbacks or {}
    spec = OPERATIONS[operation]
    names = list(spec.packages) if spec.packages is not None else (
        arguments["names"] if "names" in arguments else [arguments["name"]])
    message = {
        "id": request_id, "operation": operation, "arguments": arguments,
        "update_id": update_id,
        "callbacks": list(callbacks),
        "packages": package_definitions(names),
        "context": {"repo": str(Path(project_root).absolute() if project_root is not None else paths.repo_root()),
                    "lockfile": str(paths.lockfile_path())},
    }
    worker = Path(__file__).with_name("worker.py").resolve()
    environment = runtime_environment()
    command = _worker_command(spec, arguments, worker, environment)
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
                    callbacks[name](*response.get("args", []))
                    send({"type": "callback_result", "call": response["call"], "result": None})
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
                _raise_worker_error(response["error"])
            return response["result"]
        finally:
            stopped.set()
            if monitor is not None:
                monitor.join()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def ensure(name, *, base_env=None, explicit=False, progress=None, pause_event=None, download_progress=None) -> Runner:
    from pm.install import env_for
    from pm.registry import get_package

    if is_runtime():
        from pm.install import ensure as direct
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


def sync_venv(extras=None, *, explicit=False, plugins: PluginInput | None = None, repair=False,
              project_root: Path | None = None, evict_incompatible_plugins: bool = False) -> None:
    from pm.environments import running_from_selected_environment

    if extras and not explicit and not repair and not running_from_selected_environment(
            paths.repo_root() if project_root is None else Path(project_root)):
        # A lazy extra may only extend the environment this process runs from. From any other
        # interpreter (a build_environment test venv, a developer venv, a Nix Python) the sync would
        # commit a selection this process never activates while every process booted afterwards
        # swaps onto it — a generation without whatever the foreign interpreter carried.
        from pm.install import _refuse_lazy
        raise _refuse_lazy(
            "venv",
            f"{list(extras)}: this process is not running from the install's dependency environment "
            f"({sys.prefix}); only an explicit install may change what later processes boot into",
        )
    if isinstance(plugins, Selection) and "expected_config" not in plugins.data:
        from pm.filesystem import file_digest
        plugins = Selection({**plugins.data,
                             "expected_config": file_digest(Path(plugins.data["home"]) / "config.yaml") or "missing"})
    foreign = project_root is not None and Path(project_root).resolve() != paths.repo_root().resolve()
    if is_runtime() and not foreign:
        from pm.install import sync_venv as direct
        return direct(extras, explicit=explicit, plugins=plugins, repair=repair,
                      evict_incompatible_plugins=evict_incompatible_plugins)
    _request("sync_venv", {"extras": extras, "explicit": explicit, "repair": repair,
                          "plugins": plugin_inputs.encode(plugins),
                          "evict_incompatible_plugins": evict_incompatible_plugins}, project_root=project_root)


def ensure_tools_for_sync() -> None:
    """Publish every required tool in this tree's lockfile, then put them on PATH.

    Updates call this before the venv sync: the sync only pulls uv/python in
    through its own dependency, so a bumped ripgrep/ffmpeg/node pin was never
    installed and activation skipped the managed tool dirs on every start.
    Publishing tools first also lets native builds resolve compilers and git
    from the pinned store instead of the host (as `hermes pm install` does).
    An update is an explicit user action, so the lazy-install policy does not
    gate it; a failed download fails the update.
    """
    from pm.install import activate
    from pm.lock import Lockfile
    from pm.registry import tool_roots

    for name in tool_roots(Lockfile(paths.lockfile_path()).names()):
        ensure(name, explicit=True)
    problems = activate(allow_incomplete=True)
    if problems:
        raise RuntimeError(f"tools not on PATH before venv sync: {'; '.join(problems)}")


def stage_only(name, target, *, progress=None) -> Path:
    if is_runtime():
        from pm.install import stage_only as direct
        return direct(name, target, progress=progress)
    callbacks = {"progress": progress} if progress is not None else {}
    return Path(_request("stage_only", {"name": name, "target": target}, callbacks=callbacks))


def _python_operation(operation: str, arguments: dict):
    if is_runtime():
        return OPERATIONS[operation].resolve(operation)(**arguments)
    payload = {key: str(value.absolute()) if isinstance(value, Path) else value
               for key, value in arguments.items()}
    return _request(operation, payload)


def build_environment(
    *, source: Path, out: Path, python: Path | None = None,
    cache: Path | None = None, env: Mapping[str, str] | None = None,
    extras: Sequence[str] = (), groups: Sequence[str] = (),
    all_extras: bool = False, no_install_project: bool = False,
    frozen: bool = True, sealed: bool = False, offline: bool = False,
    explicit: bool = False, timeout: int = 1800,
) -> Path:
    """Build a validated Python environment without exposing install machinery."""
    return Path(_python_operation("build_environment", {
        "source": Path(source), "out": Path(out), "python": python, "cache": cache,
        "env": dict(env) if env is not None else None, "extras": list(extras), "groups": list(groups),
        "all_extras": all_extras, "no_install_project": no_install_project,
        "frozen": frozen, "sealed": sealed, "offline": offline,
        "explicit": explicit, "timeout": timeout,
    }))


def lock_project(
    source: Path, *, upgrade: bool = False, python: Path | None = None,
    env: Mapping[str, str] | None = None, cache: Path | None = None,
    offline: bool = False, explicit: bool = False,
) -> None:
    """Resolve a project's lock without creating or selecting an environment."""
    _python_operation("lock_project", {
        "source": Path(source), "upgrade": upgrade, "python": python,
        "env": dict(env) if env is not None else None, "cache": cache,
        "offline": offline, "explicit": explicit,
    })


def stage_manager_runtime(
    *, python: Path, destination: Path, project: Path | None = None,
    offline: bool = False, wheelhouse: Path | None = None, cache: Path | None = None,
) -> Path:
    # PM cannot dispatch the construction of its own offline runtime through
    # that runtime. The stdlib-only bootstrap shares the private build engine.
    from pm.operations import stage_manager_runtime as stage
    return stage(python=Path(python), destination=Path(destination), project=project,
                 offline=offline, wheelhouse=wheelhouse, cache=cache)


def ensure_environment(
    name: str, requirements: Sequence[str], *, root: Path | None = None,
    explicit: bool = False, timeout: int = 1800,
) -> Path:
    """Select a complete isolated dependency generation, retaining the previous one."""
    if isinstance(requirements, str):
        raise TypeError("requirements must be a sequence, not a string")
    return Path(_python_operation("ensure_environment", {
        "name": name, "requirements": list(requirements), "root": root,
        "explicit": explicit, "timeout": timeout,
    }))


def ensure_project_environment(
    name: str, project: Path, *, extras: Sequence[str] = (), groups: Sequence[str] = (),
    root: Path | None = None, explicit: bool = False, timeout: int = 1800,
) -> Path:
    """Select an isolated environment of the project's locked dependencies."""
    return Path(_python_operation("ensure_project_environment", {
        "name": name, "project": Path(project), "extras": list(extras), "groups": list(groups),
        "root": root, "explicit": explicit, "timeout": timeout,
    }))


def ensure_python_tool(
    name: str, requirements: Sequence[str], executable: str, *, root: Path | None = None,
    explicit: bool = False, timeout: int = 1800,
) -> Path:
    """Install an isolated tool and return its validated executable, not uv/uvx."""
    if isinstance(requirements, str):
        raise TypeError("requirements must be a sequence, not a string")
    return Path(_python_operation("ensure_python_tool", {
        "name": name, "requirements": list(requirements), "executable": executable,
        "root": root, "explicit": explicit, "timeout": timeout,
    }))


def venv_is_current(*, extras: list[str] | None = None, plugins: Members | Candidates | None = None,
                    project_root: Path | None = None) -> bool:
    """Check through a ready PM, never bootstrap dependencies for a probe."""
    if is_runtime() and (project_root is None or Path(project_root).resolve() == paths.repo_root().resolve()):
        from pm.install import venv_is_current as direct
        return direct(extras=extras, plugins=plugins, project_root=project_root)
    try:
        return bool(_request("venv_is_current", {"extras": extras, "plugins": plugin_inputs.encode(plugins)},
                             project_root=project_root))
    except InstallError as exc:
        if exc.package == "pm-runtime":
            return False  # Without its checker, currency cannot be established.
        raise


def check_project_lock(source: Path, *, python: Path | None = None, cache: Path | None = None,
                       env: Mapping[str, str] | None = None, offline: bool = False,
                       explicit: bool = False, quiet: bool = False) -> None:
    _python_operation("check_project_lock", {
        "source": Path(source), "python": python, "cache": cache,
        "env": dict(env) if env is not None else None, "offline": offline, "explicit": explicit,
        "quiet": quiet,
    })


def export_requirements(source: Path, out: Path, *, extras: Sequence[str] = (),
                        python: Path | None = None, cache: Path | None = None,
                        env: Mapping[str, str] | None = None, explicit: bool = False) -> None:
    _python_operation("export_requirements", {
        "source": Path(source), "out": Path(out), "extras": list(extras),
        "python": python, "cache": cache, "env": dict(env) if env is not None else None,
        "explicit": explicit,
    })


def build_requirements_environment(requirements: Sequence[str], *, out: Path,
                                   python: Path | None = None, cache: Path | None = None,
                                   env: Mapping[str, str] | None = None, wheelhouse: Path | None = None,
                                   offline: bool = False, sealed: bool = False,
                                   explicit: bool = False) -> Path:
    if isinstance(requirements, str):
        raise TypeError("requirements must be a sequence, not a string")
    return Path(_python_operation("build_requirements_environment", {
        "requirements": list(requirements), "out": Path(out), "python": python, "cache": cache,
        "env": dict(env) if env is not None else None, "wheelhouse": wheelhouse,
        "offline": offline, "sealed": sealed, "explicit": explicit,
    }))


def prepare_tools(names: Sequence[str], *, out: Path, target: str,
                  cache: Path | None = None) -> Path:
    """Realize build tools without selecting application or profile state."""
    if isinstance(names, str):
        raise TypeError("names must be a sequence, not a string")
    return Path(_python_operation("prepare_tools", {
        "names": list(names), "out": Path(out), "target": target, "cache": cache,
    }))


def stage_tools(names: Sequence[str], *, source_store: Path, out: Path, target: str) -> Path:
    """Independently copy verified pins through a ready PM; never bootstrap."""
    if isinstance(names, str):
        raise TypeError("names must be a sequence, not a string")
    return Path(_python_operation("stage_tools", {
        "names": list(names), "source_store": Path(source_store), "out": Path(out), "target": target,
    }))


def prune_cache(cache: Path, *, ci: bool = False) -> None:
    _python_operation("prune_cache", {"cache": Path(cache), "ci": ci})
