"""Cron pre-run script execution: timeouts, Windows venv bootstrap, process-tree termination,
and the claim-heartbeat thread that keeps a long script's run claim alive.

Split out of ``cron.scheduler``. Import names from this module directly (``cron.scheduler`` only
imports the few it calls itself). Origin-resident helpers and sibling split modules are reached
late-bound (``_sched`` / module refs at the bottom) so monkeypatching the defining module works.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from cron.jobs import _ensure_cron_dir
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from hermes_cli._subprocess_compat import windows_hide_flags

if TYPE_CHECKING:
    from cron.scheduler import _CancelEventLike

# Log-record parity with the origin module.
logger = logging.getLogger("cron.scheduler")


def _positive_int(raw) -> Optional[int]:
    """``int(float(raw))`` when > 0, else None; raises on unparsable input."""
    timeout = int(float(raw))
    return timeout if timeout > 0 else None


def _timeout_from_env_or_config(
    env_var: str, config_key: str, parse: Callable[[Any], Any], label: str):
    """Shared env → ``cron.<config_key>`` resolution. ``parse`` returns the value or None to keep
    looking; a parse error on the env var WARNs, on config DEBUGs. None when neither yields."""
    env_value = os.getenv(env_var, "").strip()
    if env_value:
        try:
            value = parse(env_value)
            if value is not None:
                return value
        except Exception:
            logger.warning("Invalid %s=%r; using config/default", env_var, env_value)
    try:
        cfg = _sched.load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get(config_key)
        if configured is not None:
            value = parse(configured)
            if value is not None:
                return value
    except Exception as exc:
        logger.debug("Failed to load %s from config: %s", label, exc)
    return None


def _get_script_timeout() -> int:
    """Resolve cron pre-run script timeout from module/env/config with a safe default."""
    if _sched._SCRIPT_TIMEOUT != _sched._DEFAULT_SCRIPT_TIMEOUT:
        try:
            timeout = _positive_int(_sched._SCRIPT_TIMEOUT)
            if timeout is not None:
                return timeout
        except Exception:
            logger.warning(
                "Invalid patched _SCRIPT_TIMEOUT=%r; using env/config/default",
                _sched._SCRIPT_TIMEOUT)
    resolved = _timeout_from_env_or_config(
        "HERMES_CRON_SCRIPT_TIMEOUT", "script_timeout_seconds", _positive_int,
        "cron script timeout",
    )
    return _sched._DEFAULT_SCRIPT_TIMEOUT if resolved is None else resolved


_DEFAULT_MEDIA_SEND_TIMEOUT = 300


def _get_media_send_timeout() -> int:
    """Per-attachment media-send timeout: HERMES_CRON_MEDIA_SEND_TIMEOUT env, then
    ``cron.media_send_timeout_seconds``, then 300s (long TTS audio can exceed a 30s window)."""
    resolved = _timeout_from_env_or_config(
        "HERMES_CRON_MEDIA_SEND_TIMEOUT", "media_send_timeout_seconds", _positive_int,
        "cron media-send timeout")
    return _DEFAULT_MEDIA_SEND_TIMEOUT if resolved is None else resolved


def _get_session_db_timeout() -> float:
    """Bound on run_job's SessionDB init: HERMES_CRON_SESSION_DB_TIMEOUT env, then
    ``cron.session_db_timeout_seconds`` (in DEFAULT_CONFIG), then 10s. Unlike sibling timeouts,
    0 is meaningful (unlimited, debugging opt-in), so values pass through untouched."""
    resolved = _timeout_from_env_or_config(
        "HERMES_CRON_SESSION_DB_TIMEOUT", "session_db_timeout_seconds", float,
        "cron.session_db_timeout_seconds")
    return 10.0 if resolved is None else resolved


def _read_windows_pyvenv_cfg(venv_dir: Path) -> dict[str, str]:
    try:
        lines = (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    return {
        key.strip().lower(): value.strip()
        for key, value in (raw.split("=", 1) for raw in lines if "=" in raw)
    }


def _windows_cron_python_invocation(python_exe: str) -> tuple[str, dict[str, str]]:
    """Hidden, output-capable Python invocation for Windows cron scripts. ``pythonw.exe`` loses
    captured output; uv venv launchers can re-exec the base console python and flash a window
    even with CREATE_NO_WINDOW, so run the base python directly with venv paths overlaid in env."""
    if sys.platform != "win32":
        return python_exe, {}

    interpreter = _sched.Path(python_exe)
    venv_dir = interpreter.parent.parent
    env_overlay: dict[str, str] = {}

    if interpreter.name.lower() == "pythonw.exe":
        sibling = interpreter.with_name("python.exe")
        if sibling.exists():
            interpreter = sibling

    cfg = _read_windows_pyvenv_cfg(venv_dir)
    home = cfg.get("home", "")
    site_packages = venv_dir / "Lib" / "site-packages"
    if "uv" in cfg and home:
        base_python = _sched.Path(home) / "python.exe"
        if base_python.exists() and site_packages.exists():
            interpreter = base_python
            env_overlay["VIRTUAL_ENV"] = str(venv_dir)
            pythonpath_entries = [
                str(_sched.Path(__file__).resolve().parents[1]), str(site_packages)]
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_entries.append(existing_pythonpath)
            env_overlay["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return str(interpreter), env_overlay


def _terminate_cron_script_process(proc: subprocess.Popen) -> None:
    """Best-effort hard stop of a cron script and every child it spawned."""
    if proc.poll() is not None:
        return
    if sys.platform != "win32":
        _terminate_process_group(proc)
    else:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10,
                creationflags=windows_hide_flags(), check=False)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """POSIX: TERM the script's process group, then KILL if ANY member survived (a survivor holds
    the pipe write ends open and the caller's communicate() would block on EOF forever)."""
    try:
        process_group = os.getpgid(proc.pid)
        os.killpg(process_group, signal.SIGTERM)  # windows-footgun: ok — POSIX-only branch
    except (ProcessLookupError, PermissionError, OSError):
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=1.0)
    try:
        os.killpg(process_group, 0)  # windows-footgun: ok — POSIX-only branch
    except (ProcessLookupError, OSError):
        return
    with contextlib.suppress((ProcessLookupError, PermissionError, OSError)):
        os.killpg(process_group, getattr(signal, "SIGKILL", signal.SIGTERM))


def _terminate_cron_script_tree(proc: subprocess.Popen) -> None:
    """Terminate a script tree, then fall back to the local process-group path."""
    if proc.poll() is not None:
        # Already reaped: kill_process_tree would log a spurious "no signal" warning.
        return
    def fallback(reason: str, *args, exc_info: bool = False) -> None:
        logger.warning(
            reason + "; falling back to process-group termination", *args, exc_info=exc_info)
        _terminate_cron_script_process(proc)

    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return fallback("Cron script tree-kill received invalid pid %r", pid)
    try:
        # Function-local (monkeypatchable); separate try so an import problem is not
        # misreported as a kill failure.
        from agent.deadline import kill_process_tree
    except Exception:
        return fallback("agent.deadline.kill_process_tree unavailable", exc_info=True)
    try:
        if kill_process_tree(pid):
            return
    except Exception:
        return fallback("Cron script tree-kill failed for pid %s", pid, exc_info=True)
    fallback("Cron script tree-kill reported no signal for pid %s", pid)


def _drain_script_pipes(proc: subprocess.Popen) -> None:
    """Reap a terminated script without blocking forever: a surviving descendant can hold the pipe
    write ends open, so bound the drain and abandon the pipes (output is not needed)."""
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.communicate(timeout=5.0)
        return
    with contextlib.suppress(OSError):
        proc.kill()
    for stream in (proc.stdout, proc.stderr):
        with contextlib.suppress(OSError):
            if stream is not None:
                stream.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5.0)


def _windows_cron_bootstrap_argv(
    python_exe: str, env_overlay: dict[str, str], script_path: str) -> list[str]:
    """Bootstrap a cron script under the base interpreter with ``.pth`` support. Overlay mode puts
    the venv on ``PYTHONPATH``, but ``.pth`` files are only processed by ``site.addsitedir()``, so
    editable installs would be invisible; bootstrap via addsitedir + ``runpy.run_path`` (keeps
    ``__file__``/``sys.path[0]`` semantics). Plain invocation if the venv is unresolvable."""
    site_packages = _sched.Path(env_overlay.get("VIRTUAL_ENV", "")) / "Lib" / "site-packages"
    if not site_packages.is_dir():
        # Warn: silent fallback would make "editable installs invisible" undiagnosable.
        logger.warning(
            "Windows cron script: venv site-packages %s not found; running "
            "without .pth processing (editable installs may be unimportable)",
            site_packages)
        return [python_exe, script_path]
    bootstrap = (
        "import os, runpy, site, sys;"
        f"site.addsitedir({str(site_packages)!r});"
        "script = sys.argv[1];"
        "sys.argv = [script] + sys.argv[2:];"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(script)));"
        "runpy.run_path(script, run_name='__main__')"
    )
    return [python_exe, "-c", bootstrap, script_path]


def _resolve_script_path(script_path: str) -> tuple[Optional[Path], Optional[str]]:
    """Validate a job script path; ``(path, None)`` or ``(None, error)``. Scripts MUST resolve
    inside HERMES_HOME/scripts/ (relative, absolute and ``~`` paths are all validated — path
    traversal / absolute-path injection); contract of lifecycle_guard._expand_candidate_path."""
    scripts_dir = _sched._get_hermes_home() / "scripts"
    _ensure_cron_dir(scripts_dir)
    scripts_dir_resolved = scripts_dir.resolve()

    # Reject NUL eagerly: on Windows Path ops raise ValueError *after* expanduser so the try below
    # would not catch it. str() first so the guard itself cannot raise on a non-str script_path.
    # Same ingestion contract as cron.lifecycle_guard._expand_candidate_path: a NUL-bearing value can never
    # name a real script, and on Windows the Path operations raise ValueError *after* expanduser (expanduser
    # never expands "~user" there, so the try below never fires) — reject eagerly so both platforms fail
    # cleanly instead of crashing the scheduler. str() first so the guard itself can never raise TypeError
    # on a non-str script_path (e.g. a Path passed by a future caller) — the guard must be crash-proof even
    # though every current call site passes a plain str (#86832 review).
    if "\x00" in str(script_path):
        return None, f"Blocked: script path contains a NUL byte: {script_path!r}"
    try:
        raw = _sched.Path(script_path).expanduser()
    except (ValueError, RuntimeError, OSError):
        # RuntimeError: unexpandable ``~`` (no resolvable HOME).
        return None, f"Blocked: script path is not a valid filesystem path: {script_path!r}"
    path = raw.resolve() if raw.is_absolute() else (scripts_dir / raw).resolve()

    # Traversal / absolute-path / symlink escape guard — MUST stay inside HERMES_HOME/scripts/.
    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return None, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )
    if not path.exists():
        return None, f"Script not found: {path}"
    if not path.is_file():
        return None, f"Script path is not a file: {path}"
    return path, None


def _script_argv(path: Path) -> tuple[Optional[list[str]], dict[str, str], Optional[str]]:
    """``(argv, env_overlay, error)`` for a validated script. Interpreter by extension — the
    shebang is deliberately NOT honoured (small, auditable surface): ``.sh``/``.bash`` → bash,
    else ``sys.executable`` (Windows uv-venv overlay gets the .pth bootstrap)."""
    if path.suffix.lower() in {".sh", ".bash"}:
        # which() finds Git Bash on Windows; None there → clear error instead of a "[WinError 2]".
        _bash = shutil.which("bash") or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
        if _bash is None:
            return None, {}, (
                f"Cannot run .sh/.bash script {path.name!r}: bash not found on PATH. "
                "On Windows, install Git for Windows (which ships Git Bash) "
                "or rewrite the script as Python (.py)."
            )
        return [_bash, str(path)], {}, None
    python_exe, env_overlay = _windows_cron_python_invocation(sys.executable)
    if env_overlay:
        return _windows_cron_bootstrap_argv(python_exe, env_overlay, str(path)), env_overlay, None
    return [python_exe, str(path)], env_overlay, None


def _run_job_script(
    script_path: str, workdir: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
    job_env: Optional[dict[str, str]] = None,
) -> tuple[bool, str]:
    """Execute a cron job's script and return ``(success, output)``; on failure *output* is the
    error message for the LLM to report. Env goes through ``build_subprocess_env`` (SECURITY.md
    §2.3). ``workdir`` sets the subprocess cwd only; the Python process cwd is NEVER mutated (an
    ``os.chdir()`` would leak into concurrent gateway sessions).

    Args: script_path: Path to the script. Relative paths are resolved against HERMES_HOME/scripts/.
    Absolute and ~-prefixed paths are also validated to ensure they stay within the scripts dir. workdir:
    Optional absolute path to use as the script's cwd. When set, the subprocess runs in this directory
    instead of the scripts-dir parent. See #69396.
    """
    path, err = _resolve_script_path(script_path)
    if path is None:
        return False, err
    script_timeout = _get_script_timeout()
    argv, env_overlay, err = _script_argv(path)
    if argv is None:
        return False, err

    try:
        from tools.environments.local import build_subprocess_env
        popen_kwargs: dict[str, Any] = {"start_new_session": True}
        if sys.platform == "win32":
            popen_kwargs = {
                "creationflags": windows_hide_flags()
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                # Lossy UTF-8 decode — locale-mismatched bytes from the STT command must not raise in the
                # reader threads on non-UTF-8 Windows (#45099).
                # Lossy UTF-8 decode — locale-mismatched bytes from the TTS command must not raise in the
                # reader threads on non-UTF-8 Windows (#45099).
                "encoding": "utf-8",
                "errors": "replace"}
        env = build_subprocess_env()
        env.update(env_overlay)
        if job_env:
            env.update(job_env)
        # Subprocess cwd only (default: scripts-dir parent). NEVER os.chdir() the process.
        # Use the job's workdir as the subprocess cwd when configured, otherwise default to the scripts-dir
        # parent (back-compat). NEVER mutate the Python process cwd — that would leak into concurrent
        # gateway sessions (#69396).
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=workdir or str(path.parent), env=env, **popen_kwargs)
        deadline = time.monotonic() + script_timeout
        while True:
            # Tree-kill on cancel AND timeout: killpg misses setsid grandchildren (watchdogs,
            # backgrounded shell jobs); kill_process_tree snapshots descendants BEFORE signalling.
            if cancel_event is not None and cancel_event.is_set():
                _terminate_cron_script_tree(proc)
                _drain_script_pipes(proc)
                return False, "Script cancelled because cron fire ownership was lost"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_cron_script_tree(proc)
                _drain_script_pipes(proc)
                # Phase 4a (#85125): a script timeout must leave ZERO living descendants. killpg only
                # reaches the script's own process group — a grandchild that called setsid (backgrounded
                # shell jobs, watchdogs) escapes it and keeps running after the job reports failure (#71148
                # / #59549). agent.deadline.kill_process_tree snapshots the descendant set via psutil BEFORE
                # signalling, so own-session grandchildren are reached too — the unified deadline layer's
                # tree-kill (#85147, d6a5cb9725).
                return False, f"Script timed out after {script_timeout}s: {path}"
            try:
                stdout_raw, stderr_raw = proc.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        stdout = (stdout_raw or "").strip()
        stderr = (stderr_raw or "").strip()

        # Redact secrets before ANY return path.
        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception as e:
            logger.warning("Failed to redact sensitive text from output: %s", e)
            stdout = stderr = "[REDACTED - redaction failed]"

        if proc.returncode != 0:
            parts = [f"Script exited with code {proc.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)
        return True, stdout
    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _start_heartbeat_thread(loop_fn, name: str, fail_log) -> Optional[threading.Thread]:
    """Start ``loop_fn`` on a daemon thread inside a copy of the current context (multiplexed
    profile ContextVars). On failure calls ``fail_log()`` inside the except (traceback intact) and
    returns None."""
    thread = threading.Thread(
        target=contextvars.copy_context().run, args=(loop_fn,), name=name, daemon=True)
    try:
        thread.start()
    except Exception:
        fail_log()
        return None
    return thread


def _run_job_script_with_claim_heartbeat(
    job: dict, script_path: str, workdir: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
    job_env: Optional[dict[str, str]] = None,
) -> tuple[bool, str]:
    """Run a cron script while heartbeating its owned one-shot claim. A long script can outlive
    the stale-claim TTL; without a heartbeat another scheduler would re-dispatch the one-shot.
    Recurring/unclaimed runs have no durable claim → no thread. The owner is captured from the
    dispatched job, never re-read, so a stale runner cannot extend a replacement owner's claim."""
    schedule = job.get("schedule")
    claim = job.get("run_claim")
    owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    if not (isinstance(schedule, dict) and schedule.get("kind") == "once" and owner):
        return _run_job_script(
            script_path, workdir=workdir, cancel_event=cancel_event, job_env=job_env)

    job_id = str(job.get("id") or "")
    stop = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop.wait(_sched._RUN_CLAIM_HEARTBEAT_SECONDS):
            try:
                _sched.heartbeat_run_claim(job_id, expected_owner=owner)
            except Exception:
                logger.debug("Job '%s': script run_claim heartbeat failed", job_id, exc_info=True)

    heartbeat_thread = _start_heartbeat_thread(
        _heartbeat_loop, "cron-script-claim-heartbeat",
        lambda: logger.debug(
            "Job '%s': could not start script run_claim heartbeat", job_id, exc_info=True),
    )
    if heartbeat_thread is None:
        return _run_job_script(
            script_path, workdir=workdir, cancel_event=cancel_event, job_env=job_env)

    try:
        return _run_job_script(
            script_path, workdir=workdir, cancel_event=cancel_event, job_env=job_env)
    finally:
        stop.set()
        # Bounded join: the heartbeat may be blocked on another process's jobs-file lock.
        heartbeat_thread.join(timeout=1.0)


# Late-bound origin namespace (see module docstring). Imported LAST so this module is fully
# populated before ``scheduler`` re-exports from it.
from cron import scheduler as _sched  # noqa: E402
