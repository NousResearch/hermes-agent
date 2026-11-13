"""Long-lived processes inside an environment, with both stdio pipes open.

``execute`` is request/response. A user MCP server (stdio transport) needs the other shape: start a
process once, keep talking to it over stdin/stdout from later calls in the same session, stop it when
it goes idle. Each environment owns the processes it started, by name; nothing here outlives the
environment's own object, and a name is only meaningful inside the environment that started it.

Spawning goes through :meth:`LongLivedProcessMixin._spawn_long_lived`: the local backend runs the
argv on this host with its own run environment; a backend with an exec prefix (docker, ssh, a plugin
declaring ``supports_streams``) runs it inside the sandbox through ``hermes_agent.tools.environments.streams``;
SDK backends with neither raise :class:`NotImplementedError`. A hosted backend implements the three
public methods itself (they receive the bound scope, see ``sandbox_scope.ScopedEnvironment``).
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import subprocess
import threading
import uuid
from typing import Any, Optional

from hermes_agent.tools.environments.sandbox_scope import ProcessRef, ProcessStream

logger = logging.getLogger("hermes_agent.tools.environments")


class LongLivedProcessMixin:
    """``start_process`` / ``attach_process`` / ``stop_process`` for a :class:`BaseEnvironment`."""

    def _long_lived(self) -> dict[str, subprocess.Popen]:
        state = self.__dict__.get("_long_lived_state")
        if state is None:
            # One owned state object, installed by a single setdefault: a concurrent first start
            # can never see the table without the lock, and no fallback lock can differ from
            # the starter's.
            state = self.__dict__.setdefault(
                "_long_lived_state", {"table": {}, "lock": threading.Lock()})
        return state["table"]

    def _long_lived_lock(self) -> threading.Lock:
        state = self.__dict__.get("_long_lived_state")
        if state is None:
            state = self.__dict__.setdefault(
                "_long_lived_state", {"table": {}, "lock": threading.Lock()})
        return state["lock"]

    def _spawn_long_lived(self, argv: list[str], child_env: Optional[dict]) -> subprocess.Popen:
        from hermes_agent.tools.environments.streams import exec_prefix, open_stream, sandbox_temp_dir
        try:
            prefix = exec_prefix(self)  # type: ignore[arg-type]  # a BaseEnvironment at runtime, the mixin in isolation
            if prefix is None:
                return open_stream(self, argv, child_env=child_env)  # type: ignore[arg-type]
            # A backend with an exec prefix (docker exec -i, ssh, a plugin prefix) runs the service
            # inside the sandbox behind a LOCAL client; killing that client need not stop it. The
            # service's pid is therefore recorded inside the sandbox before it execs, and
            # _reap_long_lived stops it there before closing the client. The transport is open_stream's
            # own; a ``/bin/sh -c`` wrapper records its pid and execs the service (argv passed as positional
            # words, so it keeps the pid and stdin/stdout stay the service's). The wrapper is POSIX sh by
            # absolute path: child_env is already exported when it starts, so its PATH or BASH_ENV must
            # not decide whether (or how) it runs; the service is still found through that PATH. Next to
            # the pid goes its start time (field 22 of /proc/<pid>/stat, the 20th after the last ``)``
            # since the command name may hold both; exec keeps it), so the stop never signals a process
            # that reused the pid. The fields are split inside ``$( )``, so ``"$@"`` stays the service's.
            pidfile = f"{sandbox_temp_dir(self)}/hermes_llp_{uuid.uuid4().hex[:16]}.pid"  # type: ignore[arg-type]
            record = ("echo $$ $({ read -r s < /proc/$$/stat; } 2>/dev/null; s=${s##*\\)}; set -- $s; echo \"${20:-}\")"
                      f" > {shlex.quote(pidfile)} 2>/dev/null")
            wrapper = ["/bin/sh", "-c", f"{record}; exec \"$@\"", "sh", *(str(word) for word in argv)]
            exports = {key: str(value) for key, value in (child_env or {}).items()}
            proc = open_stream(self, wrapper, child_env=exports)  # type: ignore[arg-type]
            proc._hermes_remote_pidfile = pidfile  # type: ignore[attr-defined]  # teardown metadata, not a pipe
            return proc
        except RuntimeError as exc:
            raise NotImplementedError(str(exc)) from exc

    def start_process(self, name: str, argv: list[str], *, child_env: Optional[dict] = None,
                      scope: Any = None) -> ProcessStream:
        """Start *argv* as the long-lived process *name* (stdin/stdout as byte pipes). A name that is
        still running is refused: attach to it instead. A name whose record is left over from a
        launcher that exited goes to the new process only after that record is reaped: its children
        and pipes are this environment's to stop, and overwriting them would orphan both."""
        table = self._long_lived()
        with self._long_lived_lock():
            running = table.get(name)
            if running is not None and running.poll() is None:
                raise ValueError(f"a long-lived process named {name!r} is already running; attach to it")
            stale = table.pop(name, None)
            try:
                table[name] = proc = self._spawn_long_lived(list(argv), child_env)
            except BaseException:
                # A failed respawn must not drop the leftover record on the floor: its children
                # and pipes are still this environment's to stop, and a later stop would find
                # nothing left to reap. Keep the record; the successful path reaps it below.
                if stale is not None:
                    table[name] = stale
                raise
        if stale is not None:
            self._reap_long_lived(stale, scope)
        return self._stream(name, proc)

    def attach_process(self, name: str, *, scope: Any = None) -> Optional[ProcessStream]:
        """The running process *name* this environment started, or ``None``."""
        with self._long_lived_lock():
            proc = self._long_lived().get(name)
        return self._stream(name, proc) if proc is not None and proc.poll() is None else None

    def stop_process(self, name: str, *, scope: Any = None) -> bool:
        """Stop *name* (its whole process tree where the backend can); ``False`` if none was running.
        A leader that already exited still has its group swept: a descendant it left behind is
        stopped too, and the record goes only once that is done."""
        with self._long_lived_lock():
            proc = self._long_lived().pop(name, None)
        if proc is None:
            return False
        was_running = proc.poll() is None
        self._reap_long_lived(proc, scope)
        return was_running

    def stop_all_processes(self, *, scope: Any = None) -> None:
        """Stop and reap every long-lived process this environment started (environment cleanup)."""
        with self._long_lived_lock():
            procs = list(self._long_lived().values())
            self._long_lived().clear()
        for proc in procs:
            self._reap_long_lived(proc, scope)

    def _stop_remote_service(self, proc: subprocess.Popen, scope: Any) -> None:
        """Best-effort stop of a service that runs inside a backend (docker exec / ssh / a plugin
        prefix): its pid was recorded in the sandbox at spawn; kill it there, where the local
        client's death alone would leave it running. A recorded start time must still match the
        pid's current one (else the pid was reused and nothing is signalled); a pidfile without one
        (a sandbox with no /proc) is signalled as is. Only a plain decimal pid above 1 is ever signalled
        (never a group, 0 or init). POSIX sh: it runs in the backend's shell, with errexit turned off
        so a ``set -e`` there cannot skip the pidfile's removal."""
        pidfile = getattr(proc, "_hermes_remote_pidfile", None)
        execute = getattr(self, "execute", None)
        if pidfile is None or not callable(execute):
            return
        command = (f"(set +e; f={shlex.quote(pidfile)}; p=; t=; {{ read -r p t < \"$f\"; }} 2>/dev/null; "
                   "case \"$p\" in \"\"|*[!0-9]*) p=;; esac; [ -n \"$p\" ] && [ \"$p\" -gt 1 ] 2>/dev/null || p=; "
                   "if [ -n \"$p\" ] && [ -n \"$t\" ]; then c=$(cat /proc/\"$p\"/stat 2>/dev/null); "
                   "c=${c##*\\)}; set -- $c; [ \"${20:-}\" = \"$t\" ] || p=; fi; "
                   "if [ -n \"$p\" ]; then pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; fi; "
                   "rm -f \"$f\")")
        try:
            from hermes_agent.tools.environments.sandbox_scope import accepts_scope
            if scope is not None and accepts_scope(execute):
                execute(command, scope=scope)
            else:
                execute(command)
        except Exception:
            # Best-effort by design (the client kill below still ends the transport), but never a
            # silent one: a service that survived its own stop is the next bug report.
            logger.warning("long-lived service stop command failed for %s", pidfile, exc_info=True)

    def _reap_long_lived(self, proc: subprocess.Popen, scope: Any = None) -> None:
        """Kill *proc*'s tree, reap it, close its pipes, then release a host-owned group. A backend
        service (pid recorded in the sandbox at spawn) is stopped there first: the local client's
        death need not stop it."""
        self._stop_remote_service(proc, scope)
        owned_pgid = getattr(proc, "_hermes_owned_pgid", None)
        if proc.poll() is None or owned_pgid is not None:
            self._kill_process(proc)  # type: ignore[attr-defined]  # signals the group even with the leader gone
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()
        if owned_pgid is not None:
            # A host-owned group (``spawn_owned``): stopped, so the owner-death supervisor stops covering it
            # once it is empty (a member that survived both signals stays covered and dies with Hermes).
            from hermes_cli.process_identity import release_process_group_once_empty
            release_process_group_once_empty(owned_pgid, proc=proc)

    @staticmethod
    def _stream(name: str, proc: subprocess.Popen) -> ProcessStream:
        assert proc.stdin is not None and proc.stdout is not None  # spawned with both pipes
        return ProcessStream(ref=ProcessRef("", name), stdin=proc.stdin, stdout=proc.stdout, poll=proc.poll)
