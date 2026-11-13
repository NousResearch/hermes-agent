"""Which sandbox an operation may address, derived from the bound scope alone.

A hosted deployment serves many tenants' agents from one process, and every agent's commands,
code, background processes and files live in that agent's own remote sandbox. The sandbox an
operation reaches is therefore decided here, from the bound :class:`hermes_scope.Scope`'s
``(tenant, agent)``, and from nothing a caller hands in: not a tool argument, not a task id, not
a profile path. Two calls for the same agent share one sandbox; a call for another agent can never
be routed to it.

The guard sits in core, not in the backend: :class:`ScopedEnvironment` wraps every environment a
hosted backend creates, checks the bound scope against the sandbox it was created for before each
operation, and hands the backend that scope with the call. A handle that names another agent (an
environment object, a long-lived process reference) is refused with
:class:`CrossAgentHandleRefused`, and so is any operation with no scope bound.

A local install (no scope, or a scope whose tenant is ``hermes_scope.LOCAL_TENANT``) is untouched:
:func:`hosted_scope` is ``None`` there and every caller keeps today's behaviour.
"""

from __future__ import annotations

import functools
import inspect
import os
import posixpath
from dataclasses import dataclass, field
from typing import IO, Any, Callable, Optional
from urllib.parse import quote

from hermes_scope import Scope, current_scope


class CrossAgentHandleRefused(PermissionError):
    """An operation addressed a sandbox that is not the bound agent's, or ran with no hosted scope."""


class HostedSandboxRequired(RuntimeError):
    """A hosted scope asked for a backend that would run on this host (a built-in), or for a plugin
    backend that does not declare it serves hosted scopes."""


@dataclass(frozen=True)
class SandboxIdentity:
    """The one sandbox an agent owns: ``(tenant, agent)`` and nothing else."""

    tenant: str
    agent: str

    @property
    def key(self) -> str:
        """Environment-cache key. Each part is percent-quoted, so no two identities share a key."""
        return f"sandbox:{quote(self.tenant, safe='')}/{quote(self.agent, safe='')}"


@dataclass(frozen=True)
class ProcessRef:
    """A long-lived process inside a sandbox: the sandbox it was started in and its name there."""

    sandbox: str
    name: str


@dataclass(frozen=True)
class ProcessStream:
    """A long-lived process with both stdio pipes open (bytes). ``poll`` returns the exit status, or
    ``None`` while it runs."""

    ref: ProcessRef
    stdin: IO[bytes]
    stdout: IO[bytes]
    poll: Callable[[], Optional[int]] = field(compare=False)


@dataclass(frozen=True)
class CredentialRef:
    """A credential the sandbox resolves through a broker, by name. Hosted backends receive these,
    never secret values from the secret store; no broker exists yet, so core passes none."""

    name: str


def hosted_scope() -> Optional[Scope]:
    """The bound scope when it is hosted, else ``None`` (no scope, or a local install)."""
    scope = current_scope()
    return scope if scope is not None and not scope.is_local else None


def sandbox_identity(scope: Scope) -> SandboxIdentity:
    return SandboxIdentity(scope.tenant, scope.agent)


def current_sandbox_key() -> str:
    """The bound hosted agent's sandbox key, or ``""`` outside a hosted scope."""
    scope = hosted_scope()
    return sandbox_identity(scope).key if scope is not None else ""


def sandbox_visible(sandbox_key: str) -> bool:
    """Whether a record stamped with *sandbox_key* (``""`` for a local activity's) may be seen by the
    current activity: only by the agent whose sandbox it names. A local activity sees local records
    as today; a hosted agent sees only its own, and an activity with no hosted scope bound sees no
    hosted agent's (it binds that agent's scope first). Anything else is indistinguishable from a
    missing record."""
    return sandbox_key == current_sandbox_key()


def sandbox_lexical_path(path: Any) -> Optional[str]:
    """In a hosted scope, the form a host path policy (credential denylists, protected files) judges
    *path* by: its lexical normal form. The path names a file in the agent's sandbox, so this host's
    symlinks, and whether the path exists here, say nothing about it, and resolving it here would
    stat and dereference a tenant-chosen host path. ``None`` outside a hosted scope: the caller keeps
    its host resolution."""
    if hosted_scope() is None:
        return None
    return posixpath.normpath(os.fspath(path) or ".")


def producer_fields(sandbox_key: str) -> dict:
    """The producer identity a notification about a hosted agent's record carries (nothing for a
    local one's, so local events are unchanged)."""
    return {"sandbox_key": sandbox_key} if sandbox_key else {}


def event_visible(event: dict) -> bool:
    """Whether the current activity may deliver, consume, suppress or drop *event*: only the agent
    that produced it. Anything else leaves it queued for its owner."""
    return sandbox_visible(str(event.get("sandbox_key") or ""))


def require_identity(identity: SandboxIdentity, what: str) -> Scope:
    """The bound scope, when it is hosted and names *identity*'s agent; else refuse."""
    scope = hosted_scope()
    if scope is None:
        raise CrossAgentHandleRefused(f"{what} on a hosted sandbox needs a hosted scope bound")
    if sandbox_identity(scope) != identity:
        raise CrossAgentHandleRefused(f"{what} names another agent's sandbox; refused")
    return scope


def admit_env(env: Any) -> Any:
    """*env*, when the current activity may use it: anything outside a hosted scope; inside one,
    only the bound agent's own :class:`ScopedEnvironment`. ``None`` passes through."""
    scope = hosted_scope()
    if env is None or scope is None:
        return env
    if not isinstance(env, ScopedEnvironment) or env.sandbox != sandbox_identity(scope):
        raise CrossAgentHandleRefused("a cached environment that is not this agent's sandbox was reached; refused")
    return env


def create_hosted_environment(env_type: str, provider: Any, **kwargs: Any) -> ScopedEnvironment:
    """Create the bound hosted agent's sandbox through *provider*, behind the identity guard.

    The provider receives ``scope`` and ``sandbox`` (the :class:`SandboxIdentity`), and
    ``task_id`` is the identity's key: whatever task id the caller held is not passed on. The
    environment it returns must implement the hosted operation interface (:func:`require_hosted_interface`),
    or it is refused here, before any tool can use it."""
    scope = hosted_scope()
    if scope is None:
        raise CrossAgentHandleRefused("a hosted sandbox can only be created inside a hosted scope")
    if provider is None or getattr(provider, "serves_hosted_scopes", False) is not True:
        raise HostedSandboxRequired(
            f"terminal.backend {env_type!r} cannot serve a hosted scope: tools for a hosted agent run only in "
            "a remote sandbox backend that declares serves_hosted_scopes")
    identity = sandbox_identity(scope)
    kwargs["task_id"] = identity.key
    inner = provider.create_environment(scope=scope, sandbox=identity, **kwargs)
    require_hosted_interface(inner, env_type)
    return ScopedEnvironment(inner, identity, owner=scope)


# The hosted operation interface. Every operation core makes on a hosted sandbox receives the bound
# scope as the ``scope`` keyword, so the backend can check the sandbox it is about to touch. The
# required ones must exist; an optional one, when present, must accept ``scope`` too.
# ``stop_idle`` is required: an idle hosted sandbox is suspended, never destroyed in its place.
REQUIRED_OPERATIONS = ("execute", "cleanup", "start_process", "attach_process", "stop_process", "stop_idle")
OPTIONAL_OPERATIONS = ("fetch_file", "fetch_realpath", "get_temp_dir")
# Lifecycle calls the idle reaper makes with no scope bound: they run under the owning agent's scope
# (bound for the call), and are refused from another agent. ``stop_idle`` suspends the sandbox (its
# workspace is kept, it resumes on the next call); ``cleanup`` destroys it.
_LIFECYCLE = frozenset({"cleanup", "stop_idle"})


def accepts_scope(method: Any) -> bool:
    """Whether *method* can take ``scope=``: a named parameter or a ``**kwargs`` catch-all."""
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "scope" and p.kind is not p.POSITIONAL_ONLY or p.kind is p.VAR_KEYWORD for p in params)


def require_hosted_interface(inner: Any, env_type: str) -> None:
    """Refuse a hosted environment that would silently drop the scope on any operation."""
    missing = [op for op in REQUIRED_OPERATIONS if not callable(getattr(inner, op, None))]
    blind = [op for op in (*REQUIRED_OPERATIONS, *OPTIONAL_OPERATIONS)
             if op not in missing and callable(getattr(inner, op, None)) and not accepts_scope(getattr(inner, op))]
    if missing or blind:
        raise HostedSandboxRequired(
            f"terminal.backend {env_type!r} does not implement the hosted operation interface: "
            + "; ".join(filter(None, (missing and f"missing {', '.join(missing)}",
                                      blind and f"{', '.join(blind)} must accept scope="))))


class ScopedEnvironment:
    """A hosted backend's environment behind the identity guard.

    Every method call checks the bound scope first and hands it to the backend as ``scope=``
    (:func:`require_hosted_interface` proved at creation that the interface accepts it; any other
    method gets it whenever it can take it). Lifecycle calls made with no scope bound run under the
    scope that created the sandbox. Attributes (``cwd``, ``env_type``, ...) read and write through
    to the backend's object. The file tools never take a host-filesystem fast path through it."""

    uses_host_filesystem = False
    is_local = False

    def __init__(self, inner: Any, identity: SandboxIdentity, *, owner: Optional[Scope] = None) -> None:
        if owner is not None and sandbox_identity(owner) != identity:
            raise CrossAgentHandleRefused("a sandbox's owning scope must name that sandbox's agent")
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "sandbox", identity)
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._inner, name)
        if not callable(value):
            return value
        return self._lifecycle(name, value) if name in _LIFECYCLE else self._guarded(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)

    def _guarded(self, name: str, method: Any) -> Any:
        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            scope = require_identity(self.sandbox, name)
            if accepts_scope(method):
                kwargs["scope"] = scope
            return method(*args, **kwargs)
        return call

    def _lifecycle(self, name: str, method: Any) -> Any:
        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            if hosted_scope() is not None:
                return self._guarded(name, method)(*args, **kwargs)
            if self.owner is None:
                raise CrossAgentHandleRefused(f"{name} on a hosted sandbox with no owning scope; refused")
            from hermes_agent.agent.runtime_scope import bind_state
            with bind_state(self.owner):
                return self._guarded(name, method)(*args, **kwargs)
        return call

    def start_process(self, name: str, argv: list[str], *, child_env: Optional[dict] = None) -> ProcessStream:
        """Start *argv* in this sandbox as the long-lived process *name*, stdio open both ways."""
        scope = require_identity(self.sandbox, "start_process")
        stream = self._inner.start_process(name, argv, child_env=child_env, scope=scope)
        return self._guard_stream(stream, name)

    def attach_process(self, ref: ProcessRef | str) -> Optional[ProcessStream]:
        """The running process *ref* started earlier in this sandbox, or ``None``."""
        name = self._own_name(ref, "attach_process")
        stream = self._inner.attach_process(name, scope=require_identity(self.sandbox, "attach_process"))
        return None if stream is None else self._guard_stream(stream, name)

    def stop_process(self, ref: ProcessRef | str) -> bool:
        """Stop *ref*; ``False`` when no such process was running."""
        name = self._own_name(ref, "stop_process")
        return self._inner.stop_process(name, scope=require_identity(self.sandbox, "stop_process"))

    def _guard_stream(self, stream: ProcessStream, name: str) -> ProcessStream:
        """*stream* with every pipe and ``poll`` behind this sandbox's identity check: a stream once
        acquired by A stays A's, whoever holds the object later."""
        return ProcessStream(ProcessRef(self.sandbox.key, name), ScopedPipe(stream.stdin, self.sandbox),
                             ScopedPipe(stream.stdout, self.sandbox), _scoped_call(stream.poll, self.sandbox, "poll"))

    def _own_name(self, ref: ProcessRef | str, what: str) -> str:
        if isinstance(ref, ProcessRef):
            if ref.sandbox != self.sandbox.key:
                raise CrossAgentHandleRefused(f"{what} names a process in another agent's sandbox; refused")
            return ref.name
        return ref

    def __repr__(self) -> str:
        return f"ScopedEnvironment({self._inner!r}, {self.sandbox.key})"


def _scoped_call(fn: Callable[..., Any], identity: SandboxIdentity, what: str) -> Callable[..., Any]:
    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        require_identity(identity, what)
        return fn(*args, **kwargs)
    return call


class ScopedPipe:
    """One stdio pipe of a hosted long-lived process, behind a closed byte-pipe interface: read,
    read1, readline, readlines, readinto, write, writelines, flush, close, closed, readable,
    writable, iteration and the context manager. Every operation is refused unless the bound scope
    is the sandbox's own agent. Nothing else is reachable: no ``raw``/``buffer``/``fileno``/
    ``detach`` and no attribute passthrough, so no object or descriptor that bypasses the check
    ever leaves the proxy. (An accidental cross-agent guard, not containment of in-process code.)"""

    __slots__ = ("_identity", "_pipe")

    def __init__(self, pipe: Any, identity: SandboxIdentity) -> None:
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "_identity", identity)

    def _op(self, what: str) -> Any:
        require_identity(self._identity, f"stream {what}")
        return self._pipe

    def read(self, size: int = -1) -> bytes:
        return self._op("read").read(size)

    def read1(self, size: int = -1) -> bytes:
        pipe = self._op("read1")
        return pipe.read1(size) if hasattr(pipe, "read1") else pipe.read(size)

    def readline(self, size: int = -1) -> bytes:
        return self._op("readline").readline(size)

    def readlines(self, hint: int = -1) -> list[bytes]:
        return self._op("readlines").readlines(hint)

    def readinto(self, buffer: Any) -> Optional[int]:
        return self._op("readinto").readinto(buffer)

    def write(self, data: bytes) -> Optional[int]:
        return self._op("write").write(data)

    def writelines(self, lines: Any) -> None:
        self._op("writelines").writelines(lines)

    def flush(self) -> None:
        self._op("flush").flush()

    def close(self) -> None:
        self._op("close").close()

    @property
    def closed(self) -> bool:
        return bool(self._op("closed").closed)

    def readable(self) -> bool:
        return bool(self._op("readable").readable())

    def writable(self) -> bool:
        return bool(self._op("writable").writable())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a hosted process pipe has no settable attribute {name!r}")

    def __iter__(self) -> ScopedPipe:
        self._op("iteration")
        return self

    def __next__(self) -> bytes:
        line = self._op("iteration").readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self) -> ScopedPipe:
        self._op("context")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ScopedPipe({self._identity.key})"

