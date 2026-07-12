"""Process boundary state for the privileged Canonical Brain writer.

The gateway is an unprivileged client.  It never receives a database
credential and it cannot opt into the writer-service execution context.  The
service entry point binds that context after peer authentication and supplies
the private database adapter for the duration of one typed request.

Configuration is intentionally static for a gateway process.  Tool discovery
may use :func:`writer_boundary_configured` without probing the socket, so a
temporary writer outage does not mutate the model tool schema and invalidate
the conversation prompt cache.
"""

from __future__ import annotations

import contextlib
import contextvars
import ctypes
import errno
import hashlib
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


DEFAULT_SOCKET_PATH = Path("/run/muncho-canonical-writer/writer.sock")
DEFAULT_GATEWAY_UNIT = "hermes-cloud-gateway.service"
DEFAULT_WRITER_UNIT = "muncho-canonical-writer.service"
DEFAULT_WRITER_USER = "muncho-canonical-writer"
_MAX_RUNTIME_VALUE_CHARS = 1024


@dataclass(frozen=True)
class CanonicalWriterBoundaryConfig:
    """Static, non-secret writer client configuration from ``config.yaml``."""

    enabled: bool = False
    model_tools_enabled: bool = False
    socket_path: Path = DEFAULT_SOCKET_PATH
    gateway_unit: str = DEFAULT_GATEWAY_UNIT
    writer_unit: str = DEFAULT_WRITER_UNIT
    connect_timeout_seconds: float = 2.0
    request_timeout_seconds: float = 12.0


@dataclass(frozen=True)
class WriterServiceContext:
    """Private service-local dependencies for one authenticated request."""

    database: Any
    runtime: Mapping[str, str]
    peer_pid: int
    peer_uid: int


_SERVICE_CONTEXT: contextvars.ContextVar[WriterServiceContext | None] = (
    contextvars.ContextVar("canonical_writer_service_context", default=None)
)
_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[CanonicalWriterBoundaryConfig, Any] = {}
_CONFIG_LOCK = threading.Lock()
_FROZEN_CONFIG: CanonicalWriterBoundaryConfig | None = None
_FROZEN_CONFIG_ERROR: str | None = None
_FROZEN_BOUNDARY_DECLARED_ENABLED = False
_CONFIG_IS_FROZEN = False
_PROCESS_HARDENING_LOCK = threading.Lock()
_PROCESS_HARDENED = False
_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _root_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is not None:
        return config
    try:
        from hermes_cli.config import load_config

        loaded = load_config() or {}
    except Exception:
        loaded = {}
    return loaded if isinstance(loaded, Mapping) else {}


def _boundary_declared_enabled(config: Mapping[str, Any]) -> bool:
    canonical = config.get("canonical_brain")
    canonical = canonical if isinstance(canonical, Mapping) else {}
    raw = canonical.get("writer_boundary")
    raw = raw if isinstance(raw, Mapping) else {}
    return _coerce_bool(raw.get("enabled"), False)


def load_writer_boundary_config(config: Mapping[str, Any] | None = None) -> CanonicalWriterBoundaryConfig:
    """Load the mechanical writer boundary settings.

    Behavioral settings stay in ``config.yaml``.  Environment variables are
    deliberately not accepted here because a model-controlled child process
    must not be able to redirect the privileged client to another socket.
    """

    config = _root_config(config)
    canonical = config.get("canonical_brain") if isinstance(config, Mapping) else None
    canonical = canonical if isinstance(canonical, Mapping) else {}
    audit_bridge = canonical.get("audit_bridge")
    audit_bridge = audit_bridge if isinstance(audit_bridge, Mapping) else {}
    raw = canonical.get("writer_boundary")
    raw = raw if isinstance(raw, Mapping) else {}

    socket_raw = str(raw.get("socket_path") or DEFAULT_SOCKET_PATH).strip()
    socket_path = Path(socket_raw)
    if not socket_path.is_absolute():
        raise ValueError("canonical_brain.writer_boundary.socket_path must be absolute")
    if socket_path != DEFAULT_SOCKET_PATH:
        raise ValueError(
            "canonical_brain.writer_boundary.socket_path is production-pinned"
        )
    gateway_unit = str(raw.get("gateway_unit") or DEFAULT_GATEWAY_UNIT).strip()
    if gateway_unit != DEFAULT_GATEWAY_UNIT:
        raise ValueError(
            "canonical_brain.writer_boundary.gateway_unit is production-pinned"
        )
    writer_unit = str(raw.get("writer_unit") or DEFAULT_WRITER_UNIT).strip()
    if writer_unit != DEFAULT_WRITER_UNIT:
        raise ValueError(
            "canonical_brain.writer_boundary.writer_unit is production-pinned"
        )

    connect_timeout = float(raw.get("connect_timeout_seconds") or 2.0)
    request_timeout = float(raw.get("request_timeout_seconds") or 12.0)
    if not 0.1 <= connect_timeout <= 10.0:
        raise ValueError("writer connect_timeout_seconds must be between 0.1 and 10")
    if not 0.5 <= request_timeout <= 30.0:
        raise ValueError("writer request_timeout_seconds must be between 0.5 and 30")

    return CanonicalWriterBoundaryConfig(
        enabled=_coerce_bool(raw.get("enabled"), False),
        model_tools_enabled=bool(
            _coerce_bool(canonical.get("tools_enabled"), False)
            or _coerce_bool(audit_bridge.get("enabled"), False)
        ),
        socket_path=socket_path,
        gateway_unit=gateway_unit,
        writer_unit=writer_unit,
        connect_timeout_seconds=connect_timeout,
        request_timeout_seconds=request_timeout,
    )


def writer_boundary_configured(config: Mapping[str, Any] | None = None) -> bool:
    """Return restart-only policy without a socket or database health probe."""

    try:
        return frozen_writer_boundary_config(config).enabled
    except Exception:
        return False


def canonical_model_tools_configured() -> bool:
    """Return the restart-only Canonical model-tool availability policy.

    Tool discovery must never consult mutable configuration after the first
    boundary snapshot.  Otherwise a live config edit could add or remove model
    tools and invalidate the cached conversation prefix.
    """

    try:
        config = frozen_writer_boundary_config()
    except Exception:
        return False
    return bool(config.enabled and config.model_tools_enabled)


def writer_boundary_policy_required(
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Return frozen raw enablement even when detailed config is invalid.

    An invalid enabled boundary must fail closed.  Callers use this bit to
    prevent a malformed socket/unit setting from restoring process-local
    capability authority.
    """

    try:
        frozen_writer_boundary_config(config)
    except Exception:
        pass
    return _FROZEN_BOUNDARY_DECLARED_ENABLED


def frozen_writer_boundary_config(
    config: Mapping[str, Any] | None = None,
) -> CanonicalWriterBoundaryConfig:
    """Freeze Canonical writer policy on first use for this gateway process.

    Hermes tool schemas are cached per conversation. Runtime config edits must
    therefore require a gateway restart instead of adding/removing tools or
    redirecting the writer client in a live process.
    """

    global _CONFIG_IS_FROZEN, _FROZEN_CONFIG, _FROZEN_CONFIG_ERROR
    global _FROZEN_BOUNDARY_DECLARED_ENABLED
    with _CONFIG_LOCK:
        if not _CONFIG_IS_FROZEN:
            root = _root_config(config)
            _FROZEN_BOUNDARY_DECLARED_ENABLED = _boundary_declared_enabled(root)
            try:
                _FROZEN_CONFIG = load_writer_boundary_config(root)
                _FROZEN_CONFIG_ERROR = None
            except Exception as exc:
                _FROZEN_CONFIG = None
                _FROZEN_CONFIG_ERROR = str(exc)[:500]
            _CONFIG_IS_FROZEN = True
        if _FROZEN_CONFIG is None:
            raise ValueError(
                _FROZEN_CONFIG_ERROR
                or "canonical writer boundary configuration is invalid"
            )
        return _FROZEN_CONFIG


def _reset_frozen_writer_boundary_config_for_tests() -> None:
    """Reset process policy only for isolated tests simulating a restart."""

    global _CONFIG_IS_FROZEN, _FROZEN_CONFIG, _FROZEN_CONFIG_ERROR
    global _FROZEN_BOUNDARY_DECLARED_ENABLED
    global _PROCESS_HARDENED
    with _CONFIG_LOCK:
        _CONFIG_IS_FROZEN = False
        _FROZEN_CONFIG = None
        _FROZEN_CONFIG_ERROR = None
        _FROZEN_BOUNDARY_DECLARED_ENABLED = False
    with _PROCESS_HARDENING_LOCK:
        _PROCESS_HARDENED = False


def _linux_prctl(option: int, argument: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = int(libc.prctl(option, argument, 0, 0, 0))
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number or errno.EPERM, "prctl failed")
    return result


def _disable_process_core_dumps() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def harden_gateway_process_for_writer_boundary() -> bool:
    """Make exact-PID authorization resistant to same-UID child injection."""

    global _PROCESS_HARDENED
    config = frozen_writer_boundary_config()
    if not config.enabled:
        return False
    with _PROCESS_HARDENING_LOCK:
        if _PROCESS_HARDENED:
            return True
        if sys.platform != "linux":
            raise RuntimeError(
                "canonical_writer_boundary_requires_linux_process_hardening"
            )
        _disable_process_core_dumps()
        _linux_prctl(_PR_SET_DUMPABLE, 0)
        if _linux_prctl(_PR_GET_DUMPABLE, 0) != 0:
            raise RuntimeError("gateway_process_remains_dumpable")
        _PROCESS_HARDENED = True
        return True


def writer_service_context() -> WriterServiceContext | None:
    """Return the service-only request context, if one was authenticated."""

    return _SERVICE_CONTEXT.get()


def in_writer_service() -> bool:
    return writer_service_context() is not None


def require_writer_database() -> Any:
    """Return the private database adapter only inside the writer service."""

    context = writer_service_context()
    if context is None:
        raise PermissionError("canonical_database_access_requires_writer_service")
    return context.database


@contextlib.contextmanager
def authenticated_writer_service_scope(
    *,
    database: Any,
    runtime: Mapping[str, Any],
    peer_pid: int,
    peer_uid: int,
) -> Iterator[WriterServiceContext]:
    """Bind private dependencies after the server authenticated the peer.

    This function is intentionally service plumbing, not an authentication
    primitive.  The Unix peer credential and exact systemd MainPID checks must
    succeed before the service entry point calls it.
    """

    if database is None:
        raise ValueError("writer database adapter is required")
    if int(peer_pid) <= 1 or int(peer_uid) < 0:
        raise ValueError("authenticated writer peer identity is invalid")
    clean_runtime = {
        str(key): str(value)[:_MAX_RUNTIME_VALUE_CHARS]
        for key, value in runtime.items()
        if value is not None
    }
    context = WriterServiceContext(
        database=database,
        runtime=clean_runtime,
        peer_pid=int(peer_pid),
        peer_uid=int(peer_uid),
    )
    token = _SERVICE_CONTEXT.set(context)
    session_tokens: list[Any] = []
    try:
        from gateway.session_context import set_session_vars

        session_tokens = set_session_vars(
            platform=clean_runtime.get("platform", ""),
            source=clean_runtime.get("source", ""),
            chat_id=clean_runtime.get("chat_id", ""),
            thread_id=clean_runtime.get("thread_id", ""),
            user_id=clean_runtime.get("user_id", ""),
            session_id=clean_runtime.get("session_id", ""),
            capability_epoch_sha256=clean_runtime.get(
                "capability_epoch_sha256", ""
            ),
            message_id=clean_runtime.get("message_id", ""),
            profile=clean_runtime.get("profile", ""),
        )
        yield context
    finally:
        if session_tokens:
            from gateway.session_context import clear_session_vars

            clear_session_vars(session_tokens)
        _SERVICE_CONTEXT.reset(token)


def trusted_runtime_envelope() -> dict[str, str]:
    """Build gateway-owned runtime bindings, separate from model payload.

    Raw session keys are never sent.  The writer only needs the stable digest
    for exact capability binding and replay protection.
    """

    try:
        from gateway.session_context import get_session_env
    except Exception:
        return {}

    names = {
        "platform": "HERMES_SESSION_PLATFORM",
        "source": "HERMES_SESSION_SOURCE",
        "chat_id": "HERMES_SESSION_CHAT_ID",
        "thread_id": "HERMES_SESSION_THREAD_ID",
        "user_id": "HERMES_SESSION_USER_ID",
        "message_id": "HERMES_SESSION_MESSAGE_ID",
        "session_id": "HERMES_SESSION_ID",
        "capability_epoch_sha256": "HERMES_CAPABILITY_EPOCH_SHA256",
        "profile": "HERMES_SESSION_PROFILE",
    }
    envelope = {
        key: str(get_session_env(env_name, "") or "").strip()[:_MAX_RUNTIME_VALUE_CHARS]
        for key, env_name in names.items()
    }
    session_key = str(get_session_env("HERMES_SESSION_KEY", "") or "")
    if session_key:
        envelope["session_key_sha256"] = hashlib.sha256(
            session_key.encode("utf-8", errors="strict")
        ).hexdigest()
    return {key: value for key, value in envelope.items() if value}


def canonical_writer_call(
    operation: Any,
    payload: Mapping[str, Any],
    *,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Call one typed writer operation through the cached credential-free client."""

    if in_writer_service():
        raise RuntimeError("writer_service_must_dispatch_locally")
    config = frozen_writer_boundary_config()
    if not config.enabled:
        raise RuntimeError("canonical_writer_boundary_not_configured")
    from gateway.canonical_writer_client import (
        CanonicalWriterClient,
        ExactServerMainPidAuthorizer,
        SystemctlServerMainPidProvider,
    )

    with _CLIENT_LOCK:
        client = _CLIENTS.get(config)
        if client is None:
            try:
                import pwd
            except ImportError as exc:
                raise RuntimeError(
                    "canonical_writer_requires_linux_peer_credentials"
                ) from exc
            try:
                writer_uid = pwd.getpwnam(DEFAULT_WRITER_USER).pw_uid
            except KeyError as exc:
                raise RuntimeError(
                    "canonical_writer_service_identity_missing"
                ) from exc
            client = CanonicalWriterClient(
                config.socket_path,
                connect_timeout_seconds=config.connect_timeout_seconds,
                request_timeout_seconds=config.request_timeout_seconds,
                server_authorizer=ExactServerMainPidAuthorizer(
                    server_unit=config.writer_unit,
                    expected_server_uid=writer_uid,
                    main_pid_provider=SystemctlServerMainPidProvider(),
                ),
            )
            _CLIENTS[config] = client
    result = client.call(
        operation,
        payload,
        runtime=trusted_runtime_envelope(),
        timeout_seconds=config.request_timeout_seconds,
        idempotency_key=idempotency_key,
    )
    return {
        "request_id": result.request_id,
        "status": result.status,
        **dict(result.result),
    }


def close_canonical_writer_clients() -> None:
    """Close cached clients during process shutdown and in isolated tests."""

    with _CLIENT_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass
