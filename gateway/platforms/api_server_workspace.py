"""Authenticated workspace I/O over the selected native local terminal runtime."""
import asyncio
import base64
import binascii
import errno
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from aiohttp import web
from pydantic import TypeAdapter, ValidationError

from gateway.platforms.api_server_workspace_models import (
    MAX_CHUNK_BYTES, MAX_FILE_BYTES, MAX_OUTPUT_BYTES, MAX_TIMEOUT_SECONDS,
    FileData, MissingFile, OperationReceipt, OutputRequest, OutputResponse, PathRequest, ReadRequest,
    RemoveRequest, ResolvedPath, StartRequest,
    WorkspaceRequest, WriteRequest,
)
from gateway.platforms.api_server_workspace_store import WorkspaceConflict, WorkspaceKeyExpired, WorkspaceStore
from tools.workspace_files import WorkspaceFiles, WorkspacePathError


@dataclass
class WorkspaceService:
    files: WorkspaceFiles
    store: WorkspaceStore
    tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    processes: dict[str, object] = field(default_factory=dict)
    stopping: set[str] = field(default_factory=set)

    async def process(self, owner: str, receipt: OperationReceipt, body: StartRequest):
        from tools.file_tools import _get_file_ops
        from tools.process_registry import process_registry
        from tools.terminal_tool import _resolve_container_task_id

        try:
            env = await asyncio.to_thread(_get_file_ops, body.session_id)
            if not env.env.is_local:
                raise RuntimeError("Selected native terminal environment is not local")
            # Configuration/backend selection is complete before child-only env overrides.
            # cwd's descriptor stays open through Popen's chdir, closing the validation/use gap.
            with self.files.parent(str(Path(self.files.resolve(body.cwd)) / ".cwd-probe")) as (directory, _):
                session = await asyncio.to_thread(
                    process_registry.spawn_local, body.command, cwd=self.files.resolve(body.cwd), cwd_fd=directory,
                    task_id=_resolve_container_task_id(body.session_id),
                    owner_task_id=f"workspace:{owner}", session_key=receipt.operation_id,
                    env_vars={**env.env.env, **body.env}, shell_interactive=False,
                    max_output_bytes=body.max_output_bytes, persist_on_release=True)
            self.processes[receipt.operation_id] = session
            receipt.process_id = session.id
            receipt.status = "running"
            self.store.save(owner, receipt)
            if receipt.operation_id in self.stopping:
                await asyncio.to_thread(process_registry.kill_process, session.id, source="workspace.cancel")
            completion = asyncio.create_task(asyncio.to_thread(session._completion_event.wait))
            deadline = asyncio.get_running_loop().time() + body.timeout_seconds
            timed_out = False
            while not completion.done():
                receipt.status = "stopping" if receipt.operation_id in self.stopping else "running"
                self.store.capture(owner, receipt, session.byte_capture)
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0 and not timed_out:
                    timed_out = True
                    result = await asyncio.to_thread(process_registry.kill_process, session.id, source="workspace.timeout")
                    if result["status"] == "error":
                        receipt.status, receipt.reason = "stopping", "termination_unconfirmed"
                        self.store.save(owner, receipt)
                await asyncio.wait({completion}, timeout=0.1)
            receipt.status = ("failed" if timed_out else "cancelled"
                              if receipt.operation_id in self.stopping else "completed")
            receipt.reason = "timeout" if timed_out else "cancelled" if receipt.status == "cancelled" else session.completion_reason
            receipt.termination_confirmed = not session._scope_stop_pending
            if not receipt.termination_confirmed:
                receipt.status, receipt.reason = "interrupted", "termination_unconfirmed"
            receipt.exit_code = session.exit_code
            # Terminal receipt and exact bounded output commit in the same transaction.
            self.store.capture(owner, receipt, session.byte_capture)
        except asyncio.CancelledError:
            receipt.status, receipt.reason = "interrupted", "gateway_shutdown"
            session = self.processes.get(receipt.operation_id)
            if session is not None:
                await asyncio.to_thread(process_registry.kill_process, session.id, source="workspace.shutdown")
            raise
        except Exception:
            receipt.status, receipt.reason = "failed", "execution_failed"
            session = self.processes.get(receipt.operation_id)
            receipt.termination_confirmed = session is None
            if session is not None:
                result = await asyncio.to_thread(process_registry.kill_process, session.id, source="workspace.failure")
                receipt.termination_confirmed = result["status"] != "error"
                if not receipt.termination_confirmed:
                    receipt.status = "interrupted"
            raise
        finally:
            self.store.save(owner, receipt)
            self.processes.pop(receipt.operation_id, None)


def capability(adapter):
    from hermes_cli.config import cfg_get, load_config
    from tools.terminal_tool import _get_env_config

    config = cfg_get(load_config(), "gateway", "api_server", "workspace", default={})
    enabled = config.get("enabled", False) is True
    result = {"enabled": False, "protocol_version": 1, "reason": "disabled"}
    if not enabled:
        return result
    if not adapter._expected_api_key():
        return {**result, "reason": "authentication_required"}
    terminal = _get_env_config()
    if terminal["env_type"] != "local" or os.name != "posix":
        return {**result, "reason": "unsupported_backend", "backend": terminal["env_type"]}
    files = WorkspaceFiles(terminal["cwd"])
    result = {
        "enabled": True, "protocol_version": 1, "backend": "local", "root": files.root,
        "max_file_bytes": MAX_FILE_BYTES, "max_output_bytes_per_stream": MAX_OUTPUT_BYTES,
        "max_chunk_bytes_per_stream": MAX_CHUNK_BYTES, "max_timeout_seconds": MAX_TIMEOUT_SECONDS,
        "output_encoding": "base64", "streams": ["stdout", "stderr"],
        "output_retention": "durable_with_operation", "operation_retention": "until_acknowledged",
        "max_unacknowledged_operations": WorkspaceStore.MAX_UNACKNOWLEDGED, "max_active_processes": 8,
        "operations_per_epoch": WorkspaceStore.OPERATIONS_PER_EPOCH,
        "idempotency_key_format": "<admission_epoch>:<opaque suffix>",
        "file_operations": ["read", "write", "remove", "resolve"],
        "process_operations": ["start", "status", "output", "cancel_by_original_key"],
        "restart": "completed_output_replay_active_interrupted_no_reexecution",
    }
    result["admission_epoch"] = service_for(adapter, result).store.admission_epoch
    return result


def service_for(adapter, caps):
    from hermes_constants import get_hermes_home

    adapter._workspace_closed = False
    home = get_hermes_home()
    key = str(home)
    if key not in adapter._workspace_services:
        from tools.process_registry import process_registry
        process_registry.recover_from_checkpoint()
        adapter._workspace_services[key] = WorkspaceService(
            WorkspaceFiles(caps["root"]), WorkspaceStore(home / "workspace_operations.db"))
    service = adapter._workspace_services[key]
    if service.files.root != caps["root"]:
        raise WorkspaceConflict("Workspace root changed; restart the gateway before admitting work")
    return service


def active_count(adapter):
    if hasattr(adapter, "_workspace_services") and not getattr(adapter, "_workspace_closed", False):
        capability(adapter)  # Restore unresolved receipts before fleet idle/drain decisions.
    return sum(sum(not task.done() for task in service.tasks.values()) + len(service.store.unresolved())
               for service in getattr(adapter, "_workspace_services", {}).values())


def interrupt(adapter, reason):
    from tools.process_registry import process_registry

    count = 0
    for service in getattr(adapter, "_workspace_services", {}).values():
        for operation_id, task in service.tasks.items():
            if not task.done():
                service.stopping.add(operation_id)
                count += 1
        for session in service.processes.values():
            if not session.exited:
                adapter._track_background_task(asyncio.create_task(asyncio.to_thread(
                    process_registry.kill_process, session.id, source=reason)))
    return count


async def close(adapter):
    from tools.process_registry import process_registry

    services = getattr(adapter, "_workspace_services", {})
    for service in services.values():
        for operation_id in service.tasks:
            service.stopping.add(operation_id)
        for session in service.processes.values():
            await asyncio.to_thread(process_registry.kill_process, session.id, source="workspace.shutdown")
        await asyncio.gather(*service.tasks.values(), return_exceptions=True)
        service.store.close()
    services.clear()
    adapter._workspace_closed = True


def error(kind, message, status=400):
    return web.json_response({"error": {"type": "workspace_error", "code": kind, "message": message}}, status=status)


async def handle(adapter, action, request):
    """Authenticate before parsing or resolving a runtime; scope comes from native middleware."""
    from gateway.platforms.api_server import _reserve_pending_api_work

    auth = adapter._check_auth(request)
    if auth is not None:
        return auth
    caps = capability(adapter)
    if not caps["enabled"]:
        return error(caps["reason"], "Workspace I/O is unavailable for this runtime", 503)
    if request.query:
        return error("invalid_request", "Query parameters are not supported")
    models = {"read": ReadRequest, "write": WriteRequest, "remove": RemoveRequest,
              "resolve": PathRequest, "start": StartRequest, "output": OutputRequest,
              "status": WorkspaceRequest, "cancel": WorkspaceRequest, "acknowledge": WorkspaceRequest}
    with _reserve_pending_api_work(adapter):
        adapter._reset_metrics_if_needed()
        adapter._metrics_requests_today += 1
        adapter._metrics_last_request_at = time.time()
        adapter._publish_runtime_status()
        try:
            body = models[action].model_validate(await request.json())
            db = await adapter._ensure_session_db_async()
            if db is None:
                return adapter._session_db_unavailable()
            if await asyncio.to_thread(db.get_session, body.session_id) is None:
                return error("session_not_found", "Unknown session", 404)
            service = service_for(adapter, caps)
            owner = hashlib.sha256(f"{adapter._run_idempotency_scope(request)}\0{body.session_id}".encode()).hexdigest()
            return await dispatch(adapter, service, owner, action, body, request)
        except WorkspaceKeyExpired as exc:
            return error("workspace_key_expired", str(exc), 410)
        except (ValidationError, ValueError, binascii.Error) as exc:
            return error("conflict" if isinstance(exc, WorkspaceConflict) else "invalid_request", str(exc),
                         409 if isinstance(exc, WorkspaceConflict) else 400)
        except OSError as exc:
            code = {errno.ENOENT: "missing", errno.EACCES: "permission_denied", errno.EPERM: "permission_denied",
                    errno.ELOOP: "symlink_denied", errno.ENOTDIR: "invalid_path"}.get(exc.errno, "filesystem_error")
            return error(code, os.strerror(exc.errno), 404 if exc.errno == errno.ENOENT else 400)


async def dispatch(adapter, service, owner, action, body, request):
    if action == "read":
        result = await asyncio.to_thread(service.files.read, body.path, body.max_bytes)
        return web.json_response(TypeAdapter(FileData | MissingFile).validate_python(result).model_dump())
    if action == "resolve":
        return web.json_response(ResolvedPath(path=service.files.resolve(body.path)).model_dump())
    if action in {"status", "output"}:
        receipt = service.store.get(owner, request.match_info["operation_id"])
        if receipt is None:
            return error("operation_not_found", "Unknown operation", 404)
        if action == "status":
            return web.json_response(receipt.model_dump())
        return output(service, owner, receipt, body)
    key = request.headers.get("Idempotency-Key", "")
    if action == "acknowledge":
        receipt = service.store.acknowledge(owner, key, request.match_info["operation_id"])
        service.tasks.pop(receipt.operation_id, None)
        service.processes.pop(receipt.operation_id, None)
        service.stopping.discard(receipt.operation_id)
        return web.json_response(receipt.model_dump())
    if action == "cancel":
        receipt, created = service.store.reserve(owner, key, "process", body, cancel=True)
        service.stopping.add(receipt.operation_id)
        if not created and receipt.status in {"reserved", "running", "stopping", "interrupted"}:
            recovering = receipt.status == "interrupted"
            receipt.status = "stopping"
            service.store.save(owner, receipt)
            from tools.process_registry import process_registry
            session = service.processes.get(receipt.operation_id)
            if session is None and receipt.process_id:
                session = process_registry.get(receipt.process_id)
            if session is not None and session.owner_task_id == f"workspace:{owner}" and session.session_key == receipt.operation_id:
                result = await asyncio.to_thread(process_registry.kill_process, session.id, source="workspace.cancel")
                if result["status"] == "error":
                    logging.getLogger(__name__).warning("Workspace cancellation unconfirmed: %s", result)
                    receipt.status, receipt.reason = "interrupted", "termination_unconfirmed"
                    service.store.save(owner, receipt)
                if recovering and result["status"] != "error":
                    receipt.status, receipt.reason = "interrupted", "cancelled_after_restart"
                    receipt.termination_confirmed = True
                    service.store.save(owner, receipt)
            elif receipt.operation_id not in service.tasks:
                receipt.status, receipt.reason = "interrupted", "termination_unconfirmed"
                service.store.save(owner, receipt)
        return web.json_response(service.store.get(owner, receipt.operation_id).model_dump())
    kind = "process" if action == "start" else action
    existing = service.store.by_key(owner, key)
    if existing is None:
        draining = adapter._draining_response()
        if draining is not None:
            return draining
        if action == "start" and sum(not task.done() for task in service.tasks.values()) >= 8:
            return error("capacity", "Workspace process capacity reached", 429)
    receipt, created = service.store.reserve(owner, key, kind, body)
    if not created:
        return web.json_response(receipt.model_dump())
    if action == "start":
        task = asyncio.create_task(service.process(owner, receipt, body))
        service.tasks[receipt.operation_id] = task
        task.add_done_callback(lambda finished: finished.result() if not finished.cancelled() else None)
        return web.json_response(receipt.model_dump(), status=202)
    task = asyncio.create_task(mutate(service, owner, receipt, action, body))
    service.tasks[receipt.operation_id] = task
    task.add_done_callback(lambda finished: finished.result() if not finished.cancelled() else None)
    # Disconnected clients do not release a still-executing filesystem operation from drain.
    await asyncio.shield(task)
    return web.json_response(receipt.model_dump())


async def mutate(service, owner, receipt, action, body):
    try:
        if action == "write":
            data = base64.b64decode(body.data_base64, validate=True)
            if len(data) > MAX_FILE_BYTES:
                raise WorkspacePathError("File exceeds maximum bytes")
            await asyncio.to_thread(service.files.write, body.path, data)
        else:
            await asyncio.to_thread(service.files.remove, body.path, force=body.force, recursive=body.recursive)
        receipt.status = "completed"
    except BaseException:
        receipt.status, receipt.reason = "interrupted", "mutation_incomplete"
        raise
    finally:
        service.store.save(owner, receipt)


def output(service, owner, receipt, body):
    session = service.processes.get(receipt.operation_id)
    if session is not None and receipt.status in {"running", "stopping"}:
        service.store.capture(owner, receipt, session.byte_capture)
    if receipt.acknowledged:
        return error("output_acknowledged", "Caller acknowledged the retained outcome", 410)
    if not receipt.output_available:
        return error("output_unavailable", "No output snapshot was retained", 409)
    streams = service.store.output(owner, receipt.operation_id)
    payload = {"operation_id": receipt.operation_id, "status": receipt.status,
               "complete": receipt.status in {"completed", "failed", "cancelled"} and receipt.termination_confirmed,
               "partial": receipt.status == "interrupted"}
    for stream in ("stdout", "stderr"):
        cursor = getattr(body, f"{stream}_cursor")
        if cursor > len(streams[stream]):
            return error("invalid_cursor", "Cursor exceeds retained output")
        data = streams[stream][cursor:cursor + body.max_bytes]
        payload[stream] = {"data_base64": base64.b64encode(data).decode("ascii"),
                           "next_cursor": cursor + len(data), "truncated": getattr(receipt, f"{stream}_truncated")}
    return web.json_response(OutputResponse.model_validate(payload).model_dump())


def http_routes(adapter):
    routes = [("POST", f"/api/workspace/files/{action}", partial(handle, adapter, action))
              for action in ("read", "write", "remove", "resolve")]
    routes.extend([
        ("POST", "/api/workspace/processes", partial(handle, adapter, "start")),
        ("POST", "/api/workspace/processes/cancel", partial(handle, adapter, "cancel")),
        ("POST", "/api/workspace/operations/{operation_id}/acknowledge", partial(handle, adapter, "acknowledge")),
        ("POST", "/api/workspace/processes/{operation_id}/status", partial(handle, adapter, "status")),
        ("POST", "/api/workspace/processes/{operation_id}/output", partial(handle, adapter, "output")),
    ])
    return routes
