"""Activity marker and fair request queue for the local MLX backbone.

The local MLX server is effectively a shared single-lane resource.  A listener
health check can be green while a long prompt is still monopolizing generation,
so local callers coordinate through a cross-process FIFO before opening HTTP.
The active marker then lets external jobs distinguish "server down" from "busy".
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator
from urllib.parse import urlparse
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - local MLX is currently macOS/Linux only
    fcntl = None


DEFAULT_BACKBONE_PORT = 10240


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def is_local_mlx_backbone_url(base_url: Any) -> bool:
    raw = str(base_url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    except Exception:
        return False
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return False
    return parsed.port == _env_int("HERMES_LOCAL_MLX_BACKBONE_PORT", DEFAULT_BACKBONE_PORT)


def activity_dir() -> Path:
    configured = os.environ.get("HERMES_LOCAL_MLX_ACTIVITY_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return home / "workspace" / "state" / "mlx-lifecycle"


def active_request_path() -> Path:
    return activity_dir() / "local-mlx-active-request.json"


def request_events_path() -> Path:
    return activity_dir() / "local-mlx-request-events.jsonl"


def request_queue_dir(base_url: Any) -> Path:
    raw = str(base_url or "").strip()
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    port = parsed.port or DEFAULT_BACKBONE_PORT
    return activity_dir() / "request-queue" / f"lane-{port}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def _active_marker_lock() -> Iterator[None]:
    path = activity_dir() / "local-mlx-active-request.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def _append_event(payload: dict[str, Any]) -> None:
    path = request_events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_active(path: Path | None = None) -> dict[str, Any] | None:
    target = path or active_request_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def current_active_request(*, stale_after_s: float | None = None) -> dict[str, Any] | None:
    path = active_request_path()
    with _active_marker_lock():
        data = _read_active(path)
        if not data:
            return None
        if not _pid_alive(data.get("pid")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return None
        if stale_after_s is not None:
            try:
                started = float(data.get("start_monotonic_s"))
            except (TypeError, ValueError):
                started = 0.0
            if started and time.monotonic() - started > stale_after_s:
                return None
        return data


def _pid_alive(pid: Any) -> bool:
    # psutil.pid_exists instead of os.kill(pid, 0): the signal-0 probe is NOT
    # a no-op on Windows (it sends CTRL_C_EVENT to the target's console
    # process group — bpo-14484), and psutil is already a core dependency.
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    if parsed <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(parsed))
    except Exception:
        return False


def _read_ticket(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _prune_stale_tickets(queue_dir: Path, *, now: float) -> None:
    stale_after = max(
        float(_env_int("HERMES_LOCAL_MLX_QUEUE_TIMEOUT_SECONDS", 1800)) * 2,
        7200.0,
    )
    for ticket in queue_dir.glob("ticket-*.json"):
        data = _read_ticket(ticket)
        try:
            age = now - float((data or {}).get("created_at_epoch", 0.0))
        except (TypeError, ValueError):
            age = stale_after + 1
        if data and _pid_alive(data.get("pid")) and age <= stale_after:
            continue
        try:
            ticket.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _local_mlx_fair_slot(
    *,
    base_url: Any,
    model: Any,
    request_kind: str,
    session_id: Any,
    cancel_check=None,
) -> Iterator[float]:
    """Acquire the local decoder lane in durable FIFO order."""
    if fcntl is None or not _env_enabled("HERMES_LOCAL_MLX_FAIR_QUEUE", True):
        yield 0.0
        return

    queue_dir = request_queue_dir(base_url)
    queue_dir.mkdir(parents=True, exist_ok=True)
    created_at = time.time()
    ticket_id = uuid.uuid4().hex
    ticket = queue_dir / (
        f"ticket-{time.time_ns():020d}-{os.getpid():010d}-{ticket_id}.json"
    )
    ticket_payload = {
        "kind": "local-mlx-queue-ticket",
        "ticket_id": ticket_id,
        "base_url": str(base_url or ""),
        "model": str(model or ""),
        "request_kind": str(request_kind or "unknown"),
        "session_id": str(session_id or ""),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "created_at": _utc_now(),
        "created_at_epoch": created_at,
    }
    ticket.write_text(
        json.dumps(ticket_payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_event({**ticket_payload, "event": "queue_enqueued"})

    timeout_s = float(_env_int("HERMES_LOCAL_MLX_QUEUE_TIMEOUT_SECONDS", 1800))
    poll_s = max(
        0.02,
        min(float(_env_int("HERMES_LOCAL_MLX_QUEUE_POLL_MS", 100)) / 1000.0, 1.0),
    )
    lock_handle = (queue_dir / "decoder.lock").open("a+b")
    acquired = False
    last_prune = 0.0
    waited_s = 0.0
    try:
        while True:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Local MLX queue wait interrupted")

            now = time.time()
            waited_s = now - created_at
            if timeout_s > 0 and waited_s > timeout_s:
                raise TimeoutError(
                    f"Timed out after {int(waited_s)}s waiting for local MLX decoder lane"
                )

            if now - last_prune >= 1.0:
                _prune_stale_tickets(queue_dir, now=now)
                last_prune = now

            tickets = sorted(queue_dir.glob("ticket-*.json"), key=lambda p: p.name)
            if tickets and tickets[0] == ticket:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    acquired = True
                    try:
                        ticket.unlink()
                    except FileNotFoundError:
                        pass
                    _append_event({
                        **ticket_payload,
                        "event": "queue_acquired",
                        "wait_s": round(waited_s, 3),
                    })
                    break
            time.sleep(poll_s)

        yield waited_s
    finally:
        try:
            ticket.unlink()
        except FileNotFoundError:
            pass
        if acquired:
            _append_event({
                **ticket_payload,
                "event": "queue_released",
                "wait_s": round(waited_s, 3),
            })
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        lock_handle.close()


@contextmanager
def local_mlx_request_activity(
    *,
    base_url: Any,
    model: Any,
    estimated_tokens: int,
    request_kind: str,
    session_id: Any = None,
    cancel_check=None,
) -> Iterator[None]:
    """Queue and mark a local :10240 request active for external probes.

    The queue is per server port and cross-process. It releases after each model
    response, so long tool-using sessions alternate instead of monopolizing the
    single decoder for their whole conversation.
    """
    if not is_local_mlx_backbone_url(base_url):
        yield
        return

    with _local_mlx_fair_slot(
        base_url=base_url,
        model=model,
        request_kind=request_kind,
        session_id=session_id,
        cancel_check=cancel_check,
    ) as queue_wait_s:
        request_id = uuid.uuid4().hex
        start = time.monotonic()
        payload = {
            "kind": "local-mlx-active-request",
            "request_id": request_id,
            "status": "active",
            "base_url": str(base_url or ""),
            "model": str(model or ""),
            "estimated_tokens": int(estimated_tokens or 0),
            "request_kind": str(request_kind or "unknown"),
            "session_id": str(session_id or ""),
            "queue_wait_s": round(queue_wait_s, 3),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "started_at": _utc_now(),
            "start_monotonic_s": start,
        }
        with _active_marker_lock():
            _write_json_atomic(active_request_path(), payload)
        _append_event({**payload, "event": "start"})
        error_type = ""
        try:
            yield
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            duration = round(time.monotonic() - start, 3)
            end_event = {
                **payload,
                "event": "end",
                "status": "finished" if not error_type else "error",
                "ended_at": _utc_now(),
                "duration_s": duration,
                "error_type": error_type,
            }
            _append_event(end_event)

            path = active_request_path()
            with _active_marker_lock():
                current = _read_active(path)
                if current and current.get("request_id") == request_id:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
