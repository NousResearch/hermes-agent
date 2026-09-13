"""Owner-bound consent and durable setup jobs, scoped to the serving profile."""
import contextvars
import fcntl  # windows-footgun: ok — Linux-only plugin
import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid

from .lifecycle import atomic_json
from .setup_plan import build_plan, confined
from .setup_worker import install_plan, run_child, verify_ready


def _root(service, *, create=False):
    root = confined(service.home, service.home / "plugin-data" / "hermes-realms" / "setup-jobs")
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _consent(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(service, owner, kind):
    _root(service)
    plan = build_plan(service, owner, kind)
    return {key: plan[key] for key in ("kind", "ready", "action", "summary", "details")} | {"consent": _consent(plan)}


def _path(service, job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid setup job id")
    return confined(service.home, _root(service) / (job_id + ".json"))


def _read(service, job_id):
    path = _path(service, job_id)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)  # windows-footgun: ok — Linux-only plugin
    with os.fdopen(fd, encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("id") != job_id or record.get("home") != str(service.home):
        raise PermissionError("Setup job ownership mismatch")
    return record


def _public(record):
    return {key: record[key] for key in ("id", "kind", "state", "message", "error") if key in record}


def _lock(service):
    path = confined(service.home, _root(service, create=True) / "active.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)  # windows-footgun: ok — Linux-only plugin
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ValueError("A setup job is already active in this profile") from None
    return fd


def _release(fd):
    # A lifetime supervisor may retain this same open-file description until
    # its owned units stop. LOCK_UN would release its exclusion as well.
    os.close(fd)


def status(service, owner, job_id):
    record = _read(service, job_id)
    if record.get("owner") != owner:
        raise PermissionError("Setup job ownership mismatch")
    if record["state"] == "running":
        # A process crash releases flock. No PID reuse assumptions or polling writes.
        path = confined(service.home, _root(service) / "active.lock")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)  # windows-footgun: ok — Linux-only plugin
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # The active receipt distinguishes a later job from this one.
                active = confined(service.home, _root(service) / "active.json")
                current = json.loads(active.read_text(encoding="utf-8")).get("id")
                interrupted = current != job_id
            else:
                # Reread after acquiring: completion may have raced the first read.
                record = _read(service, job_id)
                interrupted = record["state"] == "running"
            if interrupted:
                record.update(state="failed", message="Setup was interrupted. Review prerequisites and retry.", error="interrupted")
        finally:
            os.close(fd)
    return _public(record)


def latest(service, owner):
    root = _root(service)
    if not root.exists():
        return None
    owned = []
    for path in root.glob("*.json"):
        if re.fullmatch(r"[0-9a-f]{32}", path.stem):
            record = _read(service, path.stem)
            if record.get("owner") == owner:
                owned.append(record)
    return status(service, owner, max(owned, key=lambda r: r["created_at"])["id"]) if owned else None


_PHASES = {"packages": "Installing approved system packages…", "install": "Installing and verifying realm prerequisites…", "verify": "Verifying readiness…", "start": "Starting this conversation's realm…"}


def start(service, owner, kind, consent, identity):
    # No writes (including lock creation) until the exact proposal is accepted.
    _root(service)
    plan = build_plan(service, owner, kind)
    if not isinstance(consent, str) or not hmac.compare_digest(consent, _consent(plan)):
        raise ValueError("Setup consent does not match the current proposal; prepare again")
    if plan["blockers"]:
        raise ValueError("Setup prerequisites need attention; review the proposal before retrying")
    identity = {key: value for key, value in identity.items() if value is not None}
    if set(identity) - {"runtime_session_id", "stored_session_id"}:
        raise PermissionError("Invalid setup ownership identity")
    if service.owners.resolve(session_id=owner, **identity) != owner:
        raise PermissionError("Setup ownership mismatch")
    lock = _lock(service)
    record = {"id": uuid.uuid4().hex, "owner": owner, "home": str(service.home), "kind": kind,
              "state": "running", "message": "Preparing realm setup…", "created_at": time.time()}
    try:
        record["activation_generation"] = service.reserve_setup(owner)
        atomic_json(_path(service, record["id"]), record)
        atomic_json(confined(service.home, _root(service) / "active.json"), {"id": record["id"]})

        def work():
            def phase(name):
                record["message"] = _PHASES[name]
                atomic_json(_path(service, record["id"]), record)
            try:
                install_plan(plan, phase, lock_fd=lock)
                phase("verify")
                verify_ready(service.home, kind)
                phase("start")
                # Durable aliases identify the owner but do not grant an
                # activation lease. The integration atomically checks revocation
                # and frozen config before starting either kind of desktop.
                service.activate_setup(owner, kind, plan["config"],
                                       record["activation_generation"], identity)
                record.update(state="succeeded", message="Realm is ready for this conversation.")
            except Exception:
                # Installer/SSH exceptions may contain generated passwords or tokens.
                record.update(state="failed", message="Realm setup failed. Check prerequisites and administrator authorization, then retry.", error="setup_failed")
            finally:
                try:
                    atomic_json(_path(service, record["id"]), record)
                finally:
                    try:
                        service.finish_setup(owner)
                    finally:
                        _release(lock)

        context = contextvars.copy_context()
        threading.Thread(target=context.run, args=(work,), name="realms-setup-" + record["id"], daemon=True).start()
    except BaseException:
        try:
            service.finish_setup(owner)
        finally:
            _release(lock)
        raise
    return _public(record.copy())
