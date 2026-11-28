"""Exact-input, store-local idempotency for cron creation (no tombstones)."""
import hashlib
import inspect
import json
import re
from functools import wraps


# Request-transport facts the api_server records on ``origin`` for audit. A retry over a new
# connection (new peer address, different proxy hop) is still the same plan, so they stay out of
# the fingerprint; every other origin field (platform, chat, thread, principal) stays in.
TRANSPORT_ORIGIN_KEYS = frozenset({"source_ip", "peer_ip", "forwarded_for", "real_ip", "user_agent"})


def creation_fingerprint(inputs):
    """Hash original arguments, before normalization, clock reads or config defaults."""
    payload = {k: v for k, v in inputs.items() if k != "external_idempotency_key"}
    if isinstance(payload.get("origin"), dict):
        payload["origin"] = {k: v for k, v in payload["origin"].items() if k not in TRANSPORT_ORIGIN_KEYS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return "v1:" + hashlib.sha256(encoded).hexdigest()


def idempotent_creation(create):
    """Keep lookup and the existing creator's write inside the same file lock."""
    signature = inspect.signature(create)

    @wraps(create)
    def wrapped(*args, **kwargs):
        from cron import jobs

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = bound.arguments["external_idempotency_key"]
        if key is None:
            return create(*args, **kwargs)
        if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", key) is None:
            raise ValueError("cron_idempotency_invalid_key")
        fingerprint = creation_fingerprint(bound.arguments)
        with jobs._jobs_lock():
            # The legacy store permits degraded locking; keyed creation must fail closed.
            if (not getattr(jobs._jobs_lock_state, "crossprocess_locked", False)
                    or getattr(jobs._jobs_lock_state, "crossprocess_lock_path", None)
                    != jobs._jobs_lock_file().resolve()):
                raise RuntimeError("cron_idempotency_lock_unavailable")
            for job in jobs.load_jobs():
                if job.get("external_idempotency_key") == key:
                    if job.get("external_idempotency_fingerprint") != fingerprint:
                        raise ValueError("cron_idempotency_conflict")
                    return jobs._normalize_job_record(job)
            return create(*args, **kwargs)

    return wrapped
