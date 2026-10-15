"""Pure identity derivations the canonical store and the gateway both need.

stdlib-only on purpose: ``hermes_state_*`` modules must not import ``gateway/`` to compute a
digest or parse a key (the gateway imports the store, so the reverse edge is a latent cycle).
The gateway modules that historically defined these re-export them from here.
"""
import hashlib
import json
import re

# The profile scheduler's principal admits every cron fire under ``cron:<job>:<execution>``
# (``gateway/session_cron.py``). That request id is the pre-image of the run's ``local-*`` session
# id, so its admission row is the run's durable job/execution association. Runs from before the
# unified runtime were named ``cron_{job_id}_{YYYYmmdd_HHMMSS}`` (``cron/scheduler.py::run_job``).
CRON_PRINCIPAL_ID = 'cron-owner'
_CRON_REQUEST_PREFIX = 'cron:'
_LEGACY_CRON_RUN_ID = re.compile(r'^cron_(.+)_\d{8}_\d{6}$')


def cron_request_id(job_id, execution_id):
    """The admission request id of one cron fire (``execution_id`` is its ledger row id)."""
    return f'{_CRON_REQUEST_PREFIX}{job_id}:{execution_id}'


def cron_request_range(job_id):
    """Half-open ``[lo, hi)`` bounds of *job_id*'s cron request ids, for an index range scan."""
    lo = cron_request_id(job_id, '')
    return lo, lo[:-1] + chr(ord(lo[-1]) + 1)


def cron_run_from_request(request_id):
    """``(job_id, execution_id)`` of a cron fire's request id, else None. Execution ids are ledger
    hex and never hold ``:``, so the last one splits even a job id that contains one."""
    if not isinstance(request_id, str) or not request_id.startswith(_CRON_REQUEST_PREFIX):
        return None
    job_id, sep, execution_id = request_id[len(_CRON_REQUEST_PREFIX):].rpartition(':')
    return (job_id, execution_id) if sep and job_id and execution_id else None


def legacy_cron_run_job(session_id):
    """The job id a pre-unified ``cron_{job_id}_{YYYYmmdd_HHMMSS}`` run session names, else None."""
    match = _LEGACY_CRON_RUN_ID.match(str(session_id or ''))
    return match.group(1) if match else None


def admission_fingerprint(*, canonical_target: str, payload: dict) -> str:
    """Canonical payload identity after authorization and immutable media capture."""
    encoded = json.dumps(
        {'target': canonical_target, 'payload': payload}, ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8', errors='surrogatepass')
    return hashlib.sha256(encoded).hexdigest()


def local_identity(profile_id, principal_id, request_id):
    """The deterministic session id of a LOCAL creation receipt."""
    identity = json.dumps([profile_id, principal_id, request_id], separators=(',', ':'))
    return 'local-' + hashlib.sha256(identity.encode()).hexdigest()


def profile_from_session_key_namespace(namespace: str) -> str:
    """Inverse of ``gateway.session._session_key_namespace`` for the ``<ns>`` slot of a key:
    ``"default"`` for ``main``, ``"main"`` for the marked ``main~``, else the slot is the profile id."""
    if namespace == "main":
        return "default"
    return "main" if namespace == "main~" else namespace


def profile_from_session_key(session_key):
    """Profile namespace encoded in an ``agent:<ns>:...`` gateway session key, else None."""
    if not session_key:
        return None
    parts = str(session_key).split(":")
    if len(parts) < 2 or parts[0] != "agent":
        return None
    return profile_from_session_key_namespace(parts[1] or "main")
