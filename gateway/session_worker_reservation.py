"""Private owner-side reservation for a concrete already-started admission.

The producer retains the Popen handle and holds the child's bootstrap pipe until
this returns. Public worker.register remains idle-only for unreserved callers.
"""
import os
import secrets
import subprocess
from dataclasses import asdict

from gateway.session_persistence import authority_persistence, owner_rule
from persistence.core.errors import AuthorityRefusal

# A venv launcher (uv's Windows python.exe, pip's exe shim) sits between the owner's
# handle and the interpreter; the real worker is never further away than this.
MAX_LAUNCHER_HOPS = 3


def verify_worker_identity(process, hello):
    """Return the verified ``(pid, birth)`` of the interpreter behind ``process``.

    ``Popen.pid`` may be a launcher trampoline, so the worker reports its own pid and
    birth in a ``hello`` frame; the OWNER proves that pid is alive, carries that birth,
    and descends from (or is) the reserved handle. The worker never decides equality.
    """
    import psutil
    if not isinstance(process, subprocess.Popen) or process.poll() is not None:
        raise AuthorityRefusal('worker_not_live')
    if (not isinstance(hello, dict) or hello.get('type') != 'hello' or type(hello.get('pid')) is not int
            or type(hello.get('birth')) not in (int, float) or not isinstance(hello.get('ancestors'), list)
            or any(type(p) is not int for p in hello['ancestors'])):
        raise AuthorityRefusal('invalid_worker_frame')
    pid = hello['pid']
    if pid <= 0 or pid == os.getpid():
        raise AuthorityRefusal('permission_denied')
    try:
        handle = psutil.Process(process.pid)
        if handle.ppid() != os.getpid() or handle.status() == psutil.STATUS_ZOMBIE:
            raise AuthorityRefusal('permission_denied')
        worker = psutil.Process(pid)
        from hermes_cli.process_identity import worker_birth
        birth = worker_birth(worker)  # kernel start: stable across clock steps (#140580) and btime re-anchors
        if birth != hello['birth'] or not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE:
            raise AuthorityRefusal('permission_denied')
        # psutil's parent() already refuses a recycled ppid (parent born after child).
        chain = [p.pid for p in worker.parents()[:MAX_LAUNCHER_HOPS]]
    except psutil.Error as exc:
        raise AuthorityRefusal('worker_not_live') from exc
    if pid != process.pid and process.pid not in chain:
        raise AuthorityRefusal('permission_denied')
    return pid, birth


def owner_worker_subject():
    """The subject a managed worker's adoption ticket carries: the owner's own OS identity. The
    control channel mints a worker ticket only for a same-user peer (``uid:<n>`` / ``sid:<sid>``)."""
    if os.name == 'nt':
        from gateway.runtime_bootstrap_windows import _process_sid
        return 'sid:' + _process_sid(os.getpid())
    return f'uid:{os.getuid()}'  # windows-footgun: ok — POSIX branch


def reserve_admission_worker(authority, *, admission_id, process, principal_id, hello, worker_subject=None):
    """Return a private bootstrap scope; never accept caller-selected kind/target.

    Only the owner process may reserve its live direct exec child. The admission
    supplies physical session and generation; its durable principal must match
    the producer. Do not publish the returned secret on any client event stream.

    The adoption claim binds ``worker_subject``, the identity the worker will authenticate
    as (``owner_worker_subject``), when given. An automation admission's principal is a route
    (``automation:<session key>``), never a ticket subject, so a claim bound to it refused every
    adopt and each bot DM or completion wake on a managed session ended interrupted (#144139).
    """
    authority._require_admission_open()
    pid, birth = verify_worker_identity(process, hello)
    return asdict(authority_persistence(authority).worker_executions.reserve_for_admission(
        epoch=authority.epoch, admission_id=admission_id, principal_id=principal_id,
        profile_id=authority.profile_id, pid=pid, birth=birth, secret=secrets.token_urlsafe(32),
        owner=owner_rule(), worker_subject=worker_subject))


def lose_admission_worker(authority, row, scope):
    """Revoke a private assignment and pause its FIFO; no inference replay.

    This also fences a result already in the pipe at loss/shutdown. Retention
    requires a still-started admission even when the generation has not changed.
    """
    authority_persistence(authority).worker_executions.lose_reservation(
        epoch=scope['epoch'], admission_id=row['admission_id'], execution_id=scope['execution_id'],
        generation=scope['generation'])
