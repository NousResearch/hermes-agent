"""Private invocation v1 issuer journal. Trusted host code only; no model tools."""
import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager

DESCRIPTOR = {"version": 1, "header": "Private-Invocation", "directCallOnly": True}
DIGEST_VERSION = "rfc8785-sha256-v1"
BINDING_KEYS = frozenset(("version", "issuerNamespace", "invocationId", "operationId",
    "recipientPrincipal", "appId", "buildId", "toolPath", "selectedAccountsDigest",
    "action", "inputDigest", "expiresAt"))


def canonical(value):
    """Exact RFC8785 integer-only I-JSON subset; refuse floats/unsafe integers.

    Unicode scalar strings, UTF-16 property ordering, shortest JSON escapes,
    safe integer decimal representation. No alternative serializer fallback.
    """
    if value is None or type(value) is bool:
        return json.dumps(value)
    if type(value) is int and abs(value) <= 9007199254740991:
        return str(value)
    if type(value) is str:
        value.encode("utf-8", errors="strict")
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if type(value) is list:
        return "[" + ",".join(canonical(x) for x in value) + "]"
    if type(value) is dict and all(type(k) is str for k in value):
        keys = sorted(value, key=lambda k: k.encode("utf-16-be", errors="strict"))
        return "{" + ",".join(canonical(k) + ":" + canonical(value[k]) for k in keys) + "}"
    raise ValueError("unsupported RFC8785 input (integer-only I-JSON required)")


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


_REFERENCE_SEAL = object()


class PrivateReference:
    """Host-private locator without bearer export or JSON conversion."""
    __slots__ = ("_value",)

    def __init__(self, seal, value):
        if seal is not _REFERENCE_SEAL:
            raise PermissionError("issuer-owned retained reference required")
        self._value = value

    def __repr__(self):
        return "<private invocation reference>"

    def __reduce__(self):
        raise TypeError("private capability is not serializable")


class Issuer:
    """Durable issuer. policy(origin,target,action) revalidates connector and scope.

    Targets and peer certificate SHA256-to-principal map are trusted operator
    configuration. They never supply native source authority.
    """
    def __init__(self, path, namespace, targets, policy, peer_principals, *, forward_context=None):
        self.path, self.namespace = path, namespace
        self.targets, self.policy = targets, policy
        self.peer_principals = dict(peer_principals)
        self.forward_context = forward_context
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS delegations(
                  ref TEXT PRIMARY KEY, origin TEXT NOT NULL, job TEXT NOT NULL UNIQUE,
                  actions TEXT NOT NULL, expires INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS operations(
                  ref TEXT PRIMARY KEY, delegation TEXT NOT NULL, operation TEXT NOT NULL,
                  target TEXT NOT NULL, binding TEXT NOT NULL, status TEXT NOT NULL,
                  UNIQUE(delegation, operation));
                CREATE TABLE IF NOT EXISTS permits(
                  lease TEXT PRIMARY KEY, operation_ref TEXT NOT NULL UNIQUE,
                  peer TEXT NOT NULL, status TEXT NOT NULL);
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _retain_native(self, capability, job, actions, expires):
        from hermes_cli.private_invocations import _NativeOrigin
        from tools.registry import get_current_host_context
        if (type(capability) is not _NativeOrigin or not capability.active
                or get_current_host_context() is not capability):
            raise PermissionError("current native admission required")
        origin = capability.origin
        now = int(time.time())
        if (type(expires) is not int or not isinstance(job, str) or not job or not actions
                or any(type(a) is not str or a != 'call' for a in actions)
                or not now < expires <= now + 86400):
            raise PermissionError("invalid delegation bounds")
        targets = {name: dict(t) for name, t in self.targets.items()
                   if all(self.policy(origin, t, a) is True for a in actions)}
        if not targets:
            raise PermissionError("no permitted configured target")
        saved = {"native": origin, "targets": targets}
        ref = secrets.token_urlsafe(32)
        with self._db() as db:
            db.execute("INSERT INTO delegations(ref,origin,job,actions,expires) VALUES(?,?,?,?,?)",
                       (ref, canonical(saved), job, canonical(sorted(set(actions))), expires))
        return PrivateReference(_REFERENCE_SEAL, ref)

    def _load(self, db, reference, target, action):
        row = db.execute("SELECT * FROM delegations WHERE ref=?", (reference,)).fetchone()
        if row is None or row['revoked'] or row['expires'] <= time.time():
            raise PermissionError("delegation unavailable")
        saved = json.loads(row['origin'])
        actual = self.targets.get(target)
        if (actual is None or saved['targets'].get(target) != actual
                or action not in json.loads(row['actions'])
                or self.policy(saved['native'], actual, action) is not True):
            raise PermissionError("current original-scope policy denied")
        return row, saved['native'], actual

    def retained(self, job_id):
        """Trusted host restart lookup; rebind still requires current original policy.

        This returns a locator from this issuer's durable store, never from saved
        claims. Job identity is not itself a grant and this is not a model tool.
        """
        with self._db() as db:
            row = db.execute("SELECT ref FROM delegations WHERE job=?", (job_id,)).fetchone()
            if row is None:
                raise PermissionError("retained job unavailable")
            return PrivateReference(_REFERENCE_SEAL, row['ref'])

    def retained_origin(self, reference, target, action):
        """Trusted-host recovery metadata, only after original policy validation.

        Returns no reference, permit or native capability. Restored metadata
        cannot enter current()/retain(); callers still need the private locator
        for every rebind, and every operation revalidates current policy.
        """
        if type(reference) is not PrivateReference:
            raise PermissionError("private retained reference required")
        with self._db() as db:
            _, origin, _ = self._load(db, reference._value, target, action)
            return json.loads(canonical(origin))

    def rebind(self, reference, target, action):
        if type(reference) is not PrivateReference:
            raise PermissionError("private retained reference required")
        with self._db() as db:
            self._load(db, reference._value, target, action)

    def issue(self, reference, target, operation, action, input_value):
        if not isinstance(operation, str) or not operation or len(operation) > 128:
            raise PermissionError("operation id required")
        with self._db() as db:
            row, origin, actual = self._load(db, reference._value, target, action)
            binding = {"version": 1, "issuerNamespace": self.namespace,
                "invocationId": row['job'], "operationId": operation,
                "recipientPrincipal": actual['recipientPrincipal'], "appId": actual['appId'],
                "buildId": actual['buildId'], "toolPath": actual['toolPath'],
                "selectedAccountsDigest": actual['selectedAccountsDigest'], "action": action,
                "inputDigest": digest(input_value), "expiresAt": min(row['expires'], int(time.time()) + 60) * 1000}
            ref = secrets.token_urlsafe(32)
            try:
                db.execute("INSERT INTO operations VALUES(?,?,?,?,?,?)",
                    (ref, reference._value, operation, target, canonical(binding), 'issued'))
            except sqlite3.IntegrityError:
                raise PermissionError("operation already reserved; reconcile unknown effects") from None
        return ref

    @staticmethod
    def _receiver_binding(binding):
        return {"digestVersion": DIGEST_VERSION,
                "recipientPrincipal": binding['recipientPrincipal'], "appId": binding['appId'],
                "buildId": binding['buildId'], "toolPath": binding['toolPath'],
                "accountsDigest": binding['selectedAccountsDigest'], "action": binding['action'],
                "inputDigest": binding['inputDigest']}

    def _operation(self, db, reference, peer, binding):
        row = db.execute("SELECT * FROM operations WHERE ref=?", (reference,)).fetchone()
        if row is None:
            raise PermissionError("private operation unavailable")
        expected = json.loads(row['binding'])
        if (self._receiver_binding(expected) != binding or expected['expiresAt'] <= time.time() * 1000
                or self.peer_principals.get(peer) != expected['recipientPrincipal']):
            raise PermissionError("binding or authenticated peer mismatch")
        _, origin, target = self._load(db, row['delegation'], row['target'], expected['action'])
        return row, expected, origin, target

    def redeem(self, peer_certificate_digest, reference, binding):
        if type(reference) is not str or not reference.isascii() or not 1 <= len(reference) <= 512:
            raise PermissionError("invalid private redemption")
        with self._db() as db:
            row, expected, _, _ = self._operation(db, reference, peer_certificate_digest, binding)
            if row['status'] != 'issued':
                raise PermissionError("operation already reserved; outcome unknown")
            lease = secrets.token_urlsafe(32)
            db.execute("UPDATE operations SET status='unknown' WHERE ref=?", (reference,))
            db.execute("INSERT INTO permits VALUES(?,?,?,?)",
                       (lease, reference, peer_certificate_digest, 'ready'))
            return {"version": 1, "invocationId": expected['invocationId'],
                    "operationId": expected['operationId'], "expiresAt": expected['expiresAt'], "lease": lease}

    def forward(self, peer, lease, binding, binding_id, input_value):
        from hermes_cli.private_invocation_transport import forward_https
        with self._db() as db:
            permit = db.execute("SELECT * FROM permits WHERE lease=?", (lease,)).fetchone()
            if permit is None or permit['peer'] != peer or permit['status'] != 'ready':
                raise PermissionError("permit unavailable or already attempted")
            row, expected, origin, target = self._operation(db, permit['operation_ref'], peer, binding)
            if (target.get('bindingId') != binding_id or digest(input_value) != expected['inputDigest']
                    or self.forward_context is None):
                raise PermissionError("fixed forwarding binding unavailable")
            # Commit before network I/O. No retry or remote absence inference on failure.
            db.execute("UPDATE permits SET status='unknown' WHERE lease=?", (lease,))
            db.execute("UPDATE operations SET status='unknown' WHERE ref=?", (row['ref'],))
        result = forward_https(target['forwardUrl'], self.forward_context,
                               {"version": 1, "binding": expected, "origin": origin, "input": input_value})
        with self._db() as db:
            db.execute("UPDATE permits SET status='observed' WHERE lease=?", (lease,))
            db.execute("UPDATE operations SET status='observed' WHERE ref=?", (row['ref'],))
        return {"version": 1, "result": result}

    def revoke(self, reference):
        with self._db() as db:
            db.execute("UPDATE delegations SET revoked=1 WHERE ref=?", (reference._value,))
