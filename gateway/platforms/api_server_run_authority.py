"""Compact cancelled attempts only behind a proven newer room authority epoch."""
import hashlib
import json
import time


def room_run_scope(claims):
    fields = ("room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
              "member_id", "target_install_id", "target_profile")
    return hashlib.sha256("\0".join(str(claims[k]) for k in fields).encode()).hexdigest()


def room_authority(claims):
    # Home remains part of the origin namespace. Only an authenticated target-owner
    # transition may bind another home to this lineage; an epoch alone is not proof.
    fields = ("room_id", "home_install_id", "member_id", "target_install_id", "target_profile")
    key = hashlib.sha256("\0".join(str(claims[k]) for k in fields).encode()).hexdigest()
    return key, int(claims["authority_epoch"]), str(claims["authority_gateway_id"])


def room_namespace(claims):
    fields = ("room_id", "member_id", "target_install_id", "target_profile")
    return hashlib.sha256("\0".join(str(claims[key]) for key in fields).encode()).hexdigest()


# A successor grant can name any integer epoch. Hash at most this many predecessor
# epochs so a skipped epoch or an authority takeover still finds the live row, without
# walking an unbounded caller-supplied value. The newest half and the oldest half are
# both kept: a generation admitted at epoch 1 remains visible after a long gap.
_PREDECESSOR_EPOCH_CAP = 64


def _predecessor_epochs(epoch: int) -> list[int]:
    """Epochs strictly below ``epoch``, capped so one grant cannot hash millions of scopes."""
    if epoch <= 1:
        return []
    span = list(range(epoch - 1, 0, -1))
    if len(span) <= _PREDECESSOR_EPOCH_CAP:
        return span
    half = _PREDECESSOR_EPOCH_CAP // 2
    chosen = []
    seen = set()
    for older in span[:half] + list(range(1, half + 1)):
        if older in seen or older < 1 or older >= epoch:
            continue
        seen.add(older)
        chosen.append(older)
    return chosen


def predecessor_scopes(claims, previous=None, previous_home=None):
    """Scopes whose unstamped rows belong to this grant's lineage.

    Returns ``(scope, epoch)`` pairs. Predecessor epochs share every claim except
    ``authority_epoch``. A takeover also rebuilds those epochs under the previous
    ``authority_gateway_id``: that id is not part of the canonical authority key, but
    it is part of the scope hash a pre-column row was stored under. An explicit home
    move adds the named predecessor as well.
    """
    if not isinstance(claims, dict) or "authority_epoch" not in claims:
        return []
    try:
        epoch = int(claims["authority_epoch"])
    except (TypeError, ValueError):
        return []
    scopes = []
    seen = set()

    def add(prior, prior_epoch):
        try:
            scope = room_run_scope(prior)
        except KeyError:
            return
        if scope in seen:
            return
        seen.add(scope)
        scopes.append((scope, int(prior_epoch)))

    gateway_ids = []
    for gateway_id in (
        claims.get("authority_gateway_id"),
        None if previous is None else previous[2],
    ):
        if gateway_id is None:
            continue
        text = str(gateway_id)
        if text and text not in gateway_ids:
            gateway_ids.append(text)
    for gateway_id in gateway_ids:
        for older in _predecessor_epochs(epoch):
            add({**claims, "authority_epoch": older, "authority_gateway_id": gateway_id}, older)
    if previous is not None and previous_home:
        add({
            **claims,
            "home_install_id": previous_home,
            "authority_gateway_id": previous[2],
            "authority_epoch": int(previous[1]),
        }, previous[1])
    return scopes


def initialize(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_authorities (
        authority_key TEXT PRIMARY KEY, authority_epoch INTEGER NOT NULL,
        gateway_id TEXT NOT NULL, retired_through INTEGER NOT NULL DEFAULT 0)""")
    from hermes_cli.sqlite_util import add_column_if_missing
    add_column_if_missing(conn, "run_room_authorities", "home_key", "home_key TEXT")
    add_column_if_missing(conn, "run_room_authorities", "target_key", "target_key TEXT")
    from gateway.platforms import api_server_room_origins
    api_server_room_origins.initialize(conn)
    conn.execute("UPDATE run_room_authorities SET home_key=authority_key WHERE home_key IS NULL")
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_authority_aliases (
        home_key TEXT PRIMARY KEY, authority_key TEXT NOT NULL, origin_home TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_namespaces (
        namespace_key TEXT PRIMARY KEY, authority_key TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_fenced_attempts (
        authority_key TEXT NOT NULL, idempotency_key TEXT NOT NULL,
        run_id TEXT NOT NULL, status_json TEXT NOT NULL,
        PRIMARY KEY(authority_key, idempotency_key))""")
    conn.execute("CREATE INDEX IF NOT EXISTS run_room_attempts ON run_idempotency(room_authority_key, room_authority_epoch)")
    conn.execute("CREATE INDEX IF NOT EXISTS run_room_attempt_identity ON run_idempotency(room_authority_key, idempotency_key)")
    strip_stored_fence_bodies(conn)


def canonical(conn, authority):
    """Resolve only aliases previously authorized by the target, including after restart."""
    key, epoch, gateway = authority
    row = conn.execute("SELECT authority_key FROM run_room_authority_aliases WHERE home_key=?", (key,)).fetchone()
    return (row[0] if row else key), epoch, gateway


def namespace_matches(conn, namespace, authority):
    if namespace is None:
        return True
    row = conn.execute("SELECT authority_key FROM run_room_namespaces WHERE namespace_key=?", (namespace,)).fetchone()
    return row is None or row[0] == canonical(conn, authority)[0]


def origin_home(conn, authority, home):
    """Preserve the original hidden member session only across an authorized home move."""
    row = conn.execute("SELECT origin_home FROM run_room_authority_aliases WHERE home_key=?", (authority[0],)).fetchone()
    return row[0] if row else home


def successor(conn, authority, previous):
    """Validate an explicit owner-authorized transition without joining unrelated origins."""
    current = canonical(conn, authority)
    if previous is None:
        return current
    predecessor = canonical(conn, previous)
    row = conn.execute("SELECT authority_epoch,gateway_id,home_key FROM run_room_authorities WHERE authority_key=?",
                       (predecessor[0],)).fetchone()
    # A lost invitation reply may be retried: reissuing the exact current authority
    # does not advance or join anything, and still requires target-owner authorization.
    if current[0] == predecessor[0] and row == (authority[1], authority[2], authority[0]):
        return current
    # Rows written before room_authority_key existed have no watermark yet. The same
    # lineage (home, member, target) may name its previous gateway and epoch so those
    # scopes can be stamped. A different home still needs a stored watermark.
    if row is None and current[0] == predecessor[0] and authority[1] > previous[1]:
        return current
    if row != (previous[1], previous[2], previous[0]) or authority[1] <= previous[1]:
        raise ValueError("previous room authority is not current")
    if current[0] != predecessor[0] and (
            current[0] != authority[0]
            or conn.execute("SELECT 1 FROM run_room_authorities WHERE authority_key=?", (current[0],)).fetchone()):
        raise ValueError("successor home already belongs to another room origin")
    return predecessor[0], authority[1], authority[2]


def retirement_allowed(conn, authority):
    """An old scope may retire itself, never a different owner at the current epoch."""
    key, epoch, gateway = canonical(conn, authority)
    row = conn.execute("SELECT authority_epoch,gateway_id,home_key FROM run_room_authorities WHERE authority_key=?",
                       (key,)).fetchone()
    from gateway.platforms.api_server_room_origins import effective
    current = effective(conn, key)
    return current is not None and (epoch < current[0] or (
        (epoch, gateway) == current[:2] and (epoch, gateway, authority[0]) == row))


def superseded(conn, authority):
    if authority is None:
        return False
    key, epoch, gateway = canonical(conn, authority)
    from gateway.platforms.api_server_room_origins import effective
    row = effective(conn, key)
    return row is not None and (epoch <= row[2] or epoch < row[0] or (epoch == row[0] and gateway != row[1]))


def observe(conn, scope, authority, previous=None, previous_home=None, namespace=None, claims=None):
    """Bind old receipts on authenticated observation, then advance one room/member watermark."""
    key, epoch, gateway = successor(conn, authority, previous)
    if not namespace_matches(conn, namespace, (key, epoch, gateway)):
        return False
    from gateway.platforms import api_server_room_origins as origins
    if claims is not None and not origins.accepts(conn, claims, previous_home):
        return False
    if previous is not None and key != authority[0]:
        if not previous_home:
            raise ValueError("previous room home is required")
        conn.execute("""INSERT INTO run_room_authority_aliases(home_key,authority_key,origin_home)
            VALUES(?,?,?) ON CONFLICT(home_key) DO NOTHING""",
            (authority[0], key, origin_home(conn, previous, previous_home)))
    conn.execute("""UPDATE run_idempotency SET room_authority_key=?,room_authority_epoch=?
        WHERE scope=? AND room_authority_key IS NULL""", (key, epoch, scope))
    # Rows written before room_authority_key existed stay NULL. Their scope hash includes
    # the epoch, so the successor's own scope does not match them. Stamp the predecessor
    # scopes this grant is allowed to carry, keeping the old epoch so compact can still
    # tell those generations from the current one.
    for prior_scope, prior_epoch in predecessor_scopes(claims, previous, previous_home):
        conn.execute("""UPDATE run_idempotency SET room_authority_key=?,room_authority_epoch=?
            WHERE scope=? AND room_authority_key IS NULL""", (key, prior_epoch, prior_scope))
    if superseded(conn, authority):
        compact(conn, key)
        return False
    conn.execute("""INSERT INTO run_room_authorities(authority_key,authority_epoch,gateway_id,home_key) VALUES(?,?,?,?)
        ON CONFLICT(authority_key) DO UPDATE SET
        authority_epoch=excluded.authority_epoch,gateway_id=excluded.gateway_id,home_key=excluded.home_key""",
        (key, epoch, gateway, authority[0]))
    if namespace is not None:
        conn.execute("INSERT INTO run_room_namespaces(namespace_key,authority_key) VALUES(?,?) ON CONFLICT DO NOTHING",
                     (namespace, key))
    if claims is not None:
        original = origin_home(conn, authority, claims["home_install_id"])
        target = origins.observe(conn, claims, original)
        conn.execute("UPDATE run_room_authorities SET target_key=? WHERE authority_key=?", (target, key))
        for (member_key,) in conn.execute("SELECT authority_key FROM run_room_authorities WHERE target_key=?", (target,)).fetchall():
            compact(conn, member_key)
    else:
        compact(conn, key)
    return True


def retire(conn, scope, authority):
    if not retirement_allowed(conn, authority):
        raise ValueError("room retirement authority does not match retained lineage")
    observe(conn, scope, authority)
    authority = canonical(conn, authority)
    conn.execute("UPDATE run_room_authorities SET retired_through=MAX(retired_through,?) WHERE authority_key=?",
                 (authority[1], authority[0]))
    compact(conn, authority[0])


_FENCE_DROPPED_FIELDS = ("output", "error", "usage", "pending_steer")


def _absent_barrier(status):
    return (isinstance(status, dict) and status.get("admission_cancelled") is True
            and status.get("status") == "cancelled")


def _fence_body(status_json):
    """Return ``(encoded, status, discarded)`` with reply fields removed.

    ``discarded`` is true when a stored reply field was removed. A completion that never
    had those fields stays distinguishable from a reply this store has forgotten.
    """
    try:
        status = json.loads(status_json)
    except (ValueError, TypeError):
        return status_json, None, False
    if not isinstance(status, dict):
        return status_json, None, False
    discarded = any(field in status for field in _FENCE_DROPPED_FIELDS)
    for field in _FENCE_DROPPED_FIELDS:
        status.pop(field, None)
    # A fence written by an older strip removed the reply fields and did not set the
    # marker. A completion that still carries an empty output/error keeps those keys.
    if (not discarded and "body_discarded" not in status
            and status.get("status") in {"completed", "failed", "interrupted"}
            and "output" not in status and "error" not in status):
        discarded = True
    if discarded:
        status["body_discarded"] = True
    encoded = json.dumps(status, sort_keys=True, separators=(",", ":"))
    return encoded, status, discarded


def remember_fenced_attempt(conn, authority_key, idempotency_key, run_id, status_json):
    """Keep a generation key after its idempotency row is deleted, without the reply body.

    An absent cancellation barrier must not become the only fence when a real receipt
    for the same generation is already stored, or arrives later. A fence written before
    reply fields were dropped is stripped in place.
    """
    if not authority_key or not idempotency_key:
        return
    encoded, status, _discarded = _fence_body(status_json)
    existing = conn.execute(
        "SELECT status_json FROM run_room_fenced_attempts WHERE authority_key=? AND idempotency_key=?",
        (authority_key, idempotency_key)).fetchone()
    if existing is None:
        conn.execute(
            """INSERT INTO run_room_fenced_attempts(authority_key, idempotency_key, run_id, status_json)
               VALUES(?,?,?,?)""",
            (authority_key, idempotency_key, run_id, encoded))
        return
    stored_encoded, previous, stored_discarded = _fence_body(existing[0])
    if stored_discarded:
        conn.execute(
            """UPDATE run_room_fenced_attempts SET status_json=?
               WHERE authority_key=? AND idempotency_key=?""",
            (stored_encoded, authority_key, idempotency_key))
        previous = json.loads(stored_encoded)
    if _absent_barrier(previous) and not _absent_barrier(status):
        conn.execute(
            """UPDATE run_room_fenced_attempts SET run_id=?, status_json=?
               WHERE authority_key=? AND idempotency_key=?""",
            (run_id, encoded, authority_key, idempotency_key))


def strip_stored_fence_bodies(conn):
    """Drop reply fields from fences written before that strip existed."""
    rows = conn.execute(
        "SELECT authority_key, idempotency_key, run_id, status_json FROM run_room_fenced_attempts"
    ).fetchall()
    for authority_key, idempotency_key, run_id, status_json in rows:
        remember_fenced_attempt(conn, authority_key, idempotency_key, run_id, status_json)


def compact(conn, authority_key, *, now=None):
    """Retire absent barriers immediately, settled executions after their observation horizon."""
    from gateway.platforms.api_server_room_origins import effective
    from gateway.platforms.api_server_run_idempotency import terminal_observation_expired
    current = effective(conn, authority_key)
    if current is None:
        return False
    now = time.time() if now is None else now
    rows = conn.execute("""SELECT scope,idempotency_key,run_id,status_json,stop_requested,
        fingerprint,owner_pid,owner_started,updated_at,retention_until,acknowledged_at
        FROM run_idempotency WHERE room_authority_key=? AND (room_authority_epoch<? OR room_authority_epoch<=?)""",
        (authority_key, current[0], current[2])).fetchall()
    pruned = False
    for scope, key, run_id, encoded, stopped, fingerprint, owner_pid, owner_started, updated, horizon, acknowledged in rows:
        try:
            status = json.loads(encoded)
        except (ValueError, TypeError):
            continue
        if not isinstance(status, dict):
            continue
        if (status.get("status") in {"completed", "failed", "cancelled", "interrupted"}
                and (stopped or status.get("admission_cancelled"))):
            absent = (status.get("admission_cancelled") is True and status.get("status") == "cancelled"
                      and fingerprint == "" and not owner_pid and not owner_started)
            if not absent and not terminal_observation_expired(updated, horizon, acknowledged, now):
                continue
            # The row's scope includes the retired epoch. Keep the generation key so a
            # successor grant cannot reserve the same Idempotency-Key as a new run.
            # An absent barrier must not replace a real receipt already fenced for this key.
            remember_fenced_attempt(conn, authority_key, key, run_id, encoded)
            conn.execute("DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (scope, key))
            pruned = True
    return pruned
