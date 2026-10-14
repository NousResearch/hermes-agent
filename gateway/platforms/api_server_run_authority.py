"""Compact cancelled attempts only behind a proven newer room authority epoch."""
import hashlib
import json
import sqlite3
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


def hosted_member_session_id(home: str, room_id: str, member_id: str, target_profile: str) -> str:
    """Hidden session id for one home, room, and member. Gateway and epoch are not part of it.

    Pre-column ``run_idempotency`` rows store this id on their status. The scope hash cannot
    be rebuilt once the admitting gateway id has been overwritten, but this id still names
    the member that admitted the run.
    """
    seed = f"{home}\0{room_id}\0{member_id}\0{target_profile}"
    return "room_" + hashlib.sha256(seed.encode()).hexdigest()[:32]


# A successor grant can name any integer epoch. Hash at most this many predecessor
# epochs so a skipped epoch or an authority takeover still finds the live row, without
# walking an unbounded caller-supplied value. The newest half and the oldest half are
# both kept: a generation admitted at epoch 1 remains visible after a long gap.
_PREDECESSOR_EPOCH_CAP = 64


def _predecessor_epochs(epoch: int) -> list[int]:
    """Epochs strictly below ``epoch``, capped so one grant cannot hash millions of scopes.

    The kept epochs are the newest half and the oldest half. Build those ranges directly.
    ``list(range(epoch))`` is held inside the idempotency write lock; a caller-supplied
    epoch must not allocate that list before the cap is applied.
    """
    if epoch <= 1:
        return []
    span = epoch - 1
    if span <= _PREDECESSOR_EPOCH_CAP:
        return list(range(span, 0, -1))
    half = _PREDECESSOR_EPOCH_CAP // 2
    chosen = []
    seen = set()
    for older in (*range(span, span - half, -1), *range(1, half + 1)):
        if older in seen or older < 1 or older >= epoch:
            continue
        seen.add(older)
        chosen.append(older)
    return chosen


def predecessor_scopes(claims, previous=None, previous_home=None, extra_gateways=None):
    """Scopes whose unstamped rows belong to this grant's lineage.

    Returns ``(scope, epoch)`` pairs. Predecessor epochs share every claim except
    ``authority_epoch``. A takeover also rebuilds those epochs under every gateway this
    lineage has already used: that id is not part of the canonical authority key, but
    it is part of the scope hash a pre-column row was stored under. ``extra_gateways``
    carries ids recorded before this grant (the watermark, the reservation, and any
    gateway named by ``previous``). An explicit home move adds the named predecessor
    as well. The hash still includes this grant's home and member, so another
    member's or home's gateway id cannot select their row.
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
    candidates = [claims.get("authority_gateway_id")]
    if previous is not None:
        candidates.append(previous[2])
    if extra_gateways:
        candidates.extend(extra_gateways)
    for gateway_id in candidates:
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS run_idempotency_lineage ON run_idempotency(room_lineage_key, idempotency_key)")
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_seen_gateways (
        authority_key TEXT NOT NULL, gateway_id TEXT NOT NULL,
        PRIMARY KEY(authority_key, gateway_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS run_room_publication_homes (
        room_id TEXT NOT NULL, member_id TEXT NOT NULL, target_profile TEXT NOT NULL,
        authority_gateway_id TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
        home_install_id TEXT NOT NULL,
        PRIMARY KEY(room_id, member_id, target_profile, authority_gateway_id, authority_epoch))""")
    strip_stored_fence_bodies(conn)


def remember_publication_home(conn, claims) -> None:
    """Record the home this invitation is publishing, before the rooms commit.

    The peer reservation does not store a home. A later idempotency commit can fail
    after that reservation is durable, and the identical retry must be the only body
    allowed to finish it. The first home written for a gateway and epoch sticks.
    """
    if not isinstance(claims, dict):
        return
    try:
        room_id, member_id, profile = claims["room_id"], claims["member_id"], claims["target_profile"]
        gateway_id, home = str(claims["authority_gateway_id"]), str(claims["home_install_id"])
        epoch = int(claims["authority_epoch"])
    except (KeyError, TypeError, ValueError):
        return
    if not gateway_id or not home or epoch < 1:
        return
    conn.execute(
        """INSERT INTO run_room_publication_homes(
               room_id, member_id, target_profile, authority_gateway_id, authority_epoch, home_install_id)
           VALUES(?,?,?,?,?,?)
           ON CONFLICT(room_id, member_id, target_profile, authority_gateway_id, authority_epoch) DO NOTHING""",
        (str(room_id), str(member_id), str(profile), gateway_id, epoch, home))


def publication_home(conn, claims) -> str | None:
    """Home checkpointed for this member, gateway, and epoch, or None."""
    if not isinstance(claims, dict):
        return None
    try:
        epoch = int(claims["authority_epoch"])
        params = (
            str(claims["room_id"]), str(claims["member_id"]), str(claims["target_profile"]),
            str(claims["authority_gateway_id"]), epoch)
    except (KeyError, TypeError, ValueError):
        return None
    row = conn.execute(
        """SELECT home_install_id FROM run_room_publication_homes
           WHERE room_id=? AND member_id=? AND target_profile=?
             AND authority_gateway_id=? AND authority_epoch=?""",
        params).fetchone()
    return None if row is None or not row[0] else str(row[0])


def remember_seen_gateways(conn, authority_key, gateway_ids):
    """Remember gateway ids that have carried this canonical lineage.

    The scope hash is not reversible. A later grant can recompute a pre-column row's
    scope only for a gateway id it has actually stored. Ids are an append-only set:
    replacing the watermark must not forget the gateway that admitted the live row.
    """
    if not authority_key:
        return
    for gateway_id in gateway_ids or ():
        text = str(gateway_id or "")
        if not text:
            continue
        conn.execute(
            """INSERT INTO run_room_seen_gateways(authority_key, gateway_id) VALUES(?,?)
               ON CONFLICT(authority_key, gateway_id) DO NOTHING""",
            (authority_key, text))


def seen_gateway_ids(conn, authority_key):
    if not authority_key:
        return []
    return [row[0] for row in conn.execute(
        "SELECT gateway_id FROM run_room_seen_gateways WHERE authority_key=?", (authority_key,))]


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


def _stamp_unstamped_scope(conn, key, scope, epoch):
    """Bind one pre-column scope to this lineage without adopting a different member.

    Callers pass a scope they recomputed for this home and member. Every key in that
    scope is that member's generation. The session-id scan must not use this: one
    matching status would otherwise stamp every unstamped row that shares the scope,
    including a migrated public run.
    """
    conn.execute(
        """UPDATE run_idempotency SET room_authority_key=?,room_authority_epoch=?,room_lineage_key=?
           WHERE scope=? AND room_authority_key IS NULL""",
        (key, int(epoch), key, scope))


def _stamp_legacy_row(conn, key, epoch, scope, idempotency_key):
    """Bind one matched pre-column row. Sibling keys in the scope stay untouched."""
    conn.execute(
        """UPDATE run_idempotency
           SET room_authority_key=?, room_authority_epoch=?, room_lineage_key=?
           WHERE scope=? AND idempotency_key=? AND room_authority_key IS NULL AND client_replay=0""",
        (key, int(epoch), key, scope, idempotency_key))


def _explicit_compression_fork(row) -> bool:
    """A branch, delegate, reset, or tool child is not the room session's continuation."""
    if row["source"] == "tool":
        return True
    raw = row["model_config"]
    if not isinstance(raw, str) or not raw:
        return False
    try:
        cfg = json.loads(raw)
    except (ValueError, TypeError):
        return False
    if not isinstance(cfg, dict):
        return False
    parent_id = row["parent_session_id"]
    markers = (cfg.get("_branched_from"), cfg.get("_delegate_from"), cfg.get("_reset_from"))
    return parent_id in markers if parent_id else any(marker is not None for marker in markers)


def compression_lineage(session_id: str) -> list[str] | None:
    """Ids from the stored session up through compression parents, root last.

    ``None`` means the session database could not be read. A missing row is a
    one-element list so an exact hidden id still matches without a session row.
    Only a child of a session ended ``compression`` is followed, and a fork is not.

    The bound is the session store's compression-chain cap, the same one
    ``get_compression_chain`` raised so a real lineage about 180 deep still
    reaches its root. A shorter prefix drops the hidden member id.
    """
    if not isinstance(session_id, str) or not session_id:
        return []
    try:
        from hermes_state import _default_db_path
        from hermes_state_compression import _CHAIN_CAP
        path = _default_db_path()
    except (ImportError, OSError, RuntimeError):
        return None
    if path is None or not path.is_file():
        return [session_id]
    conn = None
    try:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        chain = []
        current = session_id
        seen = set()
        for _ in range(_CHAIN_CAP):
            if not current or current in seen:
                break
            seen.add(current)
            chain.append(current)
            row = conn.execute(
                "SELECT parent_session_id, source, model_config FROM sessions WHERE id=?",
                (current,)).fetchone()
            if row is None or not row["parent_session_id"]:
                break
            if _explicit_compression_fork(row):
                break
            parent = conn.execute(
                "SELECT end_reason FROM sessions WHERE id=?", (row["parent_session_id"],)).fetchone()
            if parent is None or parent["end_reason"] != "compression":
                break
            current = row["parent_session_id"]
        return chain
    except (sqlite3.Error, OSError):
        return None
    finally:
        if conn is not None:
            conn.close()


def _lineage_ids(status: dict) -> list[str] | None:
    """Session ids that name this row, including compression ancestors of the stored id.

    ``None`` means the session database could not be read and no id was already on the row.
    Ids already stored are kept when that read fails, so a canonical ``hosted_session_id``
    still matches.
    """
    found = []
    hosted = status.get("hosted_session_id")
    if isinstance(hosted, str) and hosted:
        found.append(hosted)
    stored = status.get("session_id")
    if not isinstance(stored, str) or not stored:
        return found
    if stored not in found:
        found.append(stored)
    chain = compression_lineage(stored)
    if chain is None:
        return found or None
    for item in chain:
        if item not in found:
            found.append(item)
    return found


def _foreign_room_root(ids: list[str], session_ids: set[str]) -> bool:
    """True when the lineage root is some other member's hidden session id."""
    root = ids[-1] if ids else ""
    return root.startswith("room_") and len(root) == 37 and root not in session_ids


def _legacy_member_session_ids(conn, claims):
    """Session ids this grant's member could have stored on a pre-column run.

    Main hashed the grant's own home. This build hashes the origin home once a move has
    been authorized. Both exclude the gateway id, so either one still finds the row after
    that id has been overwritten.
    """
    if not isinstance(claims, dict):
        return []
    try:
        room_id, member_id, profile = claims["room_id"], claims["member_id"], claims["target_profile"]
        own_home = claims["home_install_id"]
    except KeyError:
        return []
    homes = [own_home]
    try:
        origin = origin_home(conn, room_authority(claims), own_home)
    except (KeyError, TypeError, ValueError):
        origin = own_home
    if origin and origin not in homes:
        homes.append(origin)
    return [hosted_member_session_id(home, room_id, member_id, profile) for home in homes if home]


def _status_matches_member(status: dict, session_ids: set[str]) -> bool:
    """True when the stored session, its canonical id, or a compression ancestor is this member."""
    ids = _lineage_ids(status)
    return bool(ids) and any(item in session_ids for item in ids)


def _stamp_legacy_member_rows(conn, key, epoch, claims, excluded_scopes=None):
    """Bind pre-column rows whose session lineage is this member.

    One matching status stamps that row only. A public scope is excluded: a migrated
    ``POST /v1/runs`` can carry this member's session id and must not become the room replay.
    """
    session_ids = set(_legacy_member_session_ids(conn, claims))
    if not session_ids:
        return
    from gateway.platforms.api_server_run_idempotency import _room_generation_key
    excluded = {str(scope) for scope in (excluded_scopes or ()) if scope}
    rows = conn.execute(
        """SELECT scope, idempotency_key, status_json FROM run_idempotency
           WHERE room_authority_key IS NULL AND client_replay=0"""
    ).fetchall()
    for scope, idempotency_key, status_json in rows:
        if scope in excluded or not _room_generation_key(idempotency_key):
            continue
        try:
            status = json.loads(status_json)
        except (ValueError, TypeError):
            continue
        if not isinstance(status, dict) or not _status_matches_member(status, session_ids):
            continue
        _stamp_legacy_row(conn, key, epoch, scope, idempotency_key)


_RUNNING_LEGACY_STATUSES = frozenset({
    "queued", "running", "stopping", "waiting_for_approval", "in_progress", "requires_action"})


def legacy_run_unresolved(conn, idempotency_key, claims, excluded_scopes=None) -> bool:
    """True when a running pre-column row for this member cannot be bound.

    Admission must not insert another run for that generation. A root that is some
    other member's hidden id is not this row. A public scope is not this row.
    """
    session_ids = set(_legacy_member_session_ids(conn, claims))
    if not session_ids:
        return False
    from gateway.platforms.api_server_run_idempotency import _room_generation_key
    if not _room_generation_key(idempotency_key):
        return False
    excluded = {str(scope) for scope in (excluded_scopes or ()) if scope}
    rows = conn.execute(
        """SELECT scope, status_json FROM run_idempotency
           WHERE idempotency_key=? AND room_authority_key IS NULL AND client_replay=0""",
        (idempotency_key,)).fetchall()
    for scope, status_json in rows:
        if scope in excluded:
            continue
        try:
            status = json.loads(status_json)
        except (ValueError, TypeError):
            return True
        if not isinstance(status, dict):
            return True
        if str(status.get("status") or "") not in _RUNNING_LEGACY_STATUSES:
            continue
        if _status_matches_member(status, session_ids):
            continue
        ids = _lineage_ids(status)
        if ids is None or not _foreign_room_root(ids, session_ids):
            return True
    return False


def observe(conn, scope, authority, previous=None, previous_home=None, namespace=None, claims=None,
            extra_gateways=None, excluded_scopes=None):
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
    # A migrated row's scope includes a gateway id main may already have overwritten.
    # The hidden session id does not. Stamp that member's rows before the watermark moves.
    _stamp_legacy_member_rows(conn, key, epoch, claims, excluded_scopes)
    # Read the watermark before this grant replaces it. Production POST /v1/runs does not
    # receive previous_authority; the gateway that admitted the live row is whatever this
    # lineage last stored, plus every gateway remembered since.
    stored = conn.execute(
        "SELECT gateway_id FROM run_room_authorities WHERE authority_key=?", (key,)).fetchone()
    gateways = []
    for gateway_id in (
            gateway, None if stored is None else stored[0], *seen_gateway_ids(conn, key),
            *(extra_gateways or ())):
        text = str(gateway_id or "")
        if text and text not in gateways:
            gateways.append(text)
    remember_seen_gateways(conn, key, gateways)
    _stamp_unstamped_scope(conn, key, scope, epoch)
    # Rows written before room_authority_key existed stay NULL. Their scope hash includes
    # the gateway and the epoch, so the successor's own scope does not match them. Stamp
    # every predecessor scope this lineage is allowed to carry, keeping the old epoch so
    # compact can still tell those generations from the current one.
    for prior_scope, prior_epoch in predecessor_scopes(
            claims, previous, previous_home, extra_gateways=gateways):
        _stamp_unstamped_scope(conn, key, prior_scope, prior_epoch)
    # A row that already carries this lineage key is the same home and member at any
    # gateway and any epoch distance. Public client keys stay client_replay and are not
    # adopted. Scope recompute above is only how a pre-column NULL row is found the
    # first time; after that the key does not depend on the epoch walk.
    conn.execute(
        """UPDATE run_idempotency
           SET room_authority_key=?, room_authority_epoch=COALESCE(room_authority_epoch, ?)
           WHERE room_lineage_key=? AND room_authority_key IS NULL AND client_replay=0""",
        (key, epoch, key))
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
