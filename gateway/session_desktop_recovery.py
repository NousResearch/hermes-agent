"""Mount a cold Desktop-client row (``source='desktop'``) for a local viewer.

The Desktop client's rows written before this store carried receipts (or written by a remote
viewer through the legacy dashboard dispatcher) have no local receipt, no messaging lane and no
API binding — none of the three keys the cold-mount families key on — so after an owner restart
``session.resume`` answered bare ``not_found`` for every one of them, while the dashboard's own
legacy dispatcher kept opening the same rows straight from ``state.db``. This gives the authority
path that same floor: the row's own columns rebuild the API-family source (``chat_id`` = the
session id, exactly what ``restore_api_session`` validates real API rows against), the mount is a
plain store entry under the generated session key, and the same admission/conflict guards apply.
Rows of any other source label are never touched here.
"""
import logging

from gateway.config import Platform
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal

logger = logging.getLogger(__name__)


def _row_source(session_id, row):
    """The API-family source the row's own columns stand for; None when they name another family."""
    from gateway.session import SessionSource
    # SessionRecord.origin is the parsed ``origin_json`` (a mapping, or None).
    origin = row.origin
    if origin:
        # from_dict indexes a mapping: a JSON list/string parses to a non-dict but has no fields
        # — that is the same refusal as unparseable origin_json.
        try:
            source = SessionSource.from_dict(origin) if isinstance(origin, dict) else None
        except (TypeError, ValueError, KeyError, AttributeError):
            source = None
        # from_dict stringifies a null chat_id into 'None' (session.py:159): a truthy shared key
        # every chat_id-less row would collide on — refuse it like a missing chat_id.
        if source is not None and source.chat_id and source.chat_id != 'None':
            return source
    return SessionSource(platform=Platform.API_SERVER, chat_id=session_id,
                         user_id=row.user_id, chat_type=row.chat_type or 'dm',
                         profile=row.profile_name or None)


def restore_desktop_session(authority, session_id, actor=None):
    """Publish *session_id* as a cold Desktop row; raises ``not_found`` when the row is gone or
    stamped for another owner, and ``admission_conflict`` when the generated route already names
    another conversation.

    Ownership gate (upstream #140183 F1's identity rule, the same predicate legacy adoption
    applies in ``session_rules.legacy_local``): a row with no ``user_id`` stamp mounts, so does a
    stamp that :func:`gateway.owner_identity.same_owner` reads as the host owner's (the served
    dashboard's login labels) and a Nous account label on a desktop row (a label, not an owner).
    Any OTHER stamp — a stranger's — must stay unmounted with its stamp untouched, refusing bare
    ``not_found`` exactly like the families before this one; both call sites (the ``authorize``
    fall-through and ``resolve_local_target``'s fall-through) inherit the gate from here."""
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionRef
    persistence = authority_persistence(authority)
    row = persistence.sessions.get(session_id)
    if row is None or row.source != 'desktop':
        raise AuthorityRefusal('not_found')
    # Upstream's identity rule (#140183 F1): a desktop row stamped for someone other than the host
    # owner must stay unmounted with its stamp intact — the same refusal legacy adoption gives, so
    # the predicate is the one adoption itself reads (gateway.owner_identity.same_owner over the
    # host OwnerRule, plus the Nous account label exemption, which is a label and not an owner).
    if row.user_id:
        from gateway.owner_identity import host_owner_subject, same_owner
        from session_rules.legacy_local import legacy_account_label
        if not legacy_account_label({'source': row.source, 'user_id': row.user_id}) \
                and not same_owner(row.user_id, host_owner_subject()):
            raise AuthorityRefusal('not_found')
    # A compression continuation mounts as its lineage root or not at all — the routed family's
    # own policy for children (``_mount_plan`` refuses them; resume maps to the root): the
    # conversation's admission identity is the root, so mounting the child would fork its FIFO
    # off it. Resuming the root mounts live and executes on the tip.
    lineage = persistence.lineage.lineage(session_id)
    if lineage and lineage[0] != session_id:
        raise AuthorityRefusal('not_found')
    source = _row_source(session_id, row)
    store = authority.runner.session_store
    route = store._generate_session_key(source)
    from gateway.session import SessionEntry
    from datetime import datetime
    started = datetime.now()
    if row.started_at:
        try:
            started = datetime.fromtimestamp(row.started_at)
        except (TypeError, ValueError, OSError, OverflowError):
            pass  # a corrupt durable timestamp must not 500 the mount (session_recovery's _ts policy)
    entry = SessionEntry(session_key=route, session_id=session_id, created_at=started, updated_at=started,
                         origin=source, platform=source.platform, chat_type=source.chat_type,
                         transport_profile=row.transport_profile)
    from gateway.session_api import _pin_api_identity
    _pin_api_identity(authority.runner, source, transport_profile=row.transport_profile)
    target = authority.physical_target(SessionRef(authority.profile_id, session_id))
    entry.session_id = target
    with store._lock:
        store._ensure_loaded_locked()
        current = store._entries.get(entry.session_key)
        if current is not None and current.session_id not in persistence.lineage.lineage(session_id):
            raise AuthorityRefusal('admission_conflict')
        store._entries[entry.session_key] = current if current is not None and current.session_id == target else entry
    authority.sessions.setdefault(session_id, LiveSession(source, entry.session_key))
    logger.info('Mounted cold Desktop session %s under %s', session_id, entry.session_key)
    return SessionRef(authority.profile_id, session_id)
