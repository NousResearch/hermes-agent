"""The legacy-history rules the store applies when it binds or authorizes unrouted pre-authority local history: the login subject parser, the login label Desktop stamps, the owner rule, the client-app label rule, the Nous account label, the login-owner test and the legacy-row check.

The owner rule decides whether a stored owner stamp (a history binding's subject, a routeless row's ``user_id``) belongs to a caller. Hermes is single-user: every sign-in acts as the host owner's own OS subject, so the login stamps older builds stored (``auth:v1:[provider, issuer, user]`` subjects, the ``<provider>:<user>`` labels a served dashboard stamped) read as the owner's, while a ``local`` stamp names an OS user and matches this machine's account only. The host data it reads arrives as an ``OwnerRule``; without one (None) only an equal stamp or the subject's own login label is its owner, and no label is a client app's.

``validate_legacy_row`` takes the store's own profile name as ``own_profile``. ``AuthorityRefusal`` is the persistence contract's class, ``OwnerRule`` and the source sets the contract vocabulary's. Standard library, ``persistence.core.errors`` and ``persistence.authority.vocabulary`` only.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from persistence.authority.vocabulary import LEGACY_CREATION_SOURCES, LEGACY_LOCAL_SOURCES, NOT_CLIENT_LABELS, OwnerRule
from persistence.core.errors import AuthorityRefusal


AUTH_SUBJECT_PREFIX = 'auth:v1:'


def parse_authenticated_subject(subject):
    """``(provider, issuer, user)`` of an :func:`authenticated_subject` login form
    ``auth:v1:[provider, issuer, user]``, else None. The one parser for that format: callers
    apply their own policy (which logins may be forwarded, which map to a label) to the result."""
    if not isinstance(subject, str) or not subject.startswith(AUTH_SUBJECT_PREFIX):
        return None
    try:
        parts = json.loads(subject[len(AUTH_SUBJECT_PREFIX):])
    except ValueError:
        return None
    if not (isinstance(parts, list) and len(parts) == 3 and all(isinstance(p, str) for p in parts)):
        return None
    return tuple(parts)


def login_label(subject):
    """The ``<provider>:<user>`` label Desktop stamps on its rows (tui_gateway/session_transports.py) when it
    names exactly the dashboard subject ``auth:v1:[provider, "", user]`` (gateway/session_identity.py).
    Only an issuer-less login maps: the label drops the issuer, so an issuer-bearing subject must not claim it."""
    parts = parse_authenticated_subject(subject)
    if parts is None:
        return None
    provider, issuer, user = parts
    if issuer or ':' in provider or not provider.strip() or not user.strip():
        return None
    label = f'{provider.strip()}:{user.strip()}'
    # Never yield something that reads as a subject itself (a provider named ``auth`` + user ``v1:[...]``).
    return None if label.startswith('auth:') else label


def _owner_login(provider: str, user: str, rule: OwnerRule) -> bool:
    """Whether a stored ``(provider, user)`` stamp names the host owner: a ``local`` stamp (the native grant a
    served Desktop presented) only when its OS user is the host owner, a sign-in stamp when its provider signs in."""
    if provider == 'local':
        return user.strip() == rule.host_subject
    return provider in rule.login_providers


def _login_stamp(value: str, rule: OwnerRule) -> bool:
    """``<provider>:<user>`` for a dashboard sign-in: the label ``hermes serve`` stamped for a login. Any other
    prefix (``telegram:42``, ``uid:501``) names someone else, and a ``nous:`` stamp keeps its own rule
    (``legacy_account_label``: a Desktop row's account label)."""
    provider, sep, user = value.partition(':')
    return bool(sep and user.strip()) and provider != 'nous' and _owner_login(provider, user, rule)


def _login_subject(value: str, rule: OwnerRule) -> bool:
    """An ``auth:v1:[provider, issuer, user]`` subject a sign-in stored (a provider is named)."""
    parts = parse_authenticated_subject(value)
    return parts is not None and bool(parts[2].strip()) and _owner_login(parts[0], parts[2], rule)


def same_owner(stored: Any, subject: Any, rule: Optional[OwnerRule]) -> bool:
    """THE owner rule: whether a stored owner stamp belongs to *subject*. True when equal, when it is the
    ``<provider>:<user>`` label of *subject*'s own login, or, while *subject* is the host owner, when it is any
    sign-in stamp an older build stored (an ``auth:v1`` login subject or a served dashboard's ``<provider>:<user>``
    label). One-way: a non-owner subject never gains owner stamps."""
    if stored == subject:
        return True
    if not isinstance(stored, str) or not stored or not subject:
        return False
    if stored == login_label(subject):
        return True
    return (rule is not None and subject == rule.host_subject
            and (_login_subject(stored, rule) or _login_stamp(stored, rule)))


def owners_agree(first: Any, second: Any, rule: Optional[OwnerRule]) -> bool:
    """Whether two stored owner stamps name the same owner (:func:`same_owner` either way round), for checks that
    compare one stored stamp with another rather than with a caller."""
    return same_owner(first, second, rule) or same_owner(second, first, rule)


def _binding_owner(saved: Any) -> Any:
    """The owner half of a stored ``[profile_id, subject]`` history binding, or None when unreadable.

    The profile half is the home path that wrote the binding into this profile's own store, so it never names
    another profile; a store that moved with its profile keeps its bindings."""
    try:
        profile, subject = json.loads(saved)
    except (TypeError, ValueError):
        return None
    return subject if isinstance(profile, str) else None


def binding_names(saved: Any, subject: str, rule: Optional[OwnerRule]) -> bool:
    """Whether a stored history binding names *subject*: the same owner (a binding a Webapp login stored belongs
    to the machine owner). Every reader of a binding asks this, never a string compare."""
    owner = _binding_owner(saved)
    return owner is not None and same_owner(owner, subject, rule)


def bindings_agree(first: Optional[str], second: Optional[str], rule: Optional[OwnerRule]) -> bool:
    """Two stored history bindings name the same owner (both absent agrees; one absent does not)."""
    if first == second or first is None or second is None:
        return first == second
    first_owner, second_owner = _binding_owner(first), _binding_owner(second)
    return first_owner is not None and second_owner is not None and owners_agree(first_owner, second_owner, rule)


def not_client_labels(rule: OwnerRule) -> frozenset[str]:
    """Labels :func:`client_label` refuses, for queries that pre-filter rows."""
    return LEGACY_LOCAL_SOURCES | NOT_CLIENT_LABELS | rule.messaging_labels


def client_label(source: Any, rule: Optional[OwnerRule]) -> bool:
    """Whether *source* is a client app's own label (``mobile``, ``ios``): not a local surface, a shipped
    messaging platform, or an automation / foreign-writer label."""
    return (rule is not None and isinstance(source, str) and bool(source.strip())
            and source not in not_client_labels(rule))


def client_history_row(row, rule: Optional[OwnerRule]) -> bool:
    """A chat a client app wrote under its own label before ``session.create`` took only surfaces.

    The pre-authority ``hermes serve`` stored whatever ``source`` a client sent, so phone apps signed
    in through the dashboard login left routeless rows labelled ``mobile``, ``ios`` and the like. They
    are the owner's Desktop-style chats: adopted as ``gui``. A routed row is never one, which keeps a
    plugin platform sharing the label on its messaging route."""
    return client_label(row.get('source'), rule) and not any(row.get(k) for k in ('session_key', 'chat_id', 'origin_json'))


def legacy_history_row(row, rule: Optional[OwnerRule]) -> bool:
    """Whether resume adopts *row* as old local history: a local surface's label, or a client app's."""
    return row.get('source') in LEGACY_LOCAL_SOURCES or client_history_row(row, rule)


def legacy_creation_source(row, rule: Optional[OwnerRule]) -> str:
    """The creation source an adopted legacy row resumes on."""
    return 'gui' if client_history_row(row, rule) else LEGACY_CREATION_SOURCES.get(row['source'], row['source'])


def legacy_account_label(row):
    # main's Desktop stamped the signed-in Nous account on its chats. That is a label, not a route or
    # an authenticated owner, so it does not block adoption of an otherwise unrouted Desktop row.
    user_id = row.get('user_id')
    return row['source'] == 'desktop' and isinstance(user_id, str) and user_id.startswith('nous:')


def login_owned_by(row, subject, rule: Optional[OwnerRule]) -> bool:
    """Whether *subject* owns the login stamp a served dashboard put on this row.

    Older ``hermes serve`` builds stamped the WS-upgrade login as ``<provider>:<user id>``
    (tui_gateway/session_transports.py), and the gateway stored a Webapp login as its
    ``auth:v1:[...]`` subject. Hermes is single-user and sign-in is owner-only, so both name the
    machine owner: the host owner's subject adopts them (:func:`same_owner`).
    """
    user_id = row.get('user_id')
    if not isinstance(user_id, str) or not subject:
        return False
    return same_owner(user_id, subject, rule)


def validate_legacy_row(own_profile, row, *, subject=None, rule: Optional[OwnerRule] = None):
    # Unrouted old local history only. Existing identity, foreign profile,
    # API, messaging, delegate and canonical rows require their own authority.
    # ``subject``: the adopting caller, whose own login stamp is not a foreign identity. A ``user_id``
    # stamp equal to ``subject`` itself names that principal: it is already its history by
    # ``authorize_history``'s rule, so it adopts like an unstamped row.
    user_id = row.get('user_id')
    if (not legacy_history_row(row, rule)
            or any(row.get(k) for k in ('session_key', 'chat_id', 'origin_json'))
            or (user_id and not legacy_account_label(row) and not login_owned_by(row, subject, rule))
            or row.get('profile_name') not in (None, '', own_profile or 'default')):
        raise AuthorityRefusal('not_found')
