"""The Webapp is the machine owner: one identity for every signed-in surface of this host.

Hermes is single-user. Native Desktop, the TUI and the CLI reach the gateway as the owner's own OS
user (``uid:<n>`` / ``sid:<sid>``, minted only for a same-user peer by the control channel). A Webapp
tab, a standalone dashboard's forwarded login or a phone signed in through the dashboard login is
that same person: sign-in only decides whether a browser may open the Webapp at all, it never names a
second owner. So every authenticated non-native principal acts as :func:`host_owner_subject`, and
the login stamps older builds stored (``auth:v1:[provider, issuer, user]`` subjects, the
``<provider>:<user>`` labels ``hermes serve`` stamped on Desktop rows) read as the owner's too. A
``local`` stamp names an OS user, so it matches this machine's account only. Every comparison of a
stored owner with a caller goes through :func:`same_owner`, so the sidebar, the open, adoption and the
room controls cannot disagree about who owns a row.
"""
from __future__ import annotations


def host_owner_subject() -> str:
    """The owner's own OS identity: the subject native Desktop's ticket carries on this host."""
    from gateway.session_worker_reservation import owner_worker_subject
    return owner_worker_subject()


def principal_subject(identity: dict) -> str:
    """The subject a connection acts as. Whatever the private control channel or a native grant
    minted (provider ``local``: Desktop, the TUI, the CLI, a managed worker) keeps its subject; every
    login (a Webapp tab, the loopback session token, the server's own PTY child, or a dashboard login
    a standalone backend forwards in a local ticket) is the owner. An identity naming nobody stays
    unbound, exactly as before."""
    from gateway.session_identity import authenticated_subject, parse_authenticated_subject
    subject = authenticated_subject(identity)
    if not identity.get('user_id'):
        return subject
    provider = str(identity.get('provider') or '').strip()
    if provider == 'local':
        login = parse_authenticated_subject(subject)
        return host_owner_subject() if login is not None and _login_provider(login[0]) and login[0] != 'local' else subject
    return host_owner_subject() if _login_provider(provider) else subject


# The interactive sign-ins Hermes bundles; a plugin-registered dashboard provider counts too
# (:func:`_login_provider`). ``session-token`` is the loopback Webapp's own credential and
# ``server-internal`` the PTY child a signed-in dashboard spawns: both act for whoever opened that page,
# the owner. ``local`` (a native grant) is not a sign-in: its stamp names an OS user (:func:`_owner_login`).
_BUNDLED_LOGINS = frozenset({'basic', 'nous', 'self-hosted', 'session-token', 'server-internal'})


def _login_provider(provider: str) -> bool:
    if provider in _BUNDLED_LOGINS:
        return True
    from hermes_cli.dashboard_auth.registry import list_session_providers
    return any(p.name == provider for p in list_session_providers())


def _owner_login(provider: str, user: str) -> bool:
    """Whether a stored ``(provider, user)`` stamp names the host owner. A ``local`` stamp (the native
    grant a served Desktop presented) names an OS user, so it is the owner's only when that user is this
    machine's owner; any other OS user stays foreign. A sign-in stamp is the owner's."""
    if provider == 'local':
        return user.strip() == host_owner_subject()
    return _login_provider(provider)


def _login_label(value: str) -> bool:
    """``<provider>:<user>`` for a dashboard sign-in: the label ``hermes serve`` stamped for a login
    (session_transports). Any other prefix (``telegram:42``, ``uid:501``) names someone else, and a
    ``nous:`` stamp keeps its own rule (``legacy_account_label``: a Desktop row's account label)."""
    provider, sep, user = value.partition(':')
    return bool(sep and user.strip()) and provider != 'nous' and _owner_login(provider, user)


def _login_subject(value) -> bool:
    """An ``auth:v1:[provider, issuer, user]`` subject a sign-in stored (a provider is named)."""
    from gateway.session_identity import parse_authenticated_subject
    parts = parse_authenticated_subject(value)
    return parts is not None and bool(parts[2].strip()) and _owner_login(parts[0], parts[2])


def same_owner(stored, subject) -> bool:
    """THE owner rule: whether a stored owner stamp (a room owner, a history binding's subject, an
    admission's principal, a routeless row's ``user_id``) belongs to *subject*. True when equal, when it
    is the ``<provider>:<user>`` label of *subject*'s own login, or, while *subject* is the host owner,
    when it is any sign-in stamp an older build stored (an ``auth:v1`` login subject or a served
    dashboard's ``<provider>:<user>`` label). One-way: a non-owner subject never gains owner stamps.
    Every identity comparison on a stored owner goes through here so readers cannot disagree."""
    if stored == subject:
        return True
    if not isinstance(stored, str) or not stored or not subject:
        return False
    from hermes_state_mutation_binding import _login_label as label_of
    if stored == label_of(subject):
        return True
    return subject == host_owner_subject() and (_login_subject(stored) or _login_label(stored))


def owners_agree(first, second) -> bool:
    """Whether two stored owner stamps name the same owner (:func:`same_owner` either way round), for
    checks that compare one stored stamp with another rather than with a caller."""
    return same_owner(first, second) or same_owner(second, first)
