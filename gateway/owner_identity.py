"""The Webapp is the machine owner: one identity for every signed-in surface of this host.

Hermes is single-user. Native Desktop, the TUI and the CLI reach the gateway as the owner's own OS
user (``uid:<n>`` / ``sid:<sid>``, minted only for a same-user peer by the control channel). A Webapp
tab, a standalone dashboard's forwarded login or a phone signed in through the dashboard login is
that same person: sign-in only decides whether a browser may open the Webapp at all, it never names a
second owner. So every authenticated non-native principal acts as :func:`host_owner_subject`, and
the login stamps older builds stored (``auth:v1:[provider, issuer, user]`` subjects, the
``<provider>:<user>`` labels ``hermes serve`` stamped on Desktop rows) read as the owner's too.
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


# The interactive sign-ins Hermes bundles, plus the native grant a served Desktop presented
# (``local:uid:<n>``); a plugin-registered dashboard provider counts too (:func:`_login_provider`).
# ``session-token`` is the loopback Webapp's own credential and ``server-internal`` the PTY child a
# signed-in dashboard spawns: both act for whoever opened that page, the owner.
_BUNDLED_LOGINS = frozenset({'basic', 'nous', 'self-hosted', 'local', 'session-token', 'server-internal'})


def _login_provider(provider: str) -> bool:
    if provider in _BUNDLED_LOGINS:
        return True
    from hermes_cli.dashboard_auth.registry import list_session_providers
    return any(p.name == provider for p in list_session_providers())


def _login_label(value: str) -> bool:
    """``<provider>:<user>`` for a dashboard sign-in: the label ``hermes serve`` stamped for a login
    (session_transports). Any other prefix (``telegram:42``, ``uid:501``) names someone else, and a
    ``nous:`` stamp keeps its own rule (``legacy_account_label``: a Desktop row's account label)."""
    provider, sep, user = value.partition(':')
    return bool(sep and user.strip()) and provider != 'nous' and _login_provider(provider)


def _login_subject(value) -> bool:
    """An ``auth:v1:[provider, issuer, user]`` subject a sign-in stored (a provider is named)."""
    from gateway.session_identity import parse_authenticated_subject
    parts = parse_authenticated_subject(value)
    return parts is not None and bool(parts[2].strip()) and _login_provider(parts[0])


def same_owner(stored, subject) -> bool:
    """Whether a stored owner subject (room owner, history binding) belongs to *subject*: equal, or a
    login subject a Webapp/dashboard sign-in stored while *subject* is the host owner."""
    if stored == subject:
        return True
    return subject == host_owner_subject() and _login_subject(stored)


def owns_row_stamp(user_id, subject) -> bool:
    """Whether a routeless local row's ``user_id`` stamp names *subject*: :func:`same_owner`, the
    label of *subject*'s own login, or any ``<provider>:<user>`` login label a served dashboard
    stamped while *subject* is the host owner."""
    from hermes_state_mutation_binding import _login_label as label_of
    if same_owner(user_id, subject) or (user_id is not None and user_id == label_of(subject)):
        return True
    return isinstance(user_id, str) and subject == host_owner_subject() and _login_label(user_id)
