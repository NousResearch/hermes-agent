"""The create-retry rules the store applies when a local session's creation is retried: the request as the client sent it, the frozen-request comparison and the client-environment match under the rules frozen with the session.

Standard library only.
"""

from __future__ import annotations

import json


# "PASS" is intentionally absent: it false-positives on BYPASS_CACHE /
# COMPASS_DIR / PASSENGER_HOST while PASSWORD/PASSWD already cover credentials.
_SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSWD", "AUTH", "DSN",
                      "WEBHOOK", "CREDS", "BEARER", "APIKEY", "COOKIE")


def is_secret_env_name(name: str) -> bool:
    """The credential-name policy, shared with a local session's client environment (``agent.session_client_env``)."""
    return any(s in name.upper() for s in _SECRET_SUBSTRINGS)


_MAX_NAMES = 256
_MAX_BYTES = 256 * 1024


def _travels(name: str, secret=None) -> bool:
    """A toolchain name may travel unless it is a credential by execute_code's secret policy (even when
    listed: ``NODE_AUTH_TOKEN`` matches ``NODE_``; secrets reach tools through ``terminal.env_passthrough``)
    or a Hermes control name (a delegated child's markers, Kanban pins). ``secret``: a session's frozen
    copy of that policy's substrings."""
    hidden = is_secret_env_name(name) if secret is None else any(s in name.upper() for s in secret)
    return not hidden and not name.startswith('HERMES_')


def _kept(name: str, rules: dict) -> bool:
    return name in rules['behaviour'] or (_travels(name, rules['secret']) and (
        name in rules['names'] or name.startswith(tuple(rules['prefixes']))))


def normalise_client_env(value, rules) -> dict[str, str]:
    """The validated map filtered by ``rules`` (``None``: unfiltered, for a map frozen before rules were).
    Raises ``ValueError`` for anything but a bounded flat string-to-string map."""
    if (not isinstance(value, dict) or len(value) > _MAX_NAMES
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items())
            or sum(len(k) + len(v) for k, v in value.items()) > _MAX_BYTES):
        raise ValueError('client_env must be a map of variable names to string values')
    return {name: value[name] for name in sorted(value) if rules is None or _kept(name, rules)}


def retry_matches(committed: dict, request: dict) -> bool:
    """Whether a retried ``session.create`` (``request``: its params) carries the client environment the
    committed session (``committed``: its stored policy) froze. The complete retried map, normalised by
    the rules frozen with the session, must equal the stored one: an added name those rules keep, a
    missing stored name or a changed value is a different request; one they drop never counts. Staging
    and the transactional commit both judge by this. A session committed without a map (Desktop, one
    from before the field) still matches only a retry that sends none."""
    stored = committed.get('client_env_json')
    if stored is None or 'client_env' not in request:
        return stored is None and 'client_env' not in request
    rules = committed.get('client_env_rules_json')
    try:
        sent = normalise_client_env(request['client_env'], None if rules is None else json.loads(rules))
    except ValueError:
        return False
    return sent == json.loads(stored)


def client_request(request_json, derived=None):
    """A frozen create request as the client sent it: an owner-resolved ``-m`` route (``model_requested``)
    compares by the name the client typed, not by the model and the *derived* fields (a receipt from
    before ``route_derived``: the endpoint) its ``${VAR}`` refs expanded to at the time. ``desktop_local``
    (``gateway.session_policy.LOCAL_DESKTOP_CHAT``), which an earlier owner stamped into the request, was
    never sent."""
    request = json.loads(request_json)
    request.pop('desktop_local', None)
    if derived is None:
        request.pop('route_derived', None)
    if 'model_requested' in request:
        request.pop('model', None)
        for key in ('base_url',) if derived is None else derived:
            request.pop(key, None)
    return request


def same_create_request(saved_json, retry_json):
    """Whether a create retry carries the frozen request it resumes: one check for a retry that finds the
    session committed and one staged before it was. The saved receipt's provenance names the owner's
    fields; one from before a same-name alias got ``model_requested`` compares exactly, as it did."""
    derived = json.loads(saved_json).get('route_derived')
    if client_request(saved_json, derived) == client_request(retry_json, derived):
        return True
    retry = json.loads(retry_json)
    if retry.get('model_requested') == retry.get('model'):
        retry.pop('model_requested', None)
        retry.pop('route_derived', None)
    return json.loads(saved_json) == retry


def same_create_retry(saved_policy, retry_json, client_request):
    """The one exact-retry identity of a create: ``same_create_request`` on what the client sent, and its
    ``client_env`` (``client_request``: the retry's own, never stored) judged by the rules frozen with
    *saved_policy*. Used by a retry that finds the session committed, at commit, and by one overlapping
    the attempt that holds the session's key slot."""
    return same_create_request(saved_policy['request_json'], retry_json) and retry_matches(saved_policy, client_request)
