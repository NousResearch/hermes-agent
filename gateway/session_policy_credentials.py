"""Borrow frozen config credentials from the existing profile source, not a vault."""
import hashlib
import hmac
import json
import logging
from dataclasses import asdict
from pathlib import Path

from agent.credential_persistence import fingerprint_secret_value
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)
PREFIX = 'profile-config-v1:'
_ABSENT = object()


def _defined(root, path):
    """The value *root* defines at *path*, else ``_ABSENT`` when the key is simply not there."""
    value = root
    for key in path:
        if isinstance(value, list) and type(key) is int:
            if not 0 <= key < len(value):
                return _ABSENT
        elif not isinstance(value, dict) or key not in value:
            return _ABSENT
        value = value[key]
    return value


def _route_paths(policy):
    """Secret paths the session's own route authenticates with: its ``model:`` block, a direct alias's
    key (``api_key_from``) and the one ``providers:`` / ``custom_providers:`` entry the runtime's own
    named-provider selection routes its provider through."""
    from hermes_cli.runtime_provider_custom import (custom_provider_config_path_for_url,
                                                    named_custom_provider_config_path)
    config = json.loads(policy.config_json)
    model = config.get('model') if isinstance(config.get('model'), dict) else {}
    paths = {('model', key) for key in model}
    if isinstance(model.get('api_key_from'), list):
        paths.add(tuple(model['api_key_from']))
    provider = str(model.get('provider') or '')
    if provider.strip().lower() == 'custom':
        # A bare custom route authenticates with the entry owning its base_url (the pool's owner).
        selected = custom_provider_config_path_for_url(model.get('base_url'), config)
    else:
        selected = named_custom_provider_config_path(provider, config)
    entry = _defined(config, selected) if selected else _ABSENT
    if isinstance(entry, dict):
        paths |= {selected + (field,) for field in entry}
    return paths


class RecoveredSecrets(dict):
    """The frozen secrets still valid, plus ``dropped``: the paths whose value changed or was removed
    in config.yaml since the session started (never supplied; the policy disables what used them)."""
    dropped = ()
    session = None


def policy_identity(policy, unhashed=()):
    data = asdict(policy)
    data.pop('credential_ref')
    data.pop('config_secret_ref')
    # Fields added after release are hashed only when set, so an older session's reference still matches.
    if data.get('editor_mcp_json') is None:
        data.pop('editor_mcp_json', None)
    if not data.get('coding_posture'):
        data.pop('coding_posture', None)
    for name in ('client_env_json', 'client_env_rules_json'):
        if data.get(name) is None:
            data.pop(name, None)
    if not data.get('client_env_remote'):
        data.pop('client_env_remote', None)
    if data.get('edit_approval_policy', 'ask') == 'ask':
        data.pop('edit_approval_policy', None)
    for name in unhashed:  # _legacy_identities: a field the stamping gateway did not have yet
        data.pop(name, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# Policy fields a release added after frozen config secrets shipped, hashed unconditionally by the
# gateway that added them. A reference an older gateway stamped never hashed them; at their defaults
# they carry nothing that gateway could have seen, so the same policy without them is its policy.
# The commit that added each, to audit this list against the stamping releases:
_LATER_POLICY_FIELDS = (('skills_prompt', None),          # 2ee1cd1ebb02
                        ('kanban_json', None),            # c8f8d9ae6384
                        ('safe_mode', False),             # 53d362e18741
                        ('ignore_user_config', False))    # 53d362e18741


def _legacy_identities(policy):
    """Every whole-policy hash a gateway older than ``bound`` could have stamped for *policy*: the
    current hash, plus each subset of the later fields dropped while they are at their defaults."""
    from itertools import combinations
    unset = [name for name, default in _LATER_POLICY_FIELDS if getattr(policy, name) == default]
    return {policy_identity(policy, dropped)
            for size in range(len(unset) + 1) for dropped in combinations(unset, size)}


def _fingerprint(value):
    return fingerprint_secret_value(value if isinstance(value, str) else json.dumps(value, sort_keys=True))


# What a frozen secret authenticates: the endpoint fields of the block that holds it (``model:``, a
# ``providers:`` entry, an MCP server), plus the endpoint its route RESOLVES to when that lives in another
# block. A secret is bound to these, never to the rest of the policy. ``api`` is an entry's URL field
# (runtime_provider_custom._entry_url reads it before ``url``/``base_url``).
_ENDPOINT_KEYS = ('provider', 'base_url', 'api', 'url', 'command', 'args', 'api_mode')
_ENTRY_SECTIONS = ('providers', 'custom_providers')
# An auxiliary route naming no provider of its own (or ``main``/``auto``) goes where the session's
# ``model:`` route goes, so its key authenticates that endpoint too.
_AUX_FOLLOWS_MAIN = ('', 'auto', 'main')


def _route_owner(config, path):
    """The config path of the ROUTE block a secret at *path* rides with, else None. Every route shape a
    consumer reads: ``model:`` and its aliases (``model.aliases.<n>``, ``model_aliases.<n>``), both
    fallback sections in dictionary (one route) and list form, and each ``auxiliary.<task>`` route with
    its ``fallback_chain`` entries. A secret nested deeper in a route (headers, extra_body) rides with it."""
    head, rest = path[0], tuple(path[1:])
    model = config.get('model') if isinstance(config.get('model'), dict) else {}
    if isinstance(model.get('api_key_from'), list) and tuple(model['api_key_from']) == tuple(path):
        # ``/model <alias>`` with a literal key: config() projects the alias entry's key into ``model.api_key``,
        # so it authenticates the session's ``model:`` route, not the alias entry it is stored under.
        return ('model',)
    if head == 'model':
        if rest[:1] == ('aliases',) and len(rest) > 2:
            return ('model', 'aliases', rest[1])
        return ('model',)
    if head == 'model_aliases' and len(rest) > 1:
        return ('model_aliases', rest[0])
    if head in ('fallback_model', 'fallback_providers'):
        if isinstance(config.get(head), list) and rest and type(rest[0]) is int:
            return (head, rest[0])
        return (head,)
    if head == 'auxiliary' and len(rest) > 1:
        task = rest[0]
        if rest[1] == 'fallback_chain' and len(rest) > 3 and type(rest[2]) is int:
            return ('auxiliary', task, 'fallback_chain', rest[2])
        return ('auxiliary', task)
    return None


def _owner_block(config, path):
    """The config path of the block whose endpoint the secret at *path* authenticates: a route block
    (_route_owner) or a provider entry as a whole, else the secret's own parent block."""
    route = _route_owner(config, path)
    if route is not None:
        return route
    if path[0] in _ENTRY_SECTIONS:
        return tuple(path[:2])
    return tuple(path[:-1])


def _resolved(owner, block, config):
    """``(provider_identity, url)`` the route *block* at *owner* is actually sent to."""
    from gateway.session_route_identity import resolved_route
    if owner[0] == 'auxiliary':
        provider = str(block.get('provider') or '').strip().lower()
        if provider in _AUX_FOLLOWS_MAIN and not str(block.get('base_url') or '').strip():
            main = config.get('model') if isinstance(config.get('model'), dict) else {}
            return ['main', *resolved_route(main, config)]
    return list(resolved_route(block, config))


# Endpoint fields holding a URL: digested in normalize_route_base_url's form, so a re-spelling of the same
# endpoint (host case, a trailing slash, the scheme's default port) is not a moved endpoint.
_URL_KEYS = ('base_url', 'api', 'url')


def _url(value):
    from hermes_cli.route_identity import normalize_route_base_url
    return normalize_route_base_url(str(value or '').strip())


def _binding(config, path):
    """Digest of the endpoint the secret at *path* authenticates in the frozen *config* (a terminal-scope
    secret, ``path[0] is None``, authenticates no endpoint)."""
    owner = _owner_block(config, path) if path and path[0] is not None else None
    block = _defined(config, owner) if owner else _ABSENT
    endpoint = {k: block[k] for k in _ENDPOINT_KEYS if k in block} if isinstance(block, dict) else {}
    if isinstance(block, dict) and _route_owner(config, path) is not None:
        # A route's key authenticates what it resolves to: a named ``providers:`` entry's URL included.
        endpoint['resolved'] = _resolved(owner, block, config)
        endpoint['resolved'][-1] = _url(endpoint['resolved'][-1])
    elif isinstance(block, dict) and path[0] in _ENTRY_SECTIONS:
        from hermes_cli.runtime_provider_custom import _entry_url
        endpoint['resolved'] = _url(_entry_url(block))
    for key in _URL_KEYS:
        if isinstance(endpoint.get(key), str):
            endpoint[key] = _url(endpoint[key])
    return hashlib.sha256(json.dumps([list(path), endpoint], sort_keys=True).encode()).hexdigest()


def config_reference(authority, session_id, policy, secrets):
    home = Path(authority.db.db_path).resolve().parent
    config = json.loads(policy.config_json)
    return PREFIX + json.dumps({'home': str(home), 'profile': authority.profile_id,
        # ``policy`` (the whole-policy hash) is still written for an older gateway reading this reference;
        # this gateway checks ``bound``: each secret against the endpoint it authenticates (CFG-10).
        'session': session_id, 'policy': policy_identity(policy),
        'bound': [[list(path), _binding(config, path)] for path in secrets],
        'entries': [[list(path), _fingerprint(value)] for path, value in secrets.items()]},
        sort_keys=True)


def _field(path):
    return '.'.join(str(part) for part in path if part is not None)


def _check_scope(authority, source, policy, home, skip):
    """Refuse, by name, a reference from another home or profile, or a secret whose endpoint the policy
    moved without re-binding it. A change to anything else in the policy (model name, reasoning, toolsets,
    an unrelated provider or MCP server, a policy field a later release added) is not a credential change."""
    for name, mine in (('home', str(home)), ('profile', authority.profile_id)):
        if source[name] != mine:
            raise RuntimeStoreError('launch_credentials_unavailable', field=name,
                                    detail=f'the frozen config credentials belong to another {name}')
    if 'bound' not in source:
        # Stamped before ``bound``: its whole-policy hash is its only endpoint guard, so it must still
        # match -- allowing for the policy fields added since, at their defaults (_legacy_identities).
        # The next re-bind writes ``bound``.
        if source['policy'] not in _legacy_identities(policy):
            raise RuntimeStoreError('launch_credentials_unavailable', field='config_secret_ref',
                                    detail='stamped by an older gateway for another session policy')
        return
    config = json.loads(policy.config_json)
    for path, digest in source['bound']:
        if path[0] not in skip and not hmac.compare_digest(_binding(config, path), digest):
            raise RuntimeStoreError('launch_credentials_unavailable', field=_field(path),
                                    detail='its endpoint moved in the session policy without a re-bind')


def secrets_moved(ref, secrets, sections):
    """Whether *secrets* (the private values of *sections*, as the profile has them now) differ from the ones
    the durable reference *ref* froze there, by fingerprint. Unknown for an in-memory reference: False."""
    if ref is not None and not ref.startswith(PREFIX):
        return False
    entries = json.loads(ref[len(PREFIX):])['entries'] if ref else []
    frozen = {tuple(path): fingerprint for path, fingerprint in entries if path[0] in sections}
    return frozen != {path: _fingerprint(value) for path, value in secrets.items()}


def recover_config_secrets(authority, policy, skip=None):
    """The frozen config secrets; entries under a *skip* section are neither re-read nor checked. Every
    caller of a following session skips its followed sections: the live profile supplies them, so a key
    rotation there is not a revoked launch credential."""
    if skip is None:
        from gateway.session_policy import FOLLOW_SECTIONS
        skip = FOLLOW_SECTIONS if policy.follows_profile else ()
    ref = policy.config_secret_ref
    if not ref.startswith(PREFIX):
        values = getattr(authority, '_local_config_secrets', {}).get(ref)
        if values is None:
            raise RuntimeStoreError('launch_credentials_unavailable')
        return {path: value for path, value in values.items() if path[0] not in skip}
    try:
        source = json.loads(ref[len(PREFIX):])
        home = Path(authority.db.db_path).resolve().parent
        _check_scope(authority, source, policy, home, skip)
        cached = getattr(authority, '_local_config_secrets', {}).get(ref)
        if cached is not None:
            return {path: value for path, value in cached.items() if path[0] not in skip}
        from gateway.run import _load_gateway_config
        entries = [entry for entry in source['entries'] if entry[0][0] not in skip]
        config = _load_gateway_config(home / 'config.yaml') if any(path[0] is not None for path, _ in entries) else {}
        route = None  # computed on the first mismatch only: an unchanged source never needs it
        values, dropped = RecoveredSecrets(), []
        for path, fingerprint in entries:
            value, lookup = config, path
            if path[0] is None:
                from tools.terminal_scope import build_profile_terminal_scope
                value, lookup = build_profile_terminal_scope(home), path[1:]
            value = _defined(value, lookup)
            actual = None if value is _ABSENT else _fingerprint(value)
            if actual is None or not hmac.compare_digest(actual, fingerprint):
                if route is None:
                    route = _route_paths(policy)
                if tuple(path) in route:
                    raise RuntimeStoreError('launch_credentials_unavailable', field=_field(path),
                        detail=('removed from' if value is _ABSENT else 'changed in')
                        + ' config.yaml since this session started')
                # Any other secret (an MCP env token, an unused provider's key) is simply no longer
                # supplied: the frozen tree holds None there, never the profile's new value.
                logger.warning('Session %s: frozen config secret %s no longer matches the profile '
                               'config; skipping it', source.get('session'), path)
                dropped.append(tuple(path))
                continue
            values[tuple(path)] = value
        values.dropped, values.session = tuple(dropped), source.get('session')
        return values
    except RuntimeStoreError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        logger.warning('Frozen config reference unreadable: %s', type(exc).__name__)
        raise RuntimeStoreError('launch_credentials_unavailable', field='config_secret_ref',
                                detail=f'unreadable ({type(exc).__name__})') from exc


def _entry_owner_path(entry, config):
    """The ``providers:`` / ``custom_providers:`` entry a fallback entry authenticates through."""
    from hermes_cli.runtime_provider_custom import (custom_provider_config_path_for_url,
                                                    named_custom_provider_config_path)
    provider = str(entry.get('provider') or '').strip()
    if provider.lower() == 'custom':
        return custom_provider_config_path_for_url(entry.get('base_url'), config)
    return named_custom_provider_config_path(provider, config) if provider else None


def disable_dropped_secrets(config, dropped, session=None):
    """Disable, in the session's effective *config*, whatever authenticated with a *dropped* frozen
    secret, instead of leaving it with a null key (null means 'resolve another credential' and would
    send another account's key). A fallback using it is removed from the chain; an ``auxiliary.<task>``
    override using it is replaced by the session's primary route (``provider: main``). Never substitutes."""
    if not dropped:
        return config
    from agent.auxiliary_task_config import _AUX_ROUTE_KEYS
    auxiliary = config.get('auxiliary') if isinstance(config.get('auxiliary'), dict) else {}
    for task in sorted({path[1] for path in dropped if path[0] == 'auxiliary' and len(path) > 2}, key=str):
        block = auxiliary.get(task)
        if isinstance(block, dict):
            # The task's own route is gone with its key: the session's primary route, its own credential.
            auxiliary[task] = {**{k: v for k, v in block.items() if k not in _AUX_ROUTE_KEYS
                                  and k != 'fallback_chain'}, 'provider': 'main'}
            logger.warning('Session %s: auxiliary task %s returned to the primary route: its credential '
                           'changed in config.yaml since this session started', session, task)
    owners = {path[:2] for path in dropped if path[0] in ('providers', 'custom_providers') and len(path) > 2}
    for section in ('fallback_providers', 'fallback_model'):
        raw = config.get(section)
        if isinstance(raw, dict):
            entries, single = [raw], True
        elif isinstance(raw, list):
            entries, single = list(raw), False
        else:
            continue
        kept = []
        for index, entry in enumerate(entries):
            prefix = (section,) if single else (section, index)
            own = any(path[:len(prefix)] == prefix for path in dropped)
            owner = _entry_owner_path(entry, config) if isinstance(entry, dict) and owners else None
            if own or (owner is not None and tuple(owner) in owners):
                logger.warning('Session %s: fallback %s/%s removed: its credential changed in config.yaml '
                               'since this session started', session,
                               (entry or {}).get('provider') if isinstance(entry, dict) else '?',
                               (entry or {}).get('model') if isinstance(entry, dict) else '?')
                continue
            kept.append(entry)
        if single:
            if kept:
                config[section] = kept[0]
            else:
                config.pop(section, None)
        else:
            config[section] = kept
    return config
