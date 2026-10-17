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


def policy_identity(policy):
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
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _fingerprint(value):
    return fingerprint_secret_value(value if isinstance(value, str) else json.dumps(value, sort_keys=True))


def config_reference(authority, session_id, policy, secrets):
    home = Path(authority.db.db_path).resolve().parent
    return PREFIX + json.dumps({'home': str(home), 'profile': authority.profile_id,
        'session': session_id, 'policy': policy_identity(policy),
        'entries': [[list(path), _fingerprint(value)] for path, value in secrets.items()]},
        sort_keys=True)


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
        if source['home'] != str(home) or source['profile'] != authority.profile_id:
            raise RuntimeStoreError('launch_credentials_unavailable',
                field='home' if source['home'] != str(home) else 'profile',
                detail='receipt scope mismatch: the frozen config reference belongs to another '
                       + ('home' if source['home'] != str(home) else 'profile'))
        if source['policy'] != policy_identity(policy):
            # The stamp covers the whole frozen config_json, secret fields and not: a profile edit
            # that rewrote the snapshot after the reference froze (a repointed base_url, keys
            # unchanged) fails it without touching a single secret, stranding the session on every
            # turn -- and on the /model re-stamp and the policy refresh that would repair it. The
            # per-entry fingerprints below still gate every value against the live config.yaml and
            # still refuse a changed route credential, so a stale stamp degrades to that check.
            logger.warning('Session %s: frozen config reference predates the current policy snapshot; '
                           'verifying its entries per fingerprint', source.get('session'))
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
                    raise RuntimeStoreError('launch_credentials_unavailable',
                        field='.'.join(str(part) for part in path if part is not None),
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
        raise RuntimeStoreError('launch_credentials_unavailable') from exc


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
