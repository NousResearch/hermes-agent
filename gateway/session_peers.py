"""Authenticated peer bots. Routes are selectors, never client-supplied addresses."""
import asyncio
import hashlib
import hmac
import json
import os
import re
from pathlib import Path
import urllib.error
import urllib.parse

from hermes_state_runtime import RuntimeStoreError

_ROUTE_KEY = os.urandom(32)


def _route_id(home, alias, entry, profile):
    from hermes_constants import hermes_home_key
    value = json.dumps([hermes_home_key(home), alias, entry['url'], profile])
    return hmac.new(_ROUTE_KEY, value.encode(), hashlib.sha256).hexdigest()


def _targets(home):
    from hermes_cli.subcommands import peer
    for alias, entry in peer._load_peers().items():
        if (not isinstance(alias, str) or not peer._PEER_NAME_RE.fullmatch(alias)
                or not isinstance(entry, dict) or not isinstance(entry.get('url'), str)):
            continue
        try:
            url = urllib.parse.urlsplit(entry['url'])
            if (url.scheme not in {'http', 'https'} or not url.hostname or url.username is not None
                    or url.password is not None or url.query or url.fragment):
                continue
            url.port
        except ValueError:
            continue
        # Existing registrations name the default mirror. Optional explicit profiles do not
        # imply access: every mirror independently authenticates the configured key.
        profiles = entry.get('profiles', [None])
        if not isinstance(profiles, list):
            continue
        for profile in dict.fromkeys(p for p in profiles if p is None or isinstance(p, str)):
            if profile is not None and not peer._PROFILE_RE.fullmatch(profile):
                continue
            if profile is not None:
                from hermes_cli.profiles import normalize_profile_name
                profile = normalize_profile_name(profile)
            yield _route_id(home, alias, entry, profile), alias, entry, profile


def _resolve(home, route_id):
    for route, alias, entry, profile in _targets(home):
        if hmac.compare_digest(route, route_id):
            return alias, entry, profile
    raise RuntimeStoreError('peer_route_unavailable')


def _credential(alias):
    from agent.secret_scope import get_secret
    from hermes_cli.subcommands.peer import _peer_key_env
    return (get_secret(_peer_key_env(alias), '') or '').strip()


def _public_catalog(raw, key):
    from gateway.hosted_room_peer import GatewayRoomCatalog
    from tui_gateway.contracts.canonical_peers import PeerCatalog
    catalog = GatewayRoomCatalog.from_mapping(raw).as_mapping()
    catalog.pop('endpoint', None)
    catalog = PeerCatalog.model_validate(catalog).model_dump()
    public = json.dumps(catalog)
    if key in public or 'HERMES_PEER_' in public or re.search(r'(?i)[a-z][a-z0-9+.-]*://', public):
        raise ValueError('peer catalog contains private transport data')
    return catalog


def _probe(route, alias, entry, profile):
    from hermes_cli.subcommands import peer
    row = dict(kind='peer', route_id=route, alias=alias, profile=profile or 'default',
               reachability='unauthenticated', installation_id=None, room_link=None)
    key = _credential(alias)
    if not key:
        return row
    try:
        result = peer._request(f'{peer._base_url(entry, profile)}/v1/capabilities', key,
                               timeout=5, max_response_bytes=65536, same_origin=True)
        bot = result.get('bot') or {}
        if not isinstance(bot, dict):
            raise ValueError('invalid peer identity')
        name = bot.get('profile')
        if not isinstance(name, str) or not peer._PROFILE_RE.fullmatch(name):
            row['reachability'] = 'unsupported'
            return row
        if profile not in (None, peer.DEFAULT_PROFILE_HANDLE) and name != profile:
            raise ValueError('profile mismatch')
        if key in name or 'HERMES_PEER_' in name:
            raise ValueError('peer identity contains private data')
        if bot.get('room_link') is None:
            installation = bot.get('installation_id')
            if (not isinstance(installation, str) or not re.fullmatch(r'install:[0-9a-f]{32}', installation)
                    or key in installation):
                raise ValueError('invalid installation identity')
            row.update(profile=name, reachability='reachable', installation_id=installation)
            return row
        catalog = _public_catalog(bot['room_link'], key)
        if catalog['execution_policy']['target_profile'] != name:
            raise ValueError('profile mismatch')
        row.update(profile=name, reachability='reachable',
                   installation_id=catalog['installation_id'], room_link=catalog)
    except urllib.error.HTTPError as exc:
        row['reachability'] = 'unauthenticated' if exc.code in (401, 403) else 'unreachable'
        exc.close()
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        row['reachability'] = 'unreachable'
    return row


def roster(home):
    return {'bots': [_probe(*target) for target in _targets(home)]}


def dm(home, params, subject):
    from hermes_cli.subcommands import peer
    alias, entry, profile = _resolve(home, params['route_id'])
    key = _credential(alias)
    if not key:
        raise RuntimeStoreError('peer_unauthenticated')
    try:
        if _probe(params['route_id'], alias, entry, profile)['reachability'] != 'reachable':
            raise RuntimeStoreError('peer_delivery_failed')
        base = peer._base_url(entry, profile)
        # Stable across gateway restarts; callers and target profiles cannot collide.
        from hermes_constants import hermes_home_key
        identity = json.dumps([hermes_home_key(home), subject, alias, entry['url'], profile, params['request_id']])
        request_id = 'peer-dm-' + hashlib.sha256(identity.encode()).hexdigest()
        result = peer.deliver_bot_chat(base, key, params['message'], request_id=request_id,
                                       strict_transport=True)
        result.pop('_response', None)
        return {'route_id': params['route_id'], **result}
    except (OSError, ValueError, LookupError, RuntimeError) as exc:
        # Transport errors can contain the URL, response body, and bearer material.
        raise RuntimeStoreError('peer_delivery_failed') from exc


def handlers(connection):
    from functools import partial
    return {name: partial(_dispatch, connection, name) for name in
            ('peers.roster', 'peers.dm', 'groups.create_mixed')}


async def _dispatch(connection, method, ref, params):
    from pydantic import ValidationError
    from tui_gateway.contracts.registry import CANONICAL_METHODS
    try:
        params = CANONICAL_METHODS[method].params.model_validate(params).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise RuntimeStoreError('invalid_params') from exc
    authority, actor = connection.authority, connection.actor
    capability = {'peers.roster': 'session:read', 'peers.dm': 'session:submit',
                  'groups.create_mixed': 'session:control'}[method]
    if capability not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    home = Path(authority.profile_id)

    def invoke():
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import profile_matches_home
        from hermes_cli.config_backend import ConfigBackendUnavailable
        if (Path(authority.db.db_path).resolve().parent != home.resolve()
                or (params.get('profile') and not profile_matches_home(params['profile'], home))):
            raise RuntimeStoreError('profile_mismatch')
        with _profile_runtime_scope(home):
            try:
                if method == 'peers.roster':
                    return roster(home)
                if method == 'peers.dm':
                    return dm(home, params, actor.subject)
                from gateway.session_peers_mixed import create_mixed
                return create_mixed(connection, home, params)
            except ConfigBackendUnavailable as exc:
                raise RuntimeStoreError('config_unavailable') from exc
            except RuntimeStoreError:
                raise
            except (ValueError, TypeError) as exc:
                raise RuntimeStoreError('invalid_params') from exc
    return await asyncio.to_thread(invoke)
