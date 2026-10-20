"""Resolve explicit model changes using the existing source model switch pipeline."""
import asyncio
import json
from dataclasses import asdict, replace
from typing import NamedTuple

from gateway.session_route_identity import ROUTE_BOUND_KEYS
from hermes_state_runtime import RuntimeStoreError


class _Route(NamedTuple):
    provider: str
    model: str
    base_url: str
    api_key: object  # a callable bearer (Azure Entra) stays callable


def _current_route(authority, live, old):
    """``(config, route)`` the session runs now, both from ONE read of the owning profile.

    A following session runs the profile's current route, so its frozen creation endpoint beside
    the current key would send that key to a host the profile left. Main switched from the live
    runtime's whole tuple (``tui_gateway.model_switch._current_model_runtime``). The provider is
    the configured name: a named ``providers:`` entry resolves to the bare ``custom`` class, and a
    pick on the same entry must not read as a route change. Only the session's own route counts:
    a fallback is where a turn lands, not what the user switches from. A route that cannot
    authenticate here must not block leaving it: like main's agent-less ``--provider`` path, the
    pick then switches from the configured provider alone and resolves its own target.
    """
    from gateway.session_policy import profile_read_scope
    from hermes_cli.auth import AuthError
    with profile_read_scope(old, authority):
        config = old.config(authority)
        block = config.get('model') or {}
        try:
            model, runtime = authority.runner._resolve_session_agent_runtime(
                source=live.source, session_key=live.route, fallback=False)
        except AuthError:
            return config, _Route(block.get('provider') or '', old.current_model(authority) or '',
                                  block.get('base_url') or '', '')
    return config, _Route(block.get('provider') or runtime.get('provider') or '', model or '',
                          runtime.get('base_url') or '', runtime.get('api_key') or '')


async def prepare_model(authority, live, payload, prepared):
    from gateway.session_policy import restore_policy, bind_launch_key, bound_route_holds, route_launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    from hermes_cli.model_switch import apply_model_selection, switch_model
    from hermes_cli.config import get_compatible_custom_providers
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    config, current = _current_route(authority, live, old)
    # A pick is an explicit user action: it resolves against the owning profile's CURRENT named
    # providers, as CLI/TUI ``/model`` do (#139740). A following session's ``config`` already is
    # the profile's current read; every other session keeps ``config`` for its own route below.
    resolution = config if old.follows_profile else await asyncio.to_thread(
        _provider_resolution_config, authority, config, old.ignore_user_config)

    def switch(**kwargs):
        result = switch_model(**kwargs)
        return result, _alias_credential(authority, result)
    result, alias_credential = await asyncio.to_thread(switch, raw_input=payload['model'],
        current_provider=current.provider, current_model=current.model, current_base_url=current.base_url,
        current_api_key=current.api_key, explicit_provider=payload.get('provider', ''),
        is_global=False, user_providers=resolution.get('providers'),
        custom_providers=get_compatible_custom_providers(resolution))
    if not result.success:
        raise RuntimeStoreError('model_resolution_failed')
    refusal = await _selection_guard_refusal(authority, live, payload, prepared['snapshot'], current, result)
    if refusal is not None:
        return refusal
    frozen = old.config()
    # The committed selection freezes the ONE picked endpoint (its key stays a private secret).
    provider_secrets, retired, moved = ({}, (), {}) if old.follows_profile else _freeze_selected_provider(
        resolution, frozen, result.target_provider)
    if old.follows_profile:
        # A following session switches FROM the profile's current route: measured from its creation
        # endpoint, a pick on the current one read as a route change and dropped its key_env pointer.
        followed = config.get('model') or {}
        # An inline key (or an alias key's pointer) authenticates the route it was picked on: once the
        # followed route moved, the same-route cleanup below would keep it for the new endpoint.
        held = bound_route_holds(frozen.get('model'), frozen, followed, config)
        # Every other route-bound key follows the profile's current route.
        kept = ('api_key', 'api', 'api_key_from') if held else ()
        # ``route_moved`` names the model keys that left the frozen route. It must NOT shadow
        # ``moved``: that dict (from ``_freeze_selected_provider``) remaps the secrets paths
        # below, and shadowing it with this tuple crashed every pick on a following session
        # that carries config secrets ('tuple' object has no attribute 'get').
        route_moved = tuple(k for k in ROUTE_BOUND_KEYS if k not in kept)
        frozen['model'] = {k: v for k, v in (frozen.get('model') or {}).items() if k not in route_moved}
        # The profile's inline key and its private-secret pointer are not copied into the receipt:
        # its secrets stay at the profile's own path, never re-committed under this session.
        frozen['model'].update({k: followed[k] for k in route_moved
                                if k in followed and k not in ('api_key', 'api', 'api_key_from')})
    # The one selection shape config.yaml gets (#25106): api_mode follows the target, a context pin
    # and the endpoint-bound credential fields (inline key, key_env/api_key_env) drop on a route change.
    # Off-loop: the context-pin check can do cold-start disk I/O.
    frozen['model'] = await asyncio.to_thread(apply_model_selection, frozen.get('model'), result)
    if 'reasoning' in payload:
        # The level picked with the model lands in the same commit (as prepare_reasoning writes it).
        from gateway.session_policy import pin_reasoning_effort
        pin_reasoning_effort(frozen, payload['reasoning'], session=True)
    if old.follows_profile:
        # A committed /model holds while the profile still targets what it targets now (main's
        # composer_override_profile); a later profile edit supersedes it.
        from pathlib import Path
        from gateway.run import _load_gateway_config
        from gateway.session_policy import present_sections, profile_target
        live = await asyncio.to_thread(
            lambda: _load_gateway_config(Path(authority.db.db_path).resolve().parent / 'config.yaml'))
        frozen['model']['selection_profile'] = list(profile_target(present_sections(live)))
    # A direct alias's own credential is part of the selection (never ``result.api_key``, which the
    # switch resolved for its probe): its env pointer is committed and re-read every turn; a literal
    # key is a private config secret at the alias entry's own path (what a restarted owner re-reads)
    # and the receipt keeps only a pointer to it, which config() projects into ``model.api_key``.
    key_env, source, alias_key = alias_credential
    key_env = key_env or _selected_provider_key_env(resolution, result.target_provider)
    if key_env:
        frozen['model']['key_env'] = key_env
    if source:
        frozen['model'].update(api_key=None, api_key_from=list(source))
    elif 'api_key' not in frozen['model']:
        frozen['model'].pop('api_key_from', None)
    # Creation identity (request_json) remains immutable; runtime selection lives in the policy.
    policy = replace(old, model=result.new_model, config_json=json.dumps(frozen))
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    # The picked entry's private subtree is replaced whole: a key or header its current declaration
    # withdrew must not follow it to the new endpoint.
    # An unrelated entry the pick displaced carries its own secrets to its new index.
    secrets = {moved.get(path[:2], path[:2]) + path[2:]: value for path, value in secrets.items()}
    secrets = {path: value for path, value in secrets.items() if path[:2] not in retired}
    secrets.update(provider_secrets)
    # A cleared field's private value must not be re-hydrated into the new route by config().
    secrets = {path: value for path, value in secrets.items()
               if not (len(path) == 2 and path[0] == 'model' and path[1] not in frozen['model'])}
    if source:
        secrets.pop(('model', 'api_key'), None)
        secrets[source] = alias_key
    # Re-fingerprint frozen config references because policy identity includes model.
    # An explicit launch key belongs to the endpoint it authenticated, not to a provider name:
    # custom -> custom on another base_url must not carry it (same route identity as config keys).
    # Measured on resolved routes (a respelled provider on one endpoint is one route; a named entry
    # whose base_url moved is another): the key survives only while it still authenticates the route
    # the session runs now and the pick stays on that route.
    key = route_launch_key(authority, old, config)
    if key is None or not bound_route_holds(config.get('model'), config, frozen['model'], config):
        policy = replace(policy, credential_ref=None)
        key = None
    policy = bind_launch_key(authority, prepared['snapshot']['receipt']['session_id'], policy, key,
                             config_secrets=secrets)
    return dict(prepared, policy=asdict(policy))


def _provider_resolution_config(authority, config, bypass=False):
    """*config* with the owning profile's CURRENT named providers over its frozen ones, used only to
    resolve and freeze a pick. Reads this owner's own config.yaml (never another profile's); a
    frozen entry the profile no longer declares stays resolvable. A session that bypasses the
    profile (``ignore_user_config``, implied by safe mode) resolves only its own frozen config.
    ``${VAR}`` refs expand against the profile's CURRENT ``.env``, not the daemon's start env."""
    from pathlib import Path
    from gateway.session_local import load_profile_config
    if bypass:
        return config
    current = load_profile_config(Path(authority.db.db_path).resolve().parent)
    if not isinstance(current, dict):
        return config
    resolved = dict(config)
    frozen_providers, current_providers = config.get('providers'), current.get('providers')
    if isinstance(frozen_providers, dict) or isinstance(current_providers, dict):
        current_providers = current_providers if isinstance(current_providers, dict) else {}
        slugs = {_provider_key_slug(key) for key in current_providers}
        # A respelled key is the same provider: the profile's spelling replaces the frozen one.
        resolved['providers'] = {**{key: entry for key, entry in (
            frozen_providers if isinstance(frozen_providers, dict) else {}).items()
            if _provider_key_slug(key) not in slugs}, **current_providers}
    frozen_custom, current_custom = config.get('custom_providers'), current.get('custom_providers')
    if isinstance(frozen_custom, list) or isinstance(current_custom, list):
        current_custom = list(current_custom) if isinstance(current_custom, list) else []
        names = {_custom_provider_name(entry) for entry in current_custom if isinstance(entry, dict)}
        # Current entries keep their profile positions (the path cold recovery re-reads).
        resolved['custom_providers'] = current_custom + [
            entry for entry in (frozen_custom if isinstance(frozen_custom, list) else [])
            if not isinstance(entry, dict) or _custom_provider_name(entry) not in names]
        resolved['_profile_custom_count'] = len(current_custom)
    return resolved


def _custom_provider_name(entry):
    """The runtime's canonical identity: ``My Provider`` and ``my-provider`` are one provider."""
    from hermes_cli.providers import custom_provider_slug
    return custom_provider_slug(str(entry.get('name') or ''), str(entry.get('provider_key') or ''))


def _selected_provider_key_env(config, provider):
    """Return a selected ``providers.<slug>`` credential pointer, never its secret value."""
    providers = config.get('providers')
    entry = providers.get(provider) if isinstance(providers, dict) else None
    if not isinstance(entry, dict):
        return None
    return str(entry.get('key_env') or entry.get('api_key_env') or '').strip() or None


def _freeze_selected_provider(config, frozen, provider):
    """Copy only the picked named provider's declaration into the frozen session route.

    The next turn resolves from the session's own configuration, not a later profile edit. Private
    fields move to the policy's secret store, as policy creation does; nothing else moves. Returns
    ``(secrets, retired)``: the entry's private values and every ``(section, key)`` whose previous
    private subtree they replace, plus ``moved``: ``{old (section, index): new}`` for an unrelated entry
    the pick displaced. A legacy entry from the profile is frozen at its profile index (the path cold
    recovery re-reads); an equivalent earlier spelling is blanked, never left to win.
    """
    from copy import deepcopy
    from gateway.session_policy import _extract_config_secrets
    from hermes_cli.runtime_provider_custom import named_custom_provider_config_path
    path = named_custom_provider_config_path(provider, config)
    if path is None:
        return {}, (), {}
    section, index = path
    entries = config.get(section)
    if not isinstance(entries, (dict, list)) or not isinstance(entries[index], dict):
        return {}, (), {}
    copied = deepcopy(entries[index])
    retired, moved = {(section, index)}, {}
    if section == 'providers':
        previous = frozen.get(section) if isinstance(frozen.get(section), dict) else {}
        stale = {key for key in previous if _provider_key_slug(key) == _provider_key_slug(index)}
        frozen[section] = {**{k: v for k, v in previous.items() if k not in stale}, index: copied}
        retired |= {(section, key) for key in stale}
    else:
        frozen_entries = list(frozen.get(section) or [])
        name = _custom_provider_name(copied)
        same = [i for i, entry in enumerate(frozen_entries)
                if isinstance(entry, dict) and _custom_provider_name(entry) == name]
        if index >= config.get('_profile_custom_count', len(entries)):
            index = same[0]  # a frozen-only entry stays where it is
        frozen_entries += [{}] * (index + 1 - len(frozen_entries))
        displaced = frozen_entries[index]
        if isinstance(displaced, dict) and displaced and _custom_provider_name(displaced) != name:
            moved[(section, index)] = (section, len(frozen_entries))
            frozen_entries.append(displaced)  # still resolvable by name, with its own secrets
        for i in same:
            frozen_entries[i] = {}
        frozen_entries[index] = copied
        retired |= {(section, i) for i in same} | {(section, index)}
        frozen[section] = frozen_entries
    secrets = {}
    _extract_config_secrets(frozen[section][index], secrets, (section, index))
    return secrets, retired, moved


def _provider_key_slug(key):
    """A ``providers:`` key's canonical identity: ``Review Prov`` and ``review-prov`` are one."""
    from hermes_cli.providers import custom_provider_slug
    return custom_provider_slug('', str(key))


def _alias_credential(authority, result):
    """``(key_env, source_path, literal_key)`` the direct alias a switch resolved declares for its own
    endpoint, read from THIS owner's config.yaml: the file a restarted owner re-reads, never the
    process-global alias cache another served profile may refill after the switch. A raw
    ``api_key: "${VAR}"`` is the pointer ``VAR``; ``(None, None, None)`` when it declares none."""
    from pathlib import Path
    from gateway.run import _load_gateway_config
    from hermes_cli.config import _env_ref_var_name, read_user_config_raw
    name = result.resolved_via_alias if result.success else ''
    path = Path(authority.db.db_path).resolve().parent / 'config.yaml'
    raw = read_user_config_raw(path) if name else {}
    for section in (('model_aliases',), ('model', 'aliases')):
        model = raw.get('model')  # `model: name` (a scalar) is valid config and declares no aliases
        entries = raw.get(section[0]) if len(section) == 1 else (model.get('aliases') if isinstance(model, dict) else None)
        for written, entry in (entries.items() if isinstance(entries, dict) else ()):
            if not isinstance(entry, dict) or not entry.get('model') or str(written).strip().lower() != name:
                continue
            key = str(entry.get('api_key') or '').strip()
            pointer = _env_ref_var_name(key[2:-1]) if key.startswith('${') and key.endswith('}') else None
            if pointer:
                return pointer, None, None
            if not key:
                return str(entry.get('key_env') or '').strip() or None, None, None
            value = _load_gateway_config(path)
            for part in (*section, written, 'api_key'):
                value = value.get(part) if isinstance(value, dict) else None
            if not value:
                raise RuntimeStoreError('model_resolution_failed')
            return None, (*section, written, 'api_key'), value
    return None, None, None


def _persisted_selection_context(authority, snapshot, current):
    from agent.usage_anchor import persisted_anchor_tokens
    from hermes_cli.model_selection_guards import SelectionContext
    target = snapshot.get('target') or snapshot['receipt']['session_id']
    tokens = persisted_anchor_tokens(authority.db, target, authority.db.get_messages_as_conversation(target))
    return SelectionContext(context_tokens=tokens, current_model=current.model or None) if tokens else None


async def _selection_guard_refusal(authority, live, payload, snapshot, current, result):
    """The shared selection-guard registry (cost, data-policy, large-context switch) runs before the
    commit. A flagged target is refused with ``status: confirmation_required`` and a ``confirm``
    token bound to the resolved target, the selection it replaces and the session revision; nothing
    is written. Only that exact token in ``payload['confirm']`` applies it: a stale or foreign
    token (target re-resolved elsewhere, another switch landed, a turn settled) gets a fresh
    refusal, never a silent apply."""
    import hashlib
    from hermes_cli.model_selection_guards import combined_message, selection_context_for_agent, selection_warnings
    from hermes_cli.route_identity import normalize_route_base_url
    lookup = (getattr(authority.runner, '_resident_agent_for', None)
              or getattr(authority.runner, '_cached_agent_for', None))
    agent = lookup(live.route) if live is not None and callable(lookup) else None
    context = selection_context_for_agent(agent)
    if context is None:
        # No resident agent (evicted by the previous switch, or an owner restart): the session's
        # persisted usage anchor still measures the context at stake, so the large-context guard
        # does not go quiet on exactly the switch that follows another one.
        context = await asyncio.to_thread(_persisted_selection_context, authority, snapshot, current)
    # Off-loop: pricing lookups may hit models.dev on a cache miss.
    warnings = await asyncio.to_thread(
        selection_warnings, result.new_model, provider=result.target_provider,
        base_url=result.base_url or current.base_url, api_key=result.api_key or current.api_key,
        model_info=result.model_info, selection_context=context)
    if not warnings:
        return None
    binding = {'target': [result.new_model, result.target_provider, normalize_route_base_url(result.base_url),
                          result.api_mode or ''],
               'current': [current.model, current.provider, normalize_route_base_url(current.base_url)],
               'session': [snapshot['target'], snapshot['target_revision']],
               'guards': sorted(w.kind for w in warnings)}
    token = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if payload.get('confirm') == token:
        return None
    # A fresh dict: the prepared snapshot carries the frozen policy and never leaves the owner. No
    # ``model`` key: a client that predates the exchange must not paint the target as applied.
    return {'status': 'confirmation_required', 'confirm_required': True, 'confirm': token,
            'target_model': result.new_model, 'target_provider': result.target_provider,
            'confirm_message': combined_message(warnings),
            'warnings': [{'kind': w.kind, 'title': w.title, 'message': w.message} for w in warnings]}
