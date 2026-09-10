"""Owner-bound native settings and model inventory, never legacy execution."""
import asyncio
from functools import partial
import hashlib
import json
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {'config.get': partial(config_get, connection),
            'model.options': partial(model_options, connection)}


def _authorize(connection, ref, params):
    authority, actor = connection.authority, connection.actor
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if 'session:read' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    for name in ('profile', 'session_id'):
        if name in params and not isinstance(params[name], str):
            raise RuntimeStoreError('invalid_params')
    if params.get('profile'):
        from hermes_cli.profiles import profile_matches_home
        if not profile_matches_home(params['profile'], Path(authority.profile_id)):
            raise RuntimeStoreError('profile_mismatch')
    if ref.session_id:
        authority.authorize(actor, ref, 'session:read')
        from gateway.session_policy import policy_for_source
        return policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    return None


def _client_config(cfg):
    # Native full-config consumers are presentation-only. Do not export provider,
    # MCP, plugin or terminal configuration (which can contain arbitrary secrets).
    result = {key: cfg[key] for key in ('display', 'approvals', 'paste_collapse_threshold',
              'paste_collapse_char_threshold') if key in cfg}
    result['voice'] = {key: cfg.get('voice', {}).get(key) for key in ('record_key', 'submit_mode')
                       if key in cfg.get('voice', {})}
    from hermes_cli.config import redact_config_value
    return redact_config_value(result)


async def config_get(connection, ref, params):
    if set(params) - {'session_id', 'profile', 'key', 'cwd'}:
        raise RuntimeStoreError('invalid_params')
    key = params.get('key')
    if not isinstance(key, str):
        raise RuntimeStoreError('invalid_params')
    if key == 'busy':
        from gateway.session_busy_controls import busy_config
        return await busy_config(connection, ref, params)
    policy = _authorize(connection, ref, params)
    if 'cwd' in params and (key != 'project' or not isinstance(params['cwd'], str)):
        raise RuntimeStoreError('invalid_params')
    agent = connection.authority.agent(ref) if ref.session_id else None

    def read():
        from gateway.run import _profile_runtime_scope
        from hermes_cli.config import load_config
        from hermes_constants import DEFAULT_INDICATOR_STYLE
        from tools.approval_context import _get_approval_mode
        home = Path(connection.authority.profile_id)
        with _profile_runtime_scope(home):
            cfg = load_config()
            display = cfg.get('display') or {}
            getters = {
                'full': lambda: {'config': _client_config(cfg)},
                'mtime': lambda: {'mtime': (home / 'config.yaml').stat().st_mtime
                                 if (home / 'config.yaml').exists() else 0,
                                 # Frozen tool policy is not a live MCP reload request.
                                 'mcp_rev': hashlib.sha256((policy.config_json if policy else
                                     json.dumps({k: cfg.get(k) for k in ('mcp', 'mcp_servers', 'tools')},
                                                sort_keys=True)).encode()).hexdigest()[:12]},
                'reasoning': lambda: _reasoning(agent, policy, cfg),
                'fast': lambda: {'value': 'fast' if (getattr(agent, 'service_tier', None) or
                    (policy.config() if policy else cfg).get('agent', {}).get('service_tier'))
                    in {'priority', 'fast', 'on'} else 'normal'},
                'skin': lambda: {'value': display.get('skin', 'default')},
                'theme': lambda: {'value': display.get('tui_theme', 'auto')},
                'indicator': lambda: {'value': display.get('tui_status_indicator', DEFAULT_INDICATOR_STYLE)},
                'details_mode': lambda: {'value': display.get('details_mode', 'collapsed')},
                'approvals.mode': lambda: {'value': _get_approval_mode()},
                'approval_mode': lambda: {'value': _get_approval_mode()},
                'project': lambda: _project(params, policy, cfg),
                'profile': lambda: {'home': str(home), 'display': str(home)},
            }
            if key not in getters:
                raise RuntimeStoreError('invalid_params')
            return getters[key]()
    return await asyncio.to_thread(read)


def _reasoning(agent, policy, cfg):
    from hermes_constants import resolve_reasoning_config
    value = getattr(agent, 'reasoning_config', None)
    if not isinstance(value, dict):
        value = policy.reasoning_config if policy else resolve_reasoning_config(cfg, '')
    effort = 'none' if value and value.get('enabled') is False else (value or {}).get('effort', 'medium')
    return {'value': effort, 'display': 'show' if cfg.get('display', {}).get('show_reasoning', True) else 'hide'}


def _project(params, policy, cfg):
    from tui_gateway import git_probe
    cwd = params.get('cwd') or (policy.cwd if policy else cfg.get('terminal', {}).get('cwd'))
    if not cwd:
        from agent.runtime_cwd import resolve_agent_cwd
        cwd = str(resolve_agent_cwd())
    return {'cwd': cwd, 'branch': git_probe.branch(cwd)}


async def model_options(connection, ref, params):
    flags = {'explicit_only', 'include_unconfigured', 'refresh'}
    if (set(params) - flags - {'profile', 'session_id'}
            or any(type(params[k]) is not bool for k in flags & params.keys())):
        raise RuntimeStoreError('invalid_params')
    policy = _authorize(connection, ref, params)
    agent = connection.authority.agent(ref) if ref.session_id else None
    provider = getattr(agent, 'provider', None) or (policy.provider if policy else None)
    model = getattr(agent, 'model', None) or (policy.model if policy else None)
    base_url = getattr(agent, 'base_url', None) or (policy.base_url if policy else None)

    def discover():
        from gateway.run import _profile_runtime_scope
        from hermes_cli.inventory import load_picker_context, build_model_options_payload
        with _profile_runtime_scope(Path(connection.authority.profile_id)):
            ctx = load_picker_context()
            selected = provider
            if selected == 'custom':
                from hermes_cli.runtime_provider import canonical_custom_identity
                selected = canonical_custom_identity(base_url=base_url, config_provider=ctx.current_provider,
                                                     model=model) or selected
            return build_model_options_payload(ctx.with_overrides(current_provider=selected,
                current_model=model, current_base_url=base_url), **{k: params[k] for k in flags & params.keys()})
    return await asyncio.to_thread(discover)
