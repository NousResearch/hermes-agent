"""Frozen, explicit local-client launch policy for the existing TurnRunner."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

CREATE_FIELDS = frozenset({'request_id', 'source', 'cwd', 'model', 'toolsets',
                           'provider', 'base_url', 'reasoning', 'max_turns', 'ignore_rules', 'api_key', 'editor',
                           'safe_mode', 'ignore_user_config'})
BYPASS_FIELDS = ('safe_mode', 'ignore_user_config')
SURFACES = {'cli': 'cli', 'tui': 'tui', 'gui': 'desktop', 'acp': 'acp'}


@dataclass(frozen=True)
class LocalSessionPolicy:
    source: str
    platform: str
    cwd: str
    model: str | None
    toolsets: tuple[str, ...]
    config_json: str
    request_json: str
    terminal_json: str
    credential_ref: str | None = None
    config_secret_ref: str | None = None
    editor_mcp_json: str | None = None
    kanban_json: str | None = None
    # Troubleshooting isolation is derived by the owner at creation and frozen with the
    # route; safe_mode always implies ignore_user_config (normalized once in build_policy).
    safe_mode: bool = False
    ignore_user_config: bool = False

    def config(self, authority=None):
        config = json.loads(self.config_json)
        if self.config_secret_ref is not None and authority is not None:
            from gateway.session_policy_credentials import recover_config_secrets
            secrets = recover_config_secrets(authority, self)
            for path, value in secrets.items():
                if path[0] is None:  # private terminal projection, not config
                    continue
                target = config
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
        return config

    @property
    def provider(self):
        return self.config().get('model', {}).get('provider')

    @property
    def base_url(self):
        return self.config().get('model', {}).get('base_url')

    @property
    def ignore_rules(self):
        return self.safe_mode or json.loads(self.request_json).get('ignore_rules', False)

    @property
    def max_turns(self):
        from hermes_cli.config import resolve_turn_limit
        cfg = self.config()
        return resolve_turn_limit(cfg.get('agent', {}).get('max_turns', cfg.get('max_turns')))

    @property
    def reasoning_config(self):
        from hermes_constants import resolve_reasoning_config
        return resolve_reasoning_config(self.config(), self.model or '')


def build_policy(params, config, *, private_secrets=None, profile_terminal=True):
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import validate_toolset
    from agent.runtime_cwd import resolve_agent_cwd
    from tools.terminal_scope import build_profile_terminal_scope, default_terminal_scope
    from hermes_constants import get_hermes_home

    source = params.get('source', 'cli')
    if source == 'a2a':
        from gateway.session_a2a import build_forward_policy
        return build_forward_policy(params, config, private_secrets=private_secrets)
    if set(params) - CREATE_FIELDS or not isinstance(source, str) or source not in SURFACES:
        raise RuntimeStoreError('invalid_params')
    if any(name in params and type(params[name]) is not bool for name in BYPASS_FIELDS):
        raise RuntimeStoreError('invalid_params')
    safe_mode = params.get('safe_mode', False)
    ignore_user_config = safe_mode or params.get('ignore_user_config', False)
    if 'api_key' in params and (not isinstance(params['api_key'], str) or not params['api_key'].strip()):
        raise RuntimeStoreError('invalid_params')
    model = params.get('model')
    if 'model' in params and (not isinstance(model, str) or not model.strip()):
        raise RuntimeStoreError('invalid_params')
    cwd = params.get('cwd', str(resolve_agent_cwd()))
    if not isinstance(cwd, str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir():
        raise RuntimeStoreError('invalid_params')
    from gateway.session_local_editor import validate_editor
    validate_editor(source, params.get('editor'))
    config = json.loads(json.dumps(config))
    from urllib.parse import urlsplit
    from hermes_constants import parse_reasoning_effort
    for key in ('provider', 'base_url'):
        if key in params:
            value = params[key]
            if not isinstance(value, str) or not value.strip():
                raise RuntimeStoreError('invalid_params')
            if key == 'base_url':
                url = urlsplit(value)
                if (url.scheme not in {'http', 'https'} or not url.hostname
                        or url.username or url.password or url.query or url.fragment):
                    raise RuntimeStoreError('invalid_params')
            config.setdefault('model', {})[key] = value
    if 'ignore_rules' in params and type(params['ignore_rules']) is not bool:
        raise RuntimeStoreError('invalid_params')
    if 'max_turns' in params:
        value = params['max_turns']
        if type(value) is not int or value <= 0:
            raise RuntimeStoreError('invalid_params')
        config.setdefault('agent', {})['max_turns'] = value
    if 'reasoning' in params:
        if not isinstance(params['reasoning'], str) or parse_reasoning_effort(params['reasoning']) is None:
            raise RuntimeStoreError('invalid_params')
        config.setdefault('agent', {})['reasoning_effort'] = params['reasoning']
        # An explicit launch level wins over a per-model default, just like CLI.
        config['agent'].pop('reasoning_overrides', None)
    explicit = params.get('toolsets')
    if 'toolsets' in params:
        if (not isinstance(explicit, list) or any(not isinstance(x, str) or not validate_toolset(x) for x in explicit)
                or ('desktop_ui' in explicit and source != 'gui')):
            raise RuntimeStoreError('invalid_params')
        config.setdefault('platform_toolsets', {})['acp' if source == 'acp' else 'cli'] = explicit
    enabled = _get_platform_tools(config, 'acp' if source == 'acp' else 'cli')
    if explicit is not None:
        from toolsets import resolve_toolset
        requested = {t for name in explicit for t in resolve_toolset(name)}
        effective = {t for name in enabled for t in resolve_toolset(name)}
        if not requested <= effective:
            raise RuntimeStoreError('invalid_params')
    elif source in {'tui', 'gui'}:
        # Same surface toolsets as the native TUI factory, without its env inference.
        enabled.add('project')
        if source == 'gui':
            enabled.add('desktop_ui')
    if safe_mode:
        # Plugin toolsets are user customizations; the safe worker never imports them.
        from hermes_cli.tools_config import _get_plugin_toolset_keys
        enabled -= _get_plugin_toolset_keys()
    terminal = build_profile_terminal_scope(get_hermes_home()) if profile_terminal else default_terminal_scope()
    terminal['TERMINAL_CWD'] = cwd
    request = {k: v for k, v in params.items() if k not in {'request_id', 'api_key'}}
    request.setdefault('source', 'cli')
    from gateway.session_local_mcp import private_editor_request
    private_editor_request(request, private_secrets)
    _extract_config_secrets(config, private_secrets)
    _extract_config_secrets(terminal, private_secrets, (None,))
    return LocalSessionPolicy(source, SURFACES[source], cwd, model, tuple(sorted(enabled)),
                              json.dumps(config), json.dumps(request, sort_keys=True), json.dumps(terminal),
                              safe_mode=safe_mode, ignore_user_config=ignore_user_config)


def _extract_config_secrets(value, private, path=()):
    # Reuse the configuration owner's structural classification; opaque keys need
    # not match a vendor prefix. Only the authority keeps their original values.
    from hermes_cli.config import _SECRET_CONFIG_KEYS
    from agent.credential_persistence import _is_secret_payload_key
    containers = {'env', 'headers', 'extra_headers', 'docker_env', 'docker_extra_args',
                  'terminal_docker_env', 'terminal_docker_extra_args'}
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, child in items:
        child_path = path + (key,)
        sensitive = isinstance(key, str) and (key.lower() in _SECRET_CONFIG_KEYS
                    or key.lower() in containers or _is_secret_payload_key(key))
        if sensitive and child and child not in ('{}', '[]'):
            if private is None:
                raise RuntimeStoreError('launch_credentials_unavailable')
            private[child_path] = child
            value[key] = None
        else:
            _extract_config_secrets(child, private, child_path)


def bind_launch_key(authority, session_id, policy, api_key, *, config_secrets=None):
    """CLI keys live only in this authority lifetime, never its durable receipt.

    Restart deliberately revokes them. History remains readable; inference must
    report launch_credentials_unavailable, never select a profile fallback key.
    """
    from dataclasses import replace
    import hmac
    from gateway.session_local_mcp import PRIVATE_KEY, bind_editor_mcp
    config_secrets = dict(config_secrets or {})
    policy = bind_editor_mcp(authority, session_id, policy, config_secrets.pop(PRIVATE_KEY, None))
    if api_key is None and not config_secrets:
        return policy
    keys = getattr(authority, '_local_launch_keys', None)
    if keys is None:
        keys = authority._local_launch_keys = {}
    ref = f'{authority.instance_id}:{authority.epoch}:{session_id}'
    old = keys.get(ref)
    if old is not None and (api_key is None or not hmac.compare_digest(old, api_key)):
        raise RuntimeStoreError('admission_conflict')
    configs = getattr(authority, '_local_config_secrets', None)
    if configs is None:
        configs = authority._local_config_secrets = {}
    config_ref = ref
    if config_secrets and getattr(authority, 'db', None) is not None:
        from gateway.session_policy_credentials import config_reference
        config_ref = config_reference(authority, session_id, policy, config_secrets)
    if config_ref in configs and configs[config_ref] != config_secrets:
        raise RuntimeStoreError('admission_conflict')
    if config_secrets:
        configs[config_ref] = dict(config_secrets)
    if api_key is not None:
        keys[ref] = api_key
    return replace(policy, credential_ref=ref if api_key is not None else None,
                   config_secret_ref=config_ref if config_secrets else None)


def launch_key(authority, policy):
    if policy.credential_ref is None:
        return None
    value = getattr(authority, '_local_launch_keys', {}).get(policy.credential_ref)
    if value is None:
        raise RuntimeStoreError('launch_credentials_unavailable')
    return value


def restore_policy(data):
    """Reject incomplete private policy rather than rebuilding from current defaults."""
    try:
        policy = LocalSessionPolicy(**data)
        from gateway.session_a2a import is_forward_policy
        surfaces = {**SURFACES, 'cron': 'cron', 'kanban': 'cli'}
        if ((not is_forward_policy(policy) and
             (policy.source not in surfaces or policy.platform != surfaces[policy.source]))
                or not isinstance(policy.cwd, str) or not Path(policy.cwd).is_absolute()
                or not Path(policy.cwd).is_dir()
                or not isinstance(policy.model, str) or not policy.model.strip()
                or not isinstance(policy.toolsets, (list, tuple))
                or any(not isinstance(name, str) for name in policy.toolsets)
                or type(policy.safe_mode) is not bool or type(policy.ignore_user_config) is not bool
                or (policy.safe_mode and not policy.ignore_user_config)):
            raise ValueError('invalid policy')
        from dataclasses import replace
        for value in (policy.config_json, policy.request_json, policy.terminal_json):
            if not isinstance(json.loads(value), dict):
                raise ValueError('invalid policy object')
        if json.loads(policy.terminal_json).get('TERMINAL_CWD') != policy.cwd:
            raise ValueError('terminal policy mismatch')
        return replace(policy, toolsets=tuple(policy.toolsets))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc


def policy_for_source(runner, source):
    from gateway.session_local import LocalSessionAdapter
    from gateway.config import Platform
    if source.platform != Platform.LOCAL:
        return None
    adapter = runner._adapter_for_source(source)
    if isinstance(adapter, LocalSessionAdapter) and adapter.authorize_source(source):
        policy = adapter.policies.get(source.chat_id)
        if policy is None:
            # Cold recovery must restore the frozen policy before executing, not
            # reinterpret a LOCAL route as a default CLI launch.
            raise RuntimeStoreError('storage_unavailable')
        return policy
    return None


@contextmanager
def policy_scope(policy, *, authority=None):
    if policy is None:
        yield
        return
    from agent.runtime_cwd import set_session_cwd
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    terminal = json.loads(policy.terminal_json)
    if policy.config_secret_ref is not None:
        from gateway.session_policy_credentials import recover_config_secrets
        for path, value in recover_config_secrets(authority, policy).items():
            if path[0] is None:
                terminal[path[1]] = value
    cwd_token = set_session_cwd(policy.cwd)
    terminal_token = set_terminal_scope(terminal)
    from gateway.session_local_editor import editor_scope
    try:
        from gateway.session_local_mcp import editor_mcp_scope
        with editor_scope(policy), editor_mcp_scope(authority, policy):
            yield
    finally:
        reset_terminal_scope(terminal_token)
        cwd_token.var.reset(cwd_token)
