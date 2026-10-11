"""Frozen, explicit local-client launch policy for the existing TurnRunner."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

CREATE_FIELDS = frozenset({'request_id', 'source', 'cwd', 'model', 'toolsets',
                           'provider', 'base_url', 'reasoning', 'max_turns', 'ignore_rules', 'api_key', 'editor',
                           'yolo', 'safe_mode', 'ignore_user_config',
                           'skills', 'checkpoints', 'accept_hooks', 'pass_session_id'})
BYPASS_FIELDS = ('safe_mode', 'ignore_user_config')
_ACTIVE_POLICY: ContextVar = ContextVar('local_session_policy', default=None)
# Creation label -> agent surface. ``tool`` (third-party integrations, ``hermes chat --source tool``) and
# ``oneshot`` (finite ``chat -q`` / ``-z`` runs) run as the CLI but keep their own stored label, so human
# pickers hide them (INTERNAL_LISTING_SOURCES).
SURFACES = {'cli': 'cli', 'tui': 'tui', 'gui': 'desktop', 'acp': 'acp', 'tool': 'cli', 'oneshot': 'cli'}


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
    # `-s/--skills` blocks rendered once at creation (render_launch_skills): the session's
    # ephemeral prompt never re-reads the skill files, like the CLI's startup preload.
    skills_prompt: str | None = None

    def config(self, authority=None):
        config = present_sections(json.loads(self.config_json))
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
    def yolo(self):
        """`hermes chat --yolo`: this session's dangerous-command approvals are bypassed, exactly the
        scope `/yolo` gives a messaging chat; never the process-wide HERMES_YOLO_MODE."""
        return json.loads(self.request_json).get('yolo', False)

    @property
    def pass_session_id(self):
        """`--pass-session-id`: the session id rides the agent's system prompt."""
        return json.loads(self.request_json).get('pass_session_id', False)

    @property
    def checkpoints_enabled(self):
        """Frozen `checkpoints.enabled` (`--checkpoints` folds into it); legacy bool form accepted."""
        section = self.config().get('checkpoints', {})
        return bool(section.get('enabled', False) if isinstance(section, dict) else section)

    @property
    def max_turns(self):
        from hermes_cli.config import resolve_turn_limit
        cfg = self.config()
        return resolve_turn_limit(cfg.get('agent', {}).get('max_turns', cfg.get('max_turns')))

    @property
    def reasoning_config(self):
        from hermes_constants import resolve_reasoning_config
        return resolve_reasoning_config(self.config(), self.model or '')


def present_sections(config):
    """A bare ``gateway:`` key parses as YAML null; like load_config's merge (#58277), treat it as absent so
    ``cfg.get('gateway', {}).get(...)`` reads the default instead of crashing a fresh install's first turn."""
    return {key: value for key, value in config.items() if value is not None}


def build_policy(params, config, *, private_secrets=None, profile_terminal=True):
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
    launch_skills(params)
    config = present_sections(json.loads(json.dumps(config)))
    _apply_launch_overrides(params, config)
    enabled = _resolve_toolsets(params, config, source, safe_mode)
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


def _apply_launch_overrides(params, config):
    """Validate the explicit model/agent launch overrides and fold them into *config*."""
    from urllib.parse import urlsplit
    from hermes_cli.config import resolve_turn_limit
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
    for flag in ('ignore_rules', 'yolo', 'checkpoints', 'accept_hooks', 'pass_session_id'):
        if flag in params and type(params[flag]) is not bool:
            raise RuntimeStoreError('invalid_params')
    if params.get('checkpoints'):
        section = config.get('checkpoints')
        section = dict(section) if isinstance(section, dict) else {'enabled': bool(section)} if section else {}
        config['checkpoints'] = {**section, 'enabled': True}
    if 'max_turns' in params:
        # The spellings `--max-turns` always took: a positive cap, or 0 / -1 / "none" / "unlimited"
        # for no cap (resolve_turn_limit). Frozen normalized; anything unreadable is refused.
        value = params['max_turns']
        limit = resolve_turn_limit(value, default=0) if isinstance(value, (int, str)) else 0
        if not limit:
            raise RuntimeStoreError('invalid_params')
        config.setdefault('agent', {})['max_turns'] = limit
    if 'reasoning' in params:
        if not isinstance(params['reasoning'], str) or parse_reasoning_effort(params['reasoning']) is None:
            raise RuntimeStoreError('invalid_params')
        config.setdefault('agent', {})['reasoning_effort'] = params['reasoning']
        # An explicit launch level wins over a per-model default, just like CLI.
        config['agent'].pop('reasoning_overrides', None)


def _resolve_toolsets(params, config, source, safe_mode):
    """Validate explicit toolsets and return the session's enabled toolset set."""
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import validate_toolset
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
        # The fold-in lands after _get_platform_tools subtracted agent.disabled_toolsets, so
        # subtract again or `disabled_toolsets: [project]` is a no-op here (#54433).
        # desktop_ui is the client's own control surface, not a model toolset.
        from agent.skill_utils import parse_config_string_list
        from toolsets import CLIENT_SURFACE_TOOLSETS
        disabled = set(parse_config_string_list((config.get('agent') or {}).get('disabled_toolsets')))
        surface = set(CLIENT_SURFACE_TOOLSETS) if source == 'gui' else {'project'}
        enabled |= surface - (disabled - {'desktop_ui'})
    if safe_mode:
        # Plugin toolsets are user customizations; the safe worker never imports them.
        from hermes_cli.tools_config import _get_plugin_toolset_keys
        enabled -= _get_plugin_toolset_keys()
    return enabled


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


def launch_skills(params):
    """The validated `-s/--skills` identifiers of a create request (empty when absent). Skills
    live in the profile, which a bypass launch must never read: refused there, not dropped."""
    skills = params.get('skills', [])
    if (('skills' in params and (not isinstance(skills, list) or not skills
                                 or any(not isinstance(s, str) or not s.strip() for s in skills)))
            or (skills and any(params.get(name) is True for name in BYPASS_FIELDS))):
        raise RuntimeStoreError('invalid_params')
    return [s.strip() for s in skills]


def render_launch_skills(params, session_id):
    """`-s/--skills` preload blocks for a new session, rendered ONCE at creation and frozen in its
    policy (the CLI's startup preload). Blocking skill-tree reads: callers run it off the owner
    loop. A typo among good names is logged and skipped, like the CLI; none resolving refuses."""
    skills = launch_skills(params)
    if not skills:
        return None
    from agent.skill_commands import build_preloaded_skills_prompt, format_missing_skills
    prompt, loaded, missing = build_preloaded_skills_prompt(skills, task_id=session_id)
    if not loaded:
        raise RuntimeStoreError('unknown_skill')
    if missing:
        import logging
        logging.getLogger(__name__).warning('Skipping %s for session %s; continuing with: %s',
                                            format_missing_skills(missing), session_id, ', '.join(loaded))
    return prompt or None


def accept_launch_hooks(policy):
    """`--accept-hooks` / `HERMES_ACCEPT_HOOKS=1` on a launch: consent to the profile's configured
    shell hooks (recorded in the per-user allowlist, as the in-process CLI/TUI did) and register
    them on this owner. A bypass launch has no profile hooks to consent to (main's safe mode
    registered none either), so the flag is a no-op there rather than a refusal. Allowlist file
    lock + write: callers run it off the owner loop."""
    if json.loads(policy.request_json).get('accept_hooks') and not policy.ignore_user_config:
        from agent.shell_hooks import register_from_config
        register_from_config(policy.config(), accept_hooks=True)


def register_worker_hooks(policy):
    """Worker side of the same contract: tools run in the managed child, so its own plugin
    manager needs the session's shell hooks and outbound webhooks, registered from the frozen
    config with the launch's consent (a Kanban attempt always consents), as the classic one-shot
    CLI registered both at startup. Bypass sessions register nothing (the safe policy also
    refuses inside both registrars)."""
    if policy.ignore_user_config:
        return
    kanban = json.loads(policy.kanban_json or 'null') or {}
    accept = json.loads(policy.request_json).get('accept_hooks') is True or kanban.get('accept_hooks') is True
    from agent.outbound_webhooks import register_from_config as register_outbound_webhooks
    from agent.shell_hooks import register_from_config
    config = policy.config()
    register_from_config(config, accept_hooks=accept)
    register_outbound_webhooks(config)


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
    # Only a key that IS being bound can conflict: binding config secrets alone (a provider
    # change drops the launch key) must not be refused by the key the session already holds.
    if api_key is not None and old is not None and not hmac.compare_digest(old, api_key):
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
    # The durable ref carries the key's fingerprint (the digest config_secret_ref already
    # persists for config keys) so a restarted owner can verify a re-supplied key
    # (rebind_launch_key); the key itself is never durable.
    from agent.credential_persistence import fingerprint_secret_value
    return replace(policy, credential_ref=f'{ref}#{fingerprint_secret_value(api_key)}' if api_key is not None else None,
                   config_secret_ref=config_ref if config_secrets else None)


def _key_slot(credential_ref):
    """The in-memory slot of a durable credential ref (``<instance>:<epoch>:<sid>[#<fingerprint>]``)."""
    return credential_ref.split('#', 1)[0]


def rebind_launch_key(authority, session, api_key):
    """``session.resume`` re-supplying a launch-only key after an owner restart revoked it
    (call after authorizing *session*, a SessionRef).

    Only a session launched with a key can take one, and only that same key (its durable
    fingerprint); anything else is an override of the frozen route and is refused. Like the
    original, the key lives in this authority's memory only."""
    import hmac
    from agent.credential_persistence import fingerprint_secret_value
    from gateway.config import Platform
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeStoreError('invalid_params')
    source = authority.sessions[session.session_id].source
    if source is None or source.platform != Platform.LOCAL:
        raise RuntimeStoreError('admission_conflict')
    from hermes_state_local import local_receipt
    ref = restore_policy(local_receipt(authority.db, session.session_id)['policy']).credential_ref
    if ref is None:
        raise RuntimeStoreError('admission_conflict')
    keys = getattr(authority, '_local_launch_keys', None)
    if keys is None:
        keys = authority._local_launch_keys = {}
    held = keys.get(_key_slot(ref))
    if held is not None:
        if not hmac.compare_digest(held, api_key):
            raise RuntimeStoreError('admission_conflict')
        return
    _, _, fingerprint = ref.partition('#')
    if not fingerprint or not hmac.compare_digest(fingerprint, fingerprint_secret_value(api_key) or ''):
        raise RuntimeStoreError('admission_conflict')
    keys[_key_slot(ref)] = api_key


def release_launch_secrets(authority, session_ids):
    """Drop retired sessions' launch keys, frozen config secrets and editor MCP servers.

    They live only in this authority's memory for the session's lifetime; a deleted session can
    never run again, so keeping them would hold its credentials until the process exits."""
    import hashlib
    from gateway.session_policy_credentials import PREFIX
    retired = {str(sid) for sid in session_ids}
    if not retired:
        return
    prefix = f'{authority.instance_id}:{authority.epoch}:'
    keys = getattr(authority, '_local_launch_keys', None) or {}
    # A key re-supplied after a restart sits in the slot of the epoch that minted it.
    for ref in [ref for ref in keys if ref.split(':', 2)[-1] in retired]:
        del keys[ref]
    configs = getattr(authority, '_local_config_secrets', None) or {}
    for ref in list(configs):
        if ref.startswith(PREFIX):
            owner = json.loads(ref[len(PREFIX):]).get('session')
        else:
            owner = ref[len(prefix):] if ref.startswith(prefix) else None
        if owner in retired:
            del configs[ref]
    editors = getattr(authority, '_local_editor_mcp', None) or {}
    for sid in retired:
        editors.pop('editor-session:' + hashlib.sha256(f'{authority.profile_id}:{sid}'.encode()).hexdigest(), None)


def launch_key(authority, policy):
    if policy.credential_ref is None:
        return None
    value = getattr(authority, '_local_launch_keys', {}).get(_key_slot(policy.credential_ref))
    if value is None:
        raise RuntimeStoreError('launch_credentials_unavailable')
    return value


def restore_policy(data):
    """Reject incomplete private policy rather than rebuilding from current defaults."""
    try:
        policy = LocalSessionPolicy(**data)
        from gateway.session_a2a import is_forward_policy
        surfaces = {**SURFACES, 'cron': 'cron', 'kanban': 'cli', 'bot_room': 'bot_room'}
        if ((not is_forward_policy(policy) and
             (policy.source not in surfaces or policy.platform != surfaces[policy.source]))
                or not isinstance(policy.cwd, str) or not Path(policy.cwd).is_absolute()
                or not Path(policy.cwd).is_dir()
                or not isinstance(policy.model, str) or not policy.model.strip()
                or not isinstance(policy.toolsets, (list, tuple))
                or any(not isinstance(name, str) for name in policy.toolsets)
                or type(policy.safe_mode) is not bool or type(policy.ignore_user_config) is not bool
                or not isinstance(policy.skills_prompt, (str, type(None)))
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
    policy_token = _ACTIVE_POLICY.set(policy)
    from gateway.session_local_editor import editor_scope
    try:
        from gateway.session_local_mcp import editor_mcp_scope
        with editor_scope(policy), editor_mcp_scope(authority, policy):
            yield
    finally:
        _ACTIVE_POLICY.reset(policy_token)
        reset_terminal_scope(terminal_token)
        cwd_token.var.reset(cwd_token)


def active_policy():
    """The frozen launch policy of the turn executing in this context, or ``None`` (standalone
    serve / messaging turns). Consumers read it instead of re-deriving from live config."""
    return _ACTIVE_POLICY.get()
