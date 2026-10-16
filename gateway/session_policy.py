"""Frozen, explicit local-client launch policy for the existing TurnRunner."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

CREATE_FIELDS = frozenset({'request_id', 'source', 'cwd', 'model', 'toolsets',
                           'provider', 'base_url', 'reasoning', 'max_turns', 'ignore_rules', 'api_key', 'editor',
                           'yolo', 'safe_mode', 'ignore_user_config',
                           'skills', 'checkpoints', 'accept_hooks', 'pass_session_id', 'follow_profile_config',
                           'service_tier', 'cwd_explicit', 'run_budget', 'verbose', 'row_cwd'})
# Sections a ``follow_profile_config`` session re-reads from the owning profile's CURRENT config each
# turn (main's Bot Chat row), credentials included: a frozen key never crosses a profile route change.
FOLLOW_SECTIONS = ('model', 'providers', 'custom_providers', 'fallback_providers', 'fallback_model', 'model_aliases',
                   'platform_toolsets')
FOLLOW_AGENT_KEYS = ('reasoning_effort', 'reasoning_overrides', 'service_tier', 'disabled_toolsets', 'coding_context')
# What a pick (create-time ``model`` / committed ``/model``) owns of the ``model:`` block.
SELECTION_KEYS = ('default', 'provider', 'base_url', 'api_mode', 'selection_profile', 'api_key_from', 'key_env',
                  'api_key_env')
BYPASS_FIELDS = ('safe_mode', 'ignore_user_config')
_ACTIVE_POLICY: ContextVar = ContextVar('local_session_policy', default=None)
_TURN_PROFILE: ContextVar = ContextVar('follow_profile_read', default=None)  # (policy, live) for a turn
# Creation label -> agent surface. ``tool`` (third-party integrations, ``hermes chat --source tool``) and
# ``oneshot`` (finite ``chat -q`` / ``-z`` runs) run as the CLI but keep their own stored label, so human
# pickers hide them (INTERNAL_LISTING_SOURCES).
SURFACES = {'cli': 'cli', 'tui': 'tui', 'gui': 'desktop', 'acp': 'acp', 'tool': 'cli', 'oneshot': 'cli'}


def _refuse(field, detail=None):
    """One named refusal: the launch field the check read, and why it failed.

    ``reason`` stays the bounded wire code every caller compares as a string; ``field`` and
    ``detail`` are what tells ~20 different create checks apart, in the client and in the
    gateway log, without a second run to find out which one fired (``RuntimeStoreError`` bounds
    and flattens both)."""
    raise RuntimeStoreError('invalid_params', field=field, detail=detail)


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
    # Whether creation chose the coding posture (``agent.coding_context: focus`` in a client-named code
    # workspace). A following session replays this decision rather than re-probing its folder each turn: it
    # drops the posture when its profile leaves ``focus``, and one created outside it never gains it later.
    coding_posture: bool = False

    def config(self, authority=None, *, secrets=True):
        return self.effective(authority, secrets=secrets)[0]

    def effective(self, authority=None, *, secrets=True):
        """``(config, model)`` the next turn runs, both from ONE read of the owning profile when following
        (the turn's policy scope holds that read): a route never pairs one profile's model with another's
        endpoint. Explicit choices (create-time reasoning / tier / toolsets, a pick that still holds) are
        applied over the followed defaults; frozen secrets of followed sections are never recovered."""
        config = present_sections(json.loads(self.config_json))
        live = self.read_profile(authority)
        if self.config_secret_ref is not None and authority is not None and secrets:
            from gateway.session_policy_credentials import recover_config_secrets
            for path, value in recover_config_secrets(authority, self).items():
                if path[0] is None:  # private terminal projection, not config
                    continue
                target = config
                for key in path[:-1]:
                    # An alias entry added after creation is absent from the frozen tree.
                    target = target.setdefault(key, {}) if isinstance(target, dict) else target[key]
                target[path[-1]] = value
        model = self.model
        if live:
            frozen, request = config.get('model') or {}, json.loads(self.request_json)
            pick = frozen.get('selection_profile')
            held = pick is not None and pick == list(profile_target(live))
            pinned = {'platform_toolsets', 'disabled_toolsets'} if 'toolsets' in request else set()
            pinned |= {'reasoning_effort', 'reasoning_overrides'} if 'reasoning' in request else set()
            pinned |= {'service_tier'} if 'service_tier' in request else set()
            for key in FOLLOW_SECTIONS:
                if key in pinned:
                    continue
                if key in live:
                    config[key] = live[key]
                else:
                    config.pop(key, None)
            if held:
                config['model'] = _held_pick(frozen, config.get('model') or {})
            else:
                from gateway.run import _resolve_gateway_model
                model = _resolve_gateway_model(live) or self.model
            agent, live_agent = config.setdefault('agent', {}), live.get('agent') or {}
            for key in FOLLOW_AGENT_KEYS:
                if key in pinned:
                    continue
                if key in live_agent:
                    agent[key] = live_agent[key]
                else:
                    agent.pop(key, None)
        source = (config.get('model') or {}).get('api_key_from') if isinstance(config.get('model'), dict) else None
        if source:
            # ``/model <alias>`` with a literal key: the receipt points at the alias entry's secret.
            value = config
            for part in source:
                value = value.get(part) if isinstance(value, dict) else None
            config['model']['api_key'] = value
        return config, model

    @property
    def follows_profile(self):
        return json.loads(self.request_json).get('follow_profile_config') is True

    def read_profile(self, authority):
        """The owning profile's current config for a following session, else None; inside this policy's
        turn scope, the turn's one read."""
        if authority is None or not self.follows_profile:
            return None
        held = _TURN_PROFILE.get()
        if held is not None and held[0] == self:
            return held[1]
        from gateway.run import _load_gateway_config
        return present_sections(_load_gateway_config(Path(authority.db.db_path).resolve().parent / 'config.yaml'))

    def current_toolsets(self, config):
        """The next turn's toolsets from its resolved *config*: a following session without an explicit
        toolset pick runs the profile's current selection (main #124211), else the frozen launch set."""
        if (not self.follows_profile or self.source not in SURFACES or self.editor_mcp_json is not None
                or 'toolsets' in json.loads(self.request_json)):
            return self.toolsets
        # The workspace decision is the creation-time one: a project file the agent writes mid-session must
        # not switch this conversation's toolsets (prompt cache). Only a profile change (``focus`` off) does.
        return tuple(sorted(_resolve_toolsets({}, config, self.source, self.safe_mode, self.coding_posture)))

    def current_reasoning_config(self, runner, session_key=None):
        """The next turn's reasoning: a typed session /reasoning (the messaging handler's override), else
        the profile's current effort when following, else the frozen one."""
        peek = getattr(runner, '_peek_session_state', None)
        state = peek(session_key) if session_key and peek is not None else None
        if state is not None and state.conversation.reasoning_override is not None:
            return dict(state.conversation.reasoning_override)
        if not self.follows_profile:
            return self.reasoning_config
        from gateway.session_authorities import active_authority
        from hermes_constants import resolve_reasoning_config
        config, model = self.effective(active_authority(runner), secrets=False)
        return resolve_reasoning_config(config, model or '')

    def current_service_tier(self, runner, session_key):
        """The next turn's tier: a typed session /fast, else the policy's (frozen, or the profile's when following)."""
        from agent.fast_mode import parse_service_tier
        from gateway.session_authorities import active_authority
        from gateway.session_state import SERVICE_TIER_UNSET
        state = runner._peek_session_state(session_key) if session_key else None
        if state is not None and state.conversation.service_tier_override is not SERVICE_TIER_UNSET:
            return state.conversation.service_tier_override
        config = self.config(active_authority(runner), secrets=False)
        return parse_service_tier((config.get('agent') or {}).get('service_tier'))

    def current_model(self, authority):
        """The model this session's next turn runs: the profile's current default when following."""
        return self.effective(authority, secrets=False)[1]

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
    def verbose(self):
        """`hermes chat -v`: the session's agent runs with quiet_mode off, so its status lines reach the
        viewer (``agent.print``) as main's in-process verbose agent printed them."""
        return json.loads(self.request_json).get('verbose', False)

    @property
    def run_budget_seconds(self):
        """Frozen `agent.run_budget_seconds` (`--run-budget` folds into it); None = no budget."""
        from agent.agent_init import _normalize_run_budget_seconds
        return _normalize_run_budget_seconds(self.config().get('agent', {}).get('run_budget_seconds'))

    @property
    def reasoning_config(self):
        from hermes_constants import resolve_reasoning_config
        return resolve_reasoning_config(self.config(), self.model or '')


def profile_target(config):
    """(model, provider) a profile config selects; 'auto' is no provider (main's _config_model_target)."""
    model = config.get('model')
    if isinstance(model, dict):
        provider = str(model.get('provider') or '').strip()
        return str(model.get('default') or '').strip(), '' if provider.lower() == 'auto' else provider
    return (model.strip() if isinstance(model, str) else ''), ''


def _held_pick(frozen, live):
    """A pick that still holds keeps its selection; on the profile's own endpoint the rest of the
    ``model:`` block (a rotated key included) is the profile's. Another endpoint never gets it."""
    from hermes_cli.route_identity import normalize_route_base_url

    def route(block):
        return str(block.get('provider') or '').strip().lower(), normalize_route_base_url(block.get('base_url'))
    if route(frozen) != route(live):
        return frozen
    return {**{k: v for k, v in live.items() if k not in SELECTION_KEYS},
            **{k: frozen[k] for k in SELECTION_KEYS if k in frozen}}


def present_sections(config):
    """A bare ``gateway:`` key parses as YAML null; like load_config's merge (#58277), treat it as absent so
    ``cfg.get('gateway', {}).get(...)`` reads the default instead of crashing a fresh install's first turn."""
    config = {key: value for key, value in config.items() if value is not None}
    # A scalar ``model: <id>`` (a form every classic loader accepts) is the mapping's ``default``.
    if isinstance(config.get('model'), str):
        config['model'] = {'default': config['model']} if config['model'].strip() else {}
    return config


def build_policy(params, config, *, private_secrets=None, profile_terminal=True, workspace_posture=False):
    from agent.runtime_cwd import resolve_agent_cwd
    from tools.terminal_scope import build_profile_terminal_scope, default_terminal_scope
    from hermes_constants import get_hermes_home

    source = params.get('source', 'cli')
    if source == 'a2a':
        from gateway.session_a2a import build_forward_policy
        return build_forward_policy(params, config, private_secrets=private_secrets)
    unknown = sorted(set(params) - CREATE_FIELDS)
    if unknown or not isinstance(source, str) or source not in SURFACES:
        _refuse(f'params.{unknown[0]}' if unknown else 'source',
                'unknown create field' if unknown else f'not a surface: {source!r}')
    bad_bypass = next((name for name in BYPASS_FIELDS if name in params and type(params[name]) is not bool), None)
    if bad_bypass:
        _refuse(bad_bypass, 'must be a boolean')
    safe_mode = params.get('safe_mode', False)
    ignore_user_config = safe_mode or params.get('ignore_user_config', False)
    if 'api_key' in params and (not isinstance(params['api_key'], str) or not params['api_key'].strip()):
        _refuse('api_key', 'must be a non-empty string')
    model = params.get('model')
    if 'model' in params and (not isinstance(model, str) or not model.strip()):
        _refuse('model', 'must be a non-empty string')
    cwd = params.get('cwd', str(resolve_agent_cwd()))
    if not isinstance(cwd, str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir():
        _refuse('cwd', 'must be an absolute path to an existing directory')
    row_cwd = params.get('row_cwd')
    if 'row_cwd' in params and (not isinstance(row_cwd, str) or (row_cwd and not Path(row_cwd).is_absolute())):
        _refuse('row_cwd', 'must be an absolute path when set')
    if params.get('cwd_explicit') is False and 'cwd' in params:
        cwd = _profile_workspace_cwd(get_hermes_home()) or cwd
    from gateway.session_local_editor import validate_editor
    validate_editor(source, params.get('editor'))
    launch_skills(params)
    config = present_sections(json.loads(json.dumps(config)))
    if params.get('follow_profile_config') is True and 'model' in params:
        # The pick's provenance, recorded before the launch route folds in: it holds while the profile
        # still targets this, and a later profile edit supersedes it.
        config.setdefault('model', {})['selection_profile'] = list(profile_target(config))
    _apply_launch_overrides(params, config)
    # The posture is opt-in (``session.create`` only) and needs a folder the client named: the daemon's own
    # directory (the fallback above) is no workspace, and ``tool`` is a third-party integration. An explicit
    # pick wins anyway; skipping it here also spares the probe.
    named = workspace_posture and source != 'tool' and 'cwd' in params and 'toolsets' not in params
    coding_posture = named and _in_coding_workspace(config, source, cwd)
    enabled = _resolve_toolsets(params, config, source, safe_mode, coding_posture, bypass=ignore_user_config)
    terminal = build_profile_terminal_scope(get_hermes_home()) if profile_terminal else default_terminal_scope()
    # The client's launch folder is a path on THIS host: a remote or container backend keeps its own
    # configured cwd unless the user picked this folder for the session.
    if params.get('cwd_explicit') is True or terminal_uses_host_cwd(terminal):
        terminal['TERMINAL_CWD'] = cwd
    request = {k: v for k, v in params.items() if k not in {'request_id', 'api_key'}}
    request.setdefault('source', 'cli')
    from gateway.session_local_mcp import private_editor_request
    private_editor_request(request, private_secrets)
    _extract_config_secrets(config, private_secrets)
    _extract_config_secrets(terminal, private_secrets, (None,))
    return LocalSessionPolicy(source, SURFACES[source], cwd, model, tuple(sorted(enabled)),
                              json.dumps(config), json.dumps(request, sort_keys=True), json.dumps(terminal),
                              safe_mode=safe_mode, ignore_user_config=ignore_user_config,
                              coding_posture=coding_posture)


def terminal_uses_host_cwd(terminal):
    """Whether the terminal works in the session's host cwd: the local backend, or docker mounting it at
    ``/workspace``. ssh, modal and the other sandboxes resolve their own cwd."""
    backend = str(terminal.get('TERMINAL_ENV') or 'local').strip().lower()
    mounts = str(terminal.get('TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE') or '').strip().lower() in {'1', 'true', 'yes', 'on'}
    return backend == 'local' or (backend == 'docker' and mounts)


def _profile_workspace_cwd(home):
    """A NAMED profile's configured ``terminal.cwd``, which beats a client cwd the Desktop only inherited
    (``cwd_explicit: false``; main's ``_completion_cwd``). Placeholders and missing dirs are unset."""
    import os
    from hermes_constants import profile_name_for_home
    if profile_name_for_home(home) in {None, 'default'}:
        return None
    from hermes_cli.config import read_user_config_raw
    terminal = read_user_config_raw(Path(home) / 'config.yaml').get('terminal')
    raw = str((terminal if isinstance(terminal, dict) else {}).get('cwd') or '').strip()
    configured = os.path.abspath(os.path.expanduser(raw)) if raw not in {'', '.', 'auto', 'cwd'} else None
    return configured if configured and os.path.isdir(configured) else None


def _apply_launch_overrides(params, config):
    """Validate the explicit model/agent launch overrides and fold them into *config*."""
    from urllib.parse import urlsplit
    from hermes_cli.config import resolve_turn_limit
    from hermes_constants import parse_reasoning_effort
    for key in ('provider', 'base_url'):
        if key in params:
            value = params[key]
            if not isinstance(value, str) or not value.strip():
                _refuse(key, 'must be a non-empty string')
            if key == 'base_url':
                url = urlsplit(value)
                if (url.scheme not in {'http', 'https'} or not url.hostname
                        or url.username or url.password or url.query or url.fragment):
                    _refuse('base_url', 'must be an http(s) URL without credentials, query or fragment')
            config.setdefault('model', {})[key] = value
    bad_flag = next((flag for flag in ('ignore_rules', 'yolo', 'checkpoints', 'accept_hooks', 'pass_session_id',
                                      'follow_profile_config', 'cwd_explicit', 'verbose')
                     if flag in params and type(params[flag]) is not bool), None)
    if bad_flag:
        _refuse(bad_flag, 'must be a boolean')
    if 'service_tier' in params:
        # An explicit speed pick (Desktop Fast / Ultrafast / Normal): normal pins normal over a
        # profile tier; an unknown word is refused, never read as normal.
        from agent.fast_mode import parse_exact_service_tier
        try:
            tier = parse_exact_service_tier(params['service_tier']) if isinstance(params['service_tier'], str) else None
        except ValueError:
            tier = None
        if tier is None:
            _refuse('service_tier', f"not a known tier: {params['service_tier']!r}")
        config.setdefault('agent', {})['service_tier'] = tier or 'normal'
    if 'run_budget' in params:
        # `--run-budget SECONDS`: the wall-clock budget per run (80% wrap-up notice, capped provider
        # timeouts), frozen as the session's `agent.run_budget_seconds` like main's CLI flag over config.
        value = params['run_budget']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            _refuse('run_budget', 'must be a positive number of seconds')
        config.setdefault('agent', {})['run_budget_seconds'] = float(value)
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
            _refuse('max_turns', f'unreadable cap: {value!r}')
        config.setdefault('agent', {})['max_turns'] = limit
    if 'reasoning' in params:
        if not isinstance(params['reasoning'], str) or parse_reasoning_effort(params['reasoning']) is None:
            _refuse('reasoning', f"not a known effort level: {params['reasoning']!r}")
        config.setdefault('agent', {})['reasoning_effort'] = params['reasoning']
        # An explicit launch level wins over a per-model default, just like CLI.
        config['agent'].pop('reasoning_overrides', None)


def _resolve_toolsets(params, config, source, safe_mode, coding_posture=False, bypass=False):
    """Validate explicit toolsets and return the session's enabled toolset set. Without a pick, a
    *coding_posture* session runs ``focus_toolsets`` instead of the profile default while the profile
    stays on ``focus``; the gateway decides it, never the client."""
    from hermes_cli.tools_config import _get_platform_tools, enabled_mcp_server_names
    from hermes_cli.toolset_validation import saved_toolset_resolver
    from toolsets import validate_toolset
    explicit = params.get('toolsets')
    if 'toolsets' in params:
        if not isinstance(explicit, list):
            _refuse('toolsets', f'must be a list of toolset names, got {type(explicit).__name__}')
        # The CLI validator's rule: a configured MCP server's toolset registers only after its first
        # connect, so an offline server is inactive, not a typo (#138748, #139154). A bypass launch never
        # reads the profile, so it takes the in-process registry alone.
        known = validate_toolset if bypass else saved_toolset_resolver(config)
        configured = config.get('mcp_servers') if isinstance(config.get('mcp_servers'), dict) else {}
        enabled_servers = enabled_mcp_server_names(config) if configured else set()
        for name in explicit:
            if not isinstance(name, str) or not known(name):
                _refuse('toolsets', f'unknown or invalid toolset name: {name!r}')
            server = name[4:] if name.startswith('mcp-') and name not in configured else name
            if server in configured and server not in enabled_servers:
                # Picking a disabled server must not match nothing and fall through to every enabled one.
                _refuse('toolsets', f'MCP server {server!r} is disabled in config')
        if 'desktop_ui' in explicit and source != 'gui':
            _refuse('toolsets', f'desktop_ui is only available on the Desktop, not on {source}')
        config.setdefault('platform_toolsets', {})['acp' if source == 'acp' else 'cli'] = explicit
    posture = None
    if explicit is None and coding_posture:
        from agent.coding_context import focus_toolsets
        posture = focus_toolsets(config)
    enabled = set(posture) if posture is not None else _get_platform_tools(config, 'acp' if source == 'acp' else 'cli')
    if explicit is not None:
        from toolsets import resolve_toolset
        requested = {t for name in explicit for t in resolve_toolset(name)}
        effective = {t for name in enabled for t in resolve_toolset(name)}
        if not requested <= effective:
            _refuse('toolsets', f"not available on this surface: {', '.join(sorted(requested - effective))}")
    if explicit is None and source in {'tui', 'gui'}:
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


def _in_coding_workspace(config, source, cwd):
    """Whether *cwd* selects the coding posture (``agent.coding_context: focus`` in a code workspace).
    A detection failure keeps the profile default rather than refusing the session."""
    from agent.coding_context import in_focus_workspace
    try:
        return in_focus_workspace(platform=SURFACES[source], cwd=cwd, config=config)
    except Exception:
        logger.debug('coding posture detection failed for %s; keeping the profile default', cwd, exc_info=True)
        return False


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
    if 'skills' in params and (not isinstance(skills, list) or not skills
                               or any(not isinstance(s, str) or not s.strip() for s in skills)):
        _refuse('skills', 'must be a non-empty list of skill names')
    if skills and any(params.get(name) is True for name in BYPASS_FIELDS):
        _refuse('skills', 'a bypass launch never reads the profile, so skills cannot be preloaded')
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
    shell hooks in the per-user allowlist (no prompt; kept in memory if that write fails), which
    the open's ``refresh_profile_hooks`` then registers. A bypass launch has no profile hooks to
    consent to, so the flag is a no-op there rather than a refusal. Allowlist I/O: off the loop."""
    if json.loads(policy.request_json).get('accept_hooks') and not policy.ignore_user_config:
        from agent.shell_hooks import accept_for_launch
        from gateway.run import _load_gateway_config
        accept_for_launch(_load_gateway_config())


def refresh_profile_hooks(policy):
    """Bring this owner's shell hooks in step with the profile on every local session open
    (create and resume): hooks added to ``config.yaml`` or approved since the gateway started
    register, removed or revoked ones stop firing, as each new CLI process did on main. Reads the
    owner home's LIVE config (as session creation does), never the session's frozen copy: the
    registry is shared by every session and surface of this profile. Consent comes from the
    allowlist (the launching client's TTY prompt, or ``accept_launch_hooks``) or an opt-in in force
    on the gateway itself; this never prompts. A bypass launch registers nothing (main's safe mode
    registered none). Config read + allowlist I/O: run off the loop."""
    if policy is None or policy.ignore_user_config:
        return
    from agent.shell_hooks import sync_from_config
    from gateway.run import _load_gateway_config
    try:
        sync_from_config(_load_gateway_config)  # it never removes on an empty (failed-open) config
    except Exception:  # health: allow BLE001 -- best effort; a hook re-sync never blocks opening a session
        logger.warning("shell-hook re-sync failed; keeping the registered hooks", exc_info=True)


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
        # Nothing private remains (a followed section's secret, or a cleared key): a stale reference
        # would no longer match the re-fingerprinted policy identity.
        return replace(policy, credential_ref=None, config_secret_ref=None)
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
        _refuse('api_key', 'must be a non-empty string')
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
                or not isinstance(policy.model, str) or not policy.model.strip()
                or not isinstance(policy.toolsets, (list, tuple))
                or any(not isinstance(name, str) for name in policy.toolsets)
                or type(policy.safe_mode) is not bool or type(policy.ignore_user_config) is not bool
                or not isinstance(policy.skills_prompt, (str, type(None)))
                or type(policy.coding_posture) is not bool
                or (policy.safe_mode and not policy.ignore_user_config)):
            raise ValueError('invalid policy')
        from dataclasses import replace
        for value in (policy.config_json, policy.request_json, policy.terminal_json):
            if not isinstance(json.loads(value), dict):
                raise ValueError('invalid policy object')
        terminal = json.loads(policy.terminal_json)
        if terminal_uses_host_cwd(terminal) and terminal.get('TERMINAL_CWD') != policy.cwd:
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
    # Readiness, not structure: a removed workspace keeps the chat readable and movable, but no
    # turn runs in a missing directory.
    if not Path(policy.cwd).is_dir():
        raise RuntimeStoreError('workspace_unavailable')
    from agent.runtime_cwd import set_session_cwd
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    terminal = json.loads(policy.terminal_json)
    if policy.config_secret_ref is not None:
        from gateway.session_policy_credentials import recover_config_secrets
        for path, value in recover_config_secrets(authority, policy).items():
            if path[0] is None:
                terminal[path[1]] = value
    # A following session's turn reads its profile once; every resolution in the turn uses that read.
    profile_token = _TURN_PROFILE.set((policy, policy.read_profile(authority)))
    active = policy
    if policy.follows_profile:
        from dataclasses import replace
        active = replace(policy, toolsets=policy.current_toolsets(policy.config(authority, secrets=False)))
    cwd_token = set_session_cwd(policy.cwd)
    terminal_token = set_terminal_scope(terminal)
    policy_token = _ACTIVE_POLICY.set(active)
    from gateway.session_local_editor import editor_scope
    try:
        from gateway.session_local_mcp import editor_mcp_scope
        with editor_scope(policy), editor_mcp_scope(authority, policy):
            yield
    finally:
        _ACTIVE_POLICY.reset(policy_token)
        reset_terminal_scope(terminal_token)
        cwd_token.var.reset(cwd_token)
        _TURN_PROFILE.reset(profile_token)


def active_policy():
    """The frozen launch policy of the turn executing in this context, or ``None`` (standalone
    serve / messaging turns). Consumers read it instead of re-deriving from live config."""
    return _ACTIVE_POLICY.get()
