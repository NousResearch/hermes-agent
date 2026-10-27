"""Frozen, explicit local-client launch policy for the existing TurnRunner."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property
import contextlib
import json
import logging
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

CREATE_FIELDS = frozenset({'request_id', 'source', 'cwd', 'model', 'toolsets',
                           'provider', 'base_url', 'reasoning', 'max_turns', 'ignore_rules', 'api_key', 'editor',
                           'yolo', 'safe_mode', 'ignore_user_config',
                           'skills', 'checkpoints', 'accept_hooks', 'pass_session_id', 'follow_profile_config',
                           'service_tier', 'cwd_explicit', 'run_budget', 'verbose', 'row_cwd', 'client_env'})
# Sections a ``follow_profile_config`` session re-reads from the owning profile's CURRENT config each
# turn (main's Bot Chat row), credentials included: a frozen key never crosses a profile route change.
FOLLOW_SECTIONS = ('model', 'providers', 'custom_providers', 'fallback_providers', 'fallback_model', 'model_aliases',
                   'platform_toolsets')
# A Bot Chat an earlier build committed frozen follows its profile once a client opens it (``follow_bot_chat``).
# The marker sits in the policy's own config, so no other receipt changes shape (an older owner restores
# them all, and reads this chat as frozen), and the creation request, the exact-retry identity, stays as sent.
# A chat created on the welcome trial's model carries it too (``stage_local_session``), so a provider the user
# connects later replaces that model.
FOLLOW_ADOPTED = 'follow_adopted'
# A Desktop chat its creator's connection declared local (``process_identity.local_desktop_client``): its
# turns run as work for the local Desktop (``local_desktop_turn``). Owner-set in the policy's own config, so
# an exact create retry from a socket whose declaration has not landed carries the same creation request,
# and ``launch_request`` never echoes it.
LOCAL_DESKTOP_CHAT = 'desktop_local'
# A launch field's state as a committed session edit left it, in the policy's own config: ``True`` the
# session chose it (``/reasoning <level>``, a model pick carrying one), ``False`` a reset gave it back to
# the profile. Either wins over the creation request (the launch identity, never rewritten) and the older
# pick receipts.
SESSION_PINS = 'session_pins'


def session_pinned(config, field, launched):
    """Whether *field* is the session's own choice: its last committed session edit, else *launched*."""
    pins = config.get(SESSION_PINS)
    return pins[field] if isinstance(pins, dict) and isinstance(pins.get(field), bool) else launched


def pin_reasoning_effort(config, effort, *, session=False):
    """Write an explicit session reasoning level into *config*: it wins over a per-model default
    (``reasoning_overrides``), at creation, on ``/reasoning`` and on a model pick that carries one.
    A *session* edit also records the pin."""
    config.setdefault('agent', {})['reasoning_effort'] = effort
    config['agent'].pop('reasoning_overrides', None)
    if session:
        config[SESSION_PINS] = {**(config.get(SESSION_PINS) or {}), 'reasoning': True}


def profile_config(authority):
    """The owning profile's current ``config.yaml`` (its present sections), read fresh."""
    from gateway.run import _load_gateway_config
    return present_sections(_load_gateway_config(Path(authority.db.db_path).resolve().parent / 'config.yaml'))


def unpin_reasoning(config, profile):
    """``/reasoning reset``: the level is *profile*'s (its current config) again, re-read every turn by a
    following session and on each resume by any other; a create-time ``reasoning`` included."""
    from gateway.session_policy_refresh import _follow
    followed = profile.get('agent') if isinstance(profile.get('agent'), dict) else {}
    agent = config.setdefault('agent', {})
    for key in ('reasoning_effort', 'reasoning_overrides'):
        _follow(agent, followed, key)
    config[SESSION_PINS] = {**(config.get(SESSION_PINS) or {}), 'reasoning': False}


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
# ``tool`` and ``oneshot`` stay stored under their own labels so pickers can hide them.
# Every other creation label is rewritten to ``source='local'`` on peer refresh.
_KEEP_STORED_LABEL = frozenset({'tool', 'oneshot'})
STORED_AS_LOCAL = frozenset(SURFACES) - _KEEP_STORED_LABEL
# Routing labels and the accounting placeholder: never a creating surface.
_NOT_A_SURFACE = frozenset({None, '', 'local', 'unknown'})


def created_surface(label):
    """``created_source`` for a policy or live label.

    ``local`` and ``unknown`` are not surfaces. A label stored as ``local`` maps
    through ``SURFACES`` (``gui`` is ``desktop``). Any other label, including
    ``tool`` and ``oneshot``, is kept as itself: created_source is provenance.
    """
    if label in _NOT_A_SURFACE:
        return None
    return SURFACES[label] if label in STORED_AS_LOCAL else label


# Policy label and agent surface for one conversation, keyed by either name: ``--source gui``
# and ``--source desktop`` are the same rows. ``tool`` and ``oneshot`` are not aliases of
# ``cli`` here: they keep their own stored label.
SURFACE_GROUPS = {name: group for group in (frozenset({label, SURFACES[label]}) for label in STORED_AS_LOCAL)
                  for name in group}


def _refuse(field, detail=None):
    """One named refusal: the launch field the check read, and why it failed.

    ``reason`` stays the bounded wire code every caller compares as a string; ``field`` and
    ``detail`` are what tells ~20 different create checks apart, in the client and in the
    gateway log, without a second run to find out which one fired (``RuntimeStoreError`` bounds
    and flattens both)."""
    raise RuntimeStoreError('invalid_params', field=field, detail=detail)


# ``acp_adapter/edit_approval.py``'s AUTO_APPROVE_* values, in main's ACP mode order.
EDIT_APPROVAL_POLICIES = ('ask', 'workspace_session', 'session')


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
    # The shell-launched client's allowlisted environment (``agent.session_client_env``), bound for this
    # session's turns only; None when the client sent none (Desktop, a session from before the field).
    client_env_json: str | None = None
    # The client sent one over an attach the gateway could not prove is on its host (no gateway-minted
    # ticket): only host-free values were kept, and its tools run with the gateway's own toolchain.
    client_env_remote: bool = False
    # The rules that map was normalised by (``client_env_rules``: allowlist, secret filter, locality) as
    # plain data: an exact retry is judged by them, never by the profile's current ones, and the session's
    # tools run without every toolchain name they cover that the client did not send.
    client_env_rules_json: str | None = None
    # An ACP session's edit approval (``session.mutate edit_policy``; main's ACP modes): ``ask`` (Default),
    # ``workspace_session`` (Accept Edits) or ``session`` (Don't Ask). Sensitive paths always ask.
    edit_approval_policy: str = 'ask'

    def client_env(self):
        return None if self.client_env_json is None else json.loads(self.client_env_json)

    def client_env_scope(self):
        from agent.session_client_env import client_env_scope
        rules = None if self.client_env_rules_json is None else json.loads(self.client_env_rules_json)
        return client_env_scope(self.client_env(), local=not self.client_env_remote, rules=rules)

    def config(self, authority=None, *, secrets=True):
        return self.effective(authority, secrets=secrets)[0]

    def effective(self, authority=None, *, secrets=True):
        """``(config, model)`` the next turn runs, both from ONE read of the owning profile when following
        (the turn's policy scope holds that read): a route never pairs one profile's model with another's
        endpoint. Explicit choices (create-time reasoning / tier / toolsets, a session ``/reasoning`` until
        reset, a pick that still holds) are applied over the followed defaults; frozen secrets of followed
        sections are never recovered."""
        config = present_sections(json.loads(self.config_json))
        live = self.read_profile(authority)
        if self.config_secret_ref is not None and authority is not None and secrets:
            from gateway.session_policy_credentials import disable_dropped_secrets, recover_config_secrets
            recovered = recover_config_secrets(authority, self)
            for path, value in recovered.items():
                if path[0] is None:  # private terminal projection, not config
                    continue
                target = config
                for key in path[:-1]:
                    # An alias entry added after creation is absent from the frozen tree.
                    target = target.setdefault(key, {}) if isinstance(target, dict) else target[key]
                target[path[-1]] = value
            disable_dropped_secrets(config, getattr(recovered, 'dropped', ()), getattr(recovered, 'session', None))
        model = self.model
        if live:
            frozen, request = config.get('model') or {}, json.loads(self.request_json)
            pick = frozen.get('selection_profile')
            held = pick is not None and pick == list(profile_target(live)) and entry_serves(frozen, live)
            pinned = {'platform_toolsets', 'disabled_toolsets'} if 'toolsets' in request else set()
            chosen = session_pinned(config, 'reasoning', 'reasoning' in request)
            pinned |= {'reasoning_effort', 'reasoning_overrides'} if chosen else set()
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

    @cached_property
    def follows_profile(self):
        # Read ~8-10 times a turn on the event loop, and a frozen policy's config is a whole profile:
        # parse once per (immutable) policy; ``replace()`` builds a fresh instance with its own cache.
        return (json.loads(self.request_json).get('follow_profile_config') is True
                or json.loads(self.config_json).get(FOLLOW_ADOPTED) is True)

    def read_profile(self, authority):
        """The owning profile's current config for a following session, else None; inside this policy's
        turn scope, the turn's one read."""
        if authority is None or not self.follows_profile:
            return None
        held = _TURN_PROFILE.get()
        if held is not None and held[0] == self:
            return held[1]
        return profile_config(authority)

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


def model_route(block):
    """``(provider, endpoint)`` identity of a ``model:`` block: what an endpoint-bound credential belongs to."""
    from hermes_cli.route_identity import normalize_route_base_url
    block = block if isinstance(block, dict) else {}
    return str(block.get('provider') or '').strip().lower(), normalize_route_base_url(block.get('base_url'))


def bound_route_holds(bound, bound_config, block, config):
    """True when *block* (resolved in *config*) is the route *bound* was bound to, *bound* resolved in
    the config it was bound in: a respelled provider on one endpoint is one route, and a named entry
    whose ``base_url`` moved is another. An ``auto``/unset side keeps the literal comparison: a
    credential never follows a route the resolver picks later."""
    from gateway.session_route_identity import resolved_route, same_route
    bound = bound if isinstance(bound, dict) else {}
    identity, url = resolved_route(bound, bound_config)
    if 'auto' in (identity, resolved_route(block, config)[0]):
        return model_route(bound) == model_route(block)
    return same_route({**bound, 'base_url': url} if url else bound, block, config)


def entry_serves(block, config):
    """False when *block* pins a ``base_url`` that the config entry its credential comes from no longer
    serves in *config*: its named provider (``providers:`` / ``custom_providers:``), or the alias its
    ``api_key_from`` points into. That entry's key (inline, ``key_env``, ``key_cmd``) authenticates the
    entry's own endpoint, so a pick held on the old one ends rather than carry the new key there."""
    from hermes_cli.route_identity import normalize_route_base_url
    block = block if isinstance(block, dict) else {}
    url = normalize_route_base_url(block.get('base_url'))
    if not url:
        return True
    from gateway.session_route_identity import _named_entry
    provider = str(block.get('provider') or '').strip()
    entries = [_named_entry(provider, config)] if provider else []
    source = block.get('api_key_from')
    if isinstance(source, list) and len(source) > 1:
        alias = config
        for part in source[:-1]:
            alias = alias.get(part) if isinstance(alias, dict) else None
        entries.append(alias)
    return all(normalize_route_base_url(entry.get('base_url')) == url for entry in entries
               if isinstance(entry, dict) and str(entry.get('base_url') or '').strip())


def _held_pick(frozen, live):
    """A pick that still holds keeps its selection; on the profile's own endpoint the rest of the
    ``model:`` block (a rotated key included) is the profile's. Another endpoint never gets it."""
    if model_route(frozen) != model_route(live):
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


def build_policy(params, config, *, private_secrets=None, profile_terminal=True, workspace_posture=False,
                 client_local=False):
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
    picked_route = {'model', 'provider', 'base_url'} & set(_without_auto_pick(params))
    if params.get('follow_profile_config') is True and picked_route:
        # Any explicit route pick (a model, a provider, an endpoint) records its provenance before the
        # launch route folds in: it holds while the profile still targets this, and a later profile edit
        # supersedes it. Without it the first turn's followed route replaces the pick.
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
    client_env = rules = None
    if 'client_env' in params:
        from agent.session_client_env import client_env_rules, normalise_client_env
        rules = client_env_rules(config, local=client_local)
        try:
            client_env = json.dumps(normalise_client_env(params['client_env'], rules), sort_keys=True)
        except ValueError as exc:
            _refuse('client_env', str(exc))
        rules = json.dumps(rules, sort_keys=True)
    # Values stay out of the request: ``launch_request`` is echoed to every viewer of the session.
    request = {k: v for k, v in params.items() if k not in {'request_id', 'api_key', 'client_env'}}
    request.setdefault('source', 'cli')
    from gateway.session_local_mcp import private_editor_request
    private_editor_request(request, private_secrets)
    _extract_config_secrets(config, private_secrets)
    _extract_config_secrets(terminal, private_secrets, (None,))
    return LocalSessionPolicy(source, SURFACES[source], cwd, model, tuple(sorted(enabled)),
                              json.dumps(config), json.dumps(request, sort_keys=True), json.dumps(terminal),
                              safe_mode=safe_mode, ignore_user_config=ignore_user_config,
                              coding_posture=coding_posture, client_env_json=client_env,
                              client_env_remote=client_env is not None and not client_local,
                              client_env_rules_json=rules)


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


def _without_auto_pick(params):
    """`provider: auto` is the thin client's "no pick" sentinel: the profile's route stands as configured."""
    if isinstance(params.get('provider'), str) and params['provider'].strip().lower() == 'auto':
        return {k: v for k, v in params.items() if k != 'provider'}
    return params


def _keeps_profile_route(picked, config):
    """Whether a provider pick resolves to the profile's own route, so its endpoint and key pins stay.

    The same resolved provider under another spelling (`vllm` for `custom`, `deep-infra` for
    `deepinfra`, `custom:<name>` for a named entry) is the same route. A block with an endpoint
    and no provider (or `auto`) already routes to `custom` there.
    """
    from gateway.session_route_identity import profile_route_identity, resolved_route
    held_id = profile_route_identity(config)
    if held_id == 'auto' or not isinstance(picked, str) or not picked.strip():
        return False
    return resolved_route({'provider': picked}, config)[0] == held_id


def _with_entry_pick(params, config):
    """*params* with a provider pick of the held entry's bare config key spelled as its ``custom:<key>``.

    A configured entry keyed like a canonical built-in (`providers: deepseek`) is shadowed by the
    built-in in raw provider resolution, so the bare spelling would route, and be frozen, as a
    different provider than the entry the profile runs (#140107). Only the held entry's bare key
    is that entry (Desktop's guard). A keyed entry's display name is not; a keyless legacy
    ``custom_providers:`` entry's is (``configured_entry_key``). A bare ``custom`` pick is the billing
    class a client echoed from the runtime, not a route: on a bare ``custom`` profile with an endpoint
    it is that profile's route unless it picks another endpoint, or another entry lists a model that
    is not the profile's default; otherwise it names the configured entry serving its endpoint/model
    (the profile's own first), else (no endpoint picked) the profile's own entry. A launch key for an
    entry the model alone chose is refused. Other picks stay as they came.
    """
    from gateway.session_route_identity import configured_entry_key, profile_route_identity
    picked = params.get('provider')
    if not isinstance(picked, str):
        return params
    identity = profile_route_identity(config)
    if picked.strip().lower() == 'custom':
        from hermes_cli.runtime_provider_custom import _route_key, routed_custom_identity
        url, model = params.get('base_url'), str(params.get('model') or '')
        own_url = _route_key(model_route(config.get('model'))[1])
        if identity == 'custom' and own_url and (
                _route_key(url) == own_url if url is not None
                else model.strip().lower() == profile_target(config)[0].lower()
                or not routed_custom_identity(None, model, config)):
            return params
        held = identity if identity.startswith('custom:') else None
        healed = routed_custom_identity(url, model, config, held=held) or (
            identity if identity.startswith('custom:') and url is None else None)
        if healed and healed != identity and url is None and 'api_key' in params:
            # The model alone chose this entry: a launch key the user gave is never sent to a host they did not name.
            _refuse('api_key', f'{healed} was chosen by model; name it with --provider or --base-url to use a launch key')
        return {**params, 'provider': healed} if healed else params
    if identity == 'auto' or picked.strip().lower() != (configured_entry_key(identity, config) or None):
        return params
    return {**params, 'provider': identity}


def _apply_launch_overrides(params, config):
    """Validate the explicit model/agent launch overrides and fold them into *config*."""
    from urllib.parse import urlsplit
    from hermes_cli.config import resolve_turn_limit
    from hermes_constants import parse_reasoning_effort
    params = _with_entry_pick(_without_auto_pick(params), config)
    picks_profile_entry = 'provider' in params and _keeps_profile_route(params['provider'], config)
    if 'provider' in params and not picks_profile_entry:
        model = config.setdefault('model', {})
        # A provider pick is a route change: the profile's endpoint, wire protocol and
        # credential pointers belong to the provider being replaced, and pairing the new one
        # with them sends its key to the old host (401) or the old key down the new route.
        # They are dropped before the overrides fold in, so the target provider resolves its
        # own — the same route keys a committed /model switch rewrites (#138843). An explicit
        # base_url is applied after the drop, so a pinned endpoint survives without carrying
        # the replaced provider's other pins, and a same-provider re-pick keeps the profile's.
        from gateway.session_route_identity import ROUTE_BOUND_KEYS
        # The replaced provider's context window goes too: it was sized for the old model.
        for stale in ROUTE_BOUND_KEYS:
            model.pop(stale, None)
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
        pin_reasoning_effort(config, params['reasoning'])


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
                why = 'is disabled in config' if isinstance(configured[server], dict) else 'has no usable config entry'
                _refuse('toolsets', f'MCP server {server!r} {why}')
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


def bind_launch_key(authority, session_id, policy, api_key, *, config_secrets=None, owner_key=False,
                    client_request=None):
    """CLI keys live only in this authority lifetime, never its durable receipt.

    Restart deliberately revokes them. History remains readable; inference must
    report launch_credentials_unavailable, never select a profile fallback key.
    An ``owner_key`` (an alias's, expanded by the owner) of the same create request as the attempt
    holding the slot (``same_create_retry``, its ``client_env`` in ``client_request``) takes the held
    key: a ``.env`` rotation between overlapping attempts is not a changed request, and the policy
    then fingerprints the key it runs with.
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
    owners = getattr(authority, '_local_owner_key_requests', None)
    if owners is None:
        owners = authority._local_owner_key_requests = {}
    if owner_key and api_key is not None and old is not None and ref in owners:
        from hermes_state_local import same_create_retry
        api_key = old if same_create_retry(owners[ref], policy.request_json, client_request or {}) else api_key
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
    if api_key is not None and old is None and owner_key:
        owners[ref] = {k: getattr(policy, k) for k in ('request_json', 'client_env_json', 'client_env_rules_json')}
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


def _secret_refs(authority, session_ids):
    """The launch-key slots and config-secret references the live LOCAL policies of *session_ids* name."""
    from gateway.config import Platform
    sources = [getattr(authority.sessions.get(sid), 'source', None) for sid in session_ids]
    chats = [source.chat_id for source in sources if source is not None and source.platform == Platform.LOCAL]
    if not chats:
        return set()  # only a restored (live) LOCAL session has a policy in the adapter map
    from gateway.session_local_recovery import local_adapter_map
    policies = getattr(local_adapter_map(authority).get(Platform.LOCAL), 'policies', None) or {}
    refs = set()
    for policy in filter(None, map(policies.get, chats)):
        refs.update(ref for ref in (policy.credential_ref and _key_slot(policy.credential_ref),
                                    policy.config_secret_ref) if ref)
    return refs


def release_launch_secrets(authority, session_ids):
    """Drop retired sessions' launch keys, frozen config secrets and editor MCP servers.

    They live only in this authority's memory for the session's lifetime; a deleted session can
    never run again, so keeping them would hold its credentials until the process exits. A branch's
    policy names its parent's slots, so a slot named after a retired session stays while a live
    policy references it, and then until restart, which revokes every launch secret anyway. Only
    slots named after *session_ids* are released, so the set never exceeds release by name: no later
    sweep, and a binding not yet published always sits under its own live session's name. Call
    before the retired sessions leave ``authority.sessions``."""
    import hashlib
    from gateway.session_policy_credentials import PREFIX
    retired = {str(sid) for sid in session_ids}
    if not retired:
        return
    kept = _secret_refs(authority, set(authority.sessions) - retired)
    prefix = f'{authority.instance_id}:{authority.epoch}:'
    keys = getattr(authority, '_local_launch_keys', None) or {}
    # A key re-supplied after a restart sits in the slot of the epoch that minted it.
    owners = getattr(authority, '_local_owner_key_requests', None) or {}
    for ref in [ref for ref in keys if ref.split(':', 2)[-1] in retired and ref not in kept]:
        del keys[ref]
        owners.pop(ref, None)
    configs = getattr(authority, '_local_config_secrets', None) or {}
    for ref in list(configs):
        if ref.startswith(PREFIX):
            owner = json.loads(ref[len(PREFIX):]).get('session')
        else:
            owner = ref[len(prefix):] if ref.startswith(prefix) else None
        if owner in retired and ref not in kept:
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


def route_launch_key(authority, policy, config):
    """The launch key a turn on *config* (the route it runs, a followed profile's included) sends.
    A launch key authenticates the frozen route it was bound with: once the followed profile
    supersedes that pick and moves to another endpoint, the key stays behind with its route."""
    bound = present_sections(json.loads(policy.config_json))  # the route the key was bound with
    if not bound_route_holds(bound.get('model'), bound, config.get('model'), config):
        return None
    return launch_key(authority, policy)


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
                or not _valid_client_env(policy.client_env_json) or type(policy.client_env_remote) is not bool
                or not _valid_client_env(policy.client_env_rules_json, list)
                or policy.edit_approval_policy not in EDIT_APPROVAL_POLICIES
                or (policy.edit_approval_policy != 'ask' and policy.source != 'acp')
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


def _valid_client_env(value, kind: type = str):
    """A stored client map (names to strings) or its rules (``kind=list``: rule names to string lists)."""
    if value is None:
        return True
    env = json.loads(value) if isinstance(value, str) else None
    return isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, kind) and (
        kind is str or all(isinstance(item, str) for item in v)) for k, v in env.items())


def policy_turn_toolsets(authority, policy, config, session_id):
    """A local turn's toolsets: the policy's current set, plus ``onboarding`` for an onboarding chat
    (``agent.onboarding_session.onboarding_chat_toolsets``)."""
    from agent.onboarding_session import onboarding_chat_toolsets
    return onboarding_chat_toolsets(list(policy.current_toolsets(config)), getattr(authority, 'db', None), session_id)


def local_desktop_turn(runner, source, scope):
    """*scope* (the profile scope a turn enters), also marked as work for the local Desktop when the
    turn belongs to a Desktop chat its creator's connection declared local (``stage_local_session``);
    any other turn gets *scope* untouched. Read from the live policy without raising: a cold policy
    is restored, or refused, by the turn."""
    from gateway.config import Platform
    policy = None
    if getattr(source, 'platform', None) == Platform.LOCAL:
        policy = getattr(runner._adapter_for_source(source), 'policies', {}).get(source.chat_id)
    # The request_json mark: a chat an earlier build created, which stamped it into the creation request.
    if policy is None or policy.source != 'gui' or not any(
            json.loads(raw).get(LOCAL_DESKTOP_CHAT) is True for raw in (policy.config_json, policy.request_json)):
        return scope
    return _serving_local_desktop_within(scope)


@contextmanager
def _serving_local_desktop_within(scope):
    from hermes_cli.process_identity import serving_local_desktop
    with scope, serving_local_desktop(True):
        yield


def policy_for_source(runner, source):
    from gateway.session_local import LocalSessionAdapter
    from gateway.config import Platform
    if source is None or source.platform != Platform.LOCAL:  # None: a history mount, which runs nothing
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


def workspace_ready(policy) -> bool:
    """Readiness, not structure: a removed workspace keeps the chat readable and movable, but no
    turn runs in a missing directory."""
    return Path(policy.cwd).is_dir()


def workspace_unavailable_result(policy) -> dict:
    """A failed, never-run turn for a session whose frozen working directory is gone. Nothing ran
    and an unchanged resend fails the same way, so the reply names the folder and the way out
    instead of the generic "something went wrong" apology."""
    from agent.i18n import t
    from agent.turn_failure_copy import stamp_failure
    text = t("gateway.errors.workspace_unavailable", path=policy.cwd)
    return stamp_failure({"final_response": text, "messages": [], "api_calls": 0, "tools": [],
                          "failed": True, "completed": False, "error": text},
                         "workspace_unavailable", False)


def refused_turn_result(policy, exc) -> dict | None:
    """A failed, never-run turn for an owner refusal at scope entry that an unchanged resend repeats
    (gone working folder, changed launch key); None for any other refusal."""
    from agent.error_surface import build_error_surface_from_result
    from agent.turn_failure_copy import launch_credentials_copy, stamp_failure
    if exc.reason == 'workspace_unavailable':
        result = workspace_unavailable_result(policy)
    elif exc.reason == 'launch_credentials_unavailable':
        text = launch_credentials_copy(exc)
        result = stamp_failure({"final_response": text, "messages": [], "api_calls": 0, "tools": [],
                                "failed": True, "completed": False, "error": text},
                               "launch_credentials_unavailable", False)
    else:
        return None
    # The descriptor makes settlement commit the explanation and its card as the transcript's
    # failed-turn boundary (gateway_refusal_boundary), so a reopened chat still shows the way out.
    result["error_surface"] = build_error_surface_from_result(result)
    return result


@contextmanager
def policy_scope(policy, *, authority=None):
    if policy is None:
        yield
        return
    if not workspace_ready(policy):
        raise RuntimeStoreError('workspace_unavailable', field='cwd', detail=policy.cwd)
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
        from agent.auxiliary_task_config import session_auxiliary_scope
        from gateway.session_local_mcp import editor_mcp_scope
        # Auxiliary tasks read the session's frozen section, never live config.yaml (a rotated key there).
        auxiliary = policy.config(authority).get('auxiliary') if authority is not None else None
        aux_scope = session_auxiliary_scope(auxiliary) if authority is not None else contextlib.nullcontext()
        with (editor_scope(policy), editor_mcp_scope(authority, policy), aux_scope,
              policy.client_env_scope()):
            yield
    finally:
        _ACTIVE_POLICY.reset(policy_token)
        reset_terminal_scope(terminal_token)
        cwd_token.var.reset(cwd_token)
        _TURN_PROFILE.reset(profile_token)


def bind_turn_workspace(policy, task_id):
    """Make the session's frozen workspace its terminal cwd override for this turn's tools (#140665).

    The client's cwd (or ``--in DIR``) is frozen into the policy at admission, but the terminal tool
    mounts ``resolve_task_overrides(task_id)['cwd']`` into a per-session Docker container. Before the
    one-gateway runtime each ``hermes chat`` was its own process with its own ``TERMINAL_CWD``; now the
    turn runs here, so register it under the turn's own task id (inside the turn's routed scope, which
    qualifies the key). Docker mounting the host cwd only: the local backend already runs in the
    policy's ``TERMINAL_CWD``, and ssh and the other sandboxes keep their own cwd. Re-registered only
    when the workspace changed, so an agent's ``cd`` inside the session survives later turns.
    """
    if policy is None or not task_id:
        return
    terminal = json.loads(policy.terminal_json)
    if str(terminal.get('TERMINAL_ENV') or '').strip().lower() != 'docker' or not terminal_uses_host_cwd(terminal):
        # The session moved off a host-cwd Docker mount (e.g. to SSH with its own terminal.cwd): the
        # binding an earlier turn registered would otherwise win in _plan_execution on the new backend.
        unbind_turn_workspace(task_id)
        return
    from tools.terminal_tool import register_task_env_overrides, resolve_task_overrides, workspace_key_for
    current = resolve_task_overrides(task_id)
    if current.get('cwd') == policy.cwd and current.get('cwd_source') == 'session':
        return
    register_task_env_overrides(task_id, {**current, 'cwd': policy.cwd, 'cwd_source': 'session',
                                          'workspace_key': workspace_key_for(policy.cwd)})


def unbind_turn_workspace(task_id):
    """Drop the workspace override :func:`bind_turn_workspace` registered for *task_id*, and the cwd
    record it seeded. Only its own entry (tagged ``workspace_key``): an ACP/TUI workspace or an RL
    image override registered by someone else under the same id stays."""
    if not task_id:
        return
    from tools.terminal_tool import _qualify_task_key, _task_env_overrides, clear_session_cwd, get_session_cwd
    key = _qualify_task_key(task_id)
    current = _task_env_overrides.get(key) or {}
    if 'workspace_key' not in current:
        return
    _task_env_overrides.pop(key, None)
    if get_session_cwd(task_id) == current.get('cwd'):
        clear_session_cwd(task_id)


def release_session_workspaces(authority, session_ids):
    """Session teardown: drop each session's workspace binding under its owning profile's scope (the
    key the turn bound it under), so the override table does not grow with every closed session."""
    from gateway.session_authorities import owner_scope
    with owner_scope(authority):
        for sid in session_ids:
            unbind_turn_workspace(sid)


@contextmanager
def profile_read_scope(policy, authority):
    """Hold ONE read of a following session's profile for every resolution inside, as a turn's
    ``policy_scope`` does, without that scope's workspace and terminal setup (a session control
    reads the route, it runs nothing)."""
    token = _TURN_PROFILE.set((policy, policy.read_profile(authority)))
    try:
        yield
    finally:
        _TURN_PROFILE.reset(token)


def active_policy():
    """The frozen launch policy of the turn executing in this context, or ``None`` (standalone
    serve / messaging turns). Consumers read it instead of re-deriving from live config."""
    return _ACTIVE_POLICY.get()
