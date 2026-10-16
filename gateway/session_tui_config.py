"""Owner ``config.set`` for the TUI's profile preferences.

Main served these from the per-TUI sidecar (``tui_gateway/methods_config_set.py``); on the shared
owner they write the OWNING profile's config.yaml (raw round-trip, off-loop, under its runtime
scope) with main's per-key semantics. The session controls (busy/verbose/yolo/show_reasoning/
service_tier, the refused ``model``) stay in ``gateway.session_busy_controls``; any other key is
``invalid_params``.
"""
import asyncio
import copy
from functools import partial
from pathlib import Path
import threading

from hermes_state_runtime import RuntimeStoreError

_SESSION_CONTROL_KEYS = frozenset({'busy', 'verbose', 'yolo', 'show_reasoning', 'service_tier', 'model'})
_DETAIL_SECTIONS = ('thinking', 'tools', 'subagents', 'activity')
_DETAIL_MODES = frozenset({'hidden', 'collapsed', 'expanded'})
_STATUSBAR_MODES = ('off', 'top', 'bottom')
_BOOL_WORDS = {'on': True, 'true': True, 'yes': True, 'off': False, 'false': False, 'no': False}
_WRITE_LOCK = threading.Lock()  # one read-modify-write of config.yaml at a time


def handlers(connection):
    return {'config.set': partial(config_set, connection)}


def _word(value):
    return ('' if value is None else str(value)).strip().lower()


def _statusbar(raw):
    return 'off' if raw is False else raw.strip().lower() if (
        isinstance(raw, str) and raw.strip().lower() in _STATUSBAR_MODES) else 'top'


def _toggles():
    """key -> (display field, word -> value, flipped(display), report(value)); '' / toggle flips."""
    from tui_gateway.agent_factory import _MOUSE_TRACKING_ALIASES, _display_mouse_tracking

    def on_off(value):
        return 'on' if value else 'off'

    return {
        'density': ('tui_compact', {'on': True, 'off': False},
                    lambda d: not bool(d.get('tui_compact', False)), on_off),
        'battery': ('battery', _BOOL_WORDS, lambda d: not bool(d.get('battery', False)), on_off),
        'statusbar': ('tui_statusbar', {'on': 'top', **{m: m for m in _STATUSBAR_MODES}},
                      lambda d: 'top' if _statusbar(d.get('tui_statusbar', 'top')) == 'off' else 'off', str),
        'mouse': ('mouse_tracking', _MOUSE_TRACKING_ALIASES,
                  lambda d: 'all' if _display_mouse_tracking(d) == 'off' else 'off', str)}


def _words():
    """key -> (display field, accepted words)."""
    from hermes_constants import INDICATOR_STYLES
    return {'theme': ('tui_theme', frozenset({'auto', 'light', 'dark'})),
            'indicator': ('tui_status_indicator', frozenset(INDICATOR_STYLES))}


def _display(cfg):
    if not isinstance(cfg.get('display'), dict):
        cfg['display'] = {}
    return cfg['display']


def _sections(display):
    if not isinstance(display.get('sections'), dict):
        display['sections'] = {}
    return display['sections']


def _set_display(cfg, key, value):
    """One display preference on the raw config, main's semantics; returns the result extras."""
    display, raw = _display(cfg), _word(value)
    if key in (toggles := _toggles()):
        field, aliases, flipped, report = toggles[key]
        new = flipped(display) if raw in ('', 'toggle') else aliases.get(raw)
        if new is None:
            raise RuntimeStoreError('invalid_params')
        display[field] = new
        return {'value': report(new)}
    if key in (words := _words()):
        field, allowed = words[key]
        if raw not in allowed:
            raise RuntimeStoreError('invalid_params')
        display[field] = raw
        return {'value': raw}
    if key == 'details_mode':
        if raw not in _DETAIL_MODES:
            raise RuntimeStoreError('invalid_params')
        display['details_mode'] = raw
        _sections(display).update(dict.fromkeys(_DETAIL_SECTIONS, raw))
        return {'value': raw}
    if key.startswith('details_mode.'):
        section = key.split('.', 1)[1]
        if section not in _DETAIL_SECTIONS or (raw and raw not in _DETAIL_MODES):
            raise RuntimeStoreError('invalid_params')
        if raw:
            _sections(display)[section] = raw
        else:  # empty clears the override (the client falls back to its section default)
            _sections(display).pop(section, None)
        return {'value': raw}
    if key == 'focus':
        return _set_focus(display, raw)
    if key == 'skin':
        if not isinstance(value, str) or not value.strip():
            raise RuntimeStoreError('invalid_params')
        display['skin'] = value.strip()
        return {'value': value.strip()}
    raise RuntimeStoreError('invalid_params')


def _set_focus(display, raw):
    """/focus on stashes the configured tool-progress mode and pins it off; off restores it."""
    from hermes_cli.focus_view import FOCUS_TOOL_PROGRESS_MODE, normalize_tool_progress_mode, resolve_focus_arg
    current = bool(display.get('focus_view', False))
    configured = normalize_tool_progress_mode(display.get('tool_progress', 'all'))
    action, target = resolve_focus_arg(raw, current)
    if action == 'usage':
        raise RuntimeStoreError('invalid_params')
    if action == 'status' or target is None:
        return {'value': 'on' if current else 'off', 'tool_progress': configured}
    if target:
        display['focus_saved_tool_progress'] = normalize_tool_progress_mode(
            (current and display.get('focus_saved_tool_progress')) or configured)
        effective = FOCUS_TOOL_PROGRESS_MODE
    else:
        effective = normalize_tool_progress_mode(display.get('focus_saved_tool_progress') or 'all')
    display['tool_progress'], display['focus_view'] = effective, bool(target)
    return {'value': 'on' if target else 'off', 'tool_progress': effective}


def _write_profile_config(connection, edit):
    """``edit(raw_cfg)`` on the owning profile's config.yaml EXACTLY as written (no defaults or
    overlay persisted); saved only when it changed. Runs off the loop."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import atomic_config_replace, get_config_path, read_user_config_raw
    with _profile_runtime_scope(Path(connection.authority.profile_id)), _WRITE_LOCK:
        path = get_config_path()
        try:
            cfg = read_user_config_raw(path)
        except Exception as exc:  # unparseable: never overwrite the user's file
            raise RuntimeStoreError('storage_unavailable') from exc
        before = copy.deepcopy(cfg)
        result = edit(cfg)
        if cfg != before:
            try:
                atomic_config_replace(path, cfg)
            except Exception as exc:
                raise RuntimeStoreError('storage_unavailable') from exc
        return result


def _authorize_write(connection, ref, params):
    """Profile-level like ``config.get`` (a session is optional), plus the control capability."""
    from gateway.session_config import _authorize
    policy = _authorize(connection, ref, params)
    if 'session:control' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    return policy


# Profile settings main's sidecar wrote for Desktop's toggles (``tui_gateway/server.py::_DISPLAY_TOGGLE_KEYS``)
# and its voice-engine pick: config.yaml leaves with main's validation.
_DISPLAY_TOGGLE_KEYS = frozenset({'display.message_reactions', 'display.in_app_tips', 'display.in_app_tours'})
_VOICE_CHAT_MODES = frozenset({'chained', 'gpt-live'})


async def config_set(connection, ref, params):
    key = params.get('key')
    if key == 'reasoning' or (key == 'yolo' and 'scope' in params):
        return await _set_command_setting(connection, ref, params)
    if key in _SESSION_CONTROL_KEYS or not isinstance(key, str):
        from gateway.session_busy_controls import busy_config
        return await busy_config(connection, ref, params, write=True)
    if set(params) - {'session_id', 'profile', 'key', 'value'}:
        raise RuntimeStoreError('invalid_params')
    policy = _authorize_write(connection, ref, params)
    value = params.get('value')
    if key == 'fast':
        return await _set_fast(connection, ref, policy, value)
    if key == 'personality':
        return await asyncio.to_thread(_set_personality, connection, value)
    if key in ('approvals.mode', 'approval_mode'):
        return await asyncio.to_thread(_set_approval_mode, connection, value)
    if key in _DISPLAY_TOGGLE_KEYS or key == 'voice.voice_chat_mode':
        return {'key': key, **await asyncio.to_thread(_write_profile_config, connection,
                                                      lambda cfg: _set_profile_leaf(cfg, key, value))}
    result = await asyncio.to_thread(_write_profile_config, connection,
                                     lambda cfg: _set_display(cfg, key, value))
    if key == 'focus' and ref.session_id and 'tool_progress' in result:
        # Main's session also took the focus mode at once: this session's /verbose state.
        connection.authority.sessions[ref.session_id].tool_progress_mode = result['tool_progress']
    if key == 'skin':
        await _broadcast_skin(connection)
    return {'key': key, **result}


async def _broadcast_skin(connection):
    """Main repainted every surface of the profile: ``skin.changed`` on each attached session."""
    def resolve():
        import sys
        from gateway.run import _profile_runtime_scope
        from hermes_cli.config import load_config
        from tui_gateway.change_watcher import skin_payload
        with _profile_runtime_scope(Path(connection.authority.profile_id)):
            payload = skin_payload(load_config())
        if (server := sys.modules.get('tui_gateway.server')) is not None:
            server._note_skin_broadcast()  # as main: the config watcher must not repaint it again
        return payload
    payload = await asyncio.to_thread(resolve)
    for session_id, live in list(connection.authority.sessions.items()):
        if live.subscribers:
            live.event_stream.publish(session_id, payload, event_type='skin.changed')


def current_tier(connection, ref, policy):
    """The tier this session's next turn runs: its /fast pin, else its policy's (or the profile's)."""
    runner, live = connection.authority.runner, connection.authority.sessions[ref.session_id]
    if policy is not None:
        return policy.current_service_tier(runner, live.route)
    return runner._resolve_session_service_tier(source=live.source, session_key=live.route)


async def _set_fast(connection, ref, policy, value):
    """Session-scoped /fast like main's: validated against the session's route, never config.yaml.
    The words parse through the shared tier table (``parse_exact_service_tier``, as the classic CLI's
    ``config.set service_tier`` does); the pin is the same per-session override the next turn reads
    (``current_service_tier``). TUI-only on top: ``toggle`` / ``status`` and the route check."""
    from agent.fast_mode import STATIC_TIERS, parse_exact_service_tier, service_tier_word
    if not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    raw, current = _word(value), current_tier(connection, ref, policy)
    if raw == 'status':
        return {'key': 'fast', 'value': service_tier_word(current)}
    if raw in ('', 'toggle'):
        raw = 'normal' if current in STATIC_TIERS else 'fast'
    try:
        tier = parse_exact_service_tier(raw) or None
    except ValueError:
        raise RuntimeStoreError('invalid_params') from None
    if tier in STATIC_TIERS:
        agent = connection.authority.agent(ref)
        model = getattr(agent, 'model', None) or (policy.current_model(connection.authority) if policy else None)
        provider = getattr(agent, 'provider', None) or (policy.provider if policy else None)
        base_url = getattr(agent, 'base_url', None) or (policy.base_url if policy else None)

        def offered():
            from agent.codex_service_tiers import ultrafast_offered
            from hermes_cli.models import resolve_fast_mode_overrides
            if not model or resolve_fast_mode_overrides(model, provider=provider, base_url=base_url,
                                                        tier=tier if tier == 'ultrafast' else None) is None:
                return False
            return tier != 'ultrafast' or ultrafast_offered(model, provider=provider, base_url=base_url,
                                                            access_token=getattr(agent, 'api_key', None))
        if not await asyncio.to_thread(offered):
            raise RuntimeStoreError('fast_unavailable')
    connection.authority.runner._set_session_service_tier_override(
        connection.authority.sessions[ref.session_id].route, tier)
    return {'key': 'fast', 'value': service_tier_word(tier)}


def _set_personality(connection, value):
    """Main's /personality: validated against the profile's personalities and persisted through
    ``hermes_cli.personality`` (the only sanctioned write). Every next turn of the profile's sessions
    resolves its ephemeral prompt from that config, and the changed prompt rebuilds the cached agent."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import load_config
    from hermes_cli.personality import persist_personality, resolve_personality
    with _profile_runtime_scope(Path(connection.authority.profile_id)), _WRITE_LOCK:
        try:
            name, _prompt = resolve_personality(str(value or ''), load_config())
        except ValueError as exc:
            raise RuntimeStoreError('unknown_personality') from exc
        if not persist_personality(name):
            raise RuntimeStoreError('storage_unavailable')
    return {'key': 'personality', 'value': str(value or 'none'), 'history_reset': False}


def _set_profile_leaf(cfg, key, value):
    section, leaf = key.split('.', 1)
    if key == 'voice.voice_chat_mode':
        new = _word(value)
        if new not in _VOICE_CHAT_MODES:
            raise RuntimeStoreError('invalid_params')
    else:
        new = {**_BOOL_WORDS, '1': True, '0': False}.get(_word(value))
        if new is None:
            raise RuntimeStoreError('invalid_params')
    if not isinstance(cfg.get(section), dict):
        cfg[section] = {}
    cfg[section][leaf] = new
    return {'value': new}


def _set_approval_mode(connection, value):
    """``/approvals <mode>``'s own write (``hermes_cli.approval_mode``) for this profile."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.approval_mode import run_approval_mode_command
    with _profile_runtime_scope(Path(connection.authority.profile_id)), _WRITE_LOCK:
        result = run_approval_mode_command(_word(value) or None)
    if not result.ok:
        raise RuntimeStoreError('invalid_params' if _word(value) not in ('manual', 'smart', 'off')
                                else 'storage_unavailable')
    return {'key': 'approvals.mode', 'value': result.mode}


async def _set_command_setting(connection, ref, params):
    """``config.set`` for a setting a slash command owns: the same handler code, so the picker, the
    keybind and the typed command agree. ``reasoning`` is ``/reasoning`` (``scope=global`` =
    ``--global``; show/hide/reset/levels), ``yolo`` with ``scope=global`` is main's Shift+click zap
    (``approvals.mode`` off / manual)."""
    if (set(params) - {'session_id', 'profile', 'key', 'value', 'scope'} or not ref.session_id
            or params.get('scope', 'session') not in ('session', 'global')):
        raise RuntimeStoreError('invalid_params')
    _authorize_write(connection, ref, params)
    connection.authority.authorize(connection.actor, ref, 'session:control')
    value, scope = _word(params.get('value')), params.get('scope', 'session')
    if params['key'] == 'yolo':
        if scope != 'global':
            raise RuntimeStoreError('invalid_params')
        on = {**_BOOL_WORDS, '1': True, '0': False}.get(value)
        if on is None and value not in ('', 'toggle'):
            raise RuntimeStoreError('invalid_params')

        def flip():
            from tools.approval_context import _get_approval_mode
            from gateway.run import _profile_runtime_scope
            with _profile_runtime_scope(Path(connection.authority.profile_id)):
                enable = (_get_approval_mode() != 'off') if on is None else on
            return enable, _set_approval_mode(connection, 'off' if enable else 'manual')
        enable, _ = await asyncio.to_thread(flip)
        return {'key': 'yolo', 'value': '1' if enable else '0', 'scope': 'global'}
    if not value:
        raise RuntimeStoreError('invalid_params')
    from gateway.run import _profile_runtime_scope
    from hermes_constants import parse_reasoning_effort
    from gateway.slash_commands_model import _REASONING_DISPLAY_TOGGLES
    if value not in _REASONING_DISPLAY_TOGGLES and value != 'reset' and parse_reasoning_effort(value) is None:
        raise RuntimeStoreError('invalid_params')
    live, runner = connection.authority.sessions[ref.session_id], connection.authority.runner

    def apply():
        with _profile_runtime_scope(Path(connection.authority.profile_id)):
            return runner._apply_reasoning_selection(live.route, 'cli', value, persist_global=scope == 'global')
    await asyncio.to_thread(apply)
    show = _REASONING_DISPLAY_TOGGLES.get(value)
    reported = ('show' if show else 'hide') if show is not None else value
    return {'key': 'reasoning', 'value': reported, 'scope': scope}
