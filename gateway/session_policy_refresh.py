"""A resumed local session re-reads its config-derived settings (CFG-2); its MCP servers change the same way (CFG-3).

Main's ``hermes -c`` was a new process: toolsets, MCP servers, ``terminal.backend``, ``max_turns``, checkpoints
and the reasoning default came from the config it started with, while the stored session kept its model/provider.
The owner keeps that line on ``session.resume``: it re-derives exactly those fields from the session profile's
current config and commits them as a new policy generation. ``reload.mcp`` commits the MCP part alone for the
calling session. The creation flags (request_json), the model route and a committed ``/reasoning`` or workspace
pick stay; a session with a turn running or queued keeps the policy that turn was admitted under.
"""
import asyncio
import json
import logging
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

logger = logging.getLogger(__name__)


def _follow(target, live, key):
    if key in live:
        target[key] = live[key]
    else:
        target.pop(key, None)


def refreshed_policy(policy, live, terminal, *, picked=(), mcp_only=False):
    """``(policy, secrets, replaced)``: *policy* with its config-derived fields from *live* (the profile's
    config) and *terminal* (its terminal scope), or only its MCP servers (*mcp_only*); explicit picks (launch
    flags, the *picked* policy edits) stay. *secrets* are the refreshed sections' private values, *replaced*
    the sections (``None``: terminal) whose stored private values they supersede. *policy* itself when
    nothing moved; a rotated private value alone (its redacted tree unchanged) moves it."""
    from gateway.session_policy import _extract_config_secrets, present_sections
    from gateway.session_policy_credentials import secrets_moved
    old, config = json.loads(policy.config_json), json.loads(policy.config_json)
    live, secrets, replaced = present_sections(live), {}, set()
    if mcp_only:
        toolsets, terminal = policy.toolsets, json.loads(policy.terminal_json)
    else:
        toolsets = _settings(policy, config, live, picked)
        terminal = _terminal(policy, terminal, picked)
        _extract_config_secrets(terminal, secrets, (None,))
        replaced.add(None)
    toolsets = _mcp_servers(policy, config, live, toolsets, secrets, replaced)
    if (config == old and toolsets == policy.toolsets and terminal == json.loads(policy.terminal_json)
            and not secrets_moved(policy.config_secret_ref, secrets, replaced)):
        return policy, {}, set()
    policy = replace(policy, toolsets=toolsets, config_json=json.dumps(config), terminal_json=json.dumps(terminal))
    return policy, secrets, replaced


def _settings(policy, config, live, picked):
    """Follow the config-derived settings into *config*; the toolsets they select."""
    from gateway.session_policy import _resolve_toolsets, session_pinned
    request = json.loads(policy.request_json)
    live_agent = live.get('agent') or {}
    agent = dict(config.get('agent') or {})
    keys = [] if 'max_turns' in request else ['max_turns']
    # The session's own record (a pick, or a reset back to the profile) wins over the receipt history.
    chosen = session_pinned(config, 'reasoning', 'reasoning' in request or 'reasoning' in picked)
    keys += [] if chosen else ['reasoning_effort', 'reasoning_overrides']
    toolsets = policy.toolsets
    if 'toolsets' not in request and policy.editor_mcp_json is None:
        keys.append('disabled_toolsets')
        _follow(config, live, 'platform_toolsets')
        toolsets = tuple(sorted(_resolve_toolsets({}, live, policy.source, policy.safe_mode, policy.coding_posture)))
    for key in keys:
        _follow(agent, live_agent, key)
    if agent or 'agent' in config:
        config['agent'] = agent
    if 'max_turns' not in request:
        _follow(config, live, 'max_turns')
    _follow(config, live, 'checkpoints')
    if request.get('checkpoints') is True:
        section = config.get('checkpoints')
        section = dict(section) if isinstance(section, dict) else {'enabled': bool(section)} if section else {}
        config['checkpoints'] = {**section, 'enabled': True}
    return toolsets


def _terminal(policy, terminal, picked):
    from gateway.session_policy import terminal_uses_host_cwd
    terminal, request = dict(terminal), json.loads(policy.request_json)
    # The session's own folder, where it now is, under build_policy's rule; a committed Move to project is
    # the user's pick on every backend (it rewrote the policy, never the launch request).
    if request.get('cwd_explicit') is True or 'workspace' in picked or terminal_uses_host_cwd(terminal):
        terminal['TERMINAL_CWD'] = policy.cwd
    return terminal


def _mcp_servers(policy, config, live, toolsets, secrets, replaced):
    """The profile's MCP servers as this session's own selection takes them now (its explicit pick or platform
    allowlist, ``no_mcp`` and disabled entries respected): their definitions into *config*, private values split
    off as build_policy does (a managed worker binds this config, so it connects exactly these), and their
    names into the returned toolsets. An editor-MCP session keeps its frozen servers."""
    from gateway.session_policy import _extract_config_secrets, _resolve_toolsets
    from hermes_cli.tools_config import enabled_mcp_server_names
    from hermes_state_runtime import RuntimeStoreError
    if policy.editor_mcp_json is not None:
        return toolsets
    request = json.loads(policy.request_json)
    sections = [section if isinstance(section, dict) else {} for section in (config.get('mcp_servers'), live.get('mcp_servers'))]
    pick, picked, enabled = request.get('toolsets'), set(), enabled_mcp_server_names(live)
    if pick is not None:
        # The launch pick's servers are an allowlist: each stays only while config enables it, and once all
        # are gone the chat has none, never every enabled server. A server that is now disabled or removed
        # leaves the pick before it is re-validated (it no longer validates, and keeping the frozen
        # definitions would start the server the user turned off).
        known = set(sections[0]) | set(sections[1])
        server = partial(_server_name, known)
        picked = {server(name) for name in pick if isinstance(name, str) and server(name) in known}
        pick = [name for name in pick if not isinstance(name, str) or server(name) not in picked - enabled]
    try:
        selected = _resolve_toolsets({} if pick is None else {'toolsets': pick},
                                     json.loads(json.dumps(live)), policy.source, policy.safe_mode, policy.coding_posture)
    except RuntimeStoreError:
        return toolsets  # the rest of the launch pick no longer validates against the profile: keep the frozen servers
    # A picked server config removed keeps a disabled stub in the stored section, so every later refresh
    # (after an owner restart too) still knows that name as one of the pick's servers, not an unknown toolset.
    stubs = {name: {'enabled': False} for name in picked - set(sections[1])}
    if 'mcp_servers' in live or stubs:
        config['mcp_servers'] = json.loads(json.dumps({**sections[1], **stubs} if stubs else live['mcp_servers']))
        _extract_config_secrets(config['mcp_servers'], secrets, ('mcp_servers',))
    else:
        config.pop('mcp_servers', None)
    replaced.add('mcp_servers')
    servers = selected & enabled & picked if picked else selected & enabled
    return tuple(sorted((set(toolsets) - set(sections[0]) - set(sections[1])) | servers))


def _server_name(servers, name):
    """The server a pick name selects: ``mcp-<server>`` is its bare alias unless a server has that name."""
    return name[4:] if name.startswith('mcp-') and name not in servers else name


def _prepare(authority, session_id, mcp_only=False):
    """Off the owner loop: the stored policy, its refresh from this profile home's files, and the generation
    it was derived at. None when the session takes no refresh or nothing moved."""
    from gateway.run import _load_gateway_config
    from gateway.session_policy import SURFACES, restore_policy
    from hermes_state_local import local_receipt
    from hermes_state_mutation_prepared import policy_picks
    from tools.terminal_scope import build_profile_terminal_scope
    generation = authority.db.get_session(session_id)['runtime_generation']
    stored = local_receipt(authority.db, session_id)['policy']
    old = restore_policy(stored)
    # A bypass launch never reads the profile; cron, Kanban and room routes are not resumed CLI sessions.
    if old.ignore_user_config or old.source not in SURFACES:
        return None
    home = Path(authority.db.db_path).resolve().parent
    terminal = {} if mcp_only else build_profile_terminal_scope(home)
    policy, secrets, replaced = refreshed_policy(old, _load_gateway_config(home / 'config.yaml'), terminal,
                                                 picked=policy_picks(authority.db, session_id), mcp_only=mcp_only)
    return None if policy is old else (stored, old, policy, secrets, replaced, generation)


def _rebind(authority, session_id, old, policy, fresh, replaced):
    """Re-bind the private values like a policy mutation: policy identity covers config and terminal, and the
    refreshed sections' private values are the profile's current ones (the stored ones there are neither
    re-read nor checked: a rotated key is what the refresh picks up, not a revoked credential)."""
    from gateway.session_policy import FOLLOW_SECTIONS, bind_launch_key, launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    skip = (*(FOLLOW_SECTIONS if old.follows_profile else ()), *replaced)
    secrets = recover_config_secrets(authority, old, skip) if old.config_secret_ref else {}
    secrets.update(fresh)
    return bind_launch_key(authority, session_id, policy, launch_key(authority, old), config_secrets=secrets)


def _in_flight(authority):
    found = getattr(authority, '_policy_refreshes', None)
    if found is None:
        found = authority._policy_refreshes = {}
    return found


async def refresh_settled(authority, session_id):
    """Wait out a refresh of *session_id* in flight. Its commit lands before its terminal cleanup (a sandbox
    teardown can take 10-15 s) and its publication; an attach meanwhile would find the stored policy ahead
    of the live one, which restore_local_session refuses."""
    refreshes = _in_flight(authority)
    while (running := refreshes.get(session_id)) is not None and not running.done():
        logger.debug('Session %s waits for its policy refresh to publish', session_id)
        await asyncio.wait({running})  # its outcome is its caller's; a cancelled waiter leaves it running


async def commit_refresh(authority, ref, *, mcp_only=False):
    """Commit an idle local session's refresh as its next policy generation: True when one was committed,
    False when nothing moved. Raises the store's refusal (``session_busy`` while a turn runs or waits).
    One refresh per session at a time, run to its publication even when the caller is cancelled: a second
    waits for the first, then usually finds nothing moved."""
    from gateway.session_runtime_workers import track_mutation
    await refresh_settled(authority, ref.session_id)
    refreshes = _in_flight(authority)
    task = refreshes[ref.session_id] = track_mutation(authority, _commit_refresh(authority, ref, mcp_only))
    task.add_done_callback(lambda done: refreshes.pop(ref.session_id, None) if refreshes.get(ref.session_id)
                           is done else None)
    return await asyncio.shield(task)


async def _commit_refresh(authority, ref, mcp_only):
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_mutation_prepared import commit_refreshed_policy
    prepared = await asyncio.to_thread(_prepare, authority, ref.session_id, mcp_only)
    if prepared is None:
        return False
    stored, old, policy, secrets, replaced, generation = prepared
    moved = json.loads(policy.terminal_json) != json.loads(old.terminal_json)
    policy = _rebind(authority, ref.session_id, old, policy, secrets, replaced)
    # Compare-and-swap against the receipt as stored (JSON lists), not its restored tuple form. The
    # publication is the commit's own shielded follow-up: a caller cancelled mid-commit cannot leave the
    # stored policy ahead of the live one (restore_local_session refuses that pair).
    await tracked_write(authority, partial(commit_refreshed_policy, authority.db, epoch=authority.epoch,
                                           session_id=ref.session_id, old=stored, new=asdict(policy),
                                           generation=generation),
                        then=lambda _generation: _publish(authority, ref, moved))
    return True


async def refresh_resumed_policy(authority, ref):
    """Commit the config refresh of an idle local session before it is attached. Never fails the resume:
    a busy session, a revoked launch credential or an unreadable profile keeps the stored policy. Only a turn
    running or queued (or admitted during the commit) is expected; anything else is a WARNING."""
    try:
        await commit_refresh(authority, ref)
    except Exception as exc:  # health: allow BLE001 -- a refresh is best-effort; the resume itself must proceed
        reason = getattr(exc, 'reason', None) or type(exc).__name__
        busy = reason in {'session_busy', 'unknown_execution', 'revision_conflict'}
        logger.log(logging.INFO if busy else logging.WARNING, 'Session %s keeps its stored policy on resume: %s',
                   ref.session_id, reason)


def _drop_terminal(authority, ref):
    """The session's own cached terminal environments (keyed by its route, or its ids under per-session
    isolation): the terminal tool reuses one by key whatever backend built it. A profile-shared one stays.
    Its recorded cwd (keyed by its route or task id) goes too: it wins over the new binding's directory."""
    from gateway.session_authorities import owner_scope
    from hermes_state_local import local_receipt
    from tools.terminal_tool import _qualify_task_key, clear_session_cwd
    from tools.terminal_tool_lifecycle import cleanup_vm
    ids = {ref.session_id, local_receipt(authority.db, ref.session_id)['entry']['session_id']}
    route = authority.sessions[ref.session_id].route
    with owner_scope(authority):
        keys = {_qualify_task_key(key) for key in (f'session:{route}', *ids)}
        for key in (route, *ids):
            clear_session_cwd(key)
    for key in keys:
        cleanup_vm(key)


async def _publish(authority, ref, moved=False):
    from gateway.session_local import local_session_info, publish_local_policy
    if moved:  # before the drain can run a turn on the new binding with the old environment
        await asyncio.to_thread(_drop_terminal, authority, ref)
    publish_local_policy(authority, ref.session_id)
    # A drain that ran during the off-loop commit saw the new stored policy beside the old live one and
    # paused; with both equal again, anything admitted meanwhile runs.
    authority._schedule(ref)
    # Every viewer already attached adopts the new generation (its next fenced edit would otherwise be
    # refused as stale) and repaints from the refreshed policy, as a reasoning/workspace edit publishes.
    authority._publish_pending(ref)
    authority.sessions[ref.session_id].event_stream.publish(
        ref.session_id, local_session_info(authority, ref), event_type='session.info')
