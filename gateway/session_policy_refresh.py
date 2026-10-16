"""A resumed local session re-reads its config-derived settings (CFG-2).

Main's ``hermes -c`` was a new process: toolsets, ``terminal.backend``, ``max_turns``, checkpoints and the
reasoning default came from the config it started with, while the stored session kept its model/provider.
The owner keeps that line on ``session.resume``: it re-derives exactly those fields from the session
profile's current config and commits them as a new policy generation. The creation flags (request_json),
the model route and a committed ``/reasoning`` pick stay; a session with a turn running or queued keeps the
policy that turn was admitted under.
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


def refreshed_policy(policy, live, terminal, *, picked=()):
    """``(policy, terminal_secrets)``: *policy* with its config-derived fields from *live* (the profile's
    config) and *terminal* (its terminal scope); explicit picks (launch flags, the *picked* policy edits)
    stay. *policy* itself when nothing moved."""
    from gateway.session_policy import (_extract_config_secrets, _resolve_toolsets, present_sections,
                                        terminal_uses_host_cwd)
    request = json.loads(policy.request_json)
    old, config = json.loads(policy.config_json), json.loads(policy.config_json)
    live = present_sections(live)
    live_agent = live.get('agent') or {}
    agent = dict(config.get('agent') or {})
    keys = [] if 'max_turns' in request else ['max_turns']
    keys += [] if 'reasoning' in request or 'reasoning' in picked else ['reasoning_effort', 'reasoning_overrides']
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
    terminal, secrets = dict(terminal), {}
    # The session's own folder, where it now is, under build_policy's rule; a committed Move to project is
    # the user's pick on every backend (it rewrote the policy, never the launch request).
    if request.get('cwd_explicit') is True or 'workspace' in picked or terminal_uses_host_cwd(terminal):
        terminal['TERMINAL_CWD'] = policy.cwd
    _extract_config_secrets(terminal, secrets, (None,))
    if config == old and toolsets == policy.toolsets and terminal == json.loads(policy.terminal_json):
        return policy, {}
    return replace(policy, toolsets=toolsets, config_json=json.dumps(config), terminal_json=json.dumps(terminal)), secrets


def _prepare(authority, session_id):
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
    policy, secrets = refreshed_policy(old, _load_gateway_config(home / 'config.yaml'),
                                       build_profile_terminal_scope(home),
                                       picked=policy_picks(authority.db, session_id))
    return None if policy == old else (stored, old, policy, secrets, generation)


def _rebind(authority, session_id, old, policy, terminal_secrets):
    """Re-bind the private values like a policy mutation: policy identity covers config and terminal, and
    the terminal's own secrets are the profile's current ones."""
    from gateway.session_policy import bind_launch_key, launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    secrets = {path: value for path, value in secrets.items() if path[0] is not None}
    secrets.update(terminal_secrets)
    return bind_launch_key(authority, session_id, policy, launch_key(authority, old), config_secrets=secrets)


async def refresh_resumed_policy(authority, ref):
    """Commit the config refresh of an idle local session before it is attached. Never fails the resume:
    a busy session, a revoked launch credential or an unreadable profile keeps the stored policy."""
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_mutation_prepared import commit_refreshed_policy
    try:
        prepared = await asyncio.to_thread(_prepare, authority, ref.session_id)
        if prepared is None:
            return
        stored, old, policy, secrets, generation = prepared
        policy = _rebind(authority, ref.session_id, old, policy, secrets)
        # Compare-and-swap against the receipt as stored (JSON lists), not its restored tuple form. The
        # publication is the commit's own shielded follow-up: a caller cancelled mid-commit cannot leave the
        # stored policy ahead of the live one (restore_local_session refuses that pair).
        await tracked_write(authority, partial(commit_refreshed_policy, authority.db, epoch=authority.epoch,
                                               session_id=ref.session_id, old=stored, new=asdict(policy),
                                               generation=generation),
                            then=lambda _generation: _publish(authority, ref))
    except Exception as exc:  # health: allow BLE001 -- a refresh is best-effort; the resume itself must proceed
        logger.info('Session %s keeps its stored policy on resume: %s', ref.session_id,
                    getattr(exc, 'reason', None) or type(exc).__name__)


def _publish(authority, ref):
    from gateway.session_local import local_session_info, publish_local_policy
    publish_local_policy(authority, ref.session_id)
    # A drain that ran during the off-loop commit saw the new stored policy beside the old live one and
    # paused; with both equal again, anything admitted meanwhile runs.
    authority._schedule(ref)
    # Every viewer already attached adopts the new generation (its next fenced edit would otherwise be
    # refused as stale) and repaints from the refreshed policy, as a reasoning/workspace edit publishes.
    authority._publish_pending(ref)
    authority.sessions[ref.session_id].event_stream.publish(
        ref.session_id, local_session_info(authority, ref), event_type='session.info')
