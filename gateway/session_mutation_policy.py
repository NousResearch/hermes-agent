"""Prepare the policy-only session edits: session-scope reasoning and the workspace (Move to project).

Both rewrite the frozen LocalSessionPolicy the next turn runs under, re-bound like a model switch
(policy identity covers config/cwd, so its config-secret reference is re-fingerprinted).
"""
import asyncio
import json
from dataclasses import asdict, replace

from persistence.core.errors import AuthorityRefusal


def _rebind(authority, prepared, old, policy, **extra):
    from gateway.session_policy import bind_launch_key, launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    policy = bind_launch_key(authority, prepared['snapshot']['receipt']['session_id'], policy,
                             launch_key(authority, old), config_secrets=secrets)
    return dict(prepared, policy=asdict(policy), **extra)


async def prepare_reasoning(authority, live, payload, prepared):
    """``/reasoning <level>`` for this session: the level ``build_policy`` freezes at creation. ``None``
    (``/reasoning reset``) gives the level back to the profile."""
    from gateway.session_policy import pin_reasoning_effort, profile_config, restore_policy, unpin_reasoning
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    frozen = old.config()
    if payload['effort'] is not None:
        pin_reasoning_effort(frozen, payload['effort'], session=True)
    else:
        # A bypass launch never reads the profile: its reset is the built-in default.
        profile = {} if old.ignore_user_config else await asyncio.to_thread(profile_config, authority)
        unpin_reasoning(frozen, profile)
    return _rebind(authority, prepared, old, replace(old, config_json=json.dumps(frozen)))


async def prepare_workspace(authority, live, payload, prepared):
    """Move to project: execution cwd and TERMINAL_CWD move together (restore_policy's invariant)."""
    import os
    from gateway.session_policy import restore_policy
    from hermes_constants import translate_cwd_for_wsl_backend
    from tui_gateway import git_probe
    old = restore_policy(prepared['snapshot']['receipt']['policy'])

    def probe():
        # Off-loop: a stat and up to three `git` spawns.
        cwd = os.path.abspath(os.path.expanduser(translate_cwd_for_wsl_backend(payload['cwd'])))
        if not os.path.isdir(cwd):
            raise AuthorityRefusal('invalid_params')
        return cwd, git_probe.branch(cwd), git_probe.common_repo_root(cwd)
    cwd, branch, root = await asyncio.to_thread(probe)
    terminal = json.loads(old.terminal_json)
    terminal['TERMINAL_CWD'] = cwd
    policy = replace(old, cwd=cwd, terminal_json=json.dumps(terminal))
    return _rebind(authority, prepared, old, policy, git={'branch': branch, 'root': root})


async def apply_pending_workspace(authority, ref, live):
    """Commit the folder a ``desktop_project`` call moved the chat to while its turn ran, as the user's
    Move to project (``workspace``) on the owning principal's behalf. The drain calls this at the turn
    boundary, under the claim gate and mutation lock, beside ``apply_pending_reasoning``."""
    import logging
    import uuid
    from gateway.session_mutations import write_mutation
    from gateway.session_persistence import authority_persistence
    from persistence.core.errors import MUTATION_GUARD_REFUSALS
    cwd, live.pending_workspace = live.pending_workspace, None
    if cwd is None:
        return
    store = authority_persistence(authority)
    try:
        owner = await asyncio.to_thread(store.bindings.receipt, ref.session_id)
        row = await asyncio.to_thread(store.sessions.get, ref.session_id)
        params = {'session_id': ref.session_id, 'request_id': 'project-tool-move:' + uuid.uuid4().hex,
                  'expected_revision': row.runtime_revision, 'expected_generation': row.runtime_generation,
                  'operation': 'workspace', 'payload': {'cwd': cwd}}
        await write_mutation(authority, ref, params, None, principal_id=owner.principal_id)
    except (AuthorityRefusal, *MUTATION_GUARD_REFUSALS) as exc:
        if getattr(exc, 'reason', 'session_busy') in ('session_busy', 'unknown_execution'):
            live.pending_workspace = live.pending_workspace or cwd  # a worker still runs: the next boundary
        else:
            logging.getLogger(__name__).warning('Project move of %s not committed: %s', ref.session_id, exc.reason)
    except Exception:  # health: allow BLE001 -- before the claim: escaping would kill or wedge the drain
        logging.getLogger(__name__).exception('Project move of %s failed to commit', ref.session_id)


async def prepare_edit_policy(authority, live, payload, prepared):
    """Main's ACP modes (Default / Accept Edits / Don't Ask) as the session's frozen edit approval
    policy. Only an ACP session's edits ask an editor; selecting the current policy writes nothing."""
    from gateway.session_policy import restore_policy
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    if old.source != 'acp':
        raise AuthorityRefusal('invalid_params', field='payload.policy',
                               detail='only an ACP session has an editor edit approval policy')
    if old.edit_approval_policy == payload['policy']:
        return {'status': 'noop', 'edit_approval_policy': old.edit_approval_policy}
    return _rebind(authority, prepared, old, replace(old, edit_approval_policy=payload['policy']))


PREPARE = {'reasoning': prepare_reasoning, 'workspace': prepare_workspace, 'edit_policy': prepare_edit_policy}
