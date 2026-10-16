"""Prepare policy-only session reasoning, service tier and workspace edits.

These rewrite the frozen LocalSessionPolicy the next turn runs under, re-bound like a model switch
(policy identity covers config/cwd, so its config-secret reference is re-fingerprinted).
"""
import asyncio
import json
from dataclasses import asdict, replace

from hermes_state_runtime import RuntimeStoreError


def _rebind(authority, prepared, old, policy, **extra):
    from gateway.session_policy import bind_launch_key, launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    policy = bind_launch_key(authority, prepared['snapshot']['receipt']['session_id'], policy,
                             launch_key(authority, old), config_secrets=secrets)
    return dict(prepared, policy=asdict(policy), **extra)


async def prepare_reasoning(authority, live, payload, prepared):
    """``/reasoning <level>`` for this session: the level ``build_policy`` freezes at creation."""
    from gateway.session_policy import restore_policy
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    frozen = old.config()
    frozen.setdefault('agent', {})['reasoning_effort'] = payload['effort']
    # An explicit session level wins over a per-model default, as at creation.
    frozen['agent'].pop('reasoning_overrides', None)
    return _rebind(authority, prepared, old, replace(old, config_json=json.dumps(frozen), reasoning_pinned=True))


async def prepare_service_tier(authority, live, payload, prepared):
    """Validate an explicit paid tier against the next turn's route before freezing it."""
    from agent.fast_mode import CODEX_TIER_WORDS, STATIC_TIERS, parse_exact_service_tier
    from gateway.session_policy import restore_policy
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    tier = parse_exact_service_tier(payload['service_tier']) or None
    if tier is not None:
        from gateway.session_authorities import owner_scope

        def offered():
            from agent.codex_service_tiers import ultrafast_offered
            from hermes_cli.models import resolve_fast_mode_overrides
            # Runtime resolution and entitlement reads belong to the owning profile, off-loop.
            with owner_scope(authority):
                model, runtime = authority.runner._resolve_session_agent_runtime(
                    source=live.source, session_key=live.route)
                if (runtime.get('api_mode') == 'codex_app_server'
                        and tier in STATIC_TIERS and tier not in CODEX_TIER_WORDS):
                    return False
                route = {'provider': runtime.get('provider'), 'base_url': runtime.get('base_url')}
                if not model or resolve_fast_mode_overrides(model, **route, tier=tier) is None:
                    return False
                return tier != 'ultrafast' or ultrafast_offered(
                    model, **route, access_token=runtime.get('api_key'))
        if not await asyncio.to_thread(offered):
            raise RuntimeStoreError('fast_unavailable')
    frozen = old.config()
    frozen.setdefault('agent', {})['service_tier'] = tier or 'normal'
    policy = replace(old, config_json=json.dumps(frozen), service_tier_pinned=True)
    return _rebind(authority, prepared, old, policy)


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
            raise RuntimeStoreError('invalid_params')
        return cwd, git_probe.branch(cwd), git_probe.common_repo_root(cwd)
    cwd, branch, root = await asyncio.to_thread(probe)
    terminal = json.loads(old.terminal_json)
    terminal['TERMINAL_CWD'] = cwd
    policy = replace(old, cwd=cwd, terminal_json=json.dumps(terminal))
    return _rebind(authority, prepared, old, policy, git={'branch': branch, 'root': root})


PREPARE = {'reasoning': prepare_reasoning, 'service_tier': prepare_service_tier, 'workspace': prepare_workspace}
