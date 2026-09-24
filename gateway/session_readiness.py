"""Profile-bound setup checks without legacy session execution dispatch."""
import asyncio
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


async def readiness(authority, actor, params, *, runtime):
    if 'session:create' not in actor.capabilities or actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('permission_denied')
    allowed = {'profile', 'provider'} if runtime else {'profile'}
    if (not isinstance(params, dict) or set(params) - allowed
            or any(not isinstance(value, str) for value in params.values())):
        raise RuntimeStoreError('invalid_params')
    if params.get('profile'):
        from fastapi import HTTPException
        from hermes_cli.web_server_profiles import _resolve_profile_dir
        try:
            home = _resolve_profile_dir(params['profile']).resolve()
        except HTTPException:
            raise RuntimeStoreError('profile_mismatch') from None
        if home != Path(authority.profile_id):
            raise RuntimeStoreError('profile_mismatch')

    def probe():
        from gateway.run import _profile_runtime_scope
        from hermes_cli.main import _has_any_provider_configured
        from hermes_cli.runtime_readiness import check_runtime_readiness
        with _profile_runtime_scope(Path(authority.profile_id)):
            try:
                if runtime:
                    return check_runtime_readiness(params.get('provider') or None, strict_profile_scope=True)
                return {'provider_configured': bool(_has_any_provider_configured(strict_profile_scope=True))}
            except Exception as exc:
                return {'ok': False, 'error': str(exc)}

    return await asyncio.to_thread(probe)
