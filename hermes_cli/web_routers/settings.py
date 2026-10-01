"""Focused hosted settings; native config, auth, pairing and secret owners."""
import asyncio
import re
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, SecretStr

from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import destructive_profile, scoped_to_thread
from hermes_cli import web_settings as settings

def require_settings_token(request: Request):
    late('_require_token')(request)


router = APIRouter(prefix='/api/settings', dependencies=[Depends(require_settings_token)])


class ProfileUpdate(BaseModel):
    instructions: str | None = Field(default=None, max_length=32000)
    timezone: str | None = None


class MemoryUpdate(BaseModel):
    model: str
    learning: str
    recall: str


class Topic(BaseModel):
    id: str
    mode: Literal['inherit', 'all', 'silent']


class GroupUpdate(BaseModel):
    mode: Literal['mention', 'all'] = 'mention'
    instructions: str = Field(default='', max_length=16000)
    topics: list[Topic] = Field(default_factory=list)
    remove: bool = False


class PersonUpdate(BaseModel):
    action: Literal['add', 'remove', 'decline']
    user_id: str
    request_id: str = ''


class KeyUpdate(BaseModel):
    value: SecretStr


class AdminUpdate(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    current_password: SecretStr
    new_password: SecretStr


async def scoped(profile, fn):
    try:
        return await scoped_to_thread(profile, fn)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from None


@router.get('')
async def get_settings(profile: str | None = None):
    def read():
        from agent.secret_scope import get_secret
        from agent.reasoning_effort import codex_supported_efforts
        from hermes_cli.codex_models import get_codex_model_ids
        from hermes_cli.config import load_config
        from deploy.railway.hindsight_settings import status
        from hermes_cli.dashboard_auth import get_provider
        cfg = load_config()
        model = cfg.get('model') or {}
        if isinstance(model, str):
            model = {'default': model}
        from hermes_constants import resolve_reasoning_config
        reasoning = resolve_reasoning_config(cfg) or {}
        effort = 'none' if reasoning.get('enabled') is False else reasoning.get('effort', 'medium')
        provider = get_provider('basic')
        return {'profile': settings.profile_values(cfg),
                'chat': {'model': model.get('default', ''), 'effort': effort},
                'memory': settings.memory_values(cfg), 'memory_status': status(),
                'models': [{'id': m, 'efforts': list(codex_supported_efforts(m))} for m in get_codex_model_ids()],
                'access': settings.telegram_values(),
                'keys': {key: bool(get_secret(key, '')) for key in SERVICE_KEYS},
                'admin': {'username': getattr(provider, '_username', ''), 'available': provider is not None}}
    return await scoped(profile, read)


@router.get('/models')
async def models(profile: str | None = None, refresh: bool = False):
    def read():
        from hermes_cli.models import provider_model_ids
        from agent.reasoning_effort import codex_supported_efforts
        return [{'id': model, 'efforts': list(codex_supported_efforts(model))}
                for model in provider_model_ids('openai-codex', force_refresh=refresh)]
    return await scoped(profile, read)


@router.put('/profile')
async def update_profile(body: ProfileUpdate, profile: str | None = None):
    profile = destructive_profile(profile, '/api/settings/profile')
    restart = await scoped(profile, lambda: settings.save_profile(body.instructions, body.timezone))
    return {'ok': True, 'restart': restart, 'applies': 'gateway_restart' if restart else 'next_conversation'}


@router.put('/memory')
async def update_memory(body: MemoryUpdate, profile: str | None = None):
    profile = destructive_profile(profile, '/api/settings/memory')
    def write():
        from deploy.railway.hindsight_settings import is_owner
        if not is_owner():
            raise ValueError('Memory settings belong to the profile that owns this server’s Hindsight service.')
        settings.save_memory(body.model, body.learning, body.recall)
    await scoped(profile, write)
    return {'ok': True, 'applies': 'memory_restart'}


@router.put('/groups/{group_id}')
async def update_group(group_id: str, body: GroupUpdate, profile: str | None = None):
    profile = destructive_profile(profile, '/api/settings/groups')
    await scoped(profile, lambda: settings.save_group(group_id, body.mode, body.instructions,
                                                     [t.model_dump() for t in body.topics], body.remove))
    return {'ok': True, 'applies': 'gateway_restart'}


@router.post('/people')
async def update_person(body: PersonUpdate, profile: str | None = None):
    profile = destructive_profile(profile, '/api/settings/people')
    def write():
        from gateway.pairing import PairingStore
        from agent.secret_scope import get_secret
        group_grants = settings.id_list(get_secret('TELEGRAM_GROUP_ALLOWED_USERS', ''))
        env_grant = body.user_id in settings.id_list(get_secret('TELEGRAM_ALLOWED_USERS', '')) or body.user_id in group_grants
        if not re.fullmatch(r'[1-9][0-9]*', body.user_id):
            raise ValueError('Enter a numeric Telegram user ID.')
        if body.action != 'decline':
            settings.require_writable_people()
            settings.require_readable_config_before_write()
        store = PairingStore()
        if body.action == 'decline':
            if not store.decline_request('telegram', body.request_id):
                raise ValueError('This request has expired.')
        elif body.action == 'add':
            if body.request_id:
                result = store.approve_request('telegram', body.request_id)
                if not result:
                    raise ValueError('This request has expired.')
                user_id = str(result['user_id'])
            else:
                user_id = body.user_id
                store.approve_user('telegram', user_id)
            return settings.update_person_config(user_id, add=True)
        else:
            store.revoke('telegram', body.user_id)
            # Also remove an explicitly configured native allowlist entry.
            from gateway.pairing import _sync_allowlist_remove
            _sync_allowlist_remove('telegram', body.user_id)
            if body.user_id in group_grants:
                from hermes_cli.config import save_env_value
                save_env_value('TELEGRAM_GROUP_ALLOWED_USERS', ','.join(uid for uid in group_grants if uid != body.user_id))
            config_changed = settings.update_person_config(body.user_id)
            return config_changed or env_grant
    restart = await scoped(profile, write)
    return {'ok': True, 'restart': bool(restart)}


SERVICE_KEYS = {'TELEGRAM_BOT_TOKEN', 'OPENROUTER_API_KEY', 'BROWSER_USE_API_KEY', 'PARALLEL_API_KEY'}


async def check_key(key, value):
    requests = {
        'TELEGRAM_BOT_TOKEN': ('GET', f'https://api.telegram.org/bot{value}/getMe', {}, None),
        'OPENROUTER_API_KEY': ('GET', 'https://openrouter.ai/api/v1/key', {'Authorization': 'Bearer ' + value}, None),
        'BROWSER_USE_API_KEY': ('GET', 'https://api.browser-use.com/api/v2/billing/account', {'X-Browser-Use-API-Key': value}, None),
        'PARALLEL_API_KEY': ('POST', 'https://api.parallel.ai/v1/search', {'x-api-key': value}, {'search_queries': ['Parallel API documentation'], 'mode': 'fast', 'max_chars_total': 1000}),
    }
    method, url, headers, data = requests[key]
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.request(method, url, headers=headers, json=data)
        if not response.is_success:
            raise HTTPException(400, f'Connection check failed (HTTP {response.status_code}). The previous key was kept.')
        payload = response.json()
        if key == 'TELEGRAM_BOT_TOKEN' and not payload.get('ok'):
            raise HTTPException(400, 'Telegram rejected this token. The previous token was kept.')
        return payload.get('result', {}).get('username', '') if key == 'TELEGRAM_BOT_TOKEN' else ''
    except (httpx.HTTPError, ValueError):
        raise HTTPException(502, 'Could not verify the key. The previous key was kept; try again.') from None


@router.put('/keys/{key}')
async def update_key(key: str, body: KeyUpdate, profile: str | None = None):
    profile = destructive_profile(profile, '/api/settings/keys')
    if key not in SERVICE_KEYS:
        raise HTTPException(404, 'Unknown service')
    value = body.value.get_secret_value().strip()
    if not value or '\n' in value:
        raise HTTPException(400, 'Enter a valid key.')
    if key == 'TELEGRAM_BOT_TOKEN' and not re.fullmatch(r'\d+:[A-Za-z0-9_-]{30,}', value):
        raise HTTPException(400, 'Enter the full bot token from BotFather.')
    def check_writable():
        from hermes_cli.config import _env_write_blocked
        if _env_write_blocked(key, 'set'):
            raise HTTPException(409, 'This key is managed by the installation or administrator. Change it at that source.')
    await scoped(profile, check_writable)
    username = await check_key(key, value)
    # Recheck after the network probe, before either native writer runs.
    await scoped(profile, check_writable)
    def write():
        from hermes_cli.credential_lifecycle import save_provider_env_credential
        save_provider_env_credential(key, value)
    if key == 'TELEGRAM_BOT_TOKEN':
        from hermes_cli.web_routers.messaging import update_messaging_platform
        from hermes_cli.web_models import MessagingPlatformUpdate
        await update_messaging_platform('telegram', MessagingPlatformUpdate(enabled=True, env={key: value}), profile=profile)
    else:
        await scoped(profile, write)
    return {'ok': True, 'username': username, 'applies': 'gateway_restart'}


@router.post('/admin')
async def update_admin(body: AdminUpdate, request: Request):
    # Dashboard login belongs to the serving host, never the selected management profile.
    from hermes_cli.dashboard_auth import get_provider, InvalidCredentialsError
    provider = get_provider('basic')
    if provider is None:
        raise HTTPException(409, 'Password login is not configured for this server.')
    if not body.username.strip():
        raise HTTPException(400, 'Enter an admin username.')
    if len(body.new_password.get_secret_value()) < 12:
        raise HTTPException(400, 'Use a password of at least 12 characters.')
    def change():
        try:
            provider.complete_password_login(username=provider._username,
                                             password=body.current_password.get_secret_value())
        except InvalidCredentialsError:
            raise HTTPException(403, 'Current password is incorrect.') from None
        try:
            provider.change_credentials(body.username.strip(), body.new_password.get_secret_value())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
    await asyncio.to_thread(change)
    return {'ok': True, 'sign_in_required': True}
