"""Buzz-owned policy API. Host must mount behind Dashboard authentication.

Persistence adapted from installed Buzz plugin (5e9f111020); host discovery and
router lifecycle are deliberately not implemented here.
"""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError, field_validator, model_validator
from plugins.platforms.buzz import settings

router = APIRouter()
_POLICY_FIELDS = ('allowed_users', 'allow_all_users', 'require_mention', 'thread_require_mention')


class BuzzPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    allowed_users: list[str] | None = None
    allow_all_users: StrictBool | None = None
    require_mention: StrictBool | None = None
    thread_require_mention: StrictBool | None = None

    @model_validator(mode='before')
    @classmethod
    def nonempty(cls, value: Any):
        if not isinstance(value, dict) or not value or any(v is None for v in value.values()):
            raise ValueError('Submit at least one non-null policy field')
        return value

    @field_validator('allowed_users')
    @classmethod
    def identities(cls, values):
        normalized = []
        for index, value in enumerate(values):
            identity = settings.normalize_user_ref(value)
            if identity is None:
                raise ValueError(f'Invalid Buzz public identity at item {index + 1}')
            if identity not in normalized:
                normalized.append(identity)
        return normalized


class BuzzPolicyUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    policy: BuzzPolicy


@router.put('/policy')
async def put_policy(body: Any = Body(...), profile: str | None = None):
    # Do not let framework validation serialize rejected identities/unknown fields.
    try:
        parsed = BuzzPolicyUpdate.model_validate(body)
    except ValidationError as exc:
        errors = [{'loc': ['body', *(v for v in e['loc'] if v in ('policy', *_POLICY_FIELDS) or isinstance(v, int))],
                   'msg': e['msg'] if e['type'] == 'value_error' else 'Invalid policy field type or shape'}
                  for e in exc.errors(include_input=False, include_context=False, include_url=False)]
        raise HTTPException(422, detail=errors) from None
    import asyncio
    return await asyncio.to_thread(_save_policy, parsed, profile)


_CANONICAL_PREFIX = 'gateway.platforms.buzz.extra'
_LEGACY_PREFIXES = ('gateway.platforms.buzz', 'platforms.buzz', 'platforms.buzz.extra',
                    'gateway.buzz', 'gateway.buzz.extra', 'buzz', 'buzz.extra')


def _mapping_at(config, prefix):
    for part in prefix.split('.'):
        if not isinstance(config, dict):
            return {}
        config = config.get(part)
    return config if isinstance(config, dict) else {}


def _explicit(config):
    return settings._explicit_policy_values(config, _POLICY_FIELDS)


def _references(values):
    import re
    def contains(value):
        if isinstance(value, str):
            return re.search(r'\${[^}]+}', value) is not None
        if isinstance(value, (list, tuple)):
            return any(contains(v) for v in value)
        return False
    return {field for field, value in values.items() if contains(value)}


def _read_strict(path):
    # Existing plugin-local reader validates every supported container and never
    # expands templates or swallows corruption. Do not port a core raw-reader API.
    raw = settings._read_policy_file(path)
    for prefix in (_CANONICAL_PREFIX, *_LEGACY_PREFIXES):
        explicit = {k: v for k, v in _mapping_at(raw, prefix).items() if k in _POLICY_FIELDS}
        determinate = {k: v for k, v in explicit.items() if k not in _references(explicit)}
        settings.authorization_from_config({'buzz': determinate})
        settings.policy_from_config({'buzz': determinate})
    return raw


def _managed_state():
    from hermes_cli.managed_scope import get_managed_dir
    try:
        directory = get_managed_dir()
        raw = _read_strict(directory / 'config.yaml') if directory is not None else {}
        return _explicit(raw), False
    except Exception:
        return {}, True


def _pairing_active(home):
    import json
    try:
        active = False
        for prefix in ('platforms/pairing', 'pairing'):
            path = home / prefix / 'buzz-approved.json'
            try:
                parsed = json.loads(path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                continue
            if not isinstance(parsed, dict):
                return None
            active = active or bool(parsed)
        return active
    except Exception:
        return None


def _policy_payload_scoped(profile):
    from copy import deepcopy
    from hermes_constants import get_hermes_home
    from hermes_cli.config import is_managed
    home = get_hermes_home()
    try:
        raw = _read_strict(home / 'config.yaml')
        unavailable = False
    except Exception:
        raw, unavailable = {}, True
    managed, managed_error = _managed_state()
    explicit = _explicit(raw)
    reference_fields = _references(explicit)
    overridden = reference_fields | _references(managed)
    try:
        getenv = settings._environment_getter(None, home)
        overridden.update(field for field in _POLICY_FIELDS if getenv('BUZZ_' + field.upper()) is not None)
        global_active = bool((getenv('GATEWAY_ALLOWED_USERS') or '').strip()) or (
            (getenv('GATEWAY_ALLOW_ALL_USERS') or '').strip().lower() in {'true', '1', 'yes', 'on'})
    except Exception:
        overridden.update(_POLICY_FIELDS)
        global_active = None
    determinate = {k: v for k, v in explicit.items() if k not in reference_fields}
    policy = {**settings.authorization_from_config({'buzz': determinate}),
              **settings.policy_from_config({'buzz': determinate})}
    indeterminate = overridden | set(managed)
    if unavailable or managed_error:
        indeterminate.update(_POLICY_FIELDS)
    for field in indeterminate:
        policy[field] = None
    legacy = {field for prefix in _LEGACY_PREFIXES for field in _POLICY_FIELDS if field in _mapping_at(raw, prefix)}
    return {'profile': (profile or '').strip() or 'current', 'policy': deepcopy(policy),
            'environment_overrides': sorted(overridden), 'indeterminate_fields': sorted(indeterminate),
            'ineffective_fields': sorted(indeterminate), 'managed_fields': sorted(managed),
            'locked': unavailable or managed_error or bool(managed) or is_managed(),
            'managed_error': managed_error, 'user_policy_unavailable': unavailable,
            'legacy_fields': sorted(legacy), 'legacy_cleanup_required': bool(legacy),
            'additional_global_grants_active': global_active,
            'additional_pairing_grants_active': _pairing_active(home)}


def _policy_payload(profile):
    from hermes_cli.web_server_profiles import _config_profile_scope
    with _config_profile_scope(profile):
        return _policy_payload_scoped(profile)


def _save_policy(body, profile):
    from copy import deepcopy
    from hermes_cli.web_server_profiles import _config_profile_scope
    from hermes_cli.web_server import _CONFIG_MUTATION_LOCK
    from hermes_cli.config import is_managed, save_config
    from hermes_constants import get_hermes_home
    with _config_profile_scope(profile), _CONFIG_MUTATION_LOCK:
        managed, error = _managed_state()
        if error or managed or is_managed():
            raise HTTPException(409, detail={'error': 'managed_policy_unavailable' if error else 'managed_policy',
                                             'managed_fields': sorted(managed)})
        try:
            existing = _read_strict(get_hermes_home() / 'config.yaml')
            merged = deepcopy(existing)
            authoritative = _explicit(existing)
            authoritative.update(body.policy.model_dump(exclude_unset=True))
            canonical = merged
            for part in _CANONICAL_PREFIX.split('.'):
                canonical = canonical.setdefault(part, {})
            canonical.update(deepcopy(authoritative))
        except Exception:
            raise HTTPException(409, detail={'error': 'user_policy_unavailable'}) from None
        for prefix in _LEGACY_PREFIXES:
            node = _mapping_at(merged, prefix)
            for field in _POLICY_FIELDS:
                node.pop(field, None)
        preserve = {('gateway', 'platforms', 'buzz', 'extra', field) for field in authoritative}
        try:
            save_config(merged, preserve_keys=preserve)
        except Exception:
            raise HTTPException(409, detail={'error': 'policy_save_failed'}) from None
        return _policy_payload_scoped(profile)


@router.get('/policy')
async def get_policy(profile: str | None = None):
    import asyncio
    return await asyncio.to_thread(_policy_payload, profile)
