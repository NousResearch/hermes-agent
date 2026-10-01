"""The hosted settings view of native profile configuration (no separate store)."""
from __future__ import annotations

import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.reasoning_effort import codex_supported_efforts
from hermes_cli.config import load_config, require_readable_config_before_write, save_config
from hermes_cli.config_effective import load_user_config_effective
from hermes_cli.web_deps import LateState

CONFIG_LOCK = LateState('_CONFIG_MUTATION_LOCK')


def require_writable_config(*keys):
    from hermes_cli.config import is_managed
    from hermes_cli import managed_scope
    if is_managed():
        raise ValueError('Configuration is managed by this installation. Change it at that source.')
    for locked in managed_scope.managed_config_keys():
        if any(key == locked or key.startswith(locked + '.') or locked.startswith(key + '.') for key in keys):
            raise ValueError(f'{locked} is managed by your administrator. Change it at that source.')


def require_writable_telegram(fields):
    require_writable_config(*(f'{prefix}.{extra}{field}'
        for prefix in ('telegram', 'platforms.telegram', 'gateway.telegram', 'gateway.platforms.telegram')
        for extra in ('', 'extra.') for field in fields))


def require_writable_people():
    from hermes_cli.config import _env_write_blocked
    require_writable_telegram(('allow_from', 'allowed_users', 'group_allow_from'))
    if any(_env_write_blocked(key, 'set') for key in ('TELEGRAM_ALLOWED_USERS', 'TELEGRAM_GROUP_ALLOWED_USERS')):
        raise ValueError('The Telegram allowlist is managed by your administrator. Change it at that source.')


def profile_values(config):
    from hermes_constants import get_hermes_home
    soul_path = get_hermes_home() / 'SOUL.md'
    try:
        prompt = soul_path.read_text(encoding='utf-8-sig')
    except FileNotFoundError:
        prompt = ''
    return {'instructions': prompt,
            'timezone': config.get('timezone') or ''}


def save_profile(instructions, timezone):
    try:
        if timezone:
            ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError('Choose a valid IANA timezone, such as Europe/Prague.') from None
    with CONFIG_LOCK:
        config = require_readable_config_before_write()
        previous = profile_values(load_user_config_effective())
        timezone_changed = timezone is not None and previous['timezone'] != timezone
        if timezone_changed:
            require_writable_config('timezone')
            config['timezone'] = timezone
            save_config(config)
        if instructions is not None and instructions != previous['instructions']:
            from hermes_constants import get_hermes_home
            from utils import atomic_write_text
            # Native SOUL is frozen with the conversation. The optional config
            # system_prompt is ephemeral and must not hold these identity edits.
            atomic_write_text(get_hermes_home() / 'SOUL.md', instructions,
                              preserve_mode=True, create_mode=0o644)
        if timezone_changed:
            from hermes_time import reset_cache
            reset_cache()
        return timezone_changed


def validate_model(model, *efforts):
    from hermes_cli.models import provider_model_ids
    from agent.model_metadata import strip_codex_context_variant_suffix
    if model != strip_codex_context_variant_suffix(model) or model not in provider_model_ids('openai-codex'):
        raise ValueError('Choose a model from the Codex catalog.')
    if any(effort not in codex_supported_efforts(model) for effort in efforts):
        raise ValueError('This model does not support the selected reasoning level.')


def memory_values(config=None):
    from deploy.railway.hindsight_settings import defaults
    config = (config or load_config()).get('hindsight') or {}
    return {key: config.get(key, value) for key, value in defaults().items()}


def save_memory(model, learning, recall):
    fields = ('hindsight.llm_model', 'hindsight.llm_reasoning_effort', 'hindsight.reflect_llm_reasoning_effort')
    require_writable_config(*fields)
    validate_model(model, learning, recall)
    with CONFIG_LOCK:
        require_writable_config(*fields)
        config = require_readable_config_before_write()
        memory = config.get('hindsight') or {}
        memory.update(llm_model=model, llm_reasoning_effort=learning, reflect_llm_reasoning_effort=recall)
        config['hindsight'] = memory
        save_config(config)


def telegram_config(config):
    """Use native pure YAML bridges without mutating the dashboard process environment."""
    from gateway.config import Platform, PlatformConfig
    from gateway.config_loader import merge_platform_sections, bridge_platform_shared_keys
    from agent.secret_scope import get_secret
    data = {}
    nested = config.get('gateway') or {}
    platforms = merge_platform_sections(config, nested, data)
    bridge_platform_shared_keys(config, nested.get('platforms', {}), data, platforms, [Platform.TELEGRAM])
    telegram = PlatformConfig.from_dict(platforms.get('telegram', {})).extra
    # Match the Telegram fallback in bridge_core_env_settings without its env writes.
    if config.get('require_mention') is not None and 'require_mention' not in (config.get('telegram') or {}):
        telegram.setdefault('require_mention', config['require_mention'])
    for key in ('group_allowed_chats', 'free_response_chats', 'free_response_topics', 'allowed_chats',
                'allowed_topics', 'ignored_threads', 'allow_from', 'group_allow_from',
                'require_mention', 'guest_mode'):
        env = {'allow_from': 'TELEGRAM_ALLOWED_USERS', 'group_allow_from': 'TELEGRAM_GROUP_ALLOWED_USERS'}.get(key, 'TELEGRAM_' + key.upper())
        value = get_secret(env)
        if value is not None:
            telegram[key] = value
    return telegram


def id_list(value):
    from gateway.platforms._shared import decode_json_list_literal
    value = decode_json_list_literal(value)
    return [str(v).strip() for v in (value if isinstance(value, list) else str(value or '').split(',')) if str(v).strip()]


def truthy(value):
    return str(value).lower() in {'true', '1', 'yes', 'on'}


def update_person_config(user_id, add=False):
    """Keep explicit native adapter allowlists consistent with pairing edits."""
    with CONFIG_LOCK:
        require_writable_people()
        config = require_readable_config_before_write()
        effective = load_user_config_effective()
        changed = False
        paths = [('telegram',), ('platforms', 'telegram'), ('gateway', 'telegram'), ('gateway', 'platforms', 'telegram')]
        for path in paths:
            block, resolved = config, effective
            for part in path:
                block, resolved = block.get(part) or {}, resolved.get(part) or {}
            for section, values in ((block, resolved), (block.get('extra') or {}, resolved.get('extra') or {})):
                for key in ('allow_from', 'allowed_users', 'group_allow_from'):
                    if key in section:
                        ids = id_list(values.get(key))
                        section[key] = [value for value in ids if value != user_id] + ([user_id] if add else [])
                        changed = True
        if changed:
            save_config(config)
        return changed


def telegram_values():
    from gateway.channel_directory import load_directory
    from gateway.pairing import PairingStore
    from agent.secret_scope import get_secret
    telegram = telegram_config(load_user_config_effective())
    def ids(key):
        return id_list(telegram.get(key))
    directory = load_directory().get('platforms', {}).get('telegram', [])
    names = {str(item.get('id')): item for item in directory}
    groups = []
    for gid in ids('group_allowed_chats'):
        if not gid:
            continue
        entry = names.get(gid, {})
        group_name = entry.get('name') or gid
        topics = {str(t.get('id', t.get('thread_id'))): t.get('name', '') for t in entry.get('topics', [])}
        for item in directory:
            if str(item.get('id', '')).startswith(gid + ':'):
                tid = str(item.get('thread_id') or str(item['id']).split(':', 1)[1])
                topic_name = str(item.get('name') or '').rpartition(' / ')[2]
                topics[tid] = '' if topic_name == f'topic {tid}' else topic_name
                if group_name == gid and item.get('name'):
                    group_name = item['name'].rsplit(' / ', 1)[0]
        bindings = telegram.get('group_topics') or {}
        if isinstance(bindings, list):
            bindings = {str(item.get('chat_id')): item.get('topics', []) for item in bindings}
        bindings = {str(key): value for key, value in bindings.items()}
        for item in bindings.get(gid, []):
            topics[str(item['thread_id'])] = item.get('name', '')

        for pair in ids('free_response_topics') + ids('silent_topics'):
            if pair.startswith(gid + ':'):
                topics.setdefault(pair.split(':', 1)[1], '')
        groups.append({'id': gid, 'name': group_name,
                       'mode': 'all' if gid in ids('free_response_chats') or not truthy(telegram.get('require_mention', False)) else 'mention',
                       'instructions': (telegram.get('channel_prompts') or {}).get(gid, ''),
                       'topics': [{'id': tid, 'name': name or tid, 'mode': 'silent' if f'{gid}:{tid}' in ids('silent_topics') else 'all' if f'{gid}:{tid}' in ids('free_response_topics') else 'inherit'} for tid, name in topics.items()]})
    store = PairingStore()
    people = {p['user_id']: p for p in store.list_approved('telegram')}
    for uid in ids('allow_from') + ids('allowed_users'):
        if str(uid).strip():
            people.setdefault(str(uid).strip(), {'user_id': str(uid).strip(), 'user_name': ''})
    additional = any(telegram.get(key) for key in ('allowed_chats', 'allowed_topics', 'ignored_threads', 'group_allow_from'))
    additional = additional or truthy(telegram.get('guest_mode')) or any(get_secret(key, '') for key in
        ('GATEWAY_ALLOWED_USERS', 'GATEWAY_ALLOW_ALL_USERS', 'TELEGRAM_ALLOW_ALL_USERS')) or '*' in ids('allow_from')
    return {'groups': groups, 'people': list(people.values()), 'pending': store.list_pending('telegram'),
            'notice': 'Additional native access rules are configured. They can grant or restrict access beyond these lists.' if additional else ''}


def save_group(gid, mode, instructions, topics, remove=False):
    if not re.fullmatch(r'-[1-9][0-9]*', gid):
        raise ValueError('Enter the numeric Telegram group ID, starting with a minus sign.')
    if mode not in {'mention', 'all'} or any(t['mode'] not in {'inherit', 'all', 'silent'} or not str(t['id']).isdigit() for t in topics):
        raise ValueError('Invalid group or topic reply policy.')
    with CONFIG_LOCK:
        config = require_readable_config_before_write()
        effective = telegram_config(load_user_config_effective())
        tg = config.get('telegram') or {}
        config['telegram'] = tg
        def updated(key, match, additions):
            current = id_list(effective.get(key))
            tg[key] = [str(v) for v in current if v and not match(str(v))] + additions
        updated('group_allowed_chats', lambda v: v == gid, [] if remove else [gid])
        updated('free_response_chats', lambda v: v == gid, [gid] if not remove and mode == 'all' else [])
        for key, choice in [('free_response_topics', 'all'), ('silent_topics', 'silent')]:
            updated(key, lambda v: v.startswith(gid + ':'), [] if remove else [f"{gid}:{t['id']}" for t in topics if t['mode'] == choice])
        prompts = dict(effective.get('channel_prompts') or {})
        tg['channel_prompts'] = prompts
        if remove:
            prompts.pop(gid, None)
        else:
            prompts[gid] = instructions
        # Preserve other explicitly admitted groups when establishing the native
        # mention-by-default + free-response exceptions policy.
        if not truthy(effective.get('require_mention', False)):
            tg['free_response_chats'] = list(dict.fromkeys(tg['free_response_chats'] +
                [other for other in id_list(effective.get('group_allowed_chats')) if other != gid]))
        tg['require_mention'] = True
        # Avoid retaining a second, higher-precedence spelling of edited fields.
        edited_keys = ('group_allowed_chats', 'free_response_chats', 'free_response_topics',
                       'silent_topics', 'channel_prompts', 'require_mention')
        require_writable_telegram(edited_keys)
        for key in edited_keys:
            tg.get('extra', {}).pop(key, None)
        # Native env values take precedence. Update an existing profile override too.
        from agent.secret_scope import get_secret
        from hermes_cli.config import save_env_value, _env_write_blocked
        env_updates = {}
        for key in ('group_allowed_chats', 'free_response_chats', 'free_response_topics', 'require_mention'):
            env = 'TELEGRAM_' + key.upper()
            if get_secret(env) is not None:
                value = tg[key]
                if _env_write_blocked(env, 'set'):
                    raise ValueError(f'{env} is managed by your administrator. Change it at that source.')
                env_updates[env] = ','.join(value) if isinstance(value, list) else str(value).lower()
        for env, value in env_updates.items():
            save_env_value(env, value)
        # Empty overrides must still shadow an existing nested native policy.
        save_config(config, preserve_keys={('telegram', key) for key in edited_keys})
