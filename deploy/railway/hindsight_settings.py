"""Hindsight deployment settings read from the owning Hermes profile."""
import hashlib
import json

from agent.secret_scope import build_profile_secret_scope
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home, get_routing_process_hermes_home


def defaults():
    # The three editable exceptions to the pinned Hindsight policy.
    return {'llm_model': 'gpt-5.6-luna', 'llm_reasoning_effort': 'low',
            'reflect_llm_reasoning_effort': 'medium'}


def current():
    config = load_config().get('hindsight') or {}
    values = {name: config.get(name, value) for name, value in defaults().items()}
    values['openrouter_api_key'] = build_profile_secret_scope(get_hermes_home()).get('OPENROUTER_API_KEY', '')
    revision = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return {**values, 'revision': revision}


def is_owner():
    return get_hermes_home().resolve() == get_routing_process_hermes_home().resolve()


def status():
    if not is_owner():
        return {'state': 'unmanaged'}
    path = get_hermes_home() / 'hindsight_runtime.json'
    try:
        record = json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return {'state': 'waiting'}
    except (OSError, ValueError):
        return {'state': 'unreachable'}
    import time
    if time.time() - record.get('updated_at', 0) > 30:
        return {'state': 'unreachable'}
    if record.get('revision') != current()['revision']:
        return {'state': 'applying'}
    return {'state': record.get('state', 'starting')}
