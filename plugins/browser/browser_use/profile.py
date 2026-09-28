"""One durable Browser Use cloud profile per Hermes profile."""
import json
import requests
from hermes_constants import get_hermes_home
from hermes_cli.config import load_config_readonly
from tools.memory_tool_store import MemoryStore
from utils import atomic_write_text


def cloud_profile(base_url, headers):
    configured = load_config_readonly().get('browser', {}).get('profile_id')
    if configured:
        return configured
    path = get_hermes_home() / 'browser' / 'cloud-profile.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with MemoryStore._file_lock(path):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))['id']
        response = requests.post(base_url + '/profiles', headers=headers,
                                 json={'name': 'Hermes employee'}, timeout=30)
        response.raise_for_status()
        profile = response.json()
        if not isinstance(profile.get('id'), str):
            raise ValueError('Browser Use did not return a profile ID.')
        atomic_write_text(path, json.dumps({'id':profile['id']}))
        return profile['id']
