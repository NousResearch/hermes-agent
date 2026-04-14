#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path


ROOT = Path('/home/yeqiuqiu')
CONFIG_PATH = ROOT / '.openclaw' / 'openclaw.json'
DEFAULT_LANE_MAP = Path('/home/yeqiuqiu/clawd-architect/ops/openclaw/codex_lane_account_map.json')


def decode_jwt_payload(token: str):
    parts = token.split('.')
    if len(parts) != 3:
        return {}
    payload = parts[1] + ('=' * (-len(parts[1]) % 4))
    try:
        return json.loads(base64.urlsafe_b64decode(payload).decode('utf-8'))
    except Exception:
        return {}


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def jwt_email(profile: dict):
    access = profile.get('access') or ''
    payload = decode_jwt_payload(access) if isinstance(access, str) else {}
    return (((payload.get('https://api.openai.com/profile') or {}).get('email')) or profile.get('email') or '').strip()


def jwt_account_id(profile: dict):
    access = profile.get('access') or ''
    payload = decode_jwt_payload(access) if isinstance(access, str) else {}
    return (((payload.get('https://api.openai.com/auth') or {}).get('chatgpt_account_id')) or profile.get('accountId') or '').strip()


def find_agent_config(config: dict, agent_id: str):
    for entry in config.get('agents', {}).get('list', []):
        if entry.get('id') == agent_id:
            return entry
    return {}


def normalize_model(entry):
    model = entry.get('model')
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        return model.get('primary') or ''
    return ''


def audit_lane(spec: dict, config: dict, expected_email: str | None = None):
    agent = spec['agent']
    auth_path = ROOT / '.openclaw' / 'agents' / agent / 'agent' / 'auth-profiles.json'
    auth = load_json(auth_path, {'profiles': {}, 'order': {}, 'lastGood': {}})
    profiles = auth.get('profiles', {})
    provider_profiles = {
        pid: prof for pid, prof in profiles.items() if prof.get('provider') == 'openai-codex'
    }
    desired = spec.get('desiredProfileId')
    configured = find_agent_config(config, agent)
    config_entry_present = bool(configured)
    configured_model = normalize_model(configured)
    order = auth.get('order', {}).get('openai-codex', [])
    desired_profile = provider_profiles.get(desired)
    desired_email = jwt_email(desired_profile) if desired_profile else ''
    desired_account_id = jwt_account_id(desired_profile) if desired_profile else ''
    accounts = []
    for pid, prof in provider_profiles.items():
        accounts.append({
            'profileId': pid,
            'email': jwt_email(prof),
            'accountId': jwt_account_id(prof),
            'type': prof.get('type'),
        })

    ambiguity_reasons = []
    if not provider_profiles:
        ambiguity_reasons.append('no_openai_codex_profiles')
    if len(provider_profiles) > 1:
        ambiguity_reasons.append('multiple_openai_codex_profiles')
    if desired not in provider_profiles:
        ambiguity_reasons.append('desired_profile_missing')
    if desired in provider_profiles and not desired_account_id:
        ambiguity_reasons.append('desired_account_id_missing')
    if expected_email and desired_email and expected_email.lower() != desired_email.lower():
        ambiguity_reasons.append('desired_email_mismatch')

    config_drift_reasons = []
    if not config_entry_present:
        config_drift_reasons.append('agent_missing_in_active_config')
    elif not configured_model:
        config_drift_reasons.append('configured_model_missing')
    elif configured_model != spec.get('model'):
        config_drift_reasons.append('configured_model_mismatch')

    binding_ambiguous = len(ambiguity_reasons) > 0
    ready = (
        configured_model == spec.get('model')
        and desired in provider_profiles
        and order == [desired]
    )

    return {
        'agent': agent,
        'role': spec.get('role'),
        'configEntryPresent': config_entry_present,
        'configDriftReasons': config_drift_reasons,
        'configuredModel': configured_model,
        'expectedModel': spec.get('model'),
        'modelMatches': configured_model == spec.get('model'),
        'desiredProfileId': desired,
        'desiredProfilePresent': desired in provider_profiles,
        'order': order,
        'orderPinnedToDesired': order == [desired],
        'lastGood': auth.get('lastGood', {}).get('openai-codex'),
        'expectedEmail': expected_email,
        'accounts': accounts,
        'providerProfileCount': len(provider_profiles),
        'extraProviderProfiles': sorted([pid for pid in provider_profiles.keys() if pid != desired]),
        'desiredAccount': {
            'email': desired_email,
            'accountId': desired_account_id,
        } if desired_profile else None,
        'bindingAmbiguous': binding_ambiguous,
        'ambiguityReasons': ambiguity_reasons,
        'ready': ready,
        'dispatchReady': ready and not binding_ambiguous,
        'authPath': str(auth_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--manifest',
        default='/home/yeqiuqiu/clawd-architect/ops/openclaw/codex_worker_pool_manifest.json',
    )
    ap.add_argument(
        '--lane-map',
        default=str(DEFAULT_LANE_MAP),
    )
    args = ap.parse_args()

    manifest = load_json(Path(args.manifest), {})
    lane_map = load_json(Path(args.lane_map), {})
    config = load_json(CONFIG_PATH, {})
    expected_email_by_agent = {
        lane.get('agent'): lane.get('email')
        for lane in lane_map.get('lanes', [])
        if lane.get('agent')
    }
    specs = [manifest.get('orchestrator', {})] + manifest.get('workers', [])
    lanes = [
        audit_lane(spec, config, expected_email=expected_email_by_agent.get(spec.get('agent')))
        for spec in specs
        if spec.get('agent')
    ]

    account_to_agents = {}
    for lane in lanes:
        desired = lane.get('desiredAccount') or {}
        account_id = desired.get('accountId')
        if not account_id:
            continue
        account_to_agents.setdefault(account_id, []).append(lane['agent'])

    shared = {
        account_id: agents
        for account_id, agents in account_to_agents.items()
        if len(agents) > 1
    }

    for lane in lanes:
        desired = lane.get('desiredAccount') or {}
        account_id = desired.get('accountId')
        if account_id and account_id in shared:
            reasons = list(lane.get('ambiguityReasons') or [])
            if 'shared_desired_account' not in reasons:
                reasons.append('shared_desired_account')
            lane['ambiguityReasons'] = reasons
            lane['bindingAmbiguous'] = True
            lane['dispatchReady'] = bool(lane.get('ready')) and not lane['bindingAmbiguous']

    summary = {
        'codingModel': manifest.get('codingModel'),
        'readyAgents': [lane['agent'] for lane in lanes if lane['ready']],
        'dispatchReadyAgents': [lane['agent'] for lane in lanes if lane.get('dispatchReady')],
        'notReadyAgents': [lane['agent'] for lane in lanes if not lane['ready']],
        'ambiguousAgents': [lane['agent'] for lane in lanes if lane.get('bindingAmbiguous')],
        'configDriftAgents': [lane['agent'] for lane in lanes if lane.get('configDriftReasons')],
        'sharedDesiredAccounts': shared,
        'workerPoolReady': all(lane['ready'] for lane in lanes if lane.get('role') == 'worker'),
        'workerPoolDispatchReady': all(lane.get('dispatchReady') for lane in lanes if lane.get('role') == 'worker'),
        'orchestratorReady': any(
            lane['agent'] == manifest.get('orchestrator', {}).get('agent') and lane['ready']
            for lane in lanes
        ),
    }

    print(json.dumps({'summary': summary, 'lanes': lanes}, indent=2))


if __name__ == '__main__':
    main()
