#!/usr/bin/env python3
import argparse, base64, json, sys
from pathlib import Path

def decode_jwt_payload(token: str):
    parts = token.split('.')
    if len(parts) != 3:
        return {}
    payload = parts[1] + ('=' * (-len(parts[1]) % 4))
    try:
        return json.loads(base64.urlsafe_b64decode(payload).decode('utf-8'))
    except Exception:
        return {}

def inspect_agent(agent: str):
    p = Path(f'/home/yeqiuqiu/.openclaw/agents/{agent}/agent/auth-profiles.json')
    if not p.exists():
        return {'agent': agent, 'exists': False}
    obj = json.loads(p.read_text())
    out = {'agent': agent, 'exists': True, 'path': str(p), 'profiles': [], 'order': obj.get('order', {}), 'lastGood': obj.get('lastGood', {})}
    for profile_id, prof in obj.get('profiles', {}).items():
        row = {
            'profileId': profile_id,
            'provider': prof.get('provider'),
            'type': prof.get('type'),
            'accountId': prof.get('accountId'),
            'emailField': prof.get('email'),
        }
        access = prof.get('access')
        if isinstance(access, str) and access.count('.') == 2:
            payload = decode_jwt_payload(access)
            row['jwtEmail'] = (((payload.get('https://api.openai.com/profile') or {}).get('email')))
            row['jwtPlanType'] = (((payload.get('https://api.openai.com/auth') or {}).get('chatgpt_plan_type')))
            row['jwtAccountId'] = (((payload.get('https://api.openai.com/auth') or {}).get('chatgpt_account_id')))
            row['jwtAccountUserId'] = (((payload.get('https://api.openai.com/auth') or {}).get('chatgpt_account_user_id')))
        out['profiles'].append(row)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('agent')
    args = ap.parse_args()
    print(json.dumps(inspect_agent(args.agent), indent=2))
