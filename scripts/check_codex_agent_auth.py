#!/usr/bin/env python3
import argparse, json, base64
from pathlib import Path

def decode_jwt_payload(token: str):
    try:
        parts = token.split('.')
        if len(parts) < 2:
            return {}
        payload = parts[1]
        padding = '=' * (-len(payload) % 4)
        raw = base64.urlsafe_b64decode(payload + padding)
        return json.loads(raw.decode('utf-8'))
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('agent')
    args = ap.parse_args()
    p = Path(f'/home/yeqiuqiu/.openclaw/agents/{args.agent}/agent/auth-profiles.json')
    if not p.exists():
        print(json.dumps({'agent': args.agent, 'exists': False}, indent=2))
        return
    obj = json.loads(p.read_text())
    out = {'agent': args.agent, 'exists': True, 'profiles': []}
    for pid, prof in obj.get('profiles', {}).items():
        row = {'profileId': pid, 'provider': prof.get('provider'), 'type': prof.get('type'), 'accountId': prof.get('accountId')}
        if prof.get('type') == 'oauth' and prof.get('access'):
            payload = decode_jwt_payload(prof['access'])
            auth = payload.get('https://api.openai.com/auth', {})
            profile = payload.get('https://api.openai.com/profile', {})
            row['email'] = profile.get('email')
            row['planType'] = auth.get('chatgpt_plan_type')
            row['chatgptAccountId'] = auth.get('chatgpt_account_id')
        out['profiles'].append(row)
    out['order'] = obj.get('order', {})
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
