#!/usr/bin/env python3
import argparse, base64, json, re, sys
from pathlib import Path

def slugify(value: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '-', value).strip('-').lower() or 'profile'

def decode_jwt_payload(token: str):
    parts = token.split('.')
    if len(parts) != 3:
        return {}
    payload = parts[1] + ('=' * (-len(parts[1]) % 4))
    try:
        return json.loads(base64.urlsafe_b64decode(payload).decode('utf-8'))
    except Exception:
        return {}

def profile_email(profile: dict):
    access = profile.get('access') or ''
    payload = decode_jwt_payload(access) if isinstance(access, str) else {}
    return (((payload.get('https://api.openai.com/profile') or {}).get('email')) or profile.get('email') or '').strip()

def select_profile(profiles: dict, from_profile_id: str | None, from_email: str | None, desired: str):
    codex = {k: v for k, v in profiles.items() if v.get('provider') == 'openai-codex'}
    if not codex:
        raise SystemExit('No openai-codex profiles found in auth store')

    if desired in codex:
        return desired, codex[desired], profile_email(codex[desired])

    if from_profile_id:
        prof = codex.get(from_profile_id)
        if not prof:
            raise SystemExit(f'Profile id not found: {from_profile_id}')
        return from_profile_id, prof, profile_email(prof)

    if from_email:
        matches = [(k, v) for k, v in codex.items() if profile_email(v).lower() == from_email.lower()]
        if not matches:
            raise SystemExit(f'No profile found for email: {from_email}')
        if len(matches) != 1:
            raise SystemExit(f'Email matched multiple profiles: {from_email}')
        old_id, prof = matches[0]
        return old_id, prof, profile_email(prof)

    if len(codex) == 1:
        old_id, prof = next(iter(codex.items()))
        return old_id, prof, profile_email(prof)

    raise SystemExit(
        'Expected exactly 1 openai-codex profile, found '
        f'{len(codex)}. Re-run with --from-profile-id or --from-email.'
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('agent')
    ap.add_argument('--profile-id', help='Explicit profile id to assign (default: openai-codex:<agent>)')
    ap.add_argument('--from-profile-id', help='Source profile id to normalize when multiple profiles exist')
    ap.add_argument('--from-email', help='Source profile email to normalize when multiple profiles exist')
    ap.add_argument('--set-order-only', action='store_true', help='Do not rename; only pin order to the desired profile id')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    path = Path(f'/home/yeqiuqiu/.openclaw/agents/{args.agent}/agent/auth-profiles.json')
    obj = json.loads(path.read_text())
    profiles = obj.get('profiles', {})
    if not profiles:
        raise SystemExit('No profiles found in auth store')
    desired = args.profile_id or f'openai-codex:{args.agent}'

    old_id, prof, email = select_profile(profiles, args.from_profile_id, args.from_email, desired)
    if prof.get('provider') != 'openai-codex':
        raise SystemExit(f'Profile provider is not openai-codex: {prof.get("provider")}')

    updated = json.loads(json.dumps(obj))
    updated_profiles = updated.setdefault('profiles', {})
    if not args.set_order_only and old_id != desired:
        cred = updated_profiles.pop(old_id)
        if email and not cred.get('email'):
            cred['email'] = email
        updated_profiles[desired] = cred
    elif desired not in updated_profiles:
        raise SystemExit(f'Desired profile id does not exist for --set-order-only: {desired}')

    order = updated.setdefault('order', {})
    order['openai-codex'] = [desired]

    last_good = updated.setdefault('lastGood', {})
    if last_good.get('openai-codex') == old_id or args.set_order_only:
        last_good['openai-codex'] = desired

    usage = updated.setdefault('usageStats', {})
    if not args.set_order_only and old_id in usage and desired not in usage:
        usage[desired] = usage.pop(old_id)
    elif not args.set_order_only and old_id in usage:
        usage.pop(old_id)

    result = {
        'agent': args.agent,
        'oldProfileId': old_id,
        'newProfileId': desired,
        'email': email,
        'path': str(path)
    }
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return
    path.write_text(json.dumps(updated, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
