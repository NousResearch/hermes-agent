#!/usr/bin/env python3
"""Quick Graph API calendar fetch using MSAL cache."""
import json, requests, datetime as dt

TZ = dt.timezone(dt.timedelta(hours=1))
now = dt.datetime.now(TZ)
start = now.replace(hour=0, minute=0, second=0, microsecond=0)
end = start + dt.timedelta(days=8)

with open('/home/kensei/.config/ms-365-mcp-server/token-cache.json', encoding='utf-8') as f:
    top = json.load(f)
inner = json.loads(top['data'])

accounts = {}
for k,v in inner['Account'].items():
    accounts[v['home_account_id']] = v['username']

at_map = {}
for k,v in inner['AccessToken'].items():
    haid = v['home_account_id']
    scopes = v.get('target','')
    if 'Calendars.ReadWrite' in scopes or 'Calendars.Read' in scopes:
        if haid not in at_map:
            at_map[haid] = v.get('secret','')

issues = []
all_events = []
for haid, token in at_map.items():
    acc = accounts.get(haid, 'unknown')
    try:
        headers = {'Authorization': f'Bearer {token}', 'Prefer': 'outlook.timezone="Europe/London"'}
        r = requests.get(f'https://graph.microsoft.com/v1.0/me/calendarview?startDateTime={start.isoformat()}\u0026endDateTime={end.isoformat()}\u0026$select=subject,start,end,location,bodyPreview,webLink\u0026$top=50', headers=headers, timeout=30)
        if r.status_code == 401:
            issues.append(f'Outlook {acc}: expired (401)')
            continue
        r.raise_for_status()
        data = r.json()
        for e in data.get('value', []):
            loc_obj = e.get('location', {})
            loc_str = loc_obj.get('displayName', '') if isinstance(loc_obj, dict) else str(loc_obj)
            all_events.append({
                'source': 'Outlook',
                'account': acc,
                'calendar': 'Calendar',
                'summary': e.get('subject', 'Untitled'),
                'start': e.get('start', {}).get('dateTime', ''),
                'end': e.get('end', {}).get('dateTime', ''),
                'location': loc_str,
                'link': e.get('webLink', ''),
                'description': e.get('bodyPreview', '')[:200]
            })
    except Exception as e2:
        issues.append(f'Outlook {acc}: {str(e2)[:80]}')

print(json.dumps({'events': len(all_events), 'issues': issues, 'accounts': list(at_map.keys())}, indent=2))
# Save
import os
os.makedirs('/home/kensei/.hermes/runbooks/calendar-brief/2026-05-12', exist_ok=True)
with open('/home/kensei/.hermes/runbooks/calendar-brief/2026-05-12/outlook_events.json','w', encoding='utf-8') as f:
    json.dump({'events': all_events, 'issues': issues}, f, indent=2)
