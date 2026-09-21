#!/usr/bin/env python3
"""Direct Graph API job hunt mailbox cleaner - with token refresh."""
import json, os, sys, datetime as dt, requests, re, urllib.parse

TZ = dt.timezone(dt.timedelta(hours=1))
now = dt.datetime.now(TZ)
today = now.strftime('%d/%m/%y')
now_ts = now.strftime('%d/%m/%y %H:%M:%S')
cutoff = now - dt.timedelta(hours=24)

# Load token cache
cache_path = '/home/kensei/.config/ms-365-mcp-server/token-cache.json'
if not os.path.exists(cache_path):
    print("ERROR: Token cache not found", file=sys.stderr)
    sys.exit(1)

with open(cache_path, encoding='utf-8') as f:
    top = json.load(f)
inner = json.loads(top['data'])

# Find sahil_ss@outlook.com account
target_account = 'Sahil_SS@outlook.com'
target_haid = None
for k, v in inner.get('Account', {}).items():
    if v.get('username', '').lower() == target_account.lower():
        target_haid = v['home_account_id']
        break

if not target_haid:
    print(f"ERROR: Account {target_account} not found", file=sys.stderr)
    sys.exit(1)

# Get refresh token
refresh_token = None
for k, v in inner.get('RefreshToken', {}).items():
    if v['home_account_id'] == target_haid:
        refresh_token = v.get('secret', '')
        break

if not refresh_token:
    print(f"ERROR: No refresh token for {target_account}", file=sys.stderr)
    sys.exit(1)

# Get existing access token to check expiry
access_token = None
for k, v in inner.get('AccessToken', {}).items():
    if v['home_account_id'] == target_haid:
        access_token = v.get('secret', '')
        expires_on = int(v.get('expires_on', 0))
        break

# Refresh if expired (within 5 min buffer)
now_ts_unix = int(dt.datetime.now(dt.timezone.utc).timestamp())
if not access_token or expires_on < now_ts_unix + 300:
    print(f"Token expired (expires: {expires_on}, now: {now_ts_unix}), refreshing...", file=sys.stderr)
    # MSAL token refresh endpoint
    token_url = 'https://login.microsoftonline.com/common/oauth2/v2.0/token'
    # We need the client_id - it's embedded in the MCP server config
    # Common MSAL public client IDs for M365: d50a0a66-6c7f-4e3a-8a4f-0e0a0e0a0e0a
    # Let's try the common one used by ms-365-mcp-server
    client_id = '084a3e9f-a9f4-43f7-89f9-d229cf97853e'  # ms-365-mcp-server default global client
    
    data = {
        'client_id': client_id,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'scope': 'Calendars.ReadWrite Mail.ReadWrite Mail.Send MailboxSettings.Read MailboxSettings.ReadWrite openid profile'
    }
    
    r = requests.post(token_url, data=data, timeout=30)
    if r.status_code != 200:
        print(f"Token refresh failed: {r.status_code} {r.text[:200]}", file=sys.stderr)
        sys.exit(1)
    
    token_data = r.json()
    access_token = token_data.get('access_token', '')
    if not access_token:
        print(f"ERROR: No access_token in refresh response", file=sys.stderr)
        sys.exit(1)
    
    # Update cache
    for k, v in inner.get('AccessToken', {}).items():
        if v['home_account_id'] == target_haid:
            v['secret'] = access_token
            v['expires_on'] = str(now_ts_unix + token_data.get('expires_in', 3600))
            break
    
    # Update refresh token if provided
    if token_data.get('refresh_token'):
        for k, v in inner.get('RefreshToken', {}).items():
            if v['home_account_id'] == target_haid:
                v['secret'] = token_data['refresh_token']
                break
    
    # Write back
    top['data'] = json.dumps(inner)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(top, f)
    print(f"Token refreshed successfully", file=sys.stderr)

headers = {'Authorization': f'Bearer {access_token}'}

def graph_get(path, params=None):
    url = f'https://graph.microsoft.com/v1.0{path}'
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code == 401:
        print(f"ERROR: Token rejected for {target_account}", file=sys.stderr)
        sys.exit(1)
    r.raise_for_status()
    return r.json()

# Step 1: Get folders
print("Fetching folders...", file=sys.stderr)
folders_data = graph_get('/me/mailFolders')
folders = {}
for f in folders_data.get('value', []):
    folders[f['displayName']] = f['id']

print(f"Found {len(folders_data.get('value', []))} folders", file=sys.stderr)

# Find "Job Applications" folder
job_app_folder_id = None
for f in folders_data.get('value', []):
    if f['displayName'] == 'Job Applications':
        job_app_folder_id = f['id']
        print(f"Found 'Job Applications' folder: {job_app_folder_id}", file=sys.stderr)
        break

# Step 2: Get inbox messages
print("Fetching inbox messages...", file=sys.stderr)
inbox_id = None
for f in folders_data.get('value', []):
    if f['displayName'] == 'Inbox':
        inbox_id = f['id']
        break

if not inbox_id:
    print("ERROR: Inbox folder not found", file=sys.stderr)
    sys.exit(1)

# Get messages - try unread first, then recent
params = {
    '$top': 50,
    '$select': 'id,subject,from,receivedDateTime,bodyPreview,isRead,conversationId',
    '$orderby': 'receivedDateTime desc',
    '$filter': 'isRead eq false'
}
msgs_data = graph_get(f'/me/mailFolders/{inbox_id}/messages', params)
messages = msgs_data.get('value', [])

print(f"Found {len(messages)} unread messages", file=sys.stderr)

if not messages:
    # Try recent messages
    params2 = {
        '$top': 50,
        '$select': 'id,subject,from,receivedDateTime,bodyPreview,isRead,conversationId',
        '$orderby': 'receivedDateTime desc',
        '$filter': f'receivedDateTime ge {cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")}'
    }
    msgs_data2 = graph_get(f'/me/mailFolders/{inbox_id}/messages', params2)
    messages = msgs_data2.get('value', [])
    print(f"Found {len(messages)} recent messages (last 24h)", file=sys.stderr)

if not messages:
    print("ZERO_NEW_EMAILS")
    sys.exit(0)

# Step 3: Categorize
def get_sender(msg):
    frm = msg.get('from', {})
    if isinstance(frm, dict):
        email = frm.get('emailAddress', {})
        if isinstance(email, dict):
            return email.get('address', ''), email.get('name', '')
    return '', ''

def is_noreply(sender_email):
    return bool(re.search(r'noreply|donotreply|no-reply', sender_email, re.I))

def categorize(msg):
    subject = msg.get('subject', '') or ''
    body = msg.get('bodyPreview', '') or ''
    sender_email, sender_name = get_sender(msg)
    combined = (subject + ' ' + body).lower()
    
    # Active: interview, screening, next step, phone call, schedule
    active_patterns = [
        r'interview', r'screening', r'phone call', r'phone screen',
        r'next step', r'schedule', r'technical test', r'take.home',
        r'offer', r'pleased to offer', r'compensation', r'start date',
        r'second interview', r'final interview', r'assessment centre',
        r'video call', r'meet the team', r'culture fit'
    ]
    for p in active_patterns:
        if re.search(p, combined):
            return 'Active', f'Matched: {p}'
    
    # Rejected
    reject_patterns = [
        r'unfortunately', r'not progressing', r'other candidates',
        r'not successful', r'not moving forward', r'decided to proceed with',
        r'we will not be', r'regret to inform', r'after careful consideration.*not'
    ]
    for p in reject_patterns:
        if re.search(p, combined):
            return 'Rejected', f'Matched: {p}'
    
    # Applied: application confirmation
    applied_patterns = [
        r'thank you for applying', r'we received your application',
        r'application received', r'application submitted',
        r'your application has been', r'apply for.*position',
        r'we have received', r'application confirmation'
    ]
    for p in applied_patterns:
        if re.search(p, combined):
            return 'Applied', f'Matched: {p}'
    
    # Recruiter: direct human outreach
    if not is_noreply(sender_email) and not re.search(r'linkedin|indeed|reed|cv-library', sender_email, re.I):
        recruiter_patterns = [
            r'opportunity', r'role', r'position', r'your profile',
            r'came across', r'impressed', r'vacancy', r'looking for',
            r'would you be interested', r'reach out', r'get in touch'
        ]
        for p in recruiter_patterns:
            if re.search(p, combined):
                return 'Recruiter', f'Matched: {p}'
    
    # JobAlerts: automated digests
    alert_senders = ['linkedin', 'indeed', 'reed.co.uk', 'cv-library', 'jobsite', 'totaljobs']
    for s in alert_senders:
        if s in sender_email.lower():
            return 'JobAlerts', f'Sender: {sender_email}'
    alert_subjects = ['job alert', 'job digest', 'new jobs', 'recommended jobs', 'jobs for you']
    for s in alert_subjects:
        if s in subject.lower():
            return 'JobAlerts', f'Subject: {subject}'
    
    return 'Uncertain', 'No clear pattern match'

categorized = {'Active': [], 'Applied': [], 'Rejected': [], 'JobAlerts': [], 'Recruiter': [], 'Uncertain': []}

for msg in messages:
    cat, reason = categorize(msg)
    sender_email, sender_name = get_sender(msg)
    entry = {
        'id': msg['id'],
        'subject': msg.get('subject', 'No subject'),
        'from': sender_name or sender_email,
        'from_email': sender_email,
        'received': msg.get('receivedDateTime', ''),
        'body_preview': (msg.get('bodyPreview', '') or '')[:200],
        'is_read': msg.get('isRead', False),
        'reason': reason
    }
    categorized[cat].append(entry)

# Step 4: Generate HTML report
report_dir = f'/home/kensei/.hermes/runbooks/mailbox-cleaner/{now.strftime("%Y-%m-%d")}'
os.makedirs(report_dir, exist_ok=True)
report_path = f'{report_dir}/jobhunt-cleaner.html'

total = sum(len(v) for v in categorized.values())

def extract_apply_links(body):
    urls = re.findall(r'https?://[^\s<>"\']+', body or '')
    return urls[:3]

html = f'''<!DOCTYPE html>
<html lang="en" data-color-scheme="dark">
<head>
<meta charset="UTF-8">
<meta name="color-scheme" content="dark">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#11100f;color:#f5f5f4;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:24px;line-height:1.6}}
h1{{color:#fbbf24;font-size:1.5em;margin-bottom:4px}}
h2{{color:#fbbf24;font-size:1.2em;margin:20px 0 8px;padding-bottom:4px;border-bottom:1px solid #34302c}}
.meta{{color:#a8a29e;font-size:0.85em;margin-bottom:16px}}
.card{{background:#1c1a18;border:1px solid #34302c;border-radius:8px;padding:12px;margin-bottom:8px}}
.card.active{{border-left:3px solid #22c55e}}
.card.applied{{border-left:3px solid #3b82f6}}
.card.rejected{{border-left:3px solid #ef4444}}
.card.recruiter{{border-left:3px solid #f59e0b}}
.card.alert{{border-left:3px solid #8b5cf6}}
.card.uncertain{{border-left:3px solid #a8a29e}}
.subject{{font-weight:600;color:#f5f5f4}}
.sender{{color:#a8a29e;font-size:0.85em}}
.time{{color:#78716c;font-size:0.8em}}
.reason{{color:#a8a29e;font-size:0.8em;margin-top:4px}}
.preview{{color:#a8a29e;font-size:0.85em;margin-top:4px;border-top:1px solid #2c2a28;padding-top:4px}}
.count{{display:inline-block;background:#2c2a28;color:#fbbf24;border-radius:12px;padding:0 8px;font-size:0.85em;margin-left:6px}}
.empty{{color:#78716c;font-style:italic;padding:8px 0}}
.section{{margin-bottom:16px}}
</style>
</head>
<body>
<h1>🎯 Job Hunt Digest</h1>
<div class="meta">{today} · 06:06 · sahil_ss@outlook.com</div>
<div class="meta">{total} new · {len(categorized['Active'])} active · {len(categorized['Applied'])} applied · {len(categorized['Rejected'])} rejected · {len(categorized['JobAlerts'])} alerts · {len(categorized['Recruiter'])} recruiter · {len(categorized['Uncertain'])} uncertain</div>
'''

for section_key, section_title, section_class in [
    ('Active', '🟢 Active Opportunities', 'active'),
    ('Applied', '📋 New Applications', 'applied'),
    ('Rejected', '❌ Rejections', 'rejected'),
    ('Recruiter', '📩 Recruiter Outreach', 'recruiter'),
    ('JobAlerts', '📬 Job Alerts', 'alert'),
    ('Uncertain', '❓ Uncertain', 'uncertain')
]:
    items = categorized[section_key]
    if not items:
        continue
    html += f'<div class="section"><h2>{section_title} <span class="count">{len(items)}</span></h2>'
    for m in items:
        links_html = ''
        if section_key == 'JobAlerts':
            links = extract_apply_links(m['body_preview'])
            if links:
                links_html = '<div class="links">' + ' · '.join(f'<a href="{l}" style="color:#3b82f6">{l[:50]}...</a>' for l in links) + '</div>'
        html += f'''<div class="card {section_class}">
<div class="subject">{(m['subject'] or 'No subject')}</div>
<div class="sender">{m['from']} &lt;{m['from_email']}&gt;</div>
<div class="time">{m['received']}</div>
{links_html}
<div class="reason">ⓘ {m['reason']}</div>
</div>'''
    html += '</div>'

html += f'''
<div class="meta" style="margin-top:24px;padding-top:12px;border-top:1px solid #34302c">
Run: {now_ts} · DRY RUN — no mutations performed<br>
Inbox: sahil_ss@outlook.com · Last 24h only
</div>
</body>
</html>'''

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"HTML report written to {report_path}", file=sys.stderr)

# Output summary for Discord
print("---SUMMARY---")
print(f"🎯 **Job Hunt Digest** · {today} · 06:06")
print(f"{total} new · {len(categorized['Active'])} active · {len(categorized['Applied'])} applied · {len(categorized['Rejected'])} rejected · {len(categorized['JobAlerts'])} alerts · {len(categorized['Recruiter'])} recruiter · {len(categorized['Uncertain'])} uncertain")

if categorized['Active']:
    print(f"\n**Active opportunities** ({len(categorized['Active'])})")
    for m in categorized['Active']:
        print(f"• {(m['subject'] or 'No subject')[:80]} — from {m['from']}")

if categorized['Applied']:
    print(f"\n**New applications** ({len(categorized['Applied'])})")
    for m in categorized['Applied']:
        print(f"• {(m['subject'] or 'No subject')[:80]} — {m['from']}")

if categorized['Rejected']:
    print(f"\n**Rejections** ({len(categorized['Rejected'])})")
    for m in categorized['Rejected']:
        print(f"• {(m['subject'] or 'No subject')[:80]} — {m['from']}")

if categorized['Recruiter']:
    print(f"\n**Recruiter outreach** ({len(categorized['Recruiter'])})")
    for m in categorized['Recruiter']:
        print(f"• {(m['subject'] or 'No subject')[:80]} — {m['from']}")

if categorized['JobAlerts']:
    print(f"\n**Job alerts overnight** ({len(categorized['JobAlerts'])})")
    for m in categorized['JobAlerts'][:5]:
        print(f"• {(m['subject'] or 'No subject')[:80]} — {m['from']}")

if categorized['Uncertain']:
    print(f"\n**Uncategorised** ({len(categorized['Uncertain'])})")
    for m in categorized['Uncertain']:
        print(f"• {(m['subject'] or 'No subject')[:80]} — {m['from']}")

print(f"\nMEDIA:{report_path}")
