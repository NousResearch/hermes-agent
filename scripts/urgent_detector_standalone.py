#!/usr/bin/env python3
"""Urgent detector — sahil_ss@outlook.com, last 90 minutes, read-only.

Rule-based: subject patterns + sender filter. No LLM. DRY RUN: never mutates.
Exits 0 always. Output: JSON to stdout, HTML+MEDIA tag if matches.
"""
import json, os, sys, datetime as dt, re, requests, html

TZ = dt.timezone(dt.timedelta(hours=1))  # Europe/London (BST Jul)
now = dt.datetime.now(TZ)
now_str = now.strftime('%d/%m/%y %H:%M:%S')
cutoff_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=90)).strftime('%Y-%m-%dT%H:%M:%SZ')
now_unix = int(dt.datetime.now(dt.timezone.utc).timestamp())

# Subject trigger regex (must match one)
TRIGGERS = [
    (r'\binterview\b', 'interview'),
    (r'\bscreening\b', 'screening'),
    (r'\bphone call\b', 'phone call'),
    (r'\bschedule\b.*\b(interview|call|meeting)\b', 'schedule+meeting'),
    (r'\bnext step(s)?\b', 'next step'),
    (r'\b(offer letter|job offer|pleased to offer)\b', 'offer'),
    (r'\bassessment\b.*\b(invite|invitation|booked)\b', 'assessment'),
    (r'\btechnical\b.*\b(test|challenge|exercise)\b', 'technical'),
]

REJECT_SENDERS = re.compile(
    r'noreply@|donotreply@|no-reply@|notifications@|alerts@|digest@|daily@|'
    r'mail@indeed\.com|@cv-library\.co\.uk|@linkedin\.com|@reed\.co\.uk|@jobmails\.io|'
    r'@stepstone\.co\.uk|@adzuna\.co\.uk|@totaljobs\.com|@monster\.co\.uk',
    re.I,
)

HUMAN_SENDER = re.compile(r'^[A-Za-z][A-Za-z\.\-_]*@[A-Za-z0-9][A-Za-z0-9\.\-]*\.[a-z]{2,}')
HUMAN_VERB = re.compile(r'\b(call|chat|reach out|connect|available|interview)\b', re.I)

TOKEN_CACHE = '/home/kensei/.config/ms-365-mcp-server/token-cache.json'
MS_CLIENT_ID = '084a3e9f-a9f4-43f7-89f9-d229cf97853e'
MS_SCOPES = 'Calendars.ReadWrite Mail.ReadWrite Mail.Send MailboxSettings.Read MailboxSettings.ReadWrite openid profile'
GRAPH = 'https://graph.microsoft.com/v1.0'


def load_token_cache():
    with open(TOKEN_CACHE, encoding='utf-8') as f:
        return json.load(f)


def save_token_cache(top):
    with open(TOKEN_CACHE, 'w', encoding='utf-8') as f:
        json.dump(top, f)


def find_account(username):
    """Return (home_account_id, refresh_token, access_token, expires_on) for username or Nones."""
    top = load_token_cache()
    inner = json.loads(top['data'])
    haid = None
    for k, v in inner.get('Account', {}).items():
        if v.get('username', '').lower() == username.lower():
            haid = v['home_account_id']
            break
    if not haid:
        return None, None, None, 0
    refresh = access = ''
    expires_on = 0
    for k, v in inner.get('RefreshToken', {}).items():
        if v['home_account_id'] == haid:
            refresh = v.get('secret', '')
            break
    for k, v in inner.get('AccessToken', {}).items():
        if v['home_account_id'] == haid:
            access = v.get('secret', '')
            try:
                expires_on = int(v.get('expires_on', 0))
            except (TypeError, ValueError):
                expires_on = 0
            break
    return haid, refresh, access, expires_on


def refresh_token(haid, refresh_token_value):
    """Refresh and persist new access (and refresh) token. Returns new access token or None."""
    r = requests.post(
        'https://login.microsoftonline.com/common/oauth2/v2.0/token',
        data={
            'client_id': MS_CLIENT_ID,
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token_value,
            'scope': MS_SCOPES,
        },
        timeout=30,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    access = data.get('access_token')
    if not access:
        return None
    top = load_token_cache()
    inner = json.loads(top['data'])
    for k, v in inner.get('AccessToken', {}).items():
        if v['home_account_id'] == haid:
            v['secret'] = access
            v['expires_on'] = str(now_unix + data.get('expires_in', 3600))
            break
    if data.get('refresh_token'):
        for k, v in inner.get('RefreshToken', {}).items():
            if v['home_account_id'] == haid:
                v['secret'] = data['refresh_token']
                break
    top['data'] = json.dumps(inner)
    save_token_cache(top)
    return access


def get_access_token(username):
    """Return valid access token, refreshing if near-expiry."""
    haid, refresh, access, expires_on = find_account(username)
    if not haid:
        return None, None
    if not access or expires_on < now_unix + 300:
        access = refresh_token(haid, refresh) if refresh else None
    return haid, access


def graph_get(headers, path, params=None):
    r = requests.get(f'{GRAPH}{path}', headers=headers, params=params, timeout=30)
    if r.status_code >= 400:
        return None
    return r.json()


def find_folder_ids(folders_data):
    inbox_id = job_apps_id = None
    for f in folders_data.get('value', []):
        name = f.get('displayName', '')
        if name == 'Inbox':
            inbox_id = f['id']
        if name == 'Job Applications':
            job_apps_id = f['id']
    return inbox_id, job_apps_id


def evaluate_message(m, job_apps_id):
    """Return matched-label string or None for one message dict."""
    ea = ((m.get('from') or {}).get('emailAddress')) or {}
    sender_email = (ea.get('address') or '').lower()
    subject = m.get('subject') or ''
    if REJECT_SENDERS.search(sender_email):
        return None
    for pat, label in TRIGGERS:
        if re.search(pat, subject, re.I):
            return label
    in_job_apps = bool(job_apps_id) and m.get('parentFolderId') == job_apps_id
    is_human = bool(HUMAN_SENDER.match(sender_email))
    if in_job_apps and is_human:
        return 'JobApps+human'
    if is_human and HUMAN_VERB.search(subject):
        return 'human+verb'
    return None


def render_alert(matches, all_msgs, skipped, html_path):
    """Write HTML and return Discord-ready alert text."""

    def esc(s):
        return html.escape(str(s)) if s else ''

    cards = []
    for m in matches:
        rec = ''
        try:
            rec_dt = dt.datetime.fromisoformat(m['received'].replace('Z', '+00:00'))
            rec = rec_dt.astimezone(TZ).strftime('%d/%m/%y %H:%M')
        except Exception:
            rec = m['received']
        body_safe = m['body_html'] or esc(m['body_preview'][:600])
        cards.append(f'''
<div class="card">
  <h2>🎯 {esc(m['subject'])}</h2>
  <div class="field"><span class="label">From:</span> <span class="value">{esc(m['sender'])} &lt;{esc(m['sender_email'])}&gt;</span></div>
  <div class="field"><span class="label">Subject:</span> <span class="value">{esc(m['subject'])}</span></div>
  <div class="field"><span class="label">Received:</span> <span class="value">{esc(rec)}</span></div>
  <div class="field"><span class="label">Matched:</span> <span class="tag urgent">{esc(m['matched'])}</span></div>
  <div class="body">{body_safe}</div>
</div>''')

    html_doc = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="color-scheme" content="dark">
<title>Urgent Detector — {now_str}</title><style>
body {{ background:#11100f; color:#f5f5f4; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; max-width:720px; margin:2rem auto; padding:0 1rem; }}
h1 {{ color:#fbbf24; font-size:1.4rem; margin-bottom:0.25rem; }}
.meta {{ color:#a8a29e; font-size:0.85rem; margin-bottom:1.5rem; }}
.card {{ background:#1c1a18; border:1px solid #34302c; border-radius:8px; padding:1rem; margin-bottom:1rem; }}
.card h2 {{ color:#fbbf24; font-size:1.1rem; margin:0 0 0.5rem; }}
.field {{ margin:0.25rem 0; }}
.label {{ color:#a8a29e; font-size:0.8rem; }}
.value {{ color:#f5f5f4; }}
.body {{ margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid #34302c; color:#d4d4d4; font-size:0.9rem; white-space:pre-wrap; }}
.tag {{ display:inline-block; background:#fbbf24; color:#11100f; font-size:0.7rem; font-weight:700; padding:0.15rem 0.5rem; border-radius:4px; text-transform:uppercase; }}
.tag.urgent {{ background:#ef4444; color:#fff; }}
.summary {{ background:#2c2a28; border-radius:8px; padding:0.75rem 1rem; margin-bottom:1rem; }}
.summary span {{ color:#a8a29e; }}
</style></head><body>
<h1>🔴 Urgent Job Hunt Alert</h1>
<div class="meta">{now_str} · sahil_ss@outlook.com · 90-min window</div>
<div class="summary"><strong>{len(matches)} urgent match{'es' if len(matches)>1 else ''}</strong> · <span>{skipped} non-urgent skipped</span></div>
{''.join(cards)}
<div class="meta" style="margin-top:1.5rem;">
<p>Run: {now_str} · Emails checked: {len(all_msgs)} (last 90 min) · Matches: {len(matches)}</p>
<p>Source: sahil_ss@outlook.com · Urgent Detector cron</p></div>
</body></html>'''
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)

    discord_lines = [f'🔴 @Sahil **Urgent Job Hunt Alert** · {now.strftime("%H:%M")}']
    for m in matches[:3]:
        rec_short = ''
        try:
            rec_dt = dt.datetime.fromisoformat(m['received'].replace('Z', '+00:00'))
            rec_short = rec_dt.astimezone(TZ).strftime('%d/%m %H:%M')
        except Exception:
            rec_short = m['received']
        who_role = m['sender'] or m['sender_email'].split('@')[0]
        who_role = f"{who_role} — {m['subject']}" if m['subject'] else who_role
        ask = (m['body_preview'] or '').split('\n')[0][:140].strip()
        discord_lines.append(f"• **{who_role}** — {ask}")
        discord_lines.append(f"  ({rec_short})")
    discord_lines.append(f"MEDIA:{html_path}")
    return '\n'.join(discord_lines)


def collect_recent_messages(headers, inbox_id, job_apps_id):
    """Return (inbox_msgs, job_msgs, all_msgs)."""
    params = {
        '$top': 25,
        '$select': 'id,subject,from,receivedDateTime,bodyPreview,parentFolderId,conversationId',
        '$orderby': 'receivedDateTime desc',
        '$filter': f'receivedDateTime ge {cutoff_utc}',
    }
    inbox_msgs = (graph_get(headers, f'/me/mailFolders/{inbox_id}/messages', params) or {}).get('value', [])
    job_msgs = []
    if job_apps_id:
        job_msgs = (graph_get(headers, f'/me/mailFolders/{job_apps_id}/messages', params) or {}).get('value', [])
    return inbox_msgs, job_msgs, inbox_msgs + job_msgs


def enrich_match(headers, m, label, job_apps_id):
    """Fetch full body for a match and return a match dict."""
    full = graph_get(headers, f"/me/messages/{m['id']}",
                     params={'$select': 'id,subject,from,body,receivedDateTime'})
    body_html = ''
    if full:
        body_obj = full.get('body', {}) or {}
        body_html = body_obj.get('content', '') if body_obj.get('contentType') == 'html' \
            else html.escape(body_obj.get('content', ''))
    ea = ((m.get('from') or {}).get('emailAddress')) or {}
    return {
        'id': m['id'],
        'subject': m.get('subject') or '',
        'sender': ea.get('name') or '',
        'sender_email': (ea.get('address') or '').lower(),
        'received': m.get('receivedDateTime', ''),
        'matched': label,
        'in_job_apps': bool(job_apps_id) and m.get('parentFolderId') == job_apps_id,
        'body_html': body_html,
        'body_preview': m.get('bodyPreview') or '',
    }


def main():
    haid, access = get_access_token('sahil_ss@outlook.com')
    if not haid or not access:
        print(json.dumps({'error': 'auth failed for sahil_ss@outlook.com'}))
        return

    headers = {'Authorization': f'Bearer {access}'}
    folders = graph_get(headers, '/me/mailFolders')
    if not folders:
        print(json.dumps({'error': 'no folder access'}))
        return
    inbox_id, job_apps_id = find_folder_ids(folders)
    if not inbox_id:
        print(json.dumps({'error': 'Inbox not found'}))
        return

    inbox_msgs, job_msgs, all_msgs = collect_recent_messages(headers, inbox_id, job_apps_id)

    matches = []
    skipped = 0
    for m in all_msgs:
        sender_email = (((m.get('from') or {}).get('emailAddress')) or {}).get('address', '').lower()
        if REJECT_SENDERS.search(sender_email):
            skipped += 1
            continue
        label = evaluate_message(m, job_apps_id)
        if not label:
            continue
        matches.append(enrich_match(headers, m, label, job_apps_id))

    if not matches:
        print(json.dumps({
            'run_at': now_str,
            'cutoff_utc': cutoff_utc,
            'inbox_count': len(inbox_msgs),
            'job_apps_count': len(job_msgs),
            'matches': [],
            'skipped_rejected_sender': skipped,
        }))
        return

    date_dir = now.strftime('%Y-%m-%d')
    out_dir = f'/home/kensei/.hermes/runbooks/mailbox-cleaner/{date_dir}'
    os.makedirs(out_dir, exist_ok=True)
    html_path = f'{out_dir}/urgent-detector.html'
    print(render_alert(matches, all_msgs, skipped, html_path))


if __name__ == '__main__':
    main()
