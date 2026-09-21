#!/usr/bin/env python3
"""Main mailbox cleaner — direct API approach (DRY RUN ONLY).

Reads last 24h of mail from 6 inboxes (3 Gmail + 3 Outlook),
categorises per inbox-specific rules, produces HTML report + Discord summary.
NO mutations applied — read-only.
"""
import json
import os
import re
import sys
import datetime as dt
import urllib.parse
import requests

TZ = dt.timezone(dt.timedelta(hours=1))  # UK BST (will switch to GMT in winter)
NOW = dt.datetime.now(TZ)
TODAY = NOW.strftime('%d/%m/%y')
NOW_TS = NOW.strftime('%d/%m/%y %H:%M:%S')
CUTOFF_ISO = (NOW - dt.timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')

GMAIL_CREDS_DIR = '/home/kensei/.google_workspace_mcp/credentials'
MS_CACHE = '/home/kensei/.config/ms-365-mcp-server/token-cache.json'
MS_CLIENT_ID = '084a3e9f-a9f4-43f7-89f9-d229cf97853e'
MS_TOKEN_URL = 'https://login.microsoftonline.com/common/oauth2/v2.0/token'

# === Per-inbox rules ===
# Category assignment is done by simple pattern matching; confidence is per-sender.
GMAIL_ACCOUNTS = [
    'saghir.sahil@gmail.com',
    'sahilsaghir.ss9@gmail.com',
    'fusionfirststudios@gmail.com',
]
OUTLOOK_ACCOUNTS = [
    'sahil_ss9@hotmail.com',
    'sahil_saghir@hotmail.co.uk',
    'matchdaymaestro@outlook.com',
]

# === Gmail API ===
def get_gmail_token(creds):
    """Refresh Gmail access token. Returns (access_token, expires_at_unix)."""
    now_unix = int(dt.datetime.now(dt.timezone.utc).timestamp())
    # Try the cached token first (it's in the JSON)
    token = creds.get('token', '')
    # The cached "token" is the access_token. Check expiry by attempting use; if 401, refresh.
    return token


def refresh_gmail_token(creds):
    """Force-refresh Gmail token."""
    data = {
        'client_id': creds['client_id'],
        'client_secret': creds['client_secret'],
        'refresh_token': creds['refresh_token'],
        'grant_type': 'refresh_token',
    }
    r = requests.post(creds['token_uri'], data=data, timeout=30)
    r.raise_for_status()
    return r.json().get('access_token', '')


def gmail_search(account, query, access_token, max_results=25):
    """Search Gmail with a query string. Returns list of message metadata dicts."""
    headers = {'Authorization': f'Bearer {access_token}'}
    # Get message IDs
    params = {'q': query, 'maxResults': max_results}
    r = requests.get('https://gmail.googleapis.com/gmail/v1/users/me/messages',
                     headers=headers, params=params, timeout=30)
    if r.status_code == 401:
        return None  # Token invalid
    r.raise_for_status()
    return r.json().get('messages', [])


def gmail_get_message(msg_id, access_token):
    """Get a single Gmail message's metadata."""
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'format': 'metadata', 'metadataHeaders': ['From', 'Subject', 'Date']}
    r = requests.get(f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}',
                     headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_gmail_message(msg):
    """Parse Gmail message into normalised dict."""
    headers = {h['name'].lower(): h['value'] for h in msg.get('payload', {}).get('headers', [])}
    from_h = headers.get('from', '')
    # Extract name + email
    m = re.match(r'(.*?)\s*<([^>]+)>', from_h)
    if m:
        sender_name, sender_email = m.group(1).strip().strip('"'), m.group(2).strip()
    else:
        sender_name, sender_email = '', from_h.strip()
    return {
        'id': msg['id'],
        'subject': headers.get('subject', '(no subject)'),
        'from': sender_name or sender_email,
        'from_email': sender_email,
        'date': headers.get('date', ''),
        'internal_ts': int(msg.get('internalDate', 0)) / 1000,
    }


# === Outlook (Graph API) ===
def get_outlook_access_token(account_email):
    """Get or refresh Outlook access token. Returns access_token string."""
    with open(MS_CACHE) as f:
        top = json.load(f)
    inner = json.loads(top['data'])

    target_haid = None
    refresh_token = None
    for k, v in inner.get('Account', {}).items():
        if v.get('username', '').lower() == account_email.lower():
            target_haid = v['home_account_id']
            break
    if not target_haid:
        raise Exception(f'Account {account_email} not in token cache')

    access_token = None
    expires_on = 0
    for k, v in inner.get('AccessToken', {}).items():
        if v.get('home_account_id') == target_haid:
            access_token = v.get('secret', '')
            expires_on = int(v.get('expires_on', 0))
            break

    now_unix = int(dt.datetime.now(dt.timezone.utc).timestamp())
    if access_token and expires_on > now_unix + 300:
        return access_token

    # Need to refresh
    for k, v in inner.get('RefreshToken', {}).items():
        if v.get('home_account_id') == target_haid:
            refresh_token = v.get('secret', '')
            break
    if not refresh_token:
        raise Exception(f'No refresh token for {account_email}')

    data = {
        'client_id': MS_CLIENT_ID,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'scope': ' '.join([
            'openid', 'profile', 'offline_access',
            'Mail.ReadWrite', 'Mail.Send', 'MailboxSettings.ReadWrite',
        ]),
    }
    r = requests.post(MS_TOKEN_URL, data=data, timeout=30)
    r.raise_for_status()
    token_data = r.json()
    new_access = token_data.get('access_token', '')
    if not new_access:
        raise Exception(f'Token refresh failed: {r.text[:200]}')

    # Write back
    for k, v in inner.get('AccessToken', {}).items():
        if v.get('home_account_id') == target_haid:
            v['secret'] = new_access
            v['expires_on'] = str(now_unix + token_data.get('expires_in', 3600))
            break
    if token_data.get('refresh_token'):
        for k, v in inner.get('RefreshToken', {}).items():
            if v.get('home_account_id') == target_haid:
                v['secret'] = token_data['refresh_token']
                break
    top['data'] = json.dumps(inner)
    with open(MS_CACHE, 'w') as f:
        json.dump(top, f)
    return new_access


def outlook_list_inbox(account_email, access_token, max_results=25):
    """List recent inbox messages from Outlook."""
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {
        '$top': max_results,
        '$select': 'id,subject,from,receivedDateTime,bodyPreview,isRead,conversationId',
        '$orderby': 'receivedDateTime desc',
        '$filter': f'receivedDateTime ge {CUTOFF_ISO}',
    }
    r = requests.get('https://graph.microsoft.com/v1.0/me/mailFolders/Inbox/messages',
                     headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get('value', [])


# === Categorisation ===
def categorise_gmail(msg, account):
    """Return (category, confidence, action_text, reason)."""
    sender = (msg.get('from_email') or '').lower()
    subject = msg.get('subject') or ''
    sender_domain = sender.split('@')[-1] if '@' in sender else ''

    cat = None
    conf = 'low'
    reason = ''

    if account == 'saghir.sahil@gmail.com':
        # Aggressive
        if any(x in sender for x in ['amazon', 'tesco', 'stripe', 'apple', 'paypal', 'ebay']):
            cat, conf = 'Receipts', 'high'
        elif any(x in sender for x in ['rightmove', 'zoopla', 'openrent', 'onthemarket']):
            cat, conf = 'Property', 'high'
        elif any(x in sender for x in ['healthchecks', 'uptimerobot', 'statuspage', 'pingdom']):
            cat, conf = 'Infrastructure', 'high'
        elif any(x in sender for x in ['openai', 'anthropic', 'openrouter', 'ollama', 'huggingface', 'x.ai', 'cursor', 'replit', 'claude']):
            cat, conf = 'AI-Tools', 'high'
        elif any(x in sender_domain for x in ['substack.com', 'beehiiv.com', 'mailbrew']):
            cat, conf = 'Newsletter', 'high'
        elif 'github.com' in sender or 'stripe.com' in sender or 'supabase.com' in sender or 'vercel.com' in sender:
            cat, conf = 'Service', 'high'
        elif any(x in sender for x in ['mail.', 'newsletter', 'promo', 'deals', 'sale', 'offer', 'discount']):
            cat, conf = 'Promo', 'medium'
        else:
            cat, conf = 'Uncertain', 'low'

    elif account == 'sahilsaghir.ss9@gmail.com':
        if 'skool.com' in sender or 'skool.community' in sender:
            cat, conf = 'Skool', 'high'
        elif any(x in sender for x in ['github.com', 'railway', 'supabase', 'vercel', 'render']):
            cat, conf = 'Service', 'high'
        elif any(x in sender for x in ['coaching', 'thefa', 'fa.co.uk', 'grassroots']):
            cat, conf = 'Coaching', 'medium'
        elif any(x in sender for x in ['mail.', 'promo', 'deals', 'sale', 'offer', 'discount']):
            cat, conf = 'Promo', 'medium'
        else:
            cat, conf = 'Uncertain', 'low'

    elif account == 'fusionfirststudios@gmail.com':
        if 'apple' in sender and ('developer' in sender or 'appstore' in sender or 'receipt' in subject.lower()):
            cat, conf = 'AppStore', 'high'
        elif 'google' in sender and 'play' in sender:
            cat, conf = 'AppStore', 'high'
        elif 'privacy' in subject.lower() or 'terms' in subject.lower() or 'tos' in subject.lower():
            cat, conf = 'Legal', 'high'
        elif any(x in sender for x in ['stripe.com', 'github.com', 'vercel.com', 'aws.amazon', 'google.com']):
            cat, conf = 'Service', 'medium'
        elif any(x in sender for x in ['amazon', 'paypal', 'invoice']):
            cat, conf = 'Receipts', 'medium'
        else:
            cat, conf = 'Uncertain', 'low'

    # Map category -> proposed action (cat is always set by the if/elif chain above)
    action = CATEGORY_ACTION_GMAIL.get(cat or 'Uncertain', 'Flag in digest')
    return cat, conf, action, reason


CATEGORY_ACTION_GMAIL = {
    'Receipts': 'Label kensei/Receipts, keep in inbox',
    'Promo': 'Label kensei/Promo, propose-delete after 7d',
    'Newsletter': 'Label kensei/Newsletter, auto-archive after read',
    'Service': 'Label kensei/Service, flag in digest',
    'Property': 'Label kensei/Property, flag in digest',
    'AI-Tools': 'Label kensei/AI-Tools, keep',
    'Infrastructure': 'Label kensei/Infrastructure, skip inbox',
    'Skool': 'Label kensei/Skool, propose-delete after 7d',
    'Coaching': 'Label kensei/Coaching, keep',
    'AppStore': 'Label kensei/AppStore, keep',
    'Legal': 'Label kensei/Legal, keep, flag if action-required',
    'Uncertain': 'Flag as Uncertain in digest',
}


def categorise_outlook(msg, account):
    """Return (category, confidence, action_text, reason)."""
    sender = (msg.get('from_email') or '').lower()
    subject = msg.get('subject') or ''
    body = (msg.get('bodyPreview') or '')[:300].lower()
    combined = (subject + ' ' + body).lower()

    cat = None
    conf = 'low'
    reason = ''

    if account == 'sahil_ss9@hotmail.com':
        # Conservative — never auto-delete
        if any(x in sender for x in ['british-gas', 'britishgas', 'severn-trent', 'severntrent', 'council-tax', 'anglian-water', 'eon', 'edf', 'ovo']):
            cat, conf = 'Bills', 'high'
        elif any(x in sender for x in ['babysdays', 'famly', 'gym', 'childminder']):
            cat, conf = 'Childcare', 'medium'
        elif any(x in sender for x in ['nri-legal', 'solicitor', 'conveyancing']):
            cat, conf = 'Legal', 'high'
        elif 'account-security' in sender or 'microsoft-account' in sender or 'security-noreply' in sender:
            cat, conf = 'Security', 'high'
        elif any(x in sender for x in ['substack.com', 'mailbrew', 'beehiiv']):
            cat, conf = 'Newsletter', 'medium'
        elif any(x in sender for x in ['mail.', 'promo', 'deals', 'sale', 'offer', 'discount']):
            cat, conf = 'Promo', 'medium'
        else:
            cat, conf = 'Uncertain', 'low'

    elif account == 'sahil_saghir@hotmail.co.uk':
        if any(x in sender for x in ['holdencopley', 'estate-agent', 'rightmove', 'zoopla', 'savills', 'foxtons', 'haart']):
            cat, conf = 'Property', 'high'
        elif any(x in sender for x in ['mail.', 'promo', 'deals', 'sale', 'offer', 'discount', 'marketing']):
            cat, conf = 'Promo', 'medium'
        else:
            cat, conf = 'Uncertain', 'low'

    elif account == 'matchdaymaestro@outlook.com':
        if 'facebookmail' in sender or 'facebook' in sender or 'instagram' in sender:
            cat, conf = 'Social-Notifications', 'high'
        elif any(x in sender for x in ['kling', 'midjourney', 'openai', 'anthropic', 'ollama']):
            cat, conf = 'AI-Tools', 'medium'
        elif any(x in sender for x in ['mail.', 'promo', 'deals', 'sale', 'offer', 'discount']):
            cat, conf = 'Promo', 'medium'
        else:
            cat, conf = 'Uncertain', 'low'

    action = CATEGORY_ACTION_OUTLOOK.get(cat or 'Uncertain', 'Flag in digest')
    return cat, conf, action, reason


CATEGORY_ACTION_OUTLOOK = {
    'Bills': 'Move to KENSEI/Bills, keep',
    'Childcare': 'Move to KENSEI/Childcare, keep, flag',
    'Legal': 'Move to KENSEI/Legal, keep, flag',
    'Security': 'Move to KENSEI/Security, archive after 30d',
    'Newsletter': 'Move to KENSEI/Newsletter, auto-archive after 7d',
    'Promo': 'Move to KENSEI/Promo, propose-delete after 7d (NEVER auto-delete from sahil_ss9)',
    'Property': 'Move to KENSEI/Property, flag',
    'Social-Notifications': 'Move to KENSEI/Social-Notifications, archive',
    'AI-Tools': 'Move to KENSEI/AI-Tools, keep',
    'Uncertain': 'Flag as Uncertain in digest',
}


# === HTML report ===
def write_html_report(report_path, results):
    """results: dict {account: {category: [msg_dict_with_cat_conf_action]}}"""
    total = sum(len(msgs) for acc in results.values() for msgs in acc.values())
    attention = []
    for acc, by_cat in results.items():
        for cat, msgs in by_cat.items():
            for m in msgs:
                # Action-required for high conf + certain categories
                if cat in ('Uncertain',) or m.get('conf') == 'medium':
                    attention.append((acc, m))

    html = f'''<!DOCTYPE html>
<html lang="en" data-color-scheme="dark">
<head>
<meta charset="UTF-8">
<meta name="color-scheme" content="dark">
<style>
* {{margin:0;padding:0;box-sizing:border-box}}
body{{background:#11100f;color:#f5f5f4;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:820px;margin:0 auto;padding:20px;line-height:1.55}}
h1{{color:#fbbf24;font-size:1.5em;margin-bottom:4px}}
h2{{font-size:1.1em;margin-top:24px;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #34302c}}
h2.attention{{color:#ef4444}}
h2.organised{{color:#22c55e}}
h2.promo{{color:#f97316}}
.card{{background:#1c1a18;border:1px solid #34302c;border-radius:8px;padding:12px 16px;margin:8px 0}}
.card.attention{{border-left:3px solid #ef4444}}
.card.organised{{border-left:3px solid #22c55e}}
.card.promo{{border-left:3px solid #f97316}}
.subject{{font-weight:600}}
.tag{{display:inline-block;background:#2c2a28;color:#fbbf24;font-size:0.75em;padding:2px 8px;border-radius:4px;margin-right:4px}}
.tag.high{{background:#166534;color:#86efac}}
.tag.medium{{background:#713f12;color:#fde68a}}
.tag.low{{background:#7f1d1d;color:#fca5a5}}
.tag.account{{background:#1e3a8a;color:#93c5fd}}
.sender{{color:#a8a29e;font-size:0.85em}}
.meta{{color:#a8a29e;font-size:0.8em;margin-top:16px;border-top:1px solid #34302c;padding-top:8px}}
ul{{margin:4px 0;padding-left:20px}}
li{{margin:2px 0}}
.summary-line{{color:#a8a29e;margin-top:0}}
</style>
</head>
<body>
<h1>📬 Mailbox Cleaner — {TODAY}</h1>
<p class="summary-line">{total} new overnight · DRY RUN (no mutations) · {len(attention)} need your eye</p>
'''
    if attention:
        html += '<h2 class="attention">🚨 Needs your attention</h2>\n'
        for acc, m in attention[:10]:
            html += f'''<div class="card attention">
  <div><span class="tag account">{acc.split("@")[0]}</span> <span class="subject">{m.get("subject", "(no subject)")[:120]}</span></div>
  <div class="sender">{m.get("from","?")} &lt;{m.get("from_email","?")}&gt; — {m.get("date","")}</div>
  <div><span class="tag {m.get("conf","low")}">{m.get("conf","low").upper()}</span> {m.get("cat","?")} — {m.get("action","")}</div>
</div>
'''
        if len(attention) > 10:
            html += f'<p class="meta">Plus {len(attention)-10} more — reply "review" to see all.</p>\n'

    # Per-account summary
    for acc, by_cat in results.items():
        count = sum(len(v) for v in by_cat.values())
        if count == 0:
            continue
        html += f'<h2 class="organised">✅ {acc}</h2>\n'
        html += f'<div class="card organised"><strong>{count} messages</strong><ul>\n'
        for cat, msgs in by_cat.items():
            if not msgs:
                continue
            sample = msgs[0]
            html += f'<li>{len(msgs)} × {cat} → {sample.get("action","")[:80]}'
            if cat in ('Uncertain',) or sample.get('conf') == 'medium':
                html += f' <em>({sample.get("conf","?")} conf: {sample.get("subject","")[:60]})</em>'
            html += '</li>\n'
        html += '</ul></div>\n'

    # Promos queued
    promo_msgs = []
    for acc, by_cat in results.items():
        for m in by_cat.get('Promo', []):
            promo_msgs.append((acc, m))
    if promo_msgs:
        html += '<h2 class="promo">🗑️ Proposed for deletion (dry run)</h2>\n'
        html += '<div class="card promo"><ul>\n'
        for acc, m in promo_msgs:
            html += f'<li>{m.get("subject","?")[:80]} — from {m.get("from","?")} ({acc})</li>\n'
        html += '</ul>\n'
        html += '<p class="meta">Reply "delete promos" to confirm, or "review" to see the list. Ignore to auto-release in 7 days.</p></div>\n'

    html += f'''<div class="meta">
  <strong>Inboxes scanned:</strong> 6 (3 Gmail + 3 Outlook)<br>
  <strong>Mode:</strong> DRY RUN — no mutations applied<br>
  <strong>Time:</strong> {NOW_TS} UK<br>
  <strong>Cutoff:</strong> last 24h
</div>
</body>
</html>'''
    with open(report_path, 'w') as f:
        f.write(html)


# === Main ===
def main():
    results = {}
    errors = []

    # --- Gmail ---
    for account in GMAIL_ACCOUNTS:
        try:
            with open(f'{GMAIL_CREDS_DIR}/{account}.json') as f:
                creds = json.load(f)
            access_token = get_gmail_token(creds)
            # Use `newer_than:1d` query
            msg_list = gmail_search(account, 'newer_than:1d', access_token, max_results=25)
            if msg_list is None:
                # Token expired, refresh
                access_token = refresh_gmail_token(creds)
                msg_list = gmail_search(account, 'newer_than:1d', access_token, max_results=25)
            by_cat = {}
            for m in (msg_list or [])[:25]:
                full = gmail_get_message(m['id'], access_token)
                parsed = parse_gmail_message(full)
                cat, conf, action, reason = categorise_gmail(parsed, account)
                parsed['cat'] = cat
                parsed['conf'] = conf
                parsed['action'] = action
                by_cat.setdefault(cat, []).append(parsed)
            results[account] = by_cat
        except Exception as e:
            errors.append(f'{account}: {e}')
            results[account] = {}

    # --- Outlook ---
    for account in OUTLOOK_ACCOUNTS:
        try:
            access_token = get_outlook_access_token(account)
            messages = outlook_list_inbox(account, access_token, max_results=25)
            by_cat = {}
            for m in messages:
                # Parse from
                frm = m.get('from', {})
                if isinstance(frm, dict):
                    email_obj = frm.get('emailAddress', {})
                    if isinstance(email_obj, dict):
                        sender_email = email_obj.get('address', '')
                        sender_name = email_obj.get('name', '')
                    else:
                        sender_email, sender_name = '', ''
                else:
                    sender_email, sender_name = '', ''
                parsed = {
                    'id': m['id'],
                    'subject': m.get('subject', '(no subject)'),
                    'from': sender_name or sender_email,
                    'from_email': sender_email,
                    'date': m.get('receivedDateTime', ''),
                    'bodyPreview': m.get('bodyPreview', ''),
                }
                cat, conf, action, reason = categorise_outlook(parsed, account)
                parsed['cat'] = cat
                parsed['conf'] = conf
                parsed['action'] = action
                by_cat.setdefault(cat, []).append(parsed)
            results[account] = by_cat
        except Exception as e:
            errors.append(f'{account}: {e}')
            results[account] = {}

    # --- Generate report ---
    report_dir = f'/home/kensei/.hermes/runbooks/mailbox-cleaner/{NOW.strftime("%Y-%m-%d")}'
    os.makedirs(report_dir, exist_ok=True)
    report_path = f'{report_dir}/main-cleaner.html'
    write_html_report(report_path, results)

    # --- Discord summary ---
    total = sum(len(msgs) for acc in results.values() for msgs in acc.values())
    attention_count = 0
    for acc, by_cat in results.items():
        for cat, msgs in by_cat.items():
            if cat == 'Uncertain':
                attention_count += len(msgs)
            else:
                for m in msgs:
                    if m.get('conf') in ('medium', 'low'):
                        attention_count += 1

    print(f'📬 **Mailbox Cleaner** · {TODAY} · 06:00')
    print(f'{total} new overnight · DRY RUN (no mutations) · {attention_count} need your eye')

    if attention_count > 0:
        print('')
        print('**🚨 Needs your attention**')
        for acc, by_cat in results.items():
            for cat, msgs in by_cat.items():
                if cat == 'Uncertain':
                    for m in msgs[:3]:
                        print(f'• [{acc.split("@")[0]}] {m.get("subject","?")[:80]} — {m.get("from","?")}')
                else:
                    for m in msgs:
                        if m.get('conf') in ('medium', 'low'):
                            print(f'• [{acc.split("@")[0]}] {m.get("subject","?")[:80]} — {cat} ({m.get("conf","?")} conf)')

    if errors:
        print('')
        print(f'⚠️ Errors: {len(errors)}')
        for e in errors[:5]:
            print(f'  • {e}')

    print('')
    print(f'MEDIA:{report_path}')


if __name__ == '__main__':
    main()
