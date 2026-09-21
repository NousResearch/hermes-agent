#!/usr/bin/env python3
"""
Full mailbox digest - Gmail + Outlook.
Fetches unread message IDs, gets metadata, categorises.
"""
import json, subprocess, os, time, select, sys, re
from datetime import datetime, timezone

def get_env_val(key):
    env_path = '/home/kensei/.hermes/.env'
    with open(env_path) as f:
        for line in f:
            if line.startswith(key + '=') and not line.startswith('#'):
                val = line.split('=', 1)[1].strip()
                if val and '***' not in val:
                    return val
    return os.environ.get(key, '')

def call_mcp_batch(server_args, requests, env_overrides=None, timeout=30):
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    
    proc = subprocess.Popen(
        server_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    time.sleep(2)
    
    if proc.poll() is not None:
        return None, proc.stderr.read()
    
    init = json.dumps({
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {
            'protocolVersion': '2024-11-05',
            'capabilities': {},
            'clientInfo': {'name': 'mailbox-digest', 'version': '1.0.0'}
        }
    }) + '\n'
    proc.stdin.write(init)
    proc.stdin.flush()
    time.sleep(0.5)
    
    if proc.poll() is not None:
        return None, proc.stderr.read()
    
    notif = json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}}) + '\n'
    proc.stdin.write(notif)
    proc.stdin.flush()
    time.sleep(0.5)
    
    if proc.poll() is not None:
        return None, proc.stderr.read()
    
    for req in requests:
        proc.stdin.write(json.dumps(req) + '\n')
        proc.stdin.flush()
        time.sleep(0.3)
    
    out_data = ''
    deadline = time.time() + timeout
    while time.time() < deadline:
        r, _, _ = select.select([proc.stdout], [], [], 0.5)
        if r:
            line = proc.stdout.readline()
            if line:
                out_data += line
        else:
            break
    
    stderr_data = ''
    while True:
        r, _, _ = select.select([proc.stderr], [], [], 0.3)
        if r:
            line = proc.stderr.readline()
            if line:
                stderr_data += line
            else:
                break
        else:
            break
    
    proc.terminate()
    return out_data, stderr_data

def parse_json_responses(data):
    results = {}
    for line in data.strip().split('\n'):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if 'id' in obj:
                results[obj['id']] = obj
        except:
            pass
    return results

def extract_text_content(result_obj):
    if 'result' in result_obj:
        content = result_obj['result'].get('content', [])
        for c in content:
            if c.get('type') == 'text':
                return c['text']
        return str(result_obj['result'])
    elif 'error' in result_obj:
        return f"ERROR: {result_obj['error']}"
    return str(result_obj)

def extract_message_ids(text):
    ids = re.findall(r'Message ID: (\S+)', text)
    return ids

def gmail_search(email, query, page_size=20):
    client_id = get_env_val('GOOGLE_OAUTH_CLIENT_ID')
    client_secret = get_env_val('GOOGLE_OAUTH_CLIENT_SECRET')
    
    env_overrides = {
        'GOOGLE_OAUTH_CLIENT_ID': client_id,
        'GOOGLE_OAUTH_CLIENT_SECRET': client_secret,
        'OAUTHLIB_INSECURE_TRANSPORT': '1',
        'PATH': '/usr/local/bin:/usr/bin:/bin',
    }
    
    server_bin = '/home/kensei/.local/share/uv/tools/workspace-mcp/bin/workspace-mcp'
    server_args = [server_bin, '--single-user', '--tools', 'gmail']
    
    requests = [
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
            'name': 'search_gmail_messages',
            'arguments': {'query': query, 'user_google_email': email, 'page_size': page_size}
        }}
    ]
    
    out, err = call_mcp_batch(server_args, requests, env_overrides)
    if out is None:
        return {'error': err[:500]}
    
    results = parse_json_responses(out)
    if 2 in results:
        text = extract_text_content(results[2])
        ids = extract_message_ids(text)
        return {'text': text, 'ids': ids}
    return {'error': 'No response'}

def gmail_get_batch(email, message_ids):
    if not message_ids:
        return []
    
    client_id = get_env_val('GOOGLE_OAUTH_CLIENT_ID')
    client_secret = get_env_val('GOOGLE_OAUTH_CLIENT_SECRET')
    
    env_overrides = {
        'GOOGLE_OAUTH_CLIENT_ID': client_id,
        'GOOGLE_OAUTH_CLIENT_SECRET': client_secret,
        'OAUTHLIB_INSECURE_TRANSPORT': '1',
        'PATH': '/usr/local/bin:/usr/bin:/bin',
    }
    
    server_bin = '/home/kensei/.local/share/uv/tools/workspace-mcp/bin/workspace-mcp'
    server_args = [server_bin, '--single-user', '--tools', 'gmail']
    
    all_messages = []
    for i in range(0, len(message_ids), 10):
        batch = message_ids[i:i+10]
        requests = [
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
                'name': 'get_gmail_messages_content_batch',
                'arguments': {
                    'message_ids': batch,
                    'user_google_email': email,
                    'format': 'metadata'
                }
            }}
        ]
        
        out, err = call_mcp_batch(server_args, requests, env_overrides)
        if out is None:
            all_messages.append({'error': err[:200]})
            continue
        
        results = parse_json_responses(out)
        if 2 in results:
            text = extract_text_content(results[2])
            all_messages.append(text)
        
        time.sleep(0.5)
    
    return all_messages

def outlook_list_messages(email, top=15):
    """List Outlook messages for an account."""
    server_bin = '/home/kensei/.local/bin/node'
    server_args = [
        server_bin,
        '/home/kensei/.hermes/node/bin/ms-365-mcp-server',
        '--preset', 'mail,calendar'
    ]
    
    env_overrides = {
        'MS365_MCP_TOKEN_CACHE_PATH': '/home/kensei/.config/ms-365-mcp-server/token-cache.json',
        'PATH': '/usr/local/bin:/usr/bin:/bin',
    }
    
    requests = [
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
            'name': 'outlook_list_mail_messages',
            'arguments': {
                'account': email,
                'filter': "isRead eq false",
                'select': "id,subject,from,receivedDateTime,bodyPreview,isRead",
                'top': top
            }
        }}
    ]
    
    out, err = call_mcp_batch(server_args, requests, env_overrides)
    if out is None:
        return {'error': err[:500]}
    
    results = parse_json_responses(out)
    if 2 in results:
        text = extract_text_content(results[2])
        return {'text': text}
    return {'raw': str(results)[:500]}

def parse_gmail_metadata(text):
    """Parse Gmail batch metadata into structured items."""
    items = []
    current = {}
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('Message ID:'):
            if current and 'subject' in current:
                items.append(current)
            current = {'id': line.split(':', 1)[1].strip()}
        elif line.startswith('Subject:'):
            current['subject'] = line.split(':', 1)[1].strip()
        elif line.startswith('From:'):
            current['from'] = line.split(':', 1)[1].strip()
        elif line.startswith('Date:'):
            current['date'] = line.split(':', 1)[1].strip()
        elif line.startswith('To:'):
            current['to'] = line.split(':', 1)[1].strip()
    
    if current and 'subject' in current:
        items.append(current)
    
    return items

def categorise(subject, sender):
    """Categorise a message based on subject and sender."""
    s = (subject or '').lower()
    f = (sender or '').lower()
    
    # Action Required
    if any(k in s for k in ['invoice', 'payment', 'overdue', 'legal', 'hmrc', 'gov.uk', 
                             'childcare', 'credit control', 'application status', 're-confirm']):
        return 'action'
    if any(k in f for k in ['gov.uk', 'hmrc', 'credit control']):
        return 'action'
    
    # FYI / Monitoring
    if any(k in s for k in ['security alert', 'new sign-in', 'planned maintenance', 
                             'uptime', 'down', 'healthchecks', 'sentry']):
        return 'fyi'
    if any(k in f for k in ['security', 'accounts.google.com', 'no-reply@accounts.google.com',
                              'healthchecks.io', 'sentry']):
        return 'fyi'
    
    # Personal
    if any(k in s for k in ['bill', 'statement', 'broadband', 'energy', 'property',
                             'childcare', 'family', 'health']):
        return 'personal'
    
    # Noise
    if any(k in s for k in ['newsletter', 'promotion', 'marketing', 'job alert', 'alert',
                             'you have new', 'recommended', 'sponsored']):
        return 'noise'
    if any(k in f for k in ['jobs@', 'alerts@', 'indeed.com', 'careers.', 'newsletter',
                              'marketing', 'donotreply@']):
        return 'noise'
    
    # GitHub notifications
    if 'github.com' in f or 'notifications@github.com' in f:
        return 'fyi'
    
    # Apple Developer
    if 'apple.com' in f or 'developer' in f:
        return 'fyi'
    
    return 'unknown'

# ===== MAIN =====
now = datetime.now(timezone.utc)
print(f"MAILBOX DIGEST - {now.strftime('%d/%m/%Y %H:%M:%S')} UTC")
print("=" * 60)

all_items = []

# GMAIL ACCOUNTS
gmail_accounts = [
    ('saghir.sahil@gmail.com', 'Primary'),
    ('sahilsaghir.ss9@gmail.com', 'Dev'),
    ('fusionfirststudios@gmail.com', 'Studio')
]

for email, label in gmail_accounts:
    print(f"\n--- GMAIL: {email} ({label}) ---")
    
    result = gmail_search(email, 'is:unread', 20)
    ids = result.get('ids', [])
    print(f"  Unread count: {len(ids)}")
    
    if ids:
        batch_result = gmail_get_batch(email, ids[:15])
        for meta_text in batch_result:
            if isinstance(meta_text, str):
                items = parse_gmail_metadata(meta_text)
                for item in items:
                    item['account'] = email
                    item['account_label'] = label
                    item['category'] = categorise(item.get('subject', ''), item.get('from', ''))
                    all_items.append(item)
                    print(f"  [{item['category']}] {item.get('subject','?')[:80]} - {item.get('from','?')[:40]}")

# Category estimates for primary
print(f"\n--- CATEGORY ESTIMATES (Primary) ---")
for cat in ['promotions', 'social', 'updates', 'forums', 'purchases']:
    result = gmail_search('saghir.sahil@gmail.com', f'category:{cat} in:inbox', 1)
    ids = result.get('ids', [])
    print(f"  {cat}: {len(ids)} unread")

# OUTLOOK
print(f"\n--- OUTLOOK ---")
outlook_accounts = [
    ('sahil_ss@outlook.com', 'Job Hunt'),
    ('sahil_ss9@hotmail.com', 'Secondary'),
    ('sahil_saghir@hotmail.co.uk', 'Personal'),
    ('matchdaymaestro@outlook.com', 'Project')
]

for email, label in outlook_accounts:
    print(f"\n  OUTLOOK: {email} ({label})")
    result = outlook_list_messages(email, 15)
    if 'text' in result:
        print(f"  {result['text'][:500]}")
    else:
        print(f"  {result.get('raw', 'No response')[:200]}")

# SUMMARY
print(f"\n\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total unread items found: {len(all_items)}")
for cat in ['action', 'fyi', 'personal', 'noise', 'unknown']:
    count = sum(1 for i in all_items if i.get('category') == cat)
    if count > 0:
        print(f"  {cat}: {count}")
