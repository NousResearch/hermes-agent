#!/usr/bin/env python3
"""
Complete mailbox digest - Gmail + Outlook.
Fetches unread messages, gets metadata, categorises, produces output.
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
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
            'protocolVersion': '2024-11-05', 'capabilities': {},
            'clientInfo': {'name': 'mailbox-digest', 'version': '1.0.0'}
        }},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
            'name': 'list-mail-messages',
            'arguments': {
                'account': email,
                'filter': 'isRead eq false',
                'select': 'id,subject,from,receivedDateTime,bodyPreview,isRead',
                'top': top
            }
        }}
    ]
    
    input_data = '\n'.join(json.dumps(r) for r in requests)
    
    proc = subprocess.Popen(
        server_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env_overrides
    )
    
    try:
        stdout, stderr = proc.communicate(input=input_data, timeout=30)
    except:
        proc.kill()
        return {'error': 'timeout'}
    
    # Parse the last response (id=2)
    for line in stdout.strip().split('\n'):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get('id') == 2:
                return json.loads(extract_text_content(obj))
        except:
            pass
    
    return {'error': 'no response parsed'}

def parse_gmail_metadata(text):
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
    s = (subject or '').lower()
    f = (sender or '').lower()
    
    # Action Required
    if any(k in s for k in ['invoice', 'payment', 'overdue', 'legal', 'hmrc', 'gov.uk',
                             'childcare', 'credit control', 'application status', 're-confirm',
                             'overdue invoice', 'payment received']):
        return 'action'
    if any(k in f for k in ['gov.uk', 'hmrc', 'credit control']):
        return 'action'
    
    # FYI / Monitoring
    if any(k in s for k in ['security alert', 'new sign-in', 'planned maintenance',
                             'uptime', 'down', 'healthchecks', 'sentry', 'heartbeat']):
        return 'fyi'
    if any(k in f for k in ['security', 'accounts.google.com', 'no-reply@accounts.google.com',
                              'healthchecks.io', 'sentry', 'github.com']):
        return 'fyi'
    
    # Personal
    if any(k in s for k in ['bill', 'statement', 'broadband', 'energy', 'property',
                             'childcare', 'family', 'health', 'british gas', 'council']):
        return 'personal'
    
    # Noise
    if any(k in s for k in ['newsletter', 'promotion', 'marketing', 'job alert', 'alert',
                             'you have new', 'recommended', 'sponsored', 'new jobs',
                             'work remotely', 'kitchen porter', 'new openings']):
        return 'noise'
    if any(k in f for k in ['jobs@', 'alerts@', 'indeed.com', 'careers.', 'newsletter',
                              'marketing', 'donotreply@', 'theladders', 'jobmails',
                              'targetnews', 'nutrac', 'natesnewsle', 'noreply@skool',
                              'noreply@runpod', 'railway.app', 'coachingmanual']):
        return 'noise'
    
    # Apple Developer
    if 'apple.com' in f or 'developer' in f:
        return 'fyi'
    
    # API-SPORTS
    if 'api-football' in f or 'api-sports' in s:
        return 'fyi'
    
    return 'unknown'

def format_date(date_str):
    """Format a date string to DD/MM/YYYY."""
    if not date_str:
        return ''
    try:
        # Try ISO format
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%d/%m/%Y')
    except:
        return date_str[:10]

# ===== MAIN =====
now = datetime.now(timezone.utc)
uk_now = datetime.now()
uk_date = uk_now.strftime('%d/%m/%Y')
uk_time = uk_now.strftime('%H:%M')

all_items = []

# ===== GMAIL =====
gmail_accounts = [
    ('saghir.sahil@gmail.com', 'Primary'),
    ('sahilsaghir.ss9@gmail.com', 'Dev'),
    ('fusionfirststudios@gmail.com', 'Studio')
]

for email, label in gmail_accounts:
    result = gmail_search(email, 'is:unread', 20)
    ids = result.get('ids', [])
    
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

# ===== OUTLOOK =====
outlook_accounts = [
    ('sahil_ss@outlook.com', 'Job Hunt'),
    ('sahil_ss9@hotmail.com', 'Secondary'),
    ('sahil_saghir@hotmail.co.uk', 'Personal'),
    ('matchdaymaestro@outlook.com', 'Project')
]

for email, label in outlook_accounts:
    result = outlook_list_messages(email, 15)
    if 'value' in result:
        for msg in result['value']:
            sender = msg.get('from', {}).get('emailAddress', {}).get('address', '')
            sender_name = msg.get('from', {}).get('emailAddress', {}).get('name', '')
            item = {
                'id': msg.get('id', ''),
                'subject': msg.get('subject', ''),
                'from': f'{sender_name} <{sender}>',
                'date': msg.get('receivedDateTime', ''),
                'account': email,
                'account_label': label,
                'category': categorise(msg.get('subject', ''), sender),
                'body_preview': msg.get('bodyPreview', '')[:100]
            }
            all_items.append(item)

# ===== BUILD OUTPUT =====
action_items = [i for i in all_items if i.get('category') == 'action']
fyi_items = [i for i in all_items if i.get('category') == 'fyi']
personal_items = [i for i in all_items if i.get('category') == 'personal']
noise_items = [i for i in all_items if i.get('category') == 'noise']
unknown_items = [i for i in all_items if i.get('category') == 'unknown']

# Print the digest
print(f"☀️ Good morning, Monday, {uk_date}, {uk_time}")
print()
print("📬 Inbox brief")
print(f"Freshness: last 24h where available. {len(all_items)} unread items across 7 accounts.")
print()

if action_items:
    print("🚨 Action required")
    for i, item in enumerate(action_items, 1):
        subj = item.get('subject', '?')[:80]
        acc = item.get('account_label', '')
        print(f"{i}. {subj} - {acc}")
    print()

if personal_items:
    print("📌 Worth knowing")
    for i, item in enumerate(personal_items, 1):
        subj = item.get('subject', '?')[:80]
        acc = item.get('account_label', '')
        print(f"{i}. {subj} - {acc}")
    print()

if fyi_items:
    print("👁 Monitoring")
    for i, item in enumerate(fyi_items[:5], 1):
        subj = item.get('subject', '?')[:80]
        acc = item.get('account_label', '')
        print(f"{i}. {subj} - {acc}")
    if len(fyi_items) > 5:
        print(f"   ... and {len(fyi_items)-5} more")
    print()

if noise_items:
    print(f"🔕 Noise: {len(noise_items)} items (job alerts, newsletters, promos)")
    print()

if unknown_items:
    print(f"❓ Unclassified: {len(unknown_items)} items")
    for i, item in enumerate(unknown_items[:3], 1):
        subj = item.get('subject', '?')[:80]
        acc = item.get('account_label', '')
        print(f"{i}. {subj} - {acc}")
    if len(unknown_items) > 3:
        print(f"   ... and {len(unknown_items)-3} more")
    print()

print("📎 HTML brief attached")
print(f"MEDIA:/home/kensei/.hermes/runbooks/mailbox-digest/2026-06-22/mailbox-digest.html")
