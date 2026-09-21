#!/usr/bin/env python3
"""Probe Outlook MCP - use communicate pattern."""
import json, subprocess, os, time

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

# Build all requests
requests = []

# Initialize
requests.append(json.dumps({
    'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
    'params': {
        'protocolVersion': '2024-11-05',
        'capabilities': {},
        'clientInfo': {'name': 'mailbox-digest', 'version': '1.0.0'}
    }
}))

# Initialized notification
requests.append(json.dumps({
    'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}
}))

# List accounts
requests.append(json.dumps({
    'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
        'name': 'list-accounts', 'arguments': {}
    }
}))

# List mail messages
requests.append(json.dumps({
    'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {
        'name': 'list-mail-messages',
        'arguments': {
            'account': 'sahil_ss@outlook.com',
            'filter': 'isRead eq false',
            'select': 'id,subject,from,receivedDateTime,bodyPreview,isRead',
            'top': 5
        }
    }
}))

# Also try sahil_ss9@hotmail.com
requests.append(json.dumps({
    'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {
        'name': 'list-mail-messages',
        'arguments': {
            'account': 'sahil_ss9@hotmail.com',
            'filter': 'isRead eq false',
            'select': 'id,subject,from,receivedDateTime,bodyPreview,isRead',
            'top': 5
        }
    }
}))

# Also try sahil_saghir@hotmail.co.uk
requests.append(json.dumps({
    'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {
        'name': 'list-mail-messages',
        'arguments': {
            'account': 'sahil_saghir@hotmail.co.uk',
            'filter': 'isRead eq false',
            'select': 'id,subject,from,receivedDateTime,bodyPreview,isRead',
            'top': 5
        }
    }
}))

# Also try matchdaymaestro@outlook.com
requests.append(json.dumps({
    'jsonrpc': '2.0', 'id': 6, 'method': 'tools/call', 'params': {
        'name': 'list-mail-messages',
        'arguments': {
            'account': 'matchdaymaestro@outlook.com',
            'filter': 'isRead eq false',
            'select': 'id,subject,from,receivedDateTime,bodyPreview,isRead',
            'top': 5
        }
    }
}))

input_data = '\n'.join(requests)

proc = subprocess.Popen(
    server_args,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env_overrides
)

stdout, stderr = proc.communicate(input=input_data, timeout=30)

print("=== STDOUT ===")
print(stdout[:5000])

if stderr:
    print("\n=== STDERR (last 2000) ===")
    print(stderr[-2000:] if len(stderr) > 2000 else stderr)
