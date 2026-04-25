#!/bin/bash
# Test script — simulates Checkmk invoking the webhook handler
# Run ON the Checkmk container

HERMES_URL="http://192.168.1.8:5001/alerts"

python3 -c "
import urllib.parse, json, subprocess

form_data = 'context=' + urllib.parse.quote(json.dumps({
    'host_name': 'DC1-SPINE-01',
    'service_description': 'Interface eth0/1',
    'service_output': 'Link down — no carrier',
    'service_state': 2,
    'event_id': 'evt-001',
    'time': 1745270400
})) + '&host_name=DC1-SPINE-01&event_id=evt-001'

data = {}
for k, v in urllib.parse.parse_qsl(form_data):
    if k == 'context':
        ctx = json.loads(v)
        data['context'] = ctx
        for ck, cv in ctx.items():
            if ck not in data:
                data[ck] = cv
    else:
        data[k] = v

payload = json.dumps(data)
print('Payload:', payload[:200])

result = subprocess.run(
    ['curl', '-s', '-X', 'POST', '${HERMES_URL}',
     '-H', 'Content-Type: application/json', '-d', payload],
    capture_output=True, text=True, timeout=10
)
print('Hermes response:', result.stdout[:500])
print('curl stderr:', result.stderr[:200] if result.stderr else 'none')
print('RC:', result.returncode)
"
