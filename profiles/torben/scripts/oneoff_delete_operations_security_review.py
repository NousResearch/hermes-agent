import json
import urllib.request
import urllib.parse
import urllib.error
import datetime
import re
from pathlib import Path

BASE = Path('/Users/ericfreeman/.hermes/profiles/torben')
ACCOUNTS = {
    'personal_freeman': 'freeman.eric.m@gmail.com',
    'personal_michael': 'eric.michael.freeman@gmail.com',
    'work_interralis': 'eric@interralis.com',
    'work_magellan': 'eric@magellansec.com',
}
API = 'https://www.googleapis.com/calendar/v3'
HANDLE = 'EA-20260625-029'
START = datetime.datetime(2026, 6, 25, 0, 0, tzinfo=datetime.timezone.utc)
END = datetime.datetime(2026, 7, 31, 23, 59, tzinfo=datetime.timezone.utc)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def token(alias):
    return read_json(BASE / 'accounts' / alias / 'hermes-home/google_token.json')['token']


def req_json(method, url, tok, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {'Authorization': 'Bearer ' + tok}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors='replace')
        if method == 'DELETE' and exc.code in (404, 410):
            return {'already_gone': True, 'code': exc.code}
        raise RuntimeError('HTTP %s %s: %s' % (exc.code, url, raw[:500]))


def list_events(alias, start=START, end=END):
    params = urllib.parse.urlencode({
        'timeMin': start.isoformat().replace('+00:00', 'Z'),
        'timeMax': end.isoformat().replace('+00:00', 'Z'),
        'singleEvents': 'true',
        'orderBy': 'startTime',
        'maxResults': '2500',
    })
    url = API + '/calendars/primary/events?' + params
    return req_json('GET', url, token(alias)).get('items', [])


def parse_dt(obj):
    value = (obj or {}).get('dateTime') or (obj or {}).get('date')
    if not value:
        return None
    if len(value) == 10:
        return datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc)
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(datetime.timezone.utc)


def overlaps(first, second):
    return first[0] < second[1] and second[0] < first[1]


def event_path(event_id):
    return urllib.parse.quote(event_id, safe='')


def update_ledger(**updates):
    path = BASE / 'state/torben-action-ledger.jsonl'
    data = read_json(path)
    for record in data:
        if record.get('handle') == HANDLE:
            state = record.setdefault('executor_state', {})
            state.update({key: value for key, value in updates.items() if key != 'ledger_status'})
            record['status'] = updates.get('ledger_status', record.get('status'))
            record.setdefault('resolution_history', []).append({
                'at': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
                'status': record['status'],
                'reason': updates.get('result_summary', 'calendar deletion step completed'),
            })
            break
    write_json(path, data)


result = {
    'handle': HANDLE,
    'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
    'mutations': [],
    'errors': [],
    'verification': {},
}

source_events = []
for item in list_events('work_magellan'):
    if item.get('status') != 'cancelled' and re.search(r'Operations & Security Review', item.get('summary') or '', re.I):
        start = parse_dt(item.get('start'))
        end = parse_dt(item.get('end'))
        if start and end:
            source_events.append({
                'account': 'work_magellan',
                'id': item.get('id'),
                'start': start,
                'end': end,
                'summary': item.get('summary'),
            })

result['discovered_source_events'] = [
    {
        'account': event['account'],
        'id': event['id'],
        'start': event['start'].isoformat(),
        'end': event['end'].isoformat(),
        'summary': event['summary'],
    }
    for event in source_events
]
windows = [(event['start'], event['end']) for event in source_events]

for event in source_events:
    try:
        params = urllib.parse.urlencode({'sendUpdates': 'none'})
        url = API + '/calendars/primary/events/' + event_path(event['id']) + '?' + params
        response = req_json('DELETE', url, token(event['account']))
        result['mutations'].append({
            'type': 'delete_source_event',
            'account': event['account'],
            'event_id': event['id'],
            'start': event['start'].isoformat(),
            'end': event['end'].isoformat(),
            'response': response,
        })
    except Exception as exc:
        result['errors'].append({
            'type': 'delete_source_event',
            'account': event['account'],
            'event_id': event['id'],
            'error': str(exc),
        })

for alias in ACCOUNTS:
    for item in list_events(alias):
        ext = ((item.get('extendedProperties') or {}).get('private') or {})
        if ext.get('torben_alignment') != 'true' or ext.get('source_account') != 'work_magellan':
            continue
        start = parse_dt(item.get('start'))
        end = parse_dt(item.get('end'))
        if not start or not end or not any(overlaps((start, end), window) for window in windows):
            continue
        try:
            params = urllib.parse.urlencode({'sendUpdates': 'none'})
            url = API + '/calendars/primary/events/' + event_path(item.get('id')) + '?' + params
            response = req_json('DELETE', url, token(alias))
            result['mutations'].append({
                'type': 'delete_mirrored_busy',
                'account': alias,
                'event_id': item.get('id'),
                'start': start.isoformat(),
                'end': end.isoformat(),
                'response': response,
            })
        except Exception as exc:
            result['errors'].append({
                'type': 'delete_mirrored_busy',
                'account': alias,
                'event_id': item.get('id'),
                'error': str(exc),
            })

verify_start = datetime.datetime(2026, 6, 26, 0, 0, tzinfo=datetime.timezone.utc)
verify_end = datetime.datetime(2026, 6, 27, 4, 0, tzinfo=datetime.timezone.utc)
remaining_ops = []
remaining_mirrors = []
all_events = []
for alias in ACCOUNTS:
    for item in list_events(alias, verify_start, verify_end):
        start = parse_dt(item.get('start'))
        end = parse_dt(item.get('end'))
        record = {
            'account': alias,
            'id': item.get('id'),
            'summary': item.get('summary'),
            'start': start.isoformat() if start else None,
            'end': end.isoformat() if end else None,
        }
        all_events.append(record)
        if re.search(r'Operations & Security Review', item.get('summary') or '', re.I):
            remaining_ops.append(record)
        ext = ((item.get('extendedProperties') or {}).get('private') or {})
        if ext.get('torben_alignment') == 'true' and ext.get('source_account') == 'work_magellan' and start and end and any(overlaps((start, end), window) for window in windows):
            remaining_mirrors.append(record)

future_ops = []
for item in list_events('work_magellan'):
    if item.get('status') != 'cancelled' and re.search(r'Operations & Security Review', item.get('summary') or '', re.I):
        start = parse_dt(item.get('start'))
        end = parse_dt(item.get('end'))
        future_ops.append({
            'id': item.get('id'),
            'summary': item.get('summary'),
            'start': start.isoformat() if start else None,
            'end': end.isoformat() if end else None,
        })

result['verification'] = {
    'remaining_operations_events_tomorrow': remaining_ops,
    'remaining_mirrored_busy_blocks_tomorrow': remaining_mirrors,
    'future_operations_events_through_2026_07_31': future_ops,
    'tomorrow_calendar_events': all_events,
}
result['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
status = 'completed' if not result['errors'] and not remaining_ops and not remaining_mirrors and not future_ops else 'attention_required'
result_path = BASE / 'state/torben-calendar-cleanup-20260625-029.json'
write_json(result_path, result)
update_ledger(
    ledger_status=status,
    result_summary='Operations & Security Review deleted and mirrored Busy blocks removed',
    result_path=str(result_path),
    verification=result['verification'],
    mutation_count=len(result['mutations']),
    errors=result['errors'],
)
print(json.dumps(result, indent=2, sort_keys=True))
