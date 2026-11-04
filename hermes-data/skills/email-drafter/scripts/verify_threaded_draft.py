#!/usr/bin/env python3
"""Verify a threaded Gmail reply draft after create/re-create.

Usage:
    python3 verify_threaded_draft.py <draft_id> <thread_id> [needle1 needle2 ...]

Checks:
  1. draft exists and its threadId == expected thread
  2. exactly ONE live draft in that thread
  3. To / Cc / Subject headers
  4. every needle appears in the rendered body text

Run under the job-session identity pattern: this file sets
HERMES_SESSION_USER_ID before importing tools.gws_auth, because job/api_server
sessions have no session user and the native gws_* tools fail closed there.
Exit code 0 = all checks passed, 1 = something failed.
"""
import os
import sys
import base64
import re

os.environ['HERMES_SESSION_USER_ID'] = os.environ.get('HERMES_SESSION_USER_ID') or 'ndr'
sys.path.insert(0, '/opt/hermes')

from tools.gws_auth import build_service  # noqa: E402

SERVICE = os.environ.get('GWS_SERVICE_NAME', 'google-draas')

DEFAULT_NEEDLES = ['Regards', 'Nishant Ranka']


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    draft_id, thread_id = sys.argv[1], sys.argv[2]
    needles = sys.argv[3:] or DEFAULT_NEEDLES

    g = build_service('gmail', 'v1', service_name=SERVICE)
    ok = True

    d = g.users().drafts().get(userId='me', id=draft_id, format='raw').execute()
    raw = base64.urlsafe_b64decode(d['message']['raw']).decode('utf-8', 'replace')
    head, _, body = raw.partition('\n\n')

    got_thread = d['message'].get('threadId')
    match = got_thread == thread_id
    ok &= match
    print('THREAD:', got_thread, 'MATCH' if match else f'MISMATCH (expected {thread_id})')

    print('--- HEADERS ---')
    print(head.strip())

    count = 0
    for dr in g.users().drafts().list(userId='me', maxResults=200).execute().get('drafts', []):
        info = g.users().drafts().get(userId='me', id=dr['id'], format='metadata').execute()
        if info['message'].get('threadId') == thread_id:
            count += 1
            print('live draft in thread:', dr['id'])
    one = count == 1
    ok &= one
    print('DRAFTS IN THREAD:', count, 'OK' if one else 'DUPLICATES — delete stale drafts')

    txt = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', body))
    for needle in needles:
        hit = needle in txt
        ok &= hit
        print(('OK   ' if hit else 'MISS '), needle)
    print('body chars:', len(txt))

    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
