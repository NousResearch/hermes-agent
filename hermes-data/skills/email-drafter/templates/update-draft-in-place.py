#!/usr/bin/env python3
"""Skeleton: replace an EXISTING Gmail draft's body (plain+html) and attachment in place.

Copy, fill the CONFIG block, run. Same draft id, same To/Subject — no duplicate draft.
Read references/revising-an-existing-draft-in-place.md first (contradicting attachments,
figure-stripping rules, verification greps).

Works from a job/api_server session too: HERMES_SESSION_USER_ID must be set BEFORE importing
tools.gws_auth, and 'ndr' is the right slug for ndr@draas.com.
"""
import os, sys, base64

os.environ['HERMES_SESSION_USER_ID'] = 'ndr'          # BEFORE the import below
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# ---------------- CONFIG ----------------
DRAFT_ID = 'r-0000000000000000000'      # from drafts().list()
TO       = 'rnr@draas.com'
SUBJECT  = 'unchanged subject line'
TEXT     = open('/path/body.txt',  encoding='utf-8').read()   # or an inline string
HTML     = open('/path/body.html', encoding='utf-8').read()
PDF      = '/path/attachment.pdf'        # None to send the draft with NO attachment
PDF_NAME = 'Attachment name as the recipient sees it.pdf'
SERVICE  = 'google-draas'
STALE    = ['50,000', '43,548']          # figures you removed; verified absent below
# ----------------------------------------

msg = MIMEMultipart('mixed')
msg['To'] = TO
msg['From'] = 'Nishant Ranka <ndr@draas.com>'
msg['Subject'] = SUBJECT

alt = MIMEMultipart('alternative')
alt.attach(MIMEText(TEXT, 'plain', 'utf-8'))
alt.attach(MIMEText(HTML, 'html', 'utf-8'))
msg.attach(alt)

if PDF:
    with open(PDF, 'rb') as f:
        part = MIMEApplication(f.read(), _subtype='pdf')
    part.add_header('Content-Disposition', 'attachment', filename=PDF_NAME)
    msg.attach(part)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
svc = build_service('gmail', 'v1', service_name=SERVICE)
res = svc.users().drafts().update(userId='me', id=DRAFT_ID,
                                  body={'message': {'raw': raw}}).execute()
print('updated draft', res['id'])

# ---------------- VERIFY ----------------
d = svc.users().drafts().get(userId='me', id=DRAFT_ID, format='full').execute()
m = d['message']
h = {x['name']: x['value'] for x in m['payload']['headers']}
print('To:', h.get('To'), '| Subject:', h.get('Subject'))
print('attachments:', [c.get('filename') for c in m['payload'].get('parts', []) if c.get('filename')])


def flat(p, acc):
    if p.get('body', {}).get('data'):
        acc.append((p['mimeType'], base64.urlsafe_b64decode(p['body']['data'] + '===').decode('utf-8', 'replace')))
    for c in p.get('parts', []) or []:
        flat(c, acc)
    return acc


all_text = ''.join(t for _, t in flat(m['payload'], []))
for bad in STALE:
    assert bad not in all_text, 'STALE FIGURE STILL PRESENT: %s' % bad
html = [t for mt, t in flat(m['payload'], []) if mt == 'text/html'][0]
assert '%%' not in html, 'unformatted % left in HTML'
print('drafts total:', len(svc.users().drafts().list(userId='me').execute().get('drafts', [])))
print('VERIFIED — draft updated in place, nothing sent')
