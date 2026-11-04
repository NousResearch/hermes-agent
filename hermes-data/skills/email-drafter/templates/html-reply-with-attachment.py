#!/usr/bin/env python3
"""Template: threaded reply draft with HTML body + image attachment (hand-built raw MIME).

Copy and edit TO/CC/SUBJECT/THREAD_ID/PARENT_MSG/ATTACH/PLAIN/HTML, then run from
terminal() (NOT execute_code — the vault socket is only in the terminal environment):

    cd /opt/hermes && HERMES_SESSION_USER_ID=ndr /opt/hermes/.venv/bin/python3 \\
        html-reply-with-attachment.py

Verified 2026-09-29 (Oasis 50-cents reply). See references/html-reply-with-attachments-threaded.md.
"""
import base64
import os
import sys
import uuid

sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service

svc = build_service('gmail', 'v1', service_name='google-draas')
print('MAILBOX:', svc.users().getProfile(userId='me').execute()['emailAddress'])

# ---- EDIT THESE ----
THREAD_ID   = "<parent thread id>"
PARENT_MSG  = "<parent message id in thread>"   # for In-Reply-To / References
ATTACH      = "/data/hermes/image_cache/<img>.jpg"
TO          = "Primary Recipient <person@draas.com>"
CC          = ("Cc One <a@draas.com>, Cc Two <b@draas.com>")
SUBJECT     = "Re: <original subject>"

PLAIN = """Dear X,

Body text — one paragraph per logical line, no manual wrapping.

Regards,
Ndr
"""

HTML = """<html><body style="margin:0;padding:0;background:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">
<div style="max-width:860px;margin:0 auto;background:#ffffff;padding:24px 28px;border-top:6px solid #1F3864;font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.65;color:#222222;">
<p>Dear X,</p>
<p>Body with <b>bold highlights</b>.</p>
<p>Regards,<br>Ndr</p>
</div></body></html>"""
# -------------------

parent = svc.users().messages().get(userId='me', id=PARENT_MSG, format='metadata',
                                    metadataHeaders=['Message-ID', 'References']).execute()
phdrs = {h['name'].lower(): h['value'] for h in parent['payload']['headers']}
parent_mid = phdrs.get('message-id', '')
parent_refs = phdrs.get('references', '')

boundary = '----=_Part_%s' % uuid.uuid4().hex
alt_boundary = '----=_Alt_%s' % uuid.uuid4().hex
msgid = '<%s@mail.gmail.com>' % uuid.uuid4().hex.upper()[:24]

head = ['MIME-Version: 1.0',
        'From: Nishant Ranka <ndr@draas.com>',
        'To: %s' % TO,
        'Cc: %s' % CC,
        'Subject: %s' % SUBJECT,
        'In-Reply-To: %s' % parent_mid,
        'References: %s' % ((parent_refs + ' ' + parent_mid).strip() if parent_refs else parent_mid),
        'Message-Id: %s' % msgid,
        'Content-Type: multipart/mixed; boundary="%s"' % boundary,
        '', '']
raw = '\r\n'.join(head)

raw += ('--%s\r\nContent-Type: multipart/alternative; boundary="%s"\r\n\r\n' % (boundary, alt_boundary))
raw += ('--%s\r\nContent-Type: text/plain; charset="UTF-8"\r\nContent-Transfer-Encoding: 8bit\r\n\r\n%s\r\n\r\n'
        % (alt_boundary, PLAIN))
raw += ('--%s\r\nContent-Type: text/html; charset="UTF-8"\r\nContent-Transfer-Encoding: 8bit\r\n\r\n%s\r\n\r\n'
        % (alt_boundary, HTML))
raw += '--%s--\r\n\r\n' % alt_boundary

with open(ATTACH, 'rb') as fh:
    b64 = base64.encodebytes(fh.read()).decode().replace('\n', '\r\n')
fname = os.path.basename(ATTACH)
raw += ('--%s\r\nContent-Type: image/jpeg; name="%s"\r\n'
        'Content-Transfer-Encoding: base64\r\n'
        'Content-Disposition: attachment; filename="%s"\r\n\r\n%s\r\n' % (boundary, fname, fname, b64))
raw += '--%s--\r\n' % boundary

body = {'message': {'raw': base64.urlsafe_b64encode(raw.encode('utf-8')).decode(),
                    'threadId': THREAD_ID}}
d = svc.users().drafts().create(userId='me', body=body).execute()
print('DRAFT ID:', d['id'], '| msg:', d['message']['id'], '| thread:', d['message'].get('threadId'))

# ---- VERIFY ----
got = svc.users().drafts().get(userId='me', id=d['id'], format='full').execute()
hdrs = {h['name']: h['value'] for h in got['message']['payload']['headers']}
print('labels :', got['message'].get('labelIds'))
print('To     :', hdrs.get('To'))
print('Cc     :', hdrs.get('Cc'))
print('Subject:', hdrs.get('Subject'))
print('thread :', got['message'].get('threadId'))
names = []
def walk(part):
    if part.get('filename'):
        names.append(part['filename'])
    for c in part.get('parts', []) or []:
        walk(c)
walk(got['message']['payload'])
print('attach :', names)
assert got['message'].get('labelIds') == ['DRAFT'], 'NOT A DRAFT!'
assert got['message'].get('threadId') == THREAD_ID, 'WRONG THREAD!'
assert fname in names, 'MISSING ATTACHMENT'
print('VERIFIED OK')