#!/usr/bin/env python3
"""Create a Gmail draft (optionally threaded, with attachments) using hand-built raw MIME.

Why this exists: drafts built with email.message.EmailMessage can be stored WITHOUT
MIME-Version / Content-Type / Message-Id, and with a References chain that Gmail
folds into RFC2047 encoded-words. The draft then shows in the Drafts list but will
not open in the Gmail web UI. Hand-built raw MIME avoids that entirely.

Usage (run from terminal(), NOT execute_code -- the vault socket is only available
in the terminal environment):

    cd /opt/hermes && HERMES_SESSION_USER_ID=ndr /opt/hermes/.venv/bin/python3 \
        create-draft-with-attachments.py --config /tmp/draft_spec.py

Or import and drive the helpers directly.

Spec file format (a plain python file, so long bodies stay readable):

    TO          = "Anbarasan M <pm2.blr@draas.com>"
    CC          = "Kantesh B G <kanteshbg@draas.com>, Sinchana Gowda <sgowda@draas.com>"
    SUBJECT     = "Re: north star revised sanction dwg"
    THREAD_ID   = "1a0af9045bf47284"        # omit / None for a new standalone draft
    IN_REPLY_TO = "<CAFFuF18...@mail.gmail.com>"   # omit / None if not a reply
    REFERENCES  = "<CAFFuF18...@mail.gmail.com>"   # keep SHORT: parent (+ grandparent)
    BODY        = """Dear X, ..."""
    # attachments: paths on disk
    ATTACHMENTS = ["/tmp/att_sanction-1.pdf", "/tmp/checklist.pdf"]
    # optionally pull attachments straight off an existing message instead
    SOURCE_MESSAGE_ID = "1a0af9045bf47284"   # omit if using ATTACHMENTS only
"""
import argparse
import base64
import importlib.util
import os
import sys
import uuid

sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import _load_credentials_direct  # noqa: E402
from googleapiclient.discovery import build  # noqa: E402


def service():
    creds = _load_credentials_direct('google-draas')
    svc = build('gmail', 'v1', credentials=creds)
    # ALWAYS print the mailbox -- confirms which account the token actually is
    print('MAILBOX:', svc.users().getProfile(userId='me').execute()['emailAddress'])
    return svc


def download_attachments(svc, message_id, outdir='/tmp'):
    """Pull every attachment off a message. Returns [(filename, path)]."""
    msg = svc.users().messages().get(userId='me', id=message_id, format='full').execute()
    found = []

    def walk(part):
        if part.get('filename') and part.get('body', {}).get('attachmentId'):
            found.append((part['filename'], part['body']['attachmentId']))
        for child in part.get('parts', []) or []:
            walk(child)

    walk(msg['payload'])
    out = []
    for fn, aid in found:
        blob = svc.users().messages().attachments().get(
            userId='me', messageId=message_id, id=aid).execute()
        path = os.path.join(outdir, 'att_' + fn)
        with open(path, 'wb') as fh:
            fh.write(base64.urlsafe_b64decode(blob['data']))
        out.append((fn, path))
    print('downloaded attachments:', [fn for fn, _ in out])
    return out


def build_raw(spec, files):
    """spec: module with TO/CC/SUBJECT/BODY/IN_REPLY_TO/REFERENCES."""
    boundary = '----=_Part_%s' % uuid.uuid4().hex
    msgid = '<%s@mail.gmail.com>' % uuid.uuid4().hex.upper()[:24]

    head = ['MIME-Version: 1.0',
            'From: Nishant Ranka <ndr@draas.com>',
            'To: %s' % spec.TO]
    if getattr(spec, 'CC', None):
        head.append('Cc: %s' % spec.CC)
    head.append('Subject: %s' % spec.SUBJECT)
    if getattr(spec, 'IN_REPLY_TO', None):
        head.append('In-Reply-To: %s' % spec.IN_REPLY_TO)
    if getattr(spec, 'REFERENCES', None):
        head.append('References: %s' % spec.REFERENCES)
    head += ['Message-Id: %s' % msgid,
             'Content-Type: multipart/mixed; boundary="%s"' % boundary,
             '', '']
    raw = '\r\n'.join(head)

    parts = ['--%s\r\nContent-Type: text/plain; charset="UTF-8"\r\n'
             'Content-Transfer-Encoding: 8bit\r\n\r\n%s\r\n'
             % (boundary, spec.BODY)]
    for fn, path in files:
        with open(path, 'rb') as fh:
            b64 = base64.encodebytes(fh.read()).decode().replace('\n', '\r\n')
        parts.append('--%s\r\nContent-Type: application/pdf; name="%s"\r\n'
                     'Content-Transfer-Encoding: base64\r\n'
                     'Content-Disposition: attachment; filename="%s"\r\n\r\n%s\r\n'
                     % (boundary, fn, fn, b64))
    parts.append('--%s--\r\n' % boundary)
    return raw + ''.join(parts)


def verify(svc, draft_id, expected_files):
    got = svc.users().drafts().get(userId='me', id=draft_id, format='full').execute()
    hdrs = {h['name']: h['value'] for h in got['message']['payload']['headers']}
    labels = got['message'].get('labelIds')
    print('  labels :', labels)
    print('  To     :', hdrs.get('To'))
    print('  Cc     :', hdrs.get('Cc'))
    print('  Subject:', hdrs.get('Subject'))
    print('  thread :', got['message'].get('threadId'))
    names = []

    def walk(part):
        if part.get('filename'):
            names.append(part['filename'])
        for c in part.get('parts', []) or []:
            walk(c)

    walk(got['message']['payload'])
    print('  attach :', names)
    if labels != ['DRAFT']:
        print('  !! NOT A DRAFT -- labelIds is %s. A draft id can resolve to a SENT '
              'message after the user sends it from mobile. Do NOT recreate; report.' % labels)
    missing = [f for f in expected_files if f not in names]
    if missing:
        print('  !! MISSING ATTACHMENTS:', missing)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='python file with the draft spec')
    args = ap.parse_args()

    s = importlib.util.spec_from_file_location('draft_spec', args.config)
    spec = importlib.util.module_from_spec(s)
    s.loader.exec_module(spec)

    svc = service()

    files = []
    if getattr(spec, 'SOURCE_MESSAGE_ID', None):
        files += download_attachments(svc, spec.SOURCE_MESSAGE_ID)
    for path in getattr(spec, 'ATTACHMENTS', []) or []:
        files.append((os.path.basename(path), path))

    raw = build_raw(spec, files)
    print('raw bytes:', len(raw))
    body = {'message': {'raw': base64.urlsafe_b64encode(raw.encode('utf-8')).decode()}}
    if getattr(spec, 'THREAD_ID', None):
        body['message']['threadId'] = spec.THREAD_ID
    d = svc.users().drafts().create(userId='me', body=body).execute()
    print('DRAFT ID:', d['id'], '| msg:', d['message']['id'],
          '| thread:', d['message'].get('threadId'))
    verify(svc, d['id'], [fn for fn, _ in files])


if __name__ == '__main__':
    main()
