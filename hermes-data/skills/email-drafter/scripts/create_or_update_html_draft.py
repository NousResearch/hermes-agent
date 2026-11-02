#!/usr/bin/env python3
"""Create or revise a Gmail DRAFT from an HTML body + plain-text alternative.

Never sends. Works in job / api_server sessions where the native gws_* tools fail closed
(no session user), because it sets HERMES_SESSION_USER_ID before importing tools.gws_auth.

Usage
  python3 create_or_update_html_draft.py \
      --html email.html --text email.txt \
      --to "Dharmesh Ranka <dharmesh.ranka1@gmail.com>" \
      --cc "Manish Ranka <mdr@draas.com>" \
      --subject "Subject line" \
      [--update <draft_id>] [--service google-draas] [--state draft_id.txt]

Notes
  * --text may contain a leading "Subject:" line; it is stripped before set_content().
  * With --update the draft id is reused (message id changes) — that is the expected outcome.
  * draft id + message id are written to --state for later verification / revision.
"""
import argparse
import base64
import os
import sys
from email.message import EmailMessage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--html', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--to', required=True)
    ap.add_argument('--cc', default='')
    ap.add_argument('--subject', required=True)
    ap.add_argument('--from', dest='sender', default='Nishant Ranka <ndr@draas.com>')
    ap.add_argument('--update', default='', help='existing draft id to revise in place')
    ap.add_argument('--service', default='google-draas')
    ap.add_argument('--user', default='ndr', help='HERMES_SESSION_USER_ID to assume')
    ap.add_argument('--state', default='draft_id.txt')
    args = ap.parse_args()

    os.environ['HERMES_SESSION_USER_ID'] = args.user
    sys.path.insert(0, '/opt/hermes')
    from tools.gws_auth import build_service

    html = open(args.html, encoding='utf-8').read()
    txt = open(args.text, encoding='utf-8').read()
    lines = txt.split('\n')
    if lines and lines[0].lower().startswith('subject:'):
        txt = '\n'.join(lines[2:]).lstrip('\n')

    msg = EmailMessage()
    msg['From'] = args.sender
    msg['To'] = args.to
    if args.cc:
        msg['Cc'] = args.cc
    msg['Subject'] = args.subject
    msg.set_content(txt)
    msg.add_alternative(html, subtype='html')

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    svc = build_service('gmail', 'v1', service_name=args.service)

    draft_id = args.update
    if not draft_id and os.path.exists(args.state):
        candidate = open(args.state).read().split('\n')[0].strip()
        # only auto-reuse if the caller passes it explicitly; never guess here
        draft_id = ''

    if draft_id:
        d = svc.users().drafts().update(
            userId='me', id=draft_id, body={'message': {'raw': raw}}).execute()
        action = 'updated'
    else:
        d = svc.users().drafts().create(
            userId='me', body={'message': {'raw': raw}}).execute()
        action = 'created'

    with open(args.state, 'w') as fh:
        fh.write(d['id'] + '\n' + d['message']['id'] + '\n')

    print(f'{action} draft {d["id"]} (message {d["message"]["id"]})')
    print('NOTHING SENT — review in Gmail Drafts.')


if __name__ == '__main__':
    main()
