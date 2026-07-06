from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

REPO = Path('/Users/ericfreeman/.hermes/hermes-agent')
sys.path.insert(0, str(REPO))

from hermes_cli.signal_coo.google_auth import account_for_alias  # noqa: E402
from hermes_cli.signal_coo.google_evidence import GMAIL_API_ROOT, _read_token  # noqa: E402

HOME = Path('/Users/ericfreeman/.hermes/profiles/torben')
CONFIG = HOME / 'config' / 'google_accounts.yaml'
LEDGER = HOME / 'state' / 'torben-action-ledger.jsonl'
ACCOUNT_ALIAS = 'personal_freeman'
ORIGINAL_MESSAGE_ID = '19efc14df4896ed7'
TO = 'Kim Cooper <kimc@uandi.vc>'
BODY = """Hi Kim,

Thanks for the invite. I appreciate you thinking of me.

I’m on paternity leave right now and need to be home with the baby. My wife is still recovering from her second C-section, and this recovery has been rough.

I’m most likely not going to make this one. Please keep me in mind for the next one once things settle down at home.

Best,
Eric
"""


def google_request(url: str, token: str, *, method: str = 'GET', payload: dict | None = None) -> dict:
    data = None
    headers = {'Authorization': f'Bearer {token}'}
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8')
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Google API {method} {url} failed: HTTP {exc.code}: {detail}') from exc


def header(headers: list[dict], name: str) -> str:
    lname = name.lower()
    for item in headers:
        if str(item.get('name', '')).lower() == lname:
            return str(item.get('value') or '')
    return ''


def update_ledger(sent_payload: dict, thread_id: str) -> None:
    if not LEDGER.exists():
        return
    records = json.loads(LEDGER.read_text(encoding='utf-8'))
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    target_evidence = f'gmail:{ACCOUNT_ALIAS}:{ORIGINAL_MESSAGE_ID}'
    updated = False
    for record in records:
        if target_evidence not in (record.get('evidence_ids') or []):
            continue
        if record.get('status') in {'sent', 'executed'}:
            continue
        record['status'] = 'sent'
        record['outbound_message_id'] = sent_payload.get('id')
        state = record.setdefault('executor_state', {})
        state.update({
            'mutation_status': 'sent',
            'sent_at': now,
            'sent_message_id': sent_payload.get('id'),
            'sent_thread_id': thread_id,
            'sent_to': TO,
        })
        history = record.setdefault('resolution_history', [])
        history.append({'at': now, 'status': 'sent', 'reason': 'User explicitly approved sending the Kim Cooper U&I breakfast reply in Signal.'})
        updated = True
    if updated:
        tmp = LEDGER.with_name(f'.{LEDGER.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(records, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(tmp, LEDGER)


def main() -> int:
    account = account_for_alias(CONFIG, ACCOUNT_ALIAS)
    token = _read_token(account)
    params = urllib.parse.urlencode({
        'format': 'metadata',
        'metadataHeaders': ['Subject', 'Message-ID', 'References', 'From', 'To', 'Date'],
        'fields': 'id,threadId,payload(headers)',
    }, doseq=True)
    original = google_request(f'{GMAIL_API_ROOT}/messages/{ORIGINAL_MESSAGE_ID}?{params}', token)
    headers = (original.get('payload') or {}).get('headers') or []
    thread_id = str(original.get('threadId') or '')
    subject = header(headers, 'Subject') or 'CISO & Eng Leaders Breakfast at The Crosby Wednesday, July 8th at 9am'
    if not subject.lower().startswith('re:'):
        subject = 'Re: ' + subject
    orig_rfc_message_id = header(headers, 'Message-ID')
    refs = header(headers, 'References')

    msg = EmailMessage()
    msg['From'] = account.email
    msg['To'] = TO
    msg['Subject'] = subject
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=account.email.split('@')[-1])
    if orig_rfc_message_id:
        msg['In-Reply-To'] = orig_rfc_message_id
        msg['References'] = (refs + ' ' + orig_rfc_message_id).strip() if refs else orig_rfc_message_id
    msg.set_content(BODY)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode('ascii')
    payload = {'raw': raw}
    if thread_id:
        payload['threadId'] = thread_id
    sent = google_request(f'{GMAIL_API_ROOT}/messages/send', token, method='POST', payload=payload)
    sent_id = sent.get('id')
    if not sent_id:
        raise RuntimeError(f'Gmail send returned no message id: {sent}')

    verify = google_request(f'{GMAIL_API_ROOT}/messages/{sent_id}?format=metadata&metadataHeaders=To&metadataHeaders=Subject&fields=id,threadId,labelIds,payload(headers)', token)
    update_ledger(sent, thread_id or str(verify.get('threadId') or ''))
    print(json.dumps({
        'sent_id': sent_id,
        'thread_id': verify.get('threadId') or sent.get('threadId'),
        'labels': verify.get('labelIds'),
        'to': TO,
        'subject': subject,
        'account': account.email,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    from torben_job_contract import run_job

    raise SystemExit(run_job('send_kim_reply_once', main))
