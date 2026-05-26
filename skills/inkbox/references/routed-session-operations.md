# Inkbox routed-session operations notes

These notes capture Hermes/Inkbox runtime pitfalls that are not obvious from the SDK surface.

## Force the requested outbound channel

When a user says “text Alex”, do not rely on `send_message` with an `inkbox:Alex` contact target unless you have verified the platform adapter will choose SMS. A contact chat target can default to email when there is no recent inbound SMS modality for that contact. In one session, `send_message(target="inkbox:Alex", ...)` delivered an email to `alex@vectorly.app` even though the user requested a text.

Robust pattern for third-party SMS:

1. Resolve the contact from the routing marker, contact book, or known E.164 number.
2. Use the SDK directly: `identity.send_text(to="+E164", text="...")`.
3. Verify delivery with `identity.get_text_conversation("+E164", limit=...)` or `identity.list_texts(...)` until the outbound row shows `delivery_status="delivered"` (or report queued/failed explicitly).
4. If a mistaken email was sent, tell the user plainly: “first went as email, no bounce; I sent the actual SMS now.”

## Check sent mail/text status from the source of truth

For questions like “did it deliver / did it bounce / do you see it?”, query Inkbox, not transcript memory.

Email:

```python
# Message rows carry `status` (sent / delivered / bounced / failed / …) —
# there is no separate `delivery_status` field on email messages.
for msg in identity.iter_emails(direction="outbound"):
    print(msg.to_addresses, msg.subject, msg.created_at, msg.status)
```

SMS:

```python
for t in identity.get_text_conversation("+13129096596", limit=5):
    print(t.direction, t.text, t.delivery_status, t.created_at)
```

## Deferred follow-up after a third-party reply

If a caller asks during a live call to text a third party, then email the caller only after that third party replies:

1. Send the requested third-party SMS immediately with `identity.send_text(to="+E164", text=...)`.
2. On the `[call_ended]` turn, create a recurring cron with `deliver: "local"`; the cron output is not the deliverable.
3. In the cron prompt/body, poll Inkbox live state via the SDK (`identity.get_text_conversation(third_party_phone, limit=...)`), not session memory.
4. Detect an inbound message from the third party after the outbound question. Only then send the promised email to the caller with `identity.send_email(...)`.
5. If no reply exists yet, do nothing except print a local diagnostic such as `no reply yet`. Do not send interim SMS/email status updates and do not double-confirm.
6. Include enough context in the cron prompt to identify the original outbound question, third-party phone, caller email, and desired email subject/body.

This pattern preserves channel cleanliness: the user gets one email only when the requested fact is available, while polling noise remains local.

## Scheduling outbound voice calls from an Inkbox-routed Hermes session

If asked to “call me in N minutes,” schedule a cron job that places the call at the requested time. The call itself is the deliverable, so set cron `deliver: "local"` when the tooling supports it; do **not** send a second SMS/email “Calling you now” unless the user explicitly requested a text result.

The SDK call needs a `client_websocket_url`. In the hosted Hermes Inkbox adapter, the current websocket URL is written to the Inkbox Hermes home as `inkbox_identity_state.json` with `ws_url`. Use `$HERMES_HOME`, not `Path.home() / ".hermes"`, because Inkbox deployments may run with an isolated home such as `~/.hermes-inkbox`.

**Development deployment auth pitfall:** tool sandboxes may not inherit `INKBOX_API_KEY`, and terminal environments may carry a key without the matching `INKBOX_BASE_URL`, producing `HTTP 401: Unauthorized` against the production API. Before placing calls from a terminal, source the identity env file and pass `base_url=os.environ.get("INKBOX_BASE_URL", "https://inkbox.ai")` to `Inkbox(...)`. A known-good shell prefix is:

```bash
set -a
source "$HERMES_HOME/.env" 2>/dev/null || source /home/ec2-user/.hermes-inkbox/.env
set +a
```

Always attach call-purpose context with `?context_token=<token>`. A fresh in-call agent may be spawned when the callee picks up; without explicit context it can sound like a stranger and ask why it is calling. The first spoken turn should greet the callee by name and open with the reason/topic, e.g. “Dima, I’m calling because you asked me to call in a minute to talk about the YC application.” Do not open with “What do you want to talk about?” on an outbound scheduled call.

Minimal scheduled job body:

```bash
/home/ec2-user/inkbox-powered-hermes-agent/.venv/bin/python3 - <<'PY'
import json, os, secrets, sys
from pathlib import Path
from urllib.parse import urlencode
sys.path.insert(0, '/home/ec2-user/inkbox/sdk/python')
from inkbox import Inkbox

home = Path(os.environ.get('HERMES_HOME') or (Path.home() / '.hermes'))
state = json.loads((home / 'inkbox_identity_state.json').read_text())
ws_url = state['ws_url']

token = secrets.token_urlsafe(16)
ctx_dir = home / 'inkbox_call_contexts'
ctx_dir.mkdir(parents=True, exist_ok=True)
(ctx_dir / f'{token}.json').write_text(json.dumps({
    'reason': 'Dima asked via SMS to be called back in 1 minute to talk about the YC application',
    'opening_line': 'Dima, I’m calling because you asked me to call in a minute to talk about the YC application.',
    'conversation_summary': 'Scheduled outbound call from the SMS thread; open with the reason and then continue interactively.',
}))
sep = '&' if '?' in ws_url else '?'
ws_with_ctx = f'{ws_url}{sep}{urlencode({"context_token": token})}'

with Inkbox(api_key=os.environ['INKBOX_API_KEY']) as ink:
    identity = ink.get_identity(os.environ.get('INKBOX_IDENTITY', state.get('handle', 'inkbox-on-call-agent')))
    call = identity.place_call(to_number='+15167251294', client_websocket_url=ws_with_ctx)
    print(f'Placed outbound call: call_id={call.id} status={call.status}')
PY
```

After scheduling, reply briefly once in the originating thread: “Scheduled — I’ll call you in N minutes about <topic>.” The cron should perform the call locally and avoid double-confirming via SMS/email.
