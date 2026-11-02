# HTML threaded reply draft + bridge call style (validated 2026-09-15)

Pattern proven on the AR Portable quote follow-up (vendor quotation gap analysis → HTML reply draft).

## HTML threaded reply draft

To reply to an inbound message with a rich HTML body, use the bridge's `draft_reply_create` with
`html=True`. It sets To (from original From), Subject `Re:`, In-Reply-To/References and threadId
automatically:

```bash
HERMES_SESSION_USER_ID=7449813913 /opt/hermes/.venv/bin/python3 -c "
from tools.gws_skill_bridge import call
body = open('/tmp/reply.html').read()
print(call('draft_reply_create', service_name='google-draas',
           message_id='<SOURCE_MSG_ID>', body=body, html=True))"
```

## Pitfalls

1. `call()` takes **kwargs, NOT a dict as second arg**. Passing a dict
   (`call('draft_reply_create', {'message_id': ...})`) makes the vault read the dict
   as the service name and fails with
   `VaultError: Invalid or missing service name: '{...}'`.
2. The operation signatures differ per op — `draft_reply_create(args)` reads
   `args.message_id`, `args.body`, `args.html`; NEVER call `func({'k': v})` directly
   (AttributeError: 'dict' object has no attribute 'message_id'). Always go through
   `call(op, service_name=..., **kwargs)`.
3. Verify the draft landed on the RIGHT thread: fetch the source message first to learn
   its `threadId`, then `drafts().get(draft_id, format='full')` and compare
   `message.threadId`; also confirm In-Reply-To/References headers are present so the
   email is properly threaded.
4. For quote-gap follow-ups the draft stays draft-only — never auto-send.

## Bridge call argument style (general)

- `call(operation, service_name=..., **kwargs)` — every op arg is a keyword,
  not a nested dict.
- `sheets_get`/`sheets_update` use `sheet_id` (NOT `spreadsheet_id`).
- `sheets_update` wants `values` as a JSON **string** (`json.dumps([[...]])`),
  not a Python list — passing a list raises
  `TypeError: the JSON object must be str, bytes or bytearray, not list`.