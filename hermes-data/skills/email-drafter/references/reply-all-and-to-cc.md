# Reply-all, and controlling To/Cc on a threaded draft

## The trap

`gws_gmail_reply_draft` (the native tool) has **no `to` parameter**. It builds To from the
original message's `From` header.

So when the last message in a thread is one **NDR himself sent** — which is the normal case,
because when he says "reply all on that thread" he is usually following up his own mail — the
draft is addressed **To: ndr@draas.com (himself)**. It looks fine in the Drafts folder and the
mistake is silent.

`gws_gmail_get` on the draft shows it immediately:

```
"to": "Nishant Ranka <ndr@draas.com>"     <-- wrong
```

Never report a reply-all draft as ready without reading the addressing back.

## The fix — use the bridge operation, which honours `to` and `cc`

```python
import sys
sys.path.insert(0, "/opt/hermes")
from tools.gws_skill_bridge import call

out = call(
    "draft_reply_create",
    service_name="google-draas",
    message_id="<id of the message being replied to>",
    to="mdr@draas.com",
    cc="dharmesh.ranka1@gmail.com",
    body="<html body>",
    html=True,
)
print(out)
# {"status": "draft_created", "draft_id": "...", "message_id": "...", "threadId": "..."}
```

`draft_reply_create` sets `In-Reply-To` / `References` from the original's `Message-ID`, prefixes
`Re:` if absent, and pins `threadId` on `drafts().create` — so it stays threaded. It never sends.

**Signature pitfall:** `call(operation, service_name="google-draas", **kwargs)`. Passing the
arguments as one positional dict puts the dict into `service_name` and produces a misleading
`Invalid or missing service name: "{...}"` *credentials* error. Always use kwargs, and always pass
`service_name=` explicitly.

## The mirror trap — latest message addressed NARROWLY

Above is the case where the latest message is NDR's own. The mirror case: the latest message is
a colleague's reply addressed **To: NDR only**, while the thread carries other participants who
must stay on it.

Do NOT take the reply set from the latest message. Take the **union of everyone on the thread**,
then match **NDR's own addressing from his most recent message in that thread** — that is the
set he means by "all", and it is usually wider than the last message's To.

Worked example — OASIS saucer drain thread, 19 Oct 2026. Thread was:
Sinchana → (Bhuvanesh ×2); NDR → (Sinchana + Bhuvanesh ×2); Sinchana → (NDR only).
NDR dictated "please reply all". Because his own last mail carried Sinchana + both Bhuvanesh
addresses, the draft went To: all three. Reading the latest message alone would have produced a
reply to `ndr@draas.com` (himself) or to Sinchana alone — silently dropping the exact person he
was escalating the deadline to.

Full worked reply: `deadline-and-capacity-pushback-replies.md`.

## Working out who "reply all" actually means

Do not infer it from the `From` of the last mail. Pull the thread and read real headers:

```
gws_gmail_search(service_name="google-draas", query='subject:"<subject>"')
gws_gmail_get(service_name="google-draas", message_id="<latest id>")
```

Then decide To/Cc from everyone who is actually on the thread.

Worked example — Prestige Hermitage khata thread, 16 Oct 2026: the latest message was NDR →
Manish only, but Dharmesh was on the thread and NDR wanted him copied. Answer:
**To: mdr@draas.com, Cc: dharmesh.ranka1@gmail.com**, same subject, same threadId.

## Verify, then report the addressing

```
call("draft_get", service_name="google-draas", draft_id="<id>")
```

Confirm `to`, `cc`, `subject`, `threadId`. Report those four back to NDR in the same message that
says the draft is ready.

## Wrong recipient already drafted

`gws_gmail_draft_delete(draft_id=...)`, then recreate with explicit `to`/`cc`. Do not leave two
drafts in the folder.
