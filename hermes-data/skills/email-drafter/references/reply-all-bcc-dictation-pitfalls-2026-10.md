# Reply-All / BCC / Dictation-Conflict Pitfalls (2026-10-17)

Condensed knowledge from the Manish "Call for a meeting" reply-all+BCC draft and the
Anbu "Servant room tile repair" reply. Companion to `templates/reply-all-bcc-threaded.py`.

## 1. Native reply tool ignores `to=` — hand-build MIME for reply-all / BCC

The native `gws_gmail_reply_draft` / `gws_dwd_gmail_reply_draft` tool IGNORES its `to=`
parameter and always addresses the reply to the original sender only. Verified 2026-10-17:
called with `to="mdr@draas.com, dharmesh.ranka1@gmail.com"`, `bcc="rnr@draas.com"`; the draft
came back `to: Manish Ranka <mdr@draas.com>` — Dharmesh silently dropped, no Cc either.
It DOES honour `bcc`. The bridge's `draft_reply_create` has no `bcc` parameter at all.

So for reply-all and/or BCC, use `templates/reply-all-bcc-threaded.py`:
- fetch the original's `Message-ID` via metadata format (NOT the thread id);
- set `To` / `Cc` / `Bcc` / `Subject` / `In-Reply-To` / `References` explicitly;
- `drafts().create(body={"message": {"raw": raw, "threadId": orig["threadId"]}})`.

Reply-all recipient rule: To = original sender + the other original To/Cc participants,
minus yourself. Manish → NDR + Dharmesh: `To: mdr@draas.com, dharmesh.ranka1@gmail.com`.

Verify with `drafts().get(format='full')` and print To/Cc/Bcc. Gmail does NOT strip `bcc` on
an UNSENT draft — you can assert on it directly (`assert hb.get('bcc') == 'rnr@draas.com'`);
the "Gmail strips Bcc" claim holds only for sent mail. If a bad draft was already created,
delete it (`drafts().delete`) so only one draft sits on the thread.

## 2. A dictated term that contradicts a standing preference: KEEP, then FLAG

NDR's standing note (memory/USER.md) is to avoid the word "liquidation" in DRA wind-down
correspondence. In dictation he nevertheless used "our liquidation" and "liquidation
efforts" twice. Do NOT silently substitute the preferred word — the dictation is the
instruction, and swapping a corporate/legal term changes meaning he may have chosen
deliberately. Draft his exact words, then put ONE line in the chat reply: "you dictated X
twice, my standing note says avoid it — say the word and I'll swap it." Same rule for any
dictated term that memory or USER.md flags. Corollary: normalise obvious dictation slips to
the standard phrase without ceremony ("part and part of our liquidation" → "part and parcel
of our liquidation") and still mention that you did, in one line.

## 3. Rebut the counterparty's OWN enumerated commitments, point by point

When NDR dictates a course-correction, the counterparty has usually just replied with a
numbered list of what THEY will do. Read their list and check every item against the new
instruction. Silently bolting the new instruction on top of a reply that leaves their
contradicting commitment standing means the draft half-agrees — and their commitment wins,
because it is the thing they actually offered.

Anbu's item 2 was "I will personally visit 2401 Prestige Hermitage"; NDR's instruction was
"do not take it up yourself". Lead the draft with the reversal, then give the replacement
process. Then, in the chat reply, name the specific numbered item you overrode so NDR can
see the conflict was caught.

## 4. Reply-all "AGAIN" on a thread that already carries a Bcc — mirror it, then FLAG it

When NDR says "reply all again" on a chain where one of HIS earlier drafts already carried a
Bcc (this thread: `rnr@draas.com`), the default is to **carry the same Bcc forward** and say so
in ONE line of the chat reply, so he can drop it if this particular message should go to the
brothers only. Do not silently drop it (the audience changes without him knowing) and do not
silently keep it either. Confirmed 2026-10-19: second reply-all on the "Call for a meeting"
thread reused `To: mdr@draas.com, dharmesh.ranka1@gmail.com` + `Bcc: rnr@draas.com` and the
Bcc was flagged in the reply.

Before deciding, look up what the thread's earlier draft used — `references/dra-family-windup-matters.md`
and §1 above both record the 17-Oct-2026 draft on this thread as that exact recipient set.

**Tooling:** `scripts/create_threaded_reply_draft.py` has **no `--bcc` flag**. For a reply-all
that must carry a Bcc, hand-build the MIME: `EmailMessage` + `set_content(text, subtype="plain",
charset="utf-8")` + `add_alternative(html, subtype="html", charset="utf-8")` + explicit
`To`/`Cc`/`Bcc`/`Subject`/`In-Reply-To`/`References`, then
`drafts().create(body={"message": {"raw": ..., "threadId": orig["threadId"]}})`. Fetch the
source `Message-ID` from `format="raw"` (metadata can return it empty); the `threadId` comes
from the resource, not a header.

**Verify the BODY, not just the headers.** After creating, walk
`drafts().get(id=..., format='full')` for the `text/html` part, base64url-decode it, and assert
on character length + expected `<b>` count. A draft whose To/Cc/Bcc and `[text/plain, text/html]`
parts all read back correctly can still carry a truncated body — the decode is one call and it is
the only proof of what NDR will actually see on screen. Then re-run `drafts().list()` and confirm
exactly ONE draft sits on the thread (a stale draft from an earlier turn is the usual duplicate).
