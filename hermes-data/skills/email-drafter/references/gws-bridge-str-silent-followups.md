# gws_skill_bridge str-return + silent-counterparty follow-ups

Session: 2026-09-14 — Raghu Iyer (Ranka Amber SSA) 3rd follow-up draft, thread 19ed0777ef8c10f9.

## 1. `gws_skill_bridge.call()` may return a JSON STRING, not a dict

After `gcall("draft_reply_create", service_name="google-draas", message_id=..., to=..., cc=..., body=...)`:

```python
res = gcall("draft_reply_create", ...)
print("DRAFT RESULT:", res)          # prints {"status": "draft_created", "draft_id": "...", "message_id": "...", "threadId": "..."}
draft_id = res.get("draft_id")       # AttributeError: 'str' object has no attribute 'get'
```

The bridge returned serialized JSON (earlier sessions got a parsed dict — return type varies by bridge build/context). Normalize:

```python
if isinstance(res, str):
    res = json.loads(res)
draft_id = res.get("draft_id") or res.get("id")
thread = res.get("threadId") or res.get("thread_id")
```

Recovery if the crash already happened: the printed `DRAFT RESULT:` text carries `draft_id` and `threadId`. Read them from the output and run the verification pass as a SECOND script (`gmail.users().drafts().get(userId="me", id=draft_id, format="metadata")` → headers from `d["message"]["payload"]["headers"]`, then `format="full"` body keyword assertions). The draft was already created — do NOT recreate it or you'll leave a duplicate in Drafts.

## 2. Silent counterparty — anchor `message_id` on THEIR latest message

When NDR asks for another follow-up because the counterparty hasn't replied, the thread's newest message is usually NDR's own sent mail. Pick the anchor as the counterparty's latest message, NOT the thread-latest, so In-Reply-To/References point at their message and the new draft reads as a nudge on their pending promise.

- Anchor: counterparty's latest message id (here `1a085e90c2ac9cdc` — "returning from Europe… will come back with possible dates", 9-Sep).
- Always pass explicit `to=` and explicit `cc=` mirroring the previous sent mail's recipient set (here To Raghu Iyer oz.iyer@gmail.com, Cc Manohar Singh msingh@redsoul.co.in only — the 9-Sep sent mail dropped Prakash Singh).
- Structure: reference the earlier mails by date ("my earlier emails of 1 and 9 September"), acknowledge their stated plan ("you mentioned you would come back with dates once home"), then press: site about to come to a standstill → contractor will start imposing idling charges → request a confirmed date (even one day) → once confirmed, plan with contractor + line up bank in parallel so fund flow starts the moment the SSA is registered → protects March structure completion and pre-December OC targets.
- Register the draft in the tracker (P-085 read-modify-write on `All Tasks!A{r}:R{r}`, J = today, append `[YYYY-MM-DD, via NDR/telegram]` note with draft id, thread id, recipient set, and NOT-sent status).