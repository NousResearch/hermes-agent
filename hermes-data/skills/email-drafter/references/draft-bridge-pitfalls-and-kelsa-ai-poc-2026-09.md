# Draft-bridge pitfalls + Rakuten-Kelsa AI POC (2026-09-14)

## 1. gws_skill_bridge may return a JSON STRING, not a dict

`gcall("draft_reply_create", ...)` / `gcall("draft_create", ...)` sometimes return the
result as a `str` containing JSON (`{"status": "draft_created", "draft_id": "r...",
"threadId": "..."}`) instead of a parsed dict. Calling `.get()` on it raises
`AttributeError: 'str' object has no attribute 'get'` — **AFTER the draft was already
created** (the id is visible in the printed JSON). Hit twice on 2026-09-14 (Raghu SSA
follow-up and the Kelsa internal draft).

Guard before reading the id — never recreate the draft:

```python
res = gcall("draft_reply_create", service_name="google-draas",
            message_id=..., to=..., cc=..., body=...)
if isinstance(res, str):
    import json; res = json.loads(res)   # bridge returned a JSON string
draft_id = res["draft_id"]; thread = res["threadId"]
```

Then verify with `drafts().get(...)` after `time.sleep(2)`; `drafts().get` has no
`metadataHeaders` kwarg (known quirk).

## 2. "Reply to this email to X only" (internal recipient set) — use draft_create, NOT draft_reply_create

When NDR asks to respond to an external thread but ONLY to his own team/vendors
(e.g. Manohar + Kelsa team, no Rakuten), a threaded `draft_reply_create` inherits the
thread's participant set and risks landing in the external party's inbox. Use
`draft_create` as a NEW email, keep the context in the subject
("Rakuten - Kelsa Agentic AI + AI Wrapper: Review & Next Steps"), and mirror the
recipient set of the last NDR-sent email on the thread minus the excluded parties.

## 3. Voice garble of technical terms — normalize, don't echo

Voice dictation of technical/feature names invents proper nouns. Real garbles seen
2026-09-14: "AI Rapper" = **AI Wrapper**; "Aamir" / "Chichuraganpalli" = **KPI / SLA**
(not people or places); "Kewaar" / "Aneditor" = viewer/editor; "LLP which private
limited" = "LLP or private limited". Rule: render the sensible technical reading in
the draft and flag the mapping in the reply to NDR (he may supply the correction
himself in a mid-turn message — if he says a token is "all wrong, transcription
error", drop it and reuse the surrounding context).

## 4. "Draft from my O3 Infotech account" — capability check before promising

When NDR asks to send from a mailbox that isn't the session account: (a) probe vault
tokens first (`build_service("gmail","v1",service_name=...)` try/except across
candidate names — only google-draas/ahfl/gmail are linked; o3infotech is NOT), and
(b) check send-as aliases with `gmail.users().settings().sendAs().list(userId="me")`.
ndr@o3infotech.com has no token and no send-as alias → cannot draft from O3; state
the facts and leave the draft on draas. Also verify the user is actually marked at
that address on the thread (often they are only at ndr@draas.com / ndr@kelsa.io).

## 5. Rakuten-Kelsa AI POC context (briefing + internal reply, 2026-09-14)

- **Where the mail lives:** all in ndr@draas.com, forwarded by Manohar Singh sending
  from msingh@o3infotech.com (his O3 address — the reason NDR thinks it's the O3
  account). Original Rakuten mail (24-Aug "Kelsa (AI capability Demo)") To'ed
  ndr@kelsa.io and Cc'd ndr@draas.com. NDR is NOT marked at ndr@o3infotech.com.
- **Customer:** Rakuten Symphony India (a Rakuten Group company), Bangalore. Contacts:
  Anantha Prasanna (HR Transformation), Andrea Leena (HR Transformation Analyst),
  Prajakta Kanaglekar, Sunil T V, Satish Raju Raelekar (all @rakuten.com).
- **Requirement (28-Aug "Kelsa - AI - POC"):**
  1. **AI Agent on R-Social** — stage/status/pending/budget queries; raise a request
     (create a new process item); complete a task assigned to them/their team; upload
     bills + attach docs to the Kelsa item. NDR's verdict: all already possible for
     any Kelsa item the user can view (follower/viewer/editor roles).
  2. **Budget reports** — NDR uncertain: if Kelsa generates it from a template, the
     agent fetches the report; if manual action, must test. Needs the specific use
     case from the customer.
  3. **AI Wrapper for Dashboard** — QMS KPI/SLA ticket metrics + attendance
     regularisation dashboards. Dashboard wrapper DELIVERED 2-Sep and live at
     https://rakuten-qms-dashboard.onrender.com/. Ashwin owns the open question:
     does the MCP expose dashboard/tabular data so we can query it, answer the
     customer, and render HTML/CSS charts (customer changes charting look & feel;
     delivered on Slack/Viber if the channel renders HTML/CSS)?
  4. Customer also asked for the **cost/commercial model** for extending services.
- **Latest state:** Rakuten (Andrea Leena, 8-Sep) requested the AI Wrapper demo call
  (proposed 9-Sep 1:30 PM) and asked who presents. No confirmed outcome in the chain.
- **Kelsa team addresses (verified from thread headers):** Manohar Singh
  msingh@o3infotech.com / msingh@kelsa.io (MD); Aagney Singh aagney@kelsa.io;
  S Pavan Kumar pavan@kelsa.io; Guna gunarka@kelsa.io; Yogesh yogesh@kelsa.io;
  Ashwin Hegde ashwin@kelsa.io (also hashwin@o3infotech.com). Full roster table also
  lives in `kelsa-mcp` → `references/crm-multi-account-and-pitfalls.md` §14.