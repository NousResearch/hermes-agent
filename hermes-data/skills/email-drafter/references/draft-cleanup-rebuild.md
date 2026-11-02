# Draft clutter cleanup & rebuild (verified 2026-10-06, Ranka Amber RERA email)

## Trigger
NDR reports his Gmail Drafts are messy: "there are two empty drafts and one is still the
old non-HTML version" (or any variant of wrong/duplicated/stale drafts).

## Rule
Do NOT just create another draft on top of clutter. Sweep first:

1. **List** — `gws_gmail_draft_list` (service_name=google-draas). Note every draft id.
2. **Classify each** — `gws_gmail_get(message_id=...)` per draft:
   - *Empty draft*: `to: ""`, `subject: ""`, `body: ""`, snippet `""`.
   - *Stale draft*: correct To/Subject but old body format/version (or body from an earlier
     iteration of the same email).
3. **Delete all** — empty AND stale. `gws_gmail_draft_delete(draft_id=...)` for each, in parallel.
4. **Create exactly one fresh draft** with the final content.
5. **Verify** — `gws_gmail_draft_list` again must return `count: 1` + the new id, and
   `gws_gmail_get` must show To/Subject/From correct and snippet non-empty.

## Pitfall — gmail_get body is empty for drafts
`gws_gmail_get` / `gws_gmail_thread_get` return `"body": ""` for DRAFT messages even when
the draft has real content (headers + snippet still populate). Do not conclude the draft is
empty from the body field alone — check `snippet` (first ~100 chars of content) and To/Subject.

## Pitfall — empty drafts accumulate
Empty drafts appear from abandoned compose windows / interrupted create calls. They are safe
to delete. Always verify final count == 1.

## Technique — recover the EXACT previously-written body from session history
When you need the verbatim body of an email you wrote earlier (e.g. to rebuild it as HTML),
do not re-invent it:

1. `session_search(query="gmail_draft_create <distinctive subject words>", role_filter="user,assistant,tool", sort="newest", limit=5)`.
   The discovery result's `snippet` points to a tool message whose content embeds the
   `gws_gmail_draft_create` call with `"arguments": {..., "body": "..."}` — the FULL body
   is in the tool-call JSON inside the message content.
2. Huge results are saved to `/tmp/hermes-results/call_*.txt` — it's a JSON blob on one line.
   Parse with Python (`json.load`, walk `results[0].messages[]`, find the message whose
   content contains `gmail_draft_create`, extract `tool_calls[].function.arguments`), not
   grep, to avoid dumping 100KB+ into context.
3. Nested `session_search` scroll calls inside message content are normal — one discovery
   result may itself contain a scroll JSON; keep unwrapping until you hit the tool call args.

## HTML/CSS preference for structured email (NDR)
For emails with tabular/structured content (RERA correction tables, infrastructure
checklists), plain text with pipe-table rows is "not very readable" — NDR explicitly asks
for HTML/CSS. Build real `<table>` with:
- inline styles only (Gmail-safe), `border-collapse`, bold navy header band
- one row highlighted green for the single applicable/mandatory item
- red bold `No` styling for every non-applicable cell
Keep the same To/Cc/Subject/from and the same wording — only the presentation changes.