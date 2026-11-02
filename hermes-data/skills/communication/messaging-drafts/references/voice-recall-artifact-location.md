# Voice recall referencing a prior-session artifact ("that render we made") — locate, don't regenerate

Verified 2026-09-23 (Ankush/Infimax annotated-sketch reply to broker Ankush Musaddi, Kadathanamale land proposal).

## When this fires
NDR dictates a follow-up that references an OUTPUT produced in an earlier session — "send him that render with the survey numbers highlighted", "the proposal email", "the sketch he shared". No file path, no broker name necessarily. The deliverable is a draft USING that artifact; the artifact already exists on disk.

## Recipe
1. `session_search(query="<broker/project keyword> <artifact words>")` — add `sort='newest'`; these artifacts are normally from a session within the last few days. If the first query returns nothing, try the entity/broker name or the artifact type ("annotated", "render", "sketch").
2. Scroll into the found session (`session_search(session_id=..., around_message_id=<match>)`) and read the final assistant message(s). Completed renders/documents are typically delivered with a `MEDIA:/absolute/path` line — that path IS the artifact.
3. Confirm the file exists on disk, then use it verbatim in the draft's MIME attachment (threaded-reply-with-attachment recipe: `references/gmail-draft-attachment-verification.md`).
4. Ask NDR for the path only if the search genuinely fails — never reconstruct/guess a filename.

## Worked example
NDR: "From the proposal by Ankush Infimax earlier, we prepared that one render with the survey numbers highlighted... send it back to Ankush... there seems to be a gap in the survey number."
- `session_search("Infimax")` newest → producing session `20260923_064257_c33443a1` ("Verify Ankush's Land Proposal") + related Kelsa record #55755331.
- Final assistant message delivered `MEDIA:/data/hermes/cache/ankush_nel_land/Annotated_10Ac_Block_Kadathanamale.png`.
- That exact PNG went on the reply draft (attached via EmailMessage, verified with `drafts().get(format='full')` → filename + 3,467,342-byte body size).

## Pitfalls
- Do NOT re-render/regenerate the artifact — the user wants the version they already reviewed.
- The `MEDIA:` path in the producing session is more reliable than Kelsa S3 or Drive copies for a draft attachment (local file, no URL expiry).