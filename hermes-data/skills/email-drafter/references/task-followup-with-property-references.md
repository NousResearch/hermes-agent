# Task follow-up email that packages property references (location map + property details)

Validated 2026-09-09 on P-121 (Sinchana, Samudra 7-acre ORR site sketch). Trigger: NDR says "send the location map and the property [details] as a follow-up email" for a task, plus a WhatsApp message confirming the email with the property numbering.

## Recipe

1. **Locate the task + its artifacts (they may NOT be in Kelsa).** Start with the tracker sheet (personal-task-tracker skill) by task ID. NDR's voice phrase "from the kelsa" may refer to a Kelsa team-chat context, not a data record — the Samudra ORR property had NO Kelsa lead. Sweep Drive fullText (`fullText contains 'Samudra'`, `fullText contains 'forward purchase'`) to find the property/valuation sheet, and treat the video brief as the location map unless a real map file exists.
2. **Check for an existing unsent draft FIRST** (`drafts().list` + match subject keyword / recipient). If the same task already has a draft (e.g. created earlier that day), DELETE it and recreate the enriched version — exactly one draft per task. Duplicate drafts get sent twice.
3. **Verify the video brief without YouTube bot-block:** `curl -s "https://www.youtube.com/oembed?url=<url>&format=json"` returns title/author even when yt-dlp fails with "Sign in to confirm you're not a bot". Confirms whose walkthrough it is (e.g. "4 September 2026" by "Nishant Ranka (Geoavatar)").
4. **Extract the property summary from the reference sheet** (`sheets.spreadsheets().values().get` on each tab): extent (acres + sq ft), FAR/BUA, and the indicative value figures to quote in the email body. Do not invent numbers the sheet doesn't have.
5. **Draft as HTML** (colleague instruction-brief rule): subject keeps the task ID (P-121) so replies track in email; body = recap + video link + Drive reference link + property summary bullets + deliverable recap + "reply on this thread" ask. Verify `drafts().get`: mimeType text/html, From = ndr@draas.com, exactly one draft with the task ID remains.
6. **WhatsApp confirmation link** via `whatsapp_link` (colleague caption `*Project — Subject*`): confirm "email sent on Task P-xxx", name the property reference. When no survey/plot number exists anywhere, use the property name + extent + task ID as the numbering and FLAG to NDR that a survey number would need to come from him — never invent one.
7. **Update tracker row** (J = today, append dated note with draft id + link + recipient), Status stays Ongoing.

## Worked example (P-121)

- Tracker row 84; project "ORR Samudra — 7-acre forward purchase proposal"
- Artifacts: Drive sheet **"ORR Samudhara Land Forward Purchase Simplified Calculation"** (`1YH9_yY_C0BkYWF7Ue1462gQMA_FG-X7h7N2LdEtey6w`, owner nishantranka@gmail.com, shared with Nishant Prakash) + video https://youtu.be/xlYJeLHnu3E (Geoavatar 4-Sep-2026)
- Property summary quoted: 7.00 ac / 3,04,920 sq ft; FAR 5.20x (3.25 + 1.30 + 0.65) → 1,585,584 sq ft BUA; indicative land value ₹334.97 Cr ≈ ₹47.85 Cr/ac
- Final draft id `r8065197425280805601` (replaced the morning draft `r7722162887162937410`)

## Pitfalls

- `drafts().delete(draft_id)` uses the DRAFT resource id, not the message id.
- "From the kelsa" ≠ a Kelsa lead exists. Check tracker + Drive before concluding the record is missing.
- The WhatsApp message may say "email sent" per NDR's dictation, but the email is a DRAFT — tell NDR to send it first, then tap the link. Hermes never sends autonomously.
- Property "numbering": if the user means a survey/plot number and none exists in records, do NOT fabricate — the honest reference is property name + extent + task ID, and ask NDR for the survey number to add it.