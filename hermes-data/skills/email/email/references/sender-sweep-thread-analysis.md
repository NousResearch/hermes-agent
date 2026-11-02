# Sender-Sweep Thread Analysis (all-threads-from-one-sender deep dives)

Trigger: "review/dig into the emails from <person> about <topic>", "analyze this series of threads",
"he sent me a bunch of emails on <date> — give me feedback on each thread".

Proven on: Flameback Capital advisory review (Sep 2026) — 5 threads from kishan@flamebackcapital.com
(Dovetail, US RIA License, Team and Costs, Flameback Product, Long Shots) + a Google-Sheets-linked budget model.

## Steps

1. RESOLVE THE REAL SENDER — never trust the voice-transcribed name. The user says "Kishan, Murjani Nair"
   and the mailbox owner can be anyone. Sweep candidates:
   - `gmail.users().messages().list(q='from:<firstname> after:<date>')` (matches display name)
   - `q='from:<domain> after:<date>'` (e.g. from:flamebackcapital.com — catches everyone at that domain)
   - Match on the actual `From` header of returned messages, not the user's spelling.
2. ENUMERATE THREADS: pull metadata (format='metadata', metadataHeaders=['From','Subject','Date','To'])
   for every hit, group by threadId, and print `id | From(≤60) | Subject(≤90) | Date`. This surfaces the
   full set (including threads the user forgot to mention) and makes exclusion easy
   ("excluding the Nvidia GTC takeaway email on the 15th" = one line in the listing).
3. PICK THE SET by subject + date, then fetch each with format='full'.
4. BODY EXTRACTION: prefer text/plain; many B2B/vendor mailers are HTML-only with EMPTY text/plain.
   Fallback: recursive find_part(payload,'text/html'), decode base64url, strip tags
   (`<br>` → `\n`; `</p|div|tr|li|h1|h2|h3>` → `\n`; drop remaining `<[^>]+>`), html.unescape,
   collapse 3+ newlines. Truncate long emails at ~6000 chars in the first pass, then re-fetch
   remaining body on demand.
5. ATTACHMENTS THAT AREN'T ATTACHMENTS: budget/team/cost emails frequently carry the data as a
   Google Sheets LINK in the body (`https://docs.google.com/spreadsheets/d/<ID>/edit`), not as a file
   attachment. Read via Sheets API:
   - `sheets.spreadsheets().get(spreadsheetId=...)` → tab list + row/col counts
   - `values().get(range="'<TabName>'!A1:Z<max>")` per tab
   - Print only non-empty rows WITH row numbers (1-indexed) so the user can cite sheet+row.
   Multi-tab budget models are the norm — tabs usually mean: current vs promise vs cash-outlay state,
   dream/full-plan org chart, project-to-resource mapping, and empty placeholder tabs. Dump them all;
   the user will ask "what is sheet 2/3?".
6. VERIFY FACTS before handing the case to an analysis model: quick web checks for regulatory minimums
   (SEBI PMS ₹50L ticket / ₹5cr net worth), index benchmarks (Nifty 5Y CAGR), and counterparty existence
   (broker/hedge-fund claims). This lets the final analysis separate verified facts from claims.
7. Carry FULL message IDs end-to-end (16 hex) — never truncate in prints/diagnostics.

## Pitfalls

- Voice amounts come through mangled ("five crows" = ₹5 crore). Keep the user's spoken number as a hint,
  verify against the actual sheet/email.
- 'from:<firstname>' can return 0 hits when the display name differs ("Murjani" vs "Muniraj"); sweep the
  domain instead.
- Sheet tabs no one mentioned ("Copy of Sheet2") contain the sub-story (people→projects mapping);
  read them anyway.
- The account matters: for NDR's mail always force HERMES_SESSION_USER_ID=7449813913 (default resolves to
  another mailbox).