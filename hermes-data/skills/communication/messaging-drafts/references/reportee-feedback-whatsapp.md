# Reportee performance feedback — WhatsApp drafting recipe

NDR dictates firm-but-polite feedback to reportees (Anbu, Kantesh, …) that mixes
per-project update asks with a performance/productivity message. The tone
constraint is always the same: *"very polite, cooperative language — but it must
still say productivity at this level is not acceptable."* The 2026-09-22 Anbu
follow-up is the worked example (Udaya/Oasis/Iris/Northstar asks + productivity
paragraph + daily-updates mandate + month-end ultimatum).

## Tone formula (keep this order)

1. **Cooperative framing line** right after the greeting: *"Please read this in
   the collaborative spirit it's written in."*
2. **Per-project asks**, bold project headers, one paragraph each — make asks
   specific and small (photo + yes/no on a transformer beats "give me an
   update"). Separate "no problem / no update needed" items from open asks so
   the message reads as a genuine status round-up, not a pure complaint.
3. **Slippage note** without accusation: *"committed for yesterday — we cannot
   have slippage on days."*
4. **Feedback paragraph** — the bridge that lands the hard message softly:
   - credit the one thing that WAS delivered (e.g. officials visit + villagers
     managed),
   - then: *"at your level, your seniority, your position — that cannot be our
     productivity or our outcome every day"*,
   - the surprise bridge: *"I'm genuinely surprised, and I'm sure you also don't
     accept that productivity can be this low"*,
   - close with *"Let's work on fixing this together."*
5. **New mandatory requirement** stated as a done deal, not a request
   (*"Daily updates are now mandatory: targeted vs achieved per site, every
   day"*), and name who else has been told the same so it reads as a programme.
6. **Soft ultimatum with concession framing**: anchor in the trust NDR already
   gave (the classic is *"You asked me for three months; I gave you six"*),
   then the boundary (*"clean-cut budgets and the daily work plan — equipment,
   machinery, people — by month end, or I will have to step into engineering
   myself and bring in the necessary resources. I cannot allow this to go on."*).
   Keep explicit termination threats ("will not continue with X past month end")
   softened or flag them to NDR for a decision — in the 2026-09-22 draft the
   Kantesh threat was left implicit and NDR offered to make it explicit.
7. **Closing ask** (*"Let me know your thoughts, Anbu."*).

## Mechanics

- Caption for colleagues: first line `*Subject*` (e.g. `*Yesterday's Task
  Follow-up*`), per the WhatsApp caption rule.
- **Long messages split** — a ~2,470-char message came back as 2 parts. Before
  shipping, decode each part's `text` URL param (`urllib.parse.parse_qs` on the
  link) and confirm the boundary is at a paragraph break, not mid-sentence. The
  2026-09-22 split landed cleanly at "projects" → "feedback". Deliver each part
  as its own Telegram message, labelled Part 1 / Part 2, so Telegram's own
  splitter can't cut a link in half.
- **Voice decodes in these dictations:** "flow plan" → floor plan; "Aribind" /
  "Alvin" / "Irwin" → Arvind Jain; "Anamu" / "Andu" → Anbu.
- **Recipient number:** colleague rows often carry multiple unlabelled numbers.
  Use the roster/handle number, list the alternates, and ask NDR to confirm the
  number before he sends (see `anbu-daily-plan-project-vocabulary.md`).