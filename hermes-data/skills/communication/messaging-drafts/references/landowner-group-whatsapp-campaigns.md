# Landowner-group WhatsApp campaigns (DRAAS JDA projects)

Use when NDR dictates a message for a **landowner WhatsApp group** (Ranka North Star / Allalasandra, Ranka Amber, Ranka Udaya, …) and wants it delivered as tap-to-send links. Also applies to individual landowner messages (Swaminathan, Sundar Padmanabhan, Vaidyanathan) and to internal staff messages (Bharat, Prakash, Roshni).

## Workflow — do it in this order

1. NDR dictates the substance by voice. It is long, multi-topic, unstructured, self-correcting, and full of mangled proper nouns. Do not try to mirror his dictation — restructure it into sections.
2. **Draft the FULL message as text in the chat FIRST and wait for approval.** Do not lead with links. He reviews and iterates on wording — he asked for "informal", "polite", "no long history", "cozy and readable", and once asked for "all psychology of human psychology elements" so landowners feel positive and come forward.
3. Offer two drafts (a full version and a concise version) when he asks for a choice. He picks.
4. When he corrects a paragraph, re-show **only that paragraph** if he says "just show me the two objective portion" — do not re-dump the whole message.
5. Generate links only after approval. He taps and sends himself — **never assume anything was sent, and never claim you sent it.**

## Group messages have no phone number

Generate the link **without a phone** so NDR picks the group inside WhatsApp. Long messages get split into parts — deliver **each part as its own link in its own chat message, in order**, labelled "Part 1 of 3", etc.

Hard rules on links (non-negotiable): use the `whatsapp_link` tool when it is available in the session; never hand-build `wa.me` or `api.whatsapp.com` URLs; never pre-strip, escape or "clean" the message text before passing it (punctuation, `&`, `#`, `(`, `)`, `'`, `+`, emoji and newlines are all handled by the tool).

## Formatting that lands well

- Bold caption headline: `*🙏 RANKA NORTH STAR — <TOPIC> 🙏*`
- Bold section headings: `*WHAT WE HAVE ACHIEVED*`, `*TWO OBJECTIVES*`, `*ONE CRITICAL DECISION*`
- `•` bullets, `1)` numbered items, blank line between paragraphs
- Close: `With gratitude,` then `*Nishant Ranka*`
- **Shorter is better for landowners.** He repeatedly asked to cut the history. Give the conclusion and the ask, not the chronology.
- Warm / auspicious opening when the occasion warrants it — e.g. a Ganesha Chaturthi greeting, then the serendipity of good news landing on that day, then "With that, let me share where we stand:".
- Register per audience: polite and deferential to landowners ("Sundar sir", "Vaidyanathan sir", "Swaminathan sir", "I request you to kindly…"); crisp and directive to internal staff. Internal colleagues get a bold caption; external landowners get none.

## Voice-transcription hazards — ALWAYS reconcile names against the executed documents

| He says | It is |
|---|---|
| "legal ears" | **legal heirs** |
| "Kanshika" | **Kankshita** Swaminathan |
| "Alal Sandra" / "Allalsandra" | **Allalasandra** |
| "Sunder / Sundar Padmanabhan" | Sunder **Padmanabhan** (spellings vary by document) |
| "Sriram" | **Sreeraman** Vaidyanathan |
| "Vedyanathan" | **Vaidyanathan** |
| "Venkatraman" | **Venkataraman** |
| "Rukmini" (wife) | check against the JDA — the JDA records **Geeta as wife of V. Swaminathan** |

Numbers fail the same way: he says "row 224" when the sheet says **244**, or "TEST 17" when the file is **TEST_7**. **Verify every row number, document number and file name against the source before it goes into an external message.**

## Two failure modes he has pulled me up on

- **Never assert a fact from recollection of a document.** Read the sheet or document and cite the tab + row, or the file name + id. He asks "which index are we referring to, which row" and expects a precise answer — an unsourced assertion destroys trust in the whole reply.
- Reconcile forward-looking dates against what the counterparty actually said in the chat export before drafting. And when a message depends on someone else's commitment (a flight, a registration date), read the export rather than inferring.

## Recurring content patterns for JDA landowner groups

- Critical-milestone updates (FAR approval, in-principle approval, sanction timeline)
- All-hands meeting calls: fixed date, "set aside the whole day", logistics arranged by us (lunch, transport to the Sub-Registrar and back, overnight accommodation offered to out-of-town landowners)
- Asking for confirmation that a date works, so registration can be locked
- Recording binding positions ("whatever you all decide is binding on me") so there is no later dispute
- Explaining a legal/procedural obstacle politely without the long history (e.g. an unregistered GPA needing re-execution through the Indian Consulate)
- Requests for documents or scans from staff, with the exact index row and physical file number

When a message carries a legal position, mirror it in an email to the same parties for the record — NDR's practice is "I am putting it in an email so there is no confusion".
