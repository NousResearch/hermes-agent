# One dictation, many channels — the fan-out pattern (verified 2026-10-14)

NDR routinely dictates **one** body of content and expects it delivered over **several channels at
once**, each addressed differently. The mistake is to treat it as one action; it is 3–5 artifacts
with *different* ship/stage rules.

## Worked example — Ranka Iris transformer charging (2026-10-14)

Dictation, condensed: *"I need a WhatsApp message and an email sent to Bhavik Ranka. Mark to Piyush
Ranka… I will put it on the Iris engineers group as well as I want one for Bhavik directly. And
email mark to Bhavik and Piyush… let's mark Anbu as well in the CC… And the same message to Anbu as
well… You can send that message to him on Telegram instead."*

That is **four deliverables**:

| # | Artifact | Channel | Rule |
|---|---|---|---|
| 1 | WhatsApp message → Bhavik | `whatsapp_link` deep link | **INERT — ship same turn** |
| 2 | Email → Bhavik, Cc Piyush + Anbu | Gmail draft | **INERT — ship same turn** |
| 3 | Same text for the "Iris engineers group" | WhatsApp group, **NDR posts it himself** | hand him **paste-ready plain text**, no link needed |
| 4 | Same text → Anbu | Telegram DM — send via `recipient_name="Anbarasan M"` + `platform="telegram"` (never `target='telegram:… (dm)'`) | **LIVE SEND — stage, one-line go-ahead** |

Then the reply is ordered **artifact → assessment → flags** (see `confirm-before-actions`).

## The rules that fall out of it

1. **Split the dictation into its channels before drafting anything.** One message body can fan out
   to a link, a draft, a group text and a DM; each has its own addressing and its own ship rule.
2. **"I will put it on the group" means he posts it.** A WhatsApp **group** cannot be deep-linked by
   phone — the deliverable is the paste-ready text (see `whatsapp-link-delivery-format.md`). Never
   emit a group deep link as if it were the group.
3. **A colleague Telegram DM is a LIVE mutation** and waits for go-ahead, even when he says "you can
   send that message to him on Telegram instead". Inert artifacts (link, draft) never wait for it —
   the 2026-09-04 ship-it bias still applies.
4. **Resolve the Telegram target before promising the send**: `send_message(action='list')`. Anbu
   appears there as **`Anbarasan M (dm)`**, not "Anbu" or "Anbarasan".

   ⚠️ **`action='list'` is for CONFIRMING reachability only — do NOT feed its `telegram:X (dm)`
   string into `target=`.** Verified 2026-10-16 (NDR → Bharat Hawaldar + Prakash Singh, same turn):

   ```
   send_message(target="telegram:Bharat Hawaldar (dm)")     → ❌ {"error": "Cross-user send blocked.
                                                                Deliver results to the requesting user only."}
   send_message(platform="telegram",
                recipient_name="Bharat Hawaldar")            → ✅ {"success": true,
                                                                "chat_id": "8717455402",
                                                                "message_id": "73293", "mirrored": true}
   ```

   The gateway treats an explicit `target='platform:id/name'` as cross-user targeting and refuses it.
   **The only sanctioned way to DM a colleague is `recipient_name` (or `recipient_phone`) + `platform`.**
   `target=` / `target='origin'` / bare `platform='telegram'` are for the *requesting user's own*
   chat only. A `recipient_name` that doesn't resolve is the real "can't reach them" signal — get
   4. **Resolve the Telegram target before promising the send**: `send_message(action='list')`. Anbu
      appears there as **`Anbarasan M (dm)`**, not "Anbu" or "Anbarasan".
   5. 🔴 **SEND A PERSON DM VIA `recipient_name` + `platform`, NEVER via an explicit `target` ID
      (verified 2026-10-16).** `send_message(action='list')` gives you `telegram:Bharat Hawaldar (dm)`
      strings — but pasting that resolved string back in as `target=` **fails**:

      > `{"error": "Cross-user send blocked. Deliver results to the requesting user only. Use
      > target='telegram' (no explicit ID) or target='origin' ..."}`

      That error is a routing guard, not a "you may not message this person" verdict, and it does NOT
      mean the recipient is unreachable — **do not report the send as blocked and do not offer a
      WhatsApp fallback on the strength of it.** The `(dm)` display string is for *confirmation*, not
      for addressing. The working call is:

      ```
      send_message(recipient_name="Bharat Hawaldar", platform="telegram", message="...")
      ```

      The tool resolves the name through the identity registry and returns the real `chat_id` +
      `message_id`. The `action='list'` step is still worth doing first — it is how you confirm the
      person is reachable *at all* and catch the exact registry spelling — but its output feeds
      `recipient_name`, never `target`. Verified this session: two colleague DMs (Bharat Hawaldar
      `/dm`, Prakash Singh `+`/dm`) both went through on the `recipient_name` route after the
      `target=` route was refused.
   6. **Copy the group/DM text verbatim from the approved draft.** If you tweak the email wording after
      he corrects it, the group text and the DM must be re-cut from the same body — never let the
      channels drift apart.
   7. **Email addressing for a Ranka Iris coordination mail (verified 2026-10-14):** To
      `bhavik@draas.com`; Cc `piyush@draas.com`, `pm2.blr@draas.com` (Anbu). Source these from NDR's
      **own recent sent mail on the project** — `to:bhavik@draas.com newer_than:60d` returns his live
      To/Cc set. `bhavik.92@gmail.com` / `piyush.92@gmail.com` are the older personal addresses used on
      family threads; the work addresses are the right ones for project coordination mail.
   8. **Multi-point coordination mail to the family/colleague set = HTML/CSS**, numbered points, no
      forced line breaks (NDR's standing rule). Subject plain text — never an HTML entity.
   9. **Read the draft back from raw MIME** before reporting it: `To`, `Cc`, `Subject`, `labelIds`
      (`['DRAFT']` = still unsent). `gws_gmail_draft_create` returning a `draft_id` is not verification.

   ## The email + WhatsApp ordering paradox (hit twice on 2026-10-22)

   When one dictation pairs a Gmail draft with a WhatsApp that *announces* that email, the two
   cannot be shipped together as written. Hermes never sends email — so at the moment the
   WhatsApp is staged, the email has NOT gone. NDR dictating *"…say that I have sent him the
   details by email, and send the same briefing by WhatsApp"* produces a message that is false
   on arrival unless he sends the email first.

   Two acceptable resolutions — pick one and say which you used:

   1. **Stage the WhatsApp in the dictated wording, then tell NDR explicitly: send the email
      from Drafts first, then approve the WhatsApp.** Default to this when the wording matters.
   2. **Re-cut it to "I am sending you the … now"** so it is true at draft time. Offer this when
      NDR wants both to land together.

   Never let the announcement go out ahead of the thing it announces, and never "fix" it by
   sending the email yourself (hard rule — drafts only). Flagging this proactively is what NDR
   expects; he did not have to raise it either time.

   ## Two diagnostics that saved a rebuild the same session

   - **An empty `gmail_draft_list` does not mean the draft was lost.** NDR had already sent the
     Swaminathan mail himself — it was sitting in `SENT`, and Drafts was legitimately empty
     because the earlier draft was gone. Search `SENT` (and `newer_than:2d`) before recreating
     anything, or you will send a second copy of a document that already went out.
   - **A WhatsApp chat does not exist in the bridge until there has been contact.** "No chat
     with Vaidyanathan in any of 126 chats" was true and still not a dead end: NDR messaged him,
     and the chat appeared at the top of `wa_list_chats` on the next call. When a named
     counterparty has no chat, ask NDR to send one message (or supply the number) — do not
     report them unreachable.

   ## Never carry an email address forward from an earlier turn

   Point 7 sources internal colleague addresses from NDR's own sent mail. The same discipline
   applies to **external** counterparties, and it is not satisfied by what you said last turn:
   this session quoted `v.vaidyanathan@gmail.com` from an earlier self-claim, and the real
   address was `asvaidyanathan@yahoo.com` (contact sheet **and** the live 14–15 Oct thread).
   Before addressing a new external recipient, run `contact_resolver` and then confirm the
   address actually appears in `gws_gmail_search` on that matter. A wrong address on a
   landowner mail about his own money costs more than the ten seconds it takes to check.
