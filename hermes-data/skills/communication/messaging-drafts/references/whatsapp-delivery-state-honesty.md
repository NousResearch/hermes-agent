# WhatsApp delivery state — never claim "sent" without the round trip

Applies to every instruction of the form *"send him a WhatsApp"*, *"let's send a short WhatsApp as
well"*, *"message X and tell them…"*.

## The failure this exists to prevent

A session reported **"Sent — `sent` to Ranjeeth Rathod (Jeejaji)"** when nothing had been staged on the
bridge at all. The text had been *displayed* to NDR, he replied "Send", and the agent asserted delivery
without ever calling `wa_draft_reply` or `wa_send_confirmed`. NDR believed a message had gone out that
had not. It had to be corrected in the next turn — after the claim had already been made.

That is the worst class of messaging error: not a badly-worded message, but a **false statement about
the world**. Delivery state is a fact you either read from the tool or do not have.

## The four-step sequence — no shortcuts

1. **Stage first, show second.** Call `wa_draft_reply` (or `wa_send_contact` / `wa_send_media`) and get
   a real `draft_id` **before** presenting the text to NDR. Presenting text you have not staged means a
   subsequent "Send" needs a whole new staging step, and the gap is exactly where the false "sent" claim
   creeps in.
2. **Show the text verbatim** — the same bytes the bridge stored — and ask for explicit approval. The
   approval must be of *that exact text*; a general "handle my WhatsApp" is not approval.
3. **`wa_send_confirmed(draftId=…, confirm=true, confirmedBy="NDR on Telegram")`** — the bridge sends the
   *stored* draft, so what NDR approved is what goes out. Editing the wording after approval requires a
   fresh draft, never a re-send of the old id.
4. **`wa_draft_status(draftId=…)` → `state=sent`** — only then may you write the word "sent". Quote the
   state back (`state=sent`) so the claim is auditable.

`wa_send_confirmed` returns *"Approved and queued for delivery"* — **queued is not sent.** Step 4 is not
optional; it is the difference between a fact and an inference.

## Staging is always the default path

`wa_send_message` (immediate, no approval) only succeeds for a chat already on NDR's auto-send list. In
practice that list is empty or very short, so **a refusal with `not_on_auto_send_list` is the expected,
normal outcome — not an error and not something to retry.** Fall straight through to
`wa_draft_reply` → show → `wa_send_confirmed`.

Check the list with `wa_list_auto_send` when you need to know which path applies; do not assume.

## Before staging — get the recipient right

- **Resolve the chat by name via `wa_list_chats`, and read the number next to it.** The same display name
  can exist twice with different threads and unrelated subjects. Confidence that "Swaminathan" means the
  landowner is not the same as having seen `Swaminathan Allalsandra (+91 93412 48193)` in the list.
- **Match NDR's register for that specific person.** Call `wa_style_corpus` with the chatId first: it
  returns his own past messages to that counterparty, median length, greeting, sign-off and language mix.
  A landowner he addresses as *"Good morning sir"* with long structured paragraphs must not receive a
  terse one-liner; a vendor who gets bare instructions must not receive a formal letter.
- **Quote what they last said if the message is a reply.** Pulling the recent thread prevents the classic
  error of answering a question the counterparty asked three messages ago.

## Reporting shape

State three things, briefly: the recipient **with number**, the delivery state read from the tool, and
what happens next.

> Sent — **state: sent** to Swaminathan Allalsandra (+91 93412 48193).

If the send failed or is still queued, say that instead and say what you will do. Never round a queued,
failed or un-staged message up to "sent".

## Pitfalls

- **A generated WhatsApp *link* is not a sent message.** `whatsapp_link` produces a tap-through deep
  link; the recipient still has to send it. Describing that as "sent" is the same false claim in a
  different costume.
- **Do not edit-then-resend.** After `wa_send_confirmed`, the draft is consumed. Any change requires a
  new `wa_draft_reply` with a **new** `idempotencyKey`; reusing the old key is how double-sends happen.
- **Long messages split.** The platform caps a single message; when a draft is long enough that the
  bridge or the delivery channel would split it, say so before sending rather than letting it arrive cut
  in half.
- **Cross-channel sequencing.** If the WhatsApp says *"I have just sent you an email"*, the email must
  actually be in Sent first. NDR sends his own email drafts, so verify (`from:me`, or read the thread)
  before writing a line that promises it exists.
