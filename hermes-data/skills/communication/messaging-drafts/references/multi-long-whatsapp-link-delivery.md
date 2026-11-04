# Delivering multiple long WhatsApp deep links

## The problem
A long WhatsApp deep link is **~2,000–2,800 characters** once the message text is
percent-encoded. Telegram caps a single message at **4,096 characters**, and its splitter can
cut a URL in half — producing a link that silently doesn't work when tapped.

`whatsapp_link` reports `split: false` whenever *that one link* fits in 4,096. That tells you
nothing about whether the **reply you are about to write** fits. Two links plus prose will
exceed the cap.

## The rule
When a response must carry **two or more** WhatsApp deep links (or one link plus a long body),
do **not** put them all in the final response.

1. Generate each link with `whatsapp_link` (`platform='telegram'` → use `display_link`, never the
   bare URL).
2. Deliver each link as **its own message** with `send_message`, one call per link:
   ```
   send_message(
     target="telegram:<the requesting user's own DM>",
     message="<one line of context>\n\n<display_link>")
   ```
   Confirm the returned `chat_id` is the requesting user's own chat.
3. Keep the final response to the **summary and flags only** — no long URLs.

## Privacy
Send only to the **requesting user's own DM** (origin). Never route a link to another person's
chat, even when the link is *about* that person. Generating a link for Rahul does not mean
sending anything to Rahul — the user taps it themselves.

## Never report a link as sent
`whatsapp_link` only **generates** a deep link. The honest description is: *"a link the
recipient can tap so WhatsApp opens with the text pre-filled."* Do not write "sent",
"delivered", or "messaged Dharmesh".

## Pitfall — what you pass is what gets sent
Do not include placeholder/scratch tokens in the `message` string (e.g. a leftover marker line
like `MEDIA_MARKER`). There is no templating layer: the string is delivered verbatim. Build the
message string in one pass and read it before calling.
