# WhatsApp deep-link DISPLAY format on Telegram (NDR rule, 2026-10-01)

NDR complained: *"this is not a WhatsApp link... just some random characters. Why is the WhatsApp link tool failing?"* — when the `whatsapp_link` tool's `platform='telegram'` **display_link** (MarkdownV2-escaped text with `\*`, `\(`, `\.`, `\-`… backslashes) was pasted into the chat. His Telegram client rendered the escapes literally, so the message looked like garbage. The underlying URL was valid all along.

## Rule

- **Always deliver WhatsApp deep links to NDR as the RAW `https://api.whatsapp.com/send?phone=...&text=...` URL** (the tool's `url` field, i.e. generate with `platform` omitted or `'whatsapp'`), presented either as:
  - a normal Telegram markdown link: `[Tap to open WhatsApp](<raw url>)`, or
  - a pasteable code span with the full raw URL.
- **Do NOT paste the `display_link` / `display_text` (MarkdownV2-escaped) form.** It is correct for clients that auto-render MarkdownV2 but NDR's client shows the escapes verbatim.
- Do NOT hand-encode or "clean" the URL yourself — keep using `whatsapp_link`; only the presentation layer changes.
- Keep showing the recipient phone for verification (per `contact-phone-lookup` LABEL WINS / unlabelled-number rule) — format change is orthogonal to the verification step.

## Verify-before-claiming

If NDR says a link looks broken, first check the raw `url` value from the tool output against `https://api.whatsapp.com/send?...` — random backslashes > (escapes) mean the URL itself is fine; only the display variant was wrong.

## WhatsApp GROUPS cannot be deep-linked by phone (2026-10-13)

NDR asked for *"a WhatsApp message for the Ranka Northstar Landowners group"*. A WhatsApp **group has
no phone number** — group links are `chat.whatsapp.com/<invite>` and cannot carry pre-filled text. So:

- Call `whatsapp_link` with **`phone` omitted**. That returns
  `https://api.whatsapp.com/send?text=…`, which opens WhatsApp's compose screen with the text
  pre-filled; NDR then taps the group and sends.
- **Always hand him the paste-ready plain text as well** — for a group that is the actually reliable
  path, not the link.
- State plainly that a group can't be deep-linked, so the link opens compose rather than the group
  itself. Never imply the message was sent.

## Two pre-flight checks before promising a WhatsApp/Telegram send (2026-10-13)

1. **Is the recipient reachable at all?** Run `send_message(action='list')` *before* drafting a send
   when NDR names people ("send it to both X and Y on Telegram"). Real case: Anbu resolved
   (`Anbarasan M`); **Kantesh had no Telegram target** — only a contacts phone (+91 95918 01389).
   Surface the gap and offer the WhatsApp fallback; don't silently drop him and don't invent a chat.
   The list is for **reachability only** — to actually send, use `recipient_name` + `platform`, never
   `target='telegram:<name> (dm)'` (that shape is refused as a cross-user send). Full mechanics +
   verified error/success pair in `multi-channel-message-fanout.md` rule 4.
2. **Is the dictated name real?** Dictation mangles proper nouns. Try 3–4 spelling variants through
   `contact_resolver` (and `entity_resolver`) and check `session_search` before declaring a name
   unresolvable: `Vaidhenathan` → **Vaidyanathan LS Nitesh**; `Kantesh` → **Kantesh DRA**
   (row 1554, voice-misspelling `kantish`); `Vishwas` → **Vishwas Rao**. A *company* name that
   matches nothing in any variant is a genuine blocker — flag it and ask which vendor he meant;
   never guess (2026-10-13: "Signa and Alita" matched no contact, Kelsa record, or session).
   A hit is not a confirmation — a single **unlabelled** phone still needs his OK before you target it.
   **Registry-access caveat (2026-10-14):** the contacts sheet lives in NDR's account. A non-NDR
   session (e.g. Anbu / `pm2.blr`) can get a `gws_service` / "account is not associated with your
   session user" error from `contact_resolver`, and `account_email` overrides are refused. When that
   happens **stop after one attempt** — do not retry spelling variants, and do not go hunting for a
   token. Fall back to `session_search` for a prior resolution, then use the name as dictated and
   flag the spelling to the user. Same fallback ladder if `entity_resolver` returns a tool error:
   `session_search` → dictated name + flag.

## A LONG deep link must ship as its OWN Telegram message (2026-10-14)

A WhatsApp deep link is **~1.4–3× the message text** once percent-encoded, and the multiplier is
driven almost entirely by the **space count** (every space → `%20`, 3 chars); most other characters
stay 1×. So a message with few spaces encodes cheaply while one padded with spaces and em-dashes
gets expensive — `~3×` is a ceiling, not a rule.

Two measurements, same day (2026-10-14):

- Ranka Iris: ~1,470-char message → **~4,200-char URL** (≈2.9×) — at Telegram's 4,096-char ceiling.
- Nanjesh-invoice draft to Anbu: ~890-char, 5-paragraph message → **1,297-char URL** (≈1.46×) —
  comfortably bundleable with the draft and the flags in one reply.

**Measure, don't estimate.** `len(url)` on the tool's returned `url` field is one `execute_code`
call and settles ship-alone-vs-bundle outright. Never infer the URL size from paragraph count.

- **Do not bundle a long link with your assessment, group text, or flags in one reply.** Telegram's
  own splitter then cuts the message mid-URL and the link dies. Send the link block as its **own**
  message (`send_message` to the current chat / home channel) and put the draft summary, the
  copy-paste group text, the staged sends, and the flags in the final reply.
- **`split: false` from the `whatsapp_link` tool does NOT mean you are safe to bundle.** That flag
  measures the *URL alone* against the limit; it knows nothing about the prose you are about to wrap
  around it. Treat `split` as "the link itself is deliverable", never as "this reply will survive".
- Rule of thumb: if the dictated message is longer than ~3 short paragraphs, send the link alone.
- The link block that works: a one-line caption (`*WhatsApp → <Name>* (<phone>)`), then
  `[Open WhatsApp to <Name>](<raw url>)`, then one honest line that nothing has been sent — he taps
  and hits send himself.

## REINFORCED (2026-10-14): never paste `display_link`, and never bundle a long link

The 2026-10-01 rule above was **violated again** in the 2026-10-14 Anbu session: `whatsapp_link`
was called with `platform='telegram'` and the returned **`display_link`** — the MarkdownV2-escaped
form full of `\.`, `\(`, `\+` backslashes — was pasted into the reply as the deliverable. That is
precisely the "random characters" output the user rejected. The URL underneath was fine; the
presentation was wrong.

- **Call `whatsapp_link` with `platform` OMITTED** (or `'whatsapp'`) and deliver the **raw `url`**
  as a normal Telegram markdown link: `[Open WhatsApp](<raw url>)`. `display_text` / `display_link`
  exist only for clients that auto-render MarkdownV2 — never paste them as chat prose.
- **Self-check before sending any reply containing a link:** does the URL body show `%XX`
  (correct) or `\.` / `\(` / `\+` (wrong — you pasted the display form)?
- **Anbu's variant (only when NDR wants to send it himself):** the primary deliverable is the
  **paste-ready text in a fenced code block**, because he sends it himself. Code block FIRST, link
  at most as a secondary convenience, and the link must never replace the pasteable text.
- **But when NDR says "send it on Telegram to Anbu" (2026-10-14), paste-ready text is NOT the
  deliverable.** That is the fan-out case: a **staged Telegram DM to `Anbarasan M`** (one-line
  go-ahead) *plus* the **WhatsApp deep link shipped the same turn** as its own message. Do not
  substitute a code block for a send he actually asked for — read which verb he used.
- **Bundle rule — measure the URL, don't count paragraphs.** A ~900-char, 5-paragraph message
  measured **1,297 chars** encoded (2026-10-14) — safe to bundle with the draft and flags; a
  ~1,470-char message measured **~4,200** — never bundle that one. Run `len(url)` on the tool's
  `url` field. If it exceeds ~1,500 characters, either send the link in its **own** message or
  **omit it entirely** when the user asked for copy-paste text.

## ⚠️ The real `send_message` ceiling is ~1,800 chars, NOT 4,096 (measured 2026-10-14)

Telegram's nominal text limit is 4,096, but the Telegram adapter **escapes MarkdownV2 specials in
the URL** before sending, so the effective ceiling for a raw deep link is far lower. Measured on the
same chat, same day:

| URL length | Result |
|---|---|
| ~1,600 chars | ✅ sent |
| **2,102 chars** | ❌ `Telegram send failed: Message is too long` |

So a link measuring **> ~1,800 chars will not deliver via `send_message`**, and a link of that size
also cannot be salvaged by splitting — the `whatsapp_link` tool's `parts[]` are each ~2,000+ chars,
i.e. **every part fails too**. `split: true` is therefore not a delivery guarantee either.

**When a dictated message is long, cut the TEXT, not the link.** Worked case (2026-10-14,
Yarthiganahalli document reminder): a 1,503-char message produced a 2,102-char URL → rejected.
Rewriting to ~1,120 chars of tight, mostly-ASCII prose (em-dashes and decorative bullets replaced
with `-`, each document on one line) produced a URL that sent first time. Em-dashes cost 9 encoded
chars each (`%E2%80%94`); spaces cost 3. Drop them when the URL is near the ceiling.

## ⚠️ Presigned S3 / signed URLs cannot ride in a WhatsApp deep link (2026-10-14)

`whatsapp_link` encodes `&` as **`%EF%BC%86`** (a fullwidth ampersand) by design. Any URL carrying a
query string with multiple `&`-separated parameters — presigned S3 download links
(`?X-Amz-Algorithm=…&X-Amz-Expires=…&X-Amz-Signature=…`), most OAuth callback URLs, most tracking
links — is therefore **corrupted** in the message the recipient receives, and the link will not
resolve. Verified: a test S3 presigned URL came back with `%EF%BC%86` between every parameter.

- **Never put a signed URL with a query string in a WhatsApp deep link.** Use the bare,
  query-less object path, a short durable link (Drive/SharePoint), or send the file itself.
- A bare **private** bucket link (`https://<bucket>.s3.<region>.amazonaws.com/uploads/…`) returns
  **HTTP 403** to anyone outside the owning app — verify with `curl -o /dev/null -w "%{http_code}"`
  before promising it to a counterparty. Kelsa's bucket (`kelsa-clients-production`, `acl=private`)
  behaves exactly this way: the links are an internal record, not a shareable artefact.
