# Setting Up Webhooks

A webhook is one YAML file under the package's `webhooks/` directory. As
with Schedules, the file is the whole interface — writing it creates the
webhook and mints its URL, deleting it retires the URL.

## Whether a Webhook at All

A Schedule spends a known number of runs; a webhook lets the outside
world spend them for you — deliveries become model runs at the
sender's pace. Bursts on one key coalesce into fewer runs, but the
count is still set by the source, not by you, and you find out what a
busy source costs after the money is spent. So the choice is a
run-count comparison: estimate a normal day's deliveries and put that
against the sweep you would otherwise run. A webhook wins two ways —
events rarer than any sweep worth running, where per-event genuinely is
the fewest runs; or work that batching breaks: an outside party waiting
live — a customer mid-conversation, a counterparty expecting an answer
in minutes.
Most "as soon as possible" asks are neither: the existing sweep picks
up everything since the last run, and only the bill notices the
difference.

Before arming one: subscribe to the one event that means a unit of work
is ready — never to low-level mutations that fire several times per
unit — and put the price to the user next to the swept alternative. Arm
it on their informed yes, not on their "as soon as possible".

## Which responsibility owns the webhook

- The provider supports multiple webhook endpoints (GitHub, Stripe — most
  event APIs do): the webhook belongs to the responsibility that does the
  work. Two responsibilities needing the same events declare one webhook
  each and register both URLs upstream.
- The provider allows exactly one callback URL (WhatsApp and other
  Meta-class platforms) and its traffic can serve more than one area of
  work: charter a dedicated intake responsibility that owns the single
  webhook. Its charter is the door, not the rooms — its scope is the
  surface itself: receive, triage against the roster, then do the work
  under the responsibility it belongs to and record it in that one's
  STATE.md. The intake package's own STATE.md holds only inbox and
  routing state. Never let two responsibilities each declare a webhook
  against a single-callback provider.
- A webhook already owned by one responsibility that a second now needs
  (single-callback provider): charter the intake responsibility, move the
  webhook file into it unchanged, and name the responsibilities it serves
  in the intake charter — the serves list is the primary routing map,
  roster triage the fallback.

## The file

```yaml
# webhooks/<name>.yaml
scope: |                                      # required — what a delivery means, and which slice of the responsibility handles it
  A WhatsApp message arrived on the intake surface this responsibility
  owns.
key: entry.0.changes.0.value.messages.0.from  # optional — the field that identifies the conversation
report: "slack:C0123ABC"         # required — same grammar as Schedules
handshake: hub.challenge                      # optional — the provider's registration check
```

The scope carries what a delivery means and which slice handles it —
nothing else. As with Schedules, the work itself lives in the package;
instructions or policy restated here drift from it.

## Choosing `key`

Pick the payload field that identifies the conversation-like unit: the
sender for a messaging provider, `pull_request.number` for GitHub.
Deliveries sharing a key continue one conversation — the run remembers
earlier deliveries — and a burst on one key becomes one run instead of
many. Omit `key` where the source is a single stream (a deploy hook, an
alert feed). Dot-path into the payload; numeric list indices allowed.
A provider that posts non-JSON bodies (XML, plain text) has no fields
to key on: omit `key` there too — the body arrives verbatim for the
run to parse.

## Declaring `handshake`

Many providers verify a callback URL at registration by sending a
challenge that must be echoed back. Read the provider's webhook docs and
declare what they describe — the platform executes it; you cannot answer
a handshake live. The string form echoes a query parameter on GET (Meta:
`hub.challenge`). The object form covers the rest:

```yaml
handshake:
  method: POST                          # GET | POST (default GET)
  when: event=endpoint.url_validation   # optional guard: field equals value
  respond:                              # a dot-path to echo — or a map, rendered as JSON
    plainToken: payload.plainToken
    encryptedToken: hmac_sha256(payload.plainToken)
  secret: env:PROVIDER_WEBHOOK_SECRET     # keys the hmac; a native profile secret reference
```

The `secret` is never a literal value. It is a `env:VARIABLE_NAME` reference
into native profile secrets, resolved by the platform each time the
handshake runs — a literal is rejected at compile. Obtain the provider's
token as `{guides_root}/connections/guide.md` directs — request it when a person
holds it, store it when you obtained it yourself — and write the
returned reference into the file. The value never passes through chat
or lands in the file, and rotating it in the store leaves the file's
bytes unchanged, so the webhook's URL survives rotation. If the
referenced connection is later removed, deliveries continue — only the
provider's next verification fails; repair by requesting the credential
again and updating the reference in place.

Some providers require the computed response in a specific encoding. X
(Twitter) expects a base64 HMAC with a `sha256=` prefix:

```yaml
handshake:
  method: GET
  respond:
    response_token: hmac_sha256(crc_token)
  secret: env:PROVIDER_WEBHOOK_SECRET
  encoding: base64      # hex (default) | base64
  prefix: "sha256="     # literal glued onto the computed value
```

Both fields shape the computed `hmac_sha256` value only.

## Declaring `ack`

Most providers accept any 2xx acknowledgment and need nothing here. A
few require the acknowledgment itself to have a specific shape — Twilio
logs an error against every delivery unless the response is TwiML with
a `text/xml` content type. Read the provider's webhook docs and declare
the static response they require; the platform returns it for every
accepted delivery:

```yaml
ack:
  status: 200                 # 2xx, default 200
  content_type: text/xml      # required when body is set
  body: '<?xml version="1.0" encoding="UTF-8"?><Response/>'
```

The body is fixed, literal bytes — no placeholders, nothing computed
from the payload. A provider that wants a real reply inside the
acknowledgment doesn't get one: the acknowledgment only receives the
delivery, and the reply goes out through the provider's API as the
run's work.

## Declaring `verify`

The URL alone already authenticates senders: it is a long random
bearer credential. Add `verify` when the provider signs its deliveries
and the work warrants the extra check — signed deliveries survive a
leaked URL. Read the provider's webhook docs and declare the scheme
they describe; the platform checks every delivery before accepting it
and rejects mismatches without running anything.

```yaml
# GitHub-style: the signature covers the raw body
verify:
  secret: env:PROVIDER_WEBHOOK_SECRET
  header: X-Hub-Signature-256
  signature: sha256=hmac_sha256(body)

# Stripe-style: a timestamp joins the signed string
verify:
  secret: env:PROVIDER_WEBHOOK_SECRET
  header: Stripe-Signature.v1       # .field reads one value from a k=v header
  timestamp: Stripe-Signature.t
  signature: hmac_sha256(timestamp.body)

# Slack-style: versioned signed string, timestamp in its own header
verify:
  secret: env:PROVIDER_WEBHOOK_SECRET
  header: X-Slack-Signature
  timestamp: X-Slack-Request-Timestamp
  signature: v0=hmac_sha256(v0:timestamp:body)
```

The `secret` follows the handshake rule exactly: a `env:VARIABLE_NAME`
native profile secret reference, never a literal. `encoding: base64` covers
providers that send base64 signatures (Shopify). Timestamped schemes
tolerate five minutes of clock skew; older deliveries are rejected. If
the referenced connection is removed, deliveries fail closed until the
credential is requested again and the reference repaired in place.

## Moving and rotating

The platform tracks a webhook by its file content. Byte-identical is the
rule: a file moved, renamed, or deleted and restored unchanged within an
hour reconnects automatically — same URL, same ongoing conversations,
nothing to re-register. Any content change on the way breaks the match
and mints a fresh URL that must be registered upstream again. So when
relocating a webhook, move the file exactly as it is and edit only after
the move. Editing a file in place never changes its URL — only a
move-with-edit or a delete-and-recreate does. To deliberately rotate a
leaked URL, do the opposite: recreate the file with any edit.

## Registration

The write result comes back with the webhook's URL — that write result is
the only surface that shows it, so record the URL in the package's
`STATE.md` immediately. Registration means pointing the provider at that
URL, and it is yours to drive: use the provider's API when a connection
gives you access; otherwise hand the user the URL with exact
click-by-click steps for the provider's settings page. Either way,
confirm the registration actually succeeded — the provider's verification
passing, or a test event arriving — before reporting the webhook live.

## Testing

The URL is live from the moment the file compiles: a `curl` POST from
your computer is a genuine delivery and triggers a genuine, billed run
that delivers to the configured target. Test deliberately — shape the
test payload so the run has nothing to send.

## Checklist

- [ ] A normal day's delivery volume was estimated and compared to the
      sweep alternative: the webhook is the fewer-run shape, or batching
      genuinely breaks the work
- [ ] The user confirmed the webhook knowing its cost against the swept
      alternative
- [ ] The trigger is the one event that means a unit of work is ready,
      not a low-level mutation that fires several times per unit
- [ ] Ownership follows the ladder: multi-endpoint provider → the doing
      responsibility; single-callback provider → one intake
      responsibility, and no second webhook against that provider
- [ ] An intake responsibility's charter names the responsibilities it
      serves
- [ ] A relocated webhook was moved byte-identical; edits came only after
      the move
- [ ] `key` names the conversation-identifying field, or is deliberately
      omitted for a single-stream source
- [ ] A handshake `secret` is a `env:VARIABLE_NAME` native profile secret
      reference, never a literal value
- [ ] A provider that requires a specific acknowledgment shape has it
      declared as a static `ack`; replies go through the provider's API
- [ ] A provider that signs deliveries has `verify` declared, or the
      URL-only default was a deliberate choice
- [ ] The provider's registration verification passed (or a test event
      arrived) before the webhook was reported live
