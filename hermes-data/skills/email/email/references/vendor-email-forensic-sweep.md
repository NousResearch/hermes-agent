# Finding (or honestly failing to find) a vendor email from a half-remembered description

Trigger: *"There is an email from one of our <vendor category> partners, <name I can't
quite remember>, about <topic>. Can you find that email?"* — then, in the same breath,
an action that depends on it (*"do a reply-all asking them to..."*).

This is the case where the **email may not exist in any mailbox you can reach**. The
job is to prove that either way, not to produce a plausible-looking draft. Verified
2026-10-09 on ndr@draas.com hunting a Google Workspace partner billing email.

## 1. Enumerate the reachable mailboxes in ONE call

```
gws_resolve_account()        # no args → every known service + has_token
```

Do not assume the four accounts you remember. Do not stop at `ndr@draas.com` — the
user is describing a mail he *saw*, which may be in any of his identities.

If one account's sweep returns `invalid_grant: Token has been expired or revoked.`
(seen on `google-o3infotech`), that account alone needs re-authorization — send the
button (`send_oauth_url`; reliable route is a delegated leaf subagent with
`toolsets=["oauth"]`). The other accounts' results still stand: do not restart the
sweep, and do not report the whole hunt as failed because one mailbox was unreachable.

## 2. The query matrix (run it per account; use `in:anywhere`)

`in:anywhere` is mandatory — it is what makes Trash and Spam searchable. A plain
query silently **excludes both**, which reads as "no such mail" when the mail exists.

| Axis | Queries |
|---|---|
| Sender domain | `in:anywhere from:<vendor.com>` (all the spelling variants he might use) |
| Sender name variants | `in:anywhere brio OR briotech OR brillio OR brilo` |
| Subject | `in:anywhere subject:(workspace OR gsuite OR brio)` |
| Topic keywords | `in:anywhere (outstanding OR "payment due" OR "pending payment" OR dues OR arrears)` |
| Category | `in:anywhere (reseller OR "workspace partner" OR "google partner")` |
| **Invoice-list fingerprint** — user says *"they listed 10, 11, 12 outstanding invoices, month after month"* | `in:anywhere (invoices OR "invoice list" OR "outstanding statement" OR "debtors aging" OR "statement of account")`; `subject:(invoice OR invoices OR outstanding OR dues OR payment)`; and for ERP-export chasers `in:anywhere (DocNum OR DueDays OR PaidToDate OR "Due Due")` |
| Spam sweep | `in:spam (invoice OR payment OR subscription)` |

Emit a compact table (`date | folder | from | subject`) sorted newest-first. Budget for
it: a `messages().get` per hit costs ~0.2s, so sweep in a script with a timeout, not
one tool call per query.

## 3. The four traps that make a present email look absent

1. **No date filter.** `newer_than:120d` / `newer_than:500d` is a trap when hunting a
   half-remembered mail — the target can easily sit *outside* the window. The live
   Brio thread was Feb–Mar 2025, ~580 days back: invisible to a 500-day sweep, found
   instantly with no date filter. **Always run at least one pass with no date bound.**
2. **The vendor name the user gives is usually wrong.** "Brillo or something" was
   *Brio Technologies* (brio.co.in / briotech.com). Search the **domain**, the topic,
   and the sender's display-name family — never the spoken name alone (same lesson as
   pitfall 6 in the parent SKILL.md: voice transcription mangles entities).
3. **An address used in `To:` may never appear as `From:`.** `BK@fnw.homes` shows up
   repeatedly as a *recipient* in NDR's sent mail but never as a sender — Bhuvanesh
   actually *sends* from `bk@findingform.design`. A `from:`-only search therefore
   misses the person entirely. Search both directions.
4. **Some of the user's mailboxes are not in the vault.** Brio's billing chasers went
   to `@drahomes.in` addresses (`loganathan.k@`, `anandaraj.d@`, `naviya@`) and NSDL
   writes to `ndr@drahomes.in` — none of which is a connected service. `gws_resolve_account()`
   has no `drahomes` entry. So **"not found" must be reported as "not in the mailboxes
   I can reach", never as "it doesn't exist"** — and name the unconnected mailbox as
   the likely hiding place.
5. **Cross-domain delivery — mail *addressed* to an unconnected domain can still be
   sitting in a CONNECTED mailbox.** Brio's chasers carry `To: loganathan.k@drahomes.in`
   / `anandaraj.d@drahomes.in` yet ARE returned by searches on `ndr@draas.com` — they
   reached his mailbox by BCC or group expansion, and a Gmail result shows the
   message's *original* `To:` header, not how it was delivered. So never eliminate a
   mailbox on the strength of an address alone — run the sweep there anyway. (This cuts
   the other way too: a `To:` on the wrong domain is *not* proof the account lacks the
   mail, which is why "not in the mailboxes I can reach" is the only honest phrasing.)

## 4. Never bulk-act on the sweep's result

A topic sweep returns unrelated mail (school notices, bank statements, a colleague's
forward). Classify on subject **before** trashing/moving anything, and report counts by
class. See the Drive share-request pitfall in the parent SKILL.md for what goes wrong
when a broad sender sweep is actioned directly.

## 5. Report the negative like a result

When the mail genuinely is not there, deliver:

- **Exactly which mailboxes and which folders** were swept, and the query axes used —
  so he can see the search was exhaustive rather than assume you gave up.
- **The candidate table**: the best 2–3 real matches with dates, senders, subjects, and
  what thread/subject they belong to.
- **What the topic maps to today** — often the *problem* he described is real even when
  the email isn't (here: draas.com sat at **99% pooled storage**, nine Google warnings
  since Jul 2026, next step the 14-day grace period then read-only; `o3infotech.com`
  was on recurring *"Payment failure: Google Workspace Business Starter"*, billed
  direct to card, not via a reseller). That reframing is the useful part.
- **One precise unblocking question**, with the concrete alternative named (connect
  `ndr@drahomes.in`, or "say the word and I'll reply-all on thread X").

## 6. Anti-pattern — do not manufacture the deliverable

If the ask was **"reply all"**, the source thread is a *hard prerequisite*: reply-all
has no meaning without the parent message's headers, and `In-Reply-To`/`References`
must point at a real message. **Do not create a speculative draft addressed to the
guessed vendor** on a guessed/stale thread. A misdirected formal billing mail to a
vendor relationship the user may no longer have is a real cost; a clear blocker report
is not. Offer the draft and name the one word that unlocks it.

(Contrast: when the thread *is* known and only a detail is missing — e.g. the
certifier's individual email vs the firm address on the RERA record — DO draft, and
flag the swapped field. See `karnataka-rera-quarterly-filings` →
`references/certifier-chase-replyall.md`.)

## 7. Side-findings worth keeping (Workspace-partner landscape, ndr@draas.com)

- **Reseller of record for draas.com = Brio Technologies Pvt Ltd** — `brio.co.in` /
  `briotech.com`. Evidence trail: `accounts@brio.co.in` "G suite Renewal
  notification: draas.com"; "Payment successful for Brio Technologies Private
  Limited" (fwd'd by Naviya, Nov-2022); tax-invoice chasers from `shubha@`, `deepa@`,
  `riyaz@`; support tickets `INC-147781` / `INC-147843` with `googlesupport@briotech.com`
  and `cphelp@briotech.com`. Last live activity **5 Mar 2025**.
  - **Chaser fingerprint** (this is what "they listed 10, 11, 12 outstanding invoices
    for every month" means): an ERP-export table —
    `DocNum | DocDate | DocDueDate | DueDays | DocTotal | PaidToDate | Due | Due%` —
    one row per month's invoice. Thread subjects: `Tax Invoices - DRA aadithya south
    city projects private limited`, `Outstanding amount - Dra Aadithya Chennai Homes
    LLP`. Every named addressee is `@drahomes.in`: `loganathan.k@`, `anandaraj.d@`,
    `naviya@`, `mohamed.m@`.
  - **Re-checked 2026-10-10** across draas + ahfl + gmail + o3infotech: still nothing
    newer than **5 Mar 2025** in any connected mailbox, i.e. anything "recent" the user
    remembers seeing is almost certainly in `ndr@drahomes.in`. Unblock by asking him to
    forward it (one tap) or by authorizing that account — do NOT invent a reply-all.
- **`o3infotech.com` is billed by Google DIRECT** (card on file), not through a
  reseller — hence the recurring payment-failure notices to `ndr@o3infotech.com`.
- **Shivaami Cloud Services** (`shivaamicloud.com`) is a second Workspace partner that
  has cold-emailed the group (last: 5 Aug 2025).
- **Storage-increase options** (verified from Google's own docs, Oct 2026) — Business
  Starter = 30 GB pooled/user, **Standard = 2 TB**, Plus = 5 TB, so an edition upgrade
  is usually the cheaper lever; the **Workspace Additional Storage add-on** adds a flat
  **10 TB** per subscription, stacks, and is **annual-commitment only**; Enterprise is
  effectively unlimited. Only a **super admin** can buy; extra storage lands in ~5 min
  (up to 4 h).
