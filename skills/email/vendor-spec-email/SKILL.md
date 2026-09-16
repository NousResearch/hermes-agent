---
name: vendor-spec-email
description: "Use when emailing a vendor requesting a feature."
version: 0.1.0
author: Abe Perl (abeperl), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Email, Vendor, Spec, FeatureRequest]
    related_skills: [email-archive-research, email-inbox-triage]
---

# Vendor Feature-Request Email

Compose a professional external request when a customer/vendor thread needs a feature, integration, or parameter spec. The user (a solo dev doing client integrations) expects a ready-to-send spec email, not a chat-style summary.

## When to Use

- "Find the email where we asked X, then email vendor Y asking them to add/edit a feature"
- A client's POS / platform needs a URL link, button, or new field wired to an app you built

## Procedure

1. **Reconstruct the thread from the archive first** — see `email-archive-research` for IMAP search mechanics. Read full bodies, not headers: the real requirement often comes from a non-obvious party (a stakeholder's one-line reply can narrow a long spec), and the vendor's earlier messages carry their asks verbatim.
2. **Re-find the whole ticket by subject.** Support replies carry the ticket number in the subject (`#29180`); search `(SUBJECT "<ticket#>")` to pull every message in that thread instantly.
3. **Mine your user's own prior reply.** If an earlier user email already contained a correct spec (URL parameter list, case/encoding notes), reuse it verbatim and extend only what the stakeholders asked for — it is the user's voice and the vendor already saw it.
4. **Compose the vendor-facing spec email:**
   - One clarifying sentence stating what it is NOT ("we are not looking for a data export/report — we need a feature in the POS UI") — vendors misread feature requests as report requests.
   - The feature spec: where the UI element lives ("a button/link next to line items in the cart"), a configurable base URL field, and a parameter table (name / type / required-optional / purpose / example).
   - One sample constructed URL (with example values, URL-encoded spaces as %20).
   - Implementation notes the vendor must honor (URL-encode values, case-sensitive keys).
   - A closing offer: "happy to schedule a 10-minute technical call to walk through this" — cheaper than another wrong guess round-trip.
5. **Send only after the user approves** (and use the user's standard send path — see `email-inbox-triage`).

## Pitfalls

- Vendors answer feature requests with report/exports unless you explicitly separate the two in the first sentence.
- A comma-separated list of fields from memory is not a spec — write the table from the actual thread so Required/Optional and examples trace to real messages.
- Reconstructing a 20-field spec when the stakeholder actually wanted 4 fields wastes a round-trip; always read the whole thread before drafting.
