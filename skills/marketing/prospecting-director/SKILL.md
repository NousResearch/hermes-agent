---
name: prospecting-director
description: Orchestrate lead prospecting pipelines from source discovery to qualified lead queues, with Obsidian knowledge-base syncing and repo-backed artifacts.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
tags: [prospecting, leads, facebook-ads-library, whatsapp, obsidian, crm]
---

# Prospecting Director

Use this skill when building or operating a lead prospecting system: collecting prospects from public sources, qualifying them, writing lead queues, and syncing durable notes into Obsidian.

## Operating model

1. **Define the ICP and source constraints first.** Capture country, language, vertical, campaign type, exclusion rules, and minimum evidence.
2. **Collect evidence-backed prospects.** Every lead must include a source URL, observed offer/CTA, business name, country/locality signal, and why it matches the ICP.
3. **Qualify before outreach.** Score leads by pain intensity, deal value, urgency, reachable owner/contact path, and fit for the user's offer.
4. **Write artifacts to the project repo.** Keep CSV/Markdown queues under the requested prospecting path, e.g. `prospecting/libya/`.
5. **Sync durable knowledge to Obsidian.** Use the vault path from `OBSIDIAN_VAULT_PATH`; never hardcode another vault.
6. **Review before shipping.** Run validation: file exists, no secrets, links present, counts match, and no broken wikilinks where applicable.

## Default lead schema

Use these fields unless the user supplies a stronger schema:

- `company`
- `vertical`
- `country`
- `city_or_region`
- `source`
- `ad_or_offer_summary`
- `whatsapp_or_message_cta_evidence`
- `qualification_pain`
- `lead_score_1_5`
- `recommended_opener`
- `next_action`
- `notes`

## Facebook Ads Library workflow

For Meta/Facebook Ads Library work:

1. Apply the user's country filter first. For Libya, use `country=LY` and reject non-local advertisers unless the user explicitly broadens scope.
2. Prioritize active ads with WhatsApp/message CTAs, booking language, consultation language, quote requests, or high-friction service sales.
3. Prefer verticals where lead qualification has obvious ROI: real estate, clinics, education/training, immigration, car dealers, B2B services, home services, and high-ticket retail.
4. Store the ad/library URL and a short evidence quote. Do not include private personal data beyond what the advertiser publicly exposes.
5. Deduplicate by advertiser, then by offer. If the same advertiser has several active ads, consolidate under one lead with multiple evidence links.

## Obsidian sync

Use Obsidian for durable project knowledge, not temporary scratch:

- Project folder: `lead-qualification-os/`
- Prefer Markdown notes with YAML frontmatter.
- Use wikilinks for dashboards and vertical pages.
- After writing, validate link health when possible.

Suggested notes:

- `lead-qualification-os/README.md`
- `lead-qualification-os/Libya WhatsApp Ads Library Queue.md`
- `lead-qualification-os/Vertical Playbooks/<vertical>.md`
- `lead-qualification-os/Outreach Scripts/<segment>.md`

## Quality gate

Before reporting done:

- [ ] Country/source constraints were applied.
- [ ] Every prospect has evidence.
- [ ] Lead count matches the user's requested count or the shortfall is explained.
- [ ] Repo artifacts are in the requested path.
- [ ] Obsidian artifacts are in `OBSIDIAN_VAULT_PATH`.
- [ ] No secrets or private tokens were written.
- [ ] The final response includes exact artifact paths and unresolved blockers.

## Pitfalls

- Do not claim API integration if the data was collected manually from Ads Library pages.
- Do not mix countries when the user requested a local-only list.
- Do not treat a generic Facebook page as proof of a WhatsApp campaign; capture the ad CTA or destination evidence.
- Do not ask for CRM/outreach platform choices unless the next tool call genuinely depends on it. Use Markdown/CSV as the default lead queue.
