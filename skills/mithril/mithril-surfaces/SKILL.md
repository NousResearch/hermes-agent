---
name: mithril-surfaces
description: Open Studio, extensions, ontology, and download pages.
version: 0.1.0
author: jun (junkawasaki), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [mithril, studio, ontology, download]
    related_skills: [security-assurance, security-records]
---

# Mithril Surfaces Skill

Point the operator at the app.mithril.fund surfaces that remain hosted pages, and use the local skills or the `mithril-fund` MCP for the checks that moved. This skill does not compile a `.mith` program, publish an installer, or call an internal pipeline.

## When to Use

- The user wants App Studio, an extension listing, the DM2 ontology, the desktop download page, or the enterprise page.
- The user asks which former app or API now lives in a skill, the MCP, or still on the host.

Don't use for:

- Reimplementing the Studio compiler, OWL materialization, or SHACL validation.
- Writing lake snapshots, phishing-domain rows, Stripe webhooks, or request-ledger events.
- Starting a sandbox, a checkout, or a research-scope declaration.
- Approving a device login on the user's behalf.

## Prerequisites

- The `browser` tools when the user wants the page opened.
- `security-assurance` and `security-records` for the reviews and record checks.
- The `mithril-fund` MCP for authenticated reads. It reads `MITHRIL_API_ORIGIN` and `MITHRIL_API_TOKEN` from the environment. The token is not written into a skill file.

## How to Run

Choose the row, then either open the page or call the local tool named there.

| Former surface | Where it lives now | Action |
|---|---|---|
| `/`, `/studio/`, `POST /api/compile`, `/d/` | Hosted Studio | Open `https://app.mithril.fund/studio/` |
| `/extensions` | Hosted catalog | Open `https://app.mithril.fund/extensions` |
| `/ontology/dm2/2.02` | Hosted ontology | Open `https://app.mithril.fund/ontology/dm2/2.02` |
| `/download/` | Hosted installers | Open `https://app.mithril.fund/download/` |
| `/enterprise/` | Hosted pages | Open `https://app.mithril.fund/enterprise/` |
| Assurance, mobile, operations reviews | `security-assurance` | Run that skill's `review.py` |
| CMDB, email signals, CSF fit, service catalog | `security-records` | Run that skill's `records.py` |
| Domain reputation, account, models, billing status, cases, sandbox status | `mithril-fund` MCP | Call the matching read tool |
| `/mcp` on the host | This MCP, stdio | Config name `mithril-fund` |

## Procedure

1. Match the request to one row. Completion: the row's "where it lives now" is named in the reply.
2. For a hosted page, open that exact URL with `browser_navigate` only when the user asked to see it. Completion: the address bar is one of the URLs in the table.
3. For a moved check, follow that skill or MCP tool. Completion: the tool result includes its `boundary` sentence.
4. If the request is a lake write, Stripe webhook, sandbox launch, checkout, research-scope write, or device approval, stop and say that call stays on the API. Completion: no token was sent and no POST was made.

## Pitfalls

- Studio's compile receipt is produced by the hosted Worker. Describing the page is not a successful compile.
- Download links are the published preview artifacts. Do not invent a newer installer URL.
- A public domain check answers whether that name is in the published phishing projection. It does not investigate a person.
- Authenticated reads fail closed when `MITHRIL_API_TOKEN` is unset. Ask the user to set it. Do not paste a token into the chat.

## Verification

The reply names one owner for the request: a hosted URL, `security-assurance`, `security-records`, the `mithril-fund` MCP, or "stays on the API". A hosted page was opened only when the user asked to see it.
