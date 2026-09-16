---
name: impeccable
description: Frontend design guidance, upstream-maintained (impeccable).
version: 4.1.2
author: Paul Bakaus (pbakaus)
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, frontend, ui, ux, web-design, anti-slop]
    category: creative
    related_skills: [claude-design, popular-web-designs]
    upstream:
      repo: pbakaus/impeccable
      path: .hermes/skills/impeccable
---

# Impeccable (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [pbakaus/impeccable](https://github.com/pbakaus/impeccable): the project
> ships and verifies a Hermes-native skill bundle under `.hermes/skills/`.
> `hermes skills install impeccable` pulls the current bundle live from that
> repo (quarantined and scanned like any hub install) — this directory holds
> only the catalog metadata, so the vendored copy can never go stale.

Impeccable is a design language for AI coding agents: one skill exposing 23
sub-commands (`/impeccable init`, `craft`, `shape`, `critique`, `audit`,
`polish`, `bolder`, `quieter`, `distill`, `harden`, `onboard`, `animate`,
`colorize`, `typeset`, `layout`, `delight`, `overdrive`, `clarify`, `adapt`,
`optimize`, `extract`, `document`, `live`), explicit anti-pattern guidance
(overused fonts, purple gradients, nested cards, bounce easing), and a
61-rule deterministic detector CLI (`npx impeccable detect`) that needs no
LLM or API key.

After install, start with:

```
/impeccable init
```

Full documentation: https://impeccable.style
