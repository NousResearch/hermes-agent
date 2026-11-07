---
name: hallmark
description: "Build new web pages that look designed, upstream-kept."
version: 2.0.0
author: Hallmark contributors (Nutlope)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, frontend, web-design, landing-page, themes, anti-slop, lint]
    category: creative
    related_skills: [impeccable, claude-design, popular-web-designs, design-md]
    upstream:
      repo: Nutlope/hallmark
      path: skills/hallmark
---

# Hallmark (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [Nutlope/hallmark](https://github.com/Nutlope/hallmark) (MIT).
> `hermes skills install hallmark` pulls the current `skills/hallmark` bundle
> live from that repo (quarantined and scanned like any hub install), so the
> copy you run is never stale. This directory holds only catalog metadata.

Hallmark is a design skill for building NEW websites and pages that look made,
not generated. It sets a hard, non-waivable "floor" against the defaults models
fall into (token-only colours and fonts, no `transition: all`, no eyebrows or
tracked-caps labels, no purple-to-blue gradients, no invented metrics, hero
visible at 1280x800, about 80 words per screen), deals one of 50 material themes
so consecutive builds do not converge, and ships a deterministic linter,
`sloplint`, that checks the output (optionally in a real browser).

It is NOT for restyling an existing design unless the user names Hallmark
(`/hallmark redesign <path>`, `/hallmark audit <path>`). For critique and polish
of an existing UI, use `impeccable`; for a fixed brand system, use `design-md`.

## After install

- Start with `/hallmark <what to build>`; verbs: `redesign <path>`, `audit <path>`.
- Requires Node for the linter: `node <skill-dir>/scripts/sloplint.mjs <output>`.
  `--render` also needs a local Chrome/Chromium plus `puppeteer-core`
  (`npm i -g puppeteer-core`, or point `HALLMARK_PUPPETEER` at one); without
  it the render pass reports "not run" instead of failing.
- Optional: `TOGETHER_API_KEY` enables the upstream comp/image step
  (`scripts/imagegen.mjs`, calls api.together.xyz). Everything else is local.
  In Hermes, prefer `image_generate` for imagery when that key is not set.
- Do NOT run `scripts/install.mjs` or `scripts/install-hook.mjs`: they copy the
  skill into other agents' skill folders and write another harness's hook
  settings. `hermes skills install` already placed it; Hermes does not use them.
- The upstream body mentions an optional "Inspo" MCP server (signal 8); skip
  that signal unless such a server is configured in Hermes.

Full documentation: https://github.com/Nutlope/hallmark
