---
name: pr-lens
description: "Draw code changes as animated architecture/data-flow SVGs."
version: 1.0.0
author: Coldtea AI (adapted by Nous Research)
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [diagrams, pull-requests, code-review, svg]
    category: software-development
    related_skills: []
    upstream: https://github.com/coldteadotai/pr-lens (pinned 0993b4d)
---

# PR Lens Skill

PR Lens draws code as visually rich animated diagrams: diffs, architecture, data flows. You describe the diff or codebase as one JSON document (lanes, nodes, edges, ordered flows) and the CLI renders it as animated SVGs. There is no findings lens — PR Lens is a comprehension layer, not a review bot. There is no field for a bug, risk, or security note, and a document that invents one is rejected.

## When to Use

- Asked to diagram, visualise, or explain a code change or a system.
- A pull request should carry an architecture or data-flow diagram.
- Keywords: PR Lens, diagram, architecture, data flow, visualise, pull request.

## Prerequisites

- Node.js with `npx` (the CLI runs via `npx @coldtea/pr-lens-cli@latest`; no install step).
- `gh` (GitHub CLI) — optional, only for attaching diagrams to PRs.
- Optional canvas publishing calls the third-party service prlens.dev (see step 4b).

## How to Run

Run all commands with the terminal tool from the repository root.

1. **Read the diff.** When representing a code change: `git diff --find-renames <base>...<head>`. The base is the merge base, not the tip of the base branch. If not expressing a diff, read the code to be visualised.

2. **Write the document** to `.pr-lens/graph.json`, following `references/graph-document.md`. `references/example.graph.json` is a valid reference with three lanes, all four delta states, a hero edge, a seven-step flow, a nested drill-down tree and a six-step walkthrough. Read it before writing your first document — quicker than reading the reference.

3. **Validate, and fix.**

   ```bash
   npx @coldtea/pr-lens-cli@latest validate .pr-lens/graph.json
   ```

   Fix every failure and run it again. Do not render an invalid document; do not "work around" a failure by deleting the element it names.

4. **Render.**

   ```bash
   npx @coldtea/pr-lens-cli@latest render .pr-lens/graph.json --theme light
   ```

   Render light by default unless the user requests another theme. The SVGs, the manifest and `drawn.graph.json` land in `.pr-lens/`, which the CLI adds to the repository's .gitignore. Do not commit any of it — these files are rebuilt from the diff on demand. Each SVG is named after its view, theme and content hash; `manifest.json` lists them by lens and view.

4b. **Canvas push — OPTIONAL, opt-in.** Only when the user explicitly asks for a shareable link. This publishes `.pr-lens/drawn.graph.json` to the third-party service prlens.dev:

   ```bash
   npx @coldtea/pr-lens-cli@latest canvas push
   ```

   It prints three links. Give the user the **view link** (`https://prlens.dev/c/{id}`): the full-screen diagram, every view on one page, no login. The **edit link** (ending in `#w=…`) lets its holder overwrite the canvas — it is a secret: leave it out of the reply unless asked, never paste it anywhere public. The embed link serves the top view as an SVG for a README. Pushing the same file again updates the same canvas, so "rename that node" is: edit, validate, render, push — the link stays the same. If the push fails, say so and tell the user where the local SVGs are and which is the top view.

5. **Attach to a PR, when there is one.** Upstream documents `gh pr create/edit/comment --attach <path>`, but `--attach` arrived in GitHub CLI 2.99 — check `gh --version` first (e.g. gh 2.97 does NOT have it). With gh ≥ 2.99: write the body with a Markdown image `![alt](.pr-lens/<view>.svg)` (an HTML `<img>` is left as written and the file appended at the bottom instead; alt text is the one-line caption a reader without images gets), then repeat `--attach <path>` per referenced diagram:

   ```bash
   gh pr create --title "…" --body-file .pr-lens/body.md --attach .pr-lens/overview-light-<hash>.svg
   ```

   Without `--attach`, use a commit-free path:
   - Upload the SVGs to a gist: `gh gist create .pr-lens/<view>.svg`, then reference the raw gist URL in the PR body/comment, or
   - Publish via the canvas link (step 4b, with user consent) and link the view URL, or
   - Note the local `.pr-lens/` path in the PR body so reviewers can rebuild.

   Once published somewhere durable, let the CLI compose the comment markdown:

   ```bash
   npx @coldtea/pr-lens-cli@latest comment \
     --graph .pr-lens/drawn.graph.json \
     --manifest .pr-lens/manifest.json \
     --asset-base-url <where you published the SVGs>
   ```

   `--graph` takes `drawn.graph.json`, not the document you wrote — the CLI refuses a document its manifest does not describe. Leave out `--asset-base-url` and the markdown points at local paths no reader can fetch. The markdown goes to stdout; posting it is your business.

   Attach the views a reviewer needs and leave the rest in `.pr-lens/`: the top architecture view first, then a data flow if the change has a sequence worth following. Two diagrams usually beat four.

6. **Optional automation:** `npx @coldtea/pr-lens-cli@latest analyze --base <ref>` does steps 1–2 by asking a provider (Gemini, OpenAI, or any `/chat/completions` endpoint) with a key of your own. That is the only path here that needs one; normally you author the document yourself.

## What makes a document worth reading

- **Include what did not change.** Unchanged neighbours a change touches are the context; mark them `delta: "unchanged"`.
- **Lanes are the reader's mental model** (a runtime, a tier, a boundary), not the folder tree.
- **One hero edge**, two at the outside: the connection the change is really about.
- **Add a flow only when there is a sequence** worth animating. One good flow beats three thin ones.
- **Attach file refs**: they become the permalinks a reviewer clicks.
- Architecture views are a C4-inspired decision tree: system context → container → component, each child materially narrower. Skip empty or repetitive levels; keep data-flow views as separate roots; set `defaultOpen: true` on the highest useful architecture view.
- **Walkthroughs** (2–12 steps, aim 3–7): write one for anything non-trivial. Each step = one change (added/removed/moved), headline change first, overview last. Headings ≤48 chars built from change words; bodies ≤140 chars on behaviour, required. Write for a smart twelve-year-old; no "leverages"/"orchestrates". Keep consecutive steps on the same stage. The walkthrough field needs CLI ≥ 0.4.0 (contract 0.1.1).
- **Fixing a wrong map:** never edit the generated document — write corrections into `.github/pr-lens.yml` (see `references/config.md`), then validate it: `npx @coldtea/pr-lens-cli@latest validate .github/pr-lens.yml`. Prefer path globs over `id:` matches.

## Quick Reference

| Command | Purpose |
| --- | --- |
| `npx @coldtea/pr-lens-cli@latest validate .pr-lens/graph.json` | validate the document (also validates `.github/pr-lens.yml`) |
| `npx @coldtea/pr-lens-cli@latest render .pr-lens/graph.json --theme light` | render SVGs + manifest into `.pr-lens/` |
| `npx @coldtea/pr-lens-cli@latest canvas push` | OPTIONAL: publish to prlens.dev (opt-in only) |
| `npx @coldtea/pr-lens-cli@latest comment --graph … --manifest … --asset-base-url …` | compose PR comment markdown to stdout |
| `npx @coldtea/pr-lens-cli@latest analyze --base <ref>` | auto-author document via an LLM provider (needs API key) |

Validator failure codes:

| Code | What you did |
| --- | --- |
| `BROKEN_REFERENCE` | an edge, flow step, view or walkthrough step names an id you never declared |
| `INVALID_DOCUMENT` | an invented field; the schemas are strict, unknown keys are rejected |
| `DUPLICATE_ID` | two nodes, edges or views sharing an id |
| `UNSUPPORTED_SCHEMA_VERSION` | `schemaVersion` is not the contract version installed |

## Pitfalls

- Six rules are parser-only, not in the JSON Schema (referential integrity, inverted line ranges, disagreeing `self` endpoints, identical patch commits, too many views for a manifest, flow-step focus on the wrong stage) — always run `validate`, structured output alone is not enough.
- Do not commit anything in `.pr-lens/`; it is regenerated and gitignored by the CLI.
- The `--attach` gh flag needs gh ≥ 2.99; older gh silently lacks it — check before writing a body around it.
- The canvas edit link (`#w=…`) is a write credential — never share it unprompted or paste it publicly.
- A stored map never carries a walkthrough; a walkthrough tells the story of one change.
- `pr-lens render` reports corrections in `.github/pr-lens.yml` that matched nothing — that is drift worth fixing, not an error.

## Verification

Smoke test (live-verified 2026-09-12 with `@coldtea/pr-lens-cli` via npx, node on Linux):

```bash
cp references/example.graph.json /tmp/prlens-smoke/ && cd /tmp/prlens-smoke
npx -y @coldtea/pr-lens-cli@latest validate example.graph.json
# ✓ example.graph.json — graph document · 3 lanes, 10 nodes, 13 edges, 1 flow · 6 walkthrough steps
npx -y @coldtea/pr-lens-cli@latest render example.graph.json --theme light
# ✓ .pr-lens/manifest.json — 4 SVGs across 4 diagrams
```

Expect exit 0 on both and four `*-light-<hash>.svg` files plus `manifest.json` and `drawn.graph.json` in `.pr-lens/`.

---

Adapted from [coldteadotai/pr-lens](https://github.com/coldteadotai/pr-lens) (packages/agent-skill, pinned 0993b4d), MIT License, Copyright (c) 2026 Coldtea AI. See LICENSE.txt.
