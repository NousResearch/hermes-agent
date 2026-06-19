---
name: html-artifact
description: Build self-contained HTML files to explain, plan, or review.
version: 1.0.0
author: Anthropic (html-effectiveness gallery, MIT), adapted for Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [html, artifact, explainer, plan, report, code-review, diagram, svg, design, prototype, editor]
    related_skills: [claude-design, popular-web-designs, design-md, excalidraw, p5js]
---

# HTML Artifact Skill

Produce a single self-contained `.html` file — no build step, no dependencies, no
CDN — whenever the deliverable is something a human should *read, share, or poke at*:
a concept explainer, an implementation plan, a status/incident report, a code-review
walkthrough, a technical or educational diagram, a set of design variants, or a
throwaway editor that exports its result back to you.

HTML beats Markdown once a doc has color, layout, diagrams, tables, code, or
interaction. It opens in any browser, shares as a link, stays readable past 100
lines, and can carry SVG diagrams and live controls Markdown can't. Default to an
HTML artifact when the user says "make an HTML file/artifact", or asks you to
*explain how X works*, *write up a plan/PR/report*, *diagram* something, *compare*
options, or *prototype* an interaction — even when they don't say "HTML".

This skill **supersedes** the former `sketch`, `architecture-diagram`, and
`concept-diagrams` skills — design-variant comparison, dark-tech infra diagrams,
and educational SVG diagrams are all modes here. For matching a known brand's look
use `popular-web-designs`; for a formal design-token spec file use `design-md`; for
hand-drawn/whiteboard `.excalidraw` files use `excalidraw`; for generative/animated
canvas art use `p5js`. This skill is for everything else that ships as an HTML page.

## Reference files (load on demand)

- `references/house-style.md` — the canonical `:root` token block, type system,
  card/table/callout/code-block patterns. **Read this before authoring any artifact.**
- `references/examples.md` — 20 complete reference HTML files (Anthropic's
  html-effectiveness gallery, MIT) keyed to each mode, plus the script to fetch them.
  Read/fetch one that matches your task to calibrate the house style from a full example.
- `references/svg-diagrams.md` — hand-authored inline SVG: arrow markers, node
  groups, decision diamonds, edge semantics, coordinate-grid discipline. Read for
  any flowchart / architecture / concept diagram.
- `references/concept-archetypes.md` — the 9-ramp educational color system + a
  library of diagram archetypes (timeline, tree, quadrant, layered stack,
  before/after, hub-spoke, cross-section). Read for educational / non-software visuals.
- `references/dark-tech.md` — the dark "infra" token variant (carries the old
  architecture-diagram aesthetic). Read for cloud/infra/system architecture diagrams.
- `references/throwaway-editors.md` — the single-file editor recipe and the
  copy-to-clipboard export pattern that survives `file://`. Read when the artifact
  needs interactive controls that export state back to a prompt.
- `references/fidelity-and-verify.md` — the throwaway↔presentation fidelity dial,
  the multi-variant comparison layout, and the mandatory browser-vision verify loop.

## Templates

- `templates/base.html` — document scaffold with the house-style `<style>` block.
- `templates/diagram.html` — dual-mode diagram host (light educational + dark infra
  CSS, arrow markers, node/edge classes). Paste your SVG where marked.
- `templates/editor.html` — throwaway-editor skeleton (state → render → export).

Load one with `skill_view(name="html-artifact", file_path="templates/base.html")`.

## Workflow

1. **Pick the mode.** Match the request to one artifact type — explainer, plan,
   report, code review, diagram, variants, or editor. Each has a section in the
   references; the mode decides which template and which references to load.
2. **Decide fidelity.** Throwaway exploration or presentation-grade deliverable?
   See `references/fidelity-and-verify.md`. Don't over-polish a quick comparison;
   don't ship a sloppy report.
3. **Start from a template + the house style.** Load `templates/base.html` (or
   `diagram.html` / `editor.html`) and `references/house-style.md`. Reuse the
   `:root` tokens — never invent a new palette per file. For a richer reference,
   fetch the gallery (`bash scripts/fetch-examples.sh`) and `read_file` the worked
   example that matches your mode — see `references/examples.md` for the mode→file map.
4. **Author the artifact** with `write_file`. Keep everything inline: one `<style>`
   in `<head>`, at most one `<script>` before `</body>`. No `<link>`, no external
   fonts (use OS-native stacks), no CDN, no `<img src>` to remote URLs. All graphics
   are inline SVG or CSS.
5. **Keep JS optional and graceful.** Prefer zero JS. When you need it, keep it to
   a small vanilla IIFE and make the page render meaningfully with JS off (native
   `<details>`, anchor nav, a default-active tab/node).
6. **Verify visually.** Open the file and screenshot it — see the verify loop in
   `references/fidelity-and-verify.md`. This is mandatory for SVG diagrams, where
   hand-placed coordinates drift on edits (overlapping nodes, misaimed arrows).
7. **Report the path.** Tell the user the absolute file path so they can open it.
   Mention any interactive controls / export buttons.

## Core principles

**One design system, token-driven.** Warm paper (`--ivory`), near-black ink
(`--slate`), one terracotta accent (`--clay`), olive for success/additions, a warm
gray ramp. Semantic convention, held across every mode: **clay = focus/attention,
olive = success/added, rust = error/removed, oat = neutral fill, gray-500 =
secondary text & arrows.** Reference colors only as `var(--…)`.

**Three fonts by role.** Serif (Georgia stack) for headings, sans (system-ui) for
body, mono for every label / code / metric / eyebrow / path. All OS-native — zero
font loading. This serif-heading / mono-label / sans-body split is the house tell.

**Self-contained, always.** The file must render offline when double-clicked.
Inline the style and script; draw graphics as inline SVG or CSS; never reference a
remote asset. This is non-negotiable — it's what makes the artifact shareable.

**Graceful degradation.** Most great artifacts have *no* JS. When interactivity is
the point (sliders, drag, editors), the page must still convey its content without
JS, and exports must work from a `file://` page (clipboard fallback in
`references/throwaway-editors.md`).

**End interactive artifacts with an export.** A throwaway editor is only useful if
it hands its result back: a Copy-as-markdown / Copy-JSON / Copy-diff / Copy-prompt
button that serializes state to the clipboard for pasting into the next prompt.

## Quick reference — mode → what to build

| Request | Mode | Template | Key reference |
|---|---|---|---|
| "explain how X works" | explainer | base | house-style, svg-diagrams |
| "write up the plan / spec" | plan | base | house-style |
| "status / incident report" | report | base | house-style |
| "review this PR / diff" | code review | base | house-style (diff section) |
| "diagram the architecture / pipeline" | infra diagram | diagram | dark-tech, svg-diagrams |
| "diagram this concept / process" (science, physical, educational) | concept diagram | diagram | concept-archetypes, svg-diagrams |
| "show me N takes / compare options" | variants | base | fidelity-and-verify |
| "let me tune / triage / edit X and copy it out" | editor | editor | throwaway-editors |

## Pitfalls

- **Don't invent a palette.** Reuse the `:root` tokens from `house-style.md`. A
  per-file color scheme breaks the consistency that makes these artifacts feel pro.
- **Don't reach for a library.** No Mermaid, D3, Tailwind CDN, Prism, or web fonts.
  Diagrams are hand-authored SVG; syntax highlighting is hand-marked `<span>`s; the
  token block does the job of a build-time theme.
- **Don't skip the visual check on diagrams.** Manually computed SVG coordinates
  are the #1 source of broken output — arrows landing in whitespace, overlapping
  boxes, text overflow. Screenshot and fix before reporting done.
- **Don't add a JS export where a static `<pre>` suffices.** If the deliverable is
  one snippet, a hand-selectable code block is the bulletproof "export".
- **Don't let JS be load-bearing for content.** If the prose only exists inside a
  `render()` call, the page is blank with JS off. Put real content in the HTML;
  use JS to enhance, not to populate.
