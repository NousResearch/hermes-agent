# Reference Examples (Anthropic html-effectiveness gallery)

Twenty complete, self-contained reference HTML files — Anthropic's
[html-effectiveness gallery](https://github.com/anthropics/html-effectiveness),
MIT licensed. These are the ground truth this skill is built on. **Reading the one
that matches your mode is a required step before authoring** (workflow step 2): a
full polished example carries density, spacing, and structure that no prose summary
reproduces. The other references explain *why* the patterns are the way they are;
these show you the patterns whole.

They are **not committed into this skill** (it's someone else's living repo, ~384 KB).
Fetch them with the bundled script — it's idempotent, so just run it every time; it
clones if the examples are missing and pulls the latest otherwise.

## Fetch + read (do this before writing)

```
terminal:  bash scripts/fetch-examples.sh
read_file  references/examples/<file-for-your-mode>.html
```

The script lands the files in `references/examples/`. Always run it first — it's
cheap and self-healing, so you never have to wonder whether the examples are
present. Then read the index or jump straight to the file for your mode:

```
read_file references/examples/index.html              # categorized index of all 20
read_file references/examples/03-code-review-pr.html  # a specific example
```

Only if the fetch genuinely fails (no network) do you fall back to the distilled
pattern references alone — and say so, since you're then working without the source.

## What each file demonstrates → which to read

Pick the example closest to your mode, read it, then adapt — don't copy verbatim.

| File | Mode | Read it when you're building… |
|---|---|---|
| `01-exploration-code-approaches.html` | variants | a side-by-side comparison of code approaches with tradeoffs + a recommendation |
| `02-exploration-visual-designs.html` | variants | live design directions on a light/dark switchable surface |
| `03-code-review-pr.html` | code review | a PR/diff review — the gold-standard 3-column diff grid + risk map + comment bubbles |
| `04-code-understanding.html` | explainer | a code-flow explainer with an inline-SVG request-path diagram + callstack |
| `05-design-system.html` | report | a design-token / component reference sheet |
| `06-component-variants.html` | editor | a live component matrix driven by `:root` custom-property knobs |
| `07-prototype-animation.html` | editor | a CSS micro-interaction tuner (easing knobs, static copy-paste CSS export) |
| `08-prototype-interaction.html` | editor | a drag-to-reorder feel-test (DOM-only, no export by design) |
| `09-slide-deck.html` | report | a scroll-snap slide deck (pure-CSS paging) |
| `10-svg-illustrations.html` | diagram | standalone exportable inline-SVG illustrations |
| `11-status-report.html` | report | a weekly status report (zero-JS, shape tokens, stat band) |
| `12-incident-report.html` | report | an incident postmortem (CSS-only timeline + checklist) |
| `13-flowchart-diagram.html` | diagram | a clickable annotated flowchart with a synced detail panel (`data-k` pattern) |
| `14-research-feature-explainer.html` | explainer | "how feature X works" — sticky anchor-nav doc shell + tabbed code |
| `15-research-concept-explainer.html` | explainer | an interactive concept explainer (deterministic-hash SVG demo + glossary) |
| `16-implementation-plan.html` | plan | an implementation plan — milestone timeline, SVG architecture, DOM mockups |
| `17-pr-writeup.html` | code review | a PR walkthrough for reviewers — file-by-file tour, hand-marked diffs, TOC |
| `18-editor-triage-board.html` | editor | a drag-to-triage board with copy-as-markdown export |
| `19-editor-feature-flags.html` | editor | a config-flag editor with copy-diff + copy-full-JSON export |
| `20-editor-prompt-tuner.html` | editor | a prompt-template editor (contenteditable + live preview + copy-prompt) |

All 20 are single-file, zero-dependency, no-build — the same discipline this skill
requires. Use them to calibrate density, spacing, and the house style; the distilled
references (`house-style.md`, `svg-diagrams.md`, `throwaway-editors.md`, …) tell you
*why* each pattern is the way it is.
