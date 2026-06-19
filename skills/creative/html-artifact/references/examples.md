# Reference Examples (Anthropic html-effectiveness gallery)

Twenty complete, self-contained reference HTML files — Anthropic's
[html-effectiveness gallery](https://github.com/anthropics/html-effectiveness),
MIT licensed. These are the ground-truth artifacts the patterns in this skill were
distilled from. Reading a full polished example beats any prose description when you
need to match the house style for a given artifact type.

They are **not committed into this skill** (it's someone else's living repo, ~384 KB).
Fetch them on demand with the bundled script, then `read_file` the one that matches
your task.

## Fetch them

```
terminal: bash scripts/fetch-examples.sh
```

Idempotent — clones on first run, pulls latest after. Files land in
`references/examples/`. Then read whichever fits:

```
read_file references/examples/index.html              # categorized index of all 20
read_file references/examples/03-code-review-pr.html  # a specific example
```

If the fetch fails (no network), fall back to the distilled patterns in the other
references — they capture the same conventions. The examples are a richer supplement,
not a hard dependency.

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
