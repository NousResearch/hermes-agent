# Fidelity and Verification

Two cross-cutting concerns: how polished to make an artifact (the fidelity dial,
carried over from the former `sketch` skill), and how to catch the broken output
before you report done (the browser-vision loop — mandatory for diagrams).

## The fidelity dial

Match effort to intent. Over-polishing a quick comparison wastes time; shipping a
sloppy report undercuts the point of using HTML at all.

**Throwaway / sketch fidelity** — fast, low-ceremony, meant to be reacted to and
discarded. Use when the user says "sketch", "rough", "show me what X could look
like", "a quick take", "compare A vs B", "mock this before I build". Signals:
- Realistic *fake* content (don't make the user imagine — fill it in).
- System fonts, the house tokens, minimal custom CSS. One or two states of
  interactivity, not a full app.
- Multiple variants over one perfect screen (see below).
- Explicitly disposable: a sketch worth keeping should be promoted into real project
  code, not curated as a deliverable.

**Presentation fidelity** — a real deliverable someone will read end-to-end and
share: an explainer, plan, report, PR write-up, or a diagram going into docs. Full
house style, careful spacing, verified diagrams, graceful-degradation checked.

When unsure, ask one question ("quick throwaway or polished deliverable?") rather
than guessing — the two need very different amounts of effort.

## Multi-variant comparison

When the user wants to *choose a direction*, generate 3–6 distinct variants and lay
them out for side-by-side comparison in **one** HTML file. Three proven layouts:

- **Static tradeoff columns** — equal-weight columns, each with the approach, a code
  or visual sample, a small tradeoffs table, and uniform metric chips
  (`Bundle: +0.2kb`, `Reuse: high`). Close with one opinionated **recommendation**.
  Best for comparing *code approaches* or strategies.
- **Live artboards on a switchable surface** — a 2×N grid of `.artboard` cards each
  rendering a real variant, with a light/dark toggle so each is proven on both
  surfaces. Best for *visual design* directions. Per-stage theme via scoped tokens:
  ```css
  .stage      { --fg: var(--slate); --panel: var(--white); --line: var(--gray-300); }
  .stage.dark { --fg: #F0EEE6; --panel: #1F1E1B; --line: #3D3D3A; }
  ```
  Variants reference only `var(--fg/--panel/--line)`, so flipping `.dark` re-themes all.
- **Live token matrix** — a toolbar of controls (slider / segmented / checkbox) that
  writes to `:root` custom properties so every variant cell updates at once. Best for
  a *component* explored across a parameter space (density × border × shadow).

Always: vary layout/tone/density meaningfully (not cosmetic tweaks), label each
variant with the tradeoff it's making, and state your pick.

## The browser-vision verify loop (mandatory for diagrams)

Hand-placed SVG coordinates drift: arrows land in whitespace, boxes overlap, text
overflows its rect, the legend collides with a boundary. Static review of the markup
does **not** catch this — you must look at the rendered pixels.

1. Write the file with `write_file`.
2. Open it: `browser_navigate(url="file:///absolute/path/to/artifact.html")`.
3. Inspect it: `browser_vision(question="Are any arrows pointing into empty space?
   Any overlapping boxes or text overflowing its container? Is the legend clear of
   the diagram? Is anything cut off?")`. (Or `browser_screenshot` and read it.)
4. Fix what the screenshot reveals — recompute the offending coordinates, widen a
   box to fit its text, bump the viewBox height, move the legend.
5. Re-render and re-check until clean.

For non-diagram artifacts (reports, plans, explainers) a single screenshot pass is
enough to catch layout breakage — overflow, broken grids, unreadable contrast,
clipped content. Always do at least one visual pass before telling the user it's done;
"it's valid HTML" is not the same as "it renders correctly".

## Graceful-degradation check

If the artifact has JS, confirm the page still conveys its content with JS disabled:
real prose lives in the HTML (not only inside a `render()` call), collapsibles use
native `<details>`, tabs default one to `.on`, interactive diagrams set a
default-active node. The artifact should never be blank without JavaScript.
