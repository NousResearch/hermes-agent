# House Style

Every artifact uses one design system. Reuse these tokens verbatim — do not invent
a per-file palette. This is the single biggest lever on whether the output looks
professional or improvised.

## The canonical `:root` block

Paste this into every artifact's `<style>` (it's already in `templates/base.html`):

```css
:root {
  /* surfaces */
  --ivory:    #FAF9F5;   /* page background (warm paper) */
  --white:    #FFFFFF;   /* cards / panels */
  --slate:    #141413;   /* near-black text & inverted/dark panels */
  /* accents (semantic — see convention below) */
  --clay:     #D97757;   /* primary accent: focus / attention */
  --olive:    #788C5D;   /* success / additions / "after" / done */
  --rust:     #B04A3F;   /* error / deletions / failure path */
  --oat:      #E3DACC;   /* warm neutral fill / highlight */
  /* warm gray ramp */
  --gray-150: #F0EEE6;
  --gray-300: #D1CFC5;
  --gray-500: #87867F;   /* secondary text, arrows, muted labels */
  --gray-700: #3D3D3A;
  /* shape tokens */
  --border:        1.5px solid var(--gray-300);
  --radius-panel:  12px;
  --radius-row:    8px;
  --radius-pill:   999px;
  /* fonts (OS-native — zero loading) */
  --serif: ui-serif, Georgia, "Times New Roman", serif;
  --sans:  system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono:  ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
```

## Semantic color convention

Color encodes **meaning**, applied identically across every artifact mode:

| Token | Means |
|---|---|
| `--clay` | the thing in focus / attention / primary accent / "hot path" |
| `--olive` | success, positive delta, added lines, "after", done |
| `--rust` | error, negative, deleted lines, failure path (only add when a doc has errors) |
| `--oat` | neutral highlight / warm fill / generic badge |
| `--gray-500` | secondary text, arrowheads, muted metadata |

Never cycle colors like a rainbow. 2–3 accents per artifact.

## Type system — three fonts by role

- **Serif** (`--serif`) → all headings and big display numbers. `font-weight: 500`
  (medium, never bold), `letter-spacing: -0.01em`.
- **Sans** (`--sans`) → body copy. `line-height: 1.55–1.65`.
- **Mono** (`--mono`) → every label, code, path, metric, timestamp, pill, eyebrow.

The "eyebrow" header pattern opens most docs:

```css
.eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
           text-transform: uppercase; color: var(--gray-500); }
h1 { font-family: var(--serif); font-weight: 500; letter-spacing: -0.01em; }
```

## Boilerplate

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--ivory); color: var(--gray-700);
  font-family: var(--sans); line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  padding: 56px 24px 120px;   /* generous bottom gutter */
}
.page { max-width: 860px; margin: 0 auto; }   /* tune width per density */
html { scroll-behavior: smooth; }
```

**`.page` max-width by density:** 820–860px single-column reports/explainers;
1040–1120px two-column plans/PRs; ~780px for slide-inner.

## The card pattern (the workhorse)

White card on ivory, hairline border, rounded, optional accent border-left. This one
recipe produces stat cards, callouts, TL;DR boxes, panels, mockup frames:

```css
.card {
  background: var(--white); border: var(--border);
  border-radius: var(--radius-panel); padding: 20px;
}
.card.warn { border-left: 4px solid var(--clay); }   /* or --olive / --rust */
```

## Layout

CSS Grid for structure, Flexbox for alignment. Two-column doc shell:

```css
.layout { display: grid; grid-template-columns: 220px minmax(0,1fr); gap: 40px; }
/* minmax(0,1fr) prevents the content column from overflowing */
aside { position: sticky; top: 32px; align-self: start; }   /* in-page nav / TOC */
h2 { scroll-margin-top: 24px; }   /* so anchor jumps clear the top */

@media (max-width: 860px) {        /* the entire responsive strategy: */
  .layout { grid-template-columns: 1fr; }   /* collapse to one column */
  aside { display: none; }                  /* hide the sidebar */
}
```

Stat/summary bands: `display: grid; grid-template-columns: repeat(4, 1fr);` with one
breakpoint to `repeat(2,1fr)`.

## Tables

Real `<table>` for tabular data: `border-collapse`, a `--gray-150` `<thead>` with
small uppercase mono headers, hairline row borders, wrapped in a rounded card with
`overflow: hidden` to clip the corners. Use a `display:grid` "table" of `.row`/`.cell`
divs only when cells need rich content or must restack responsively (swap
`border-left` for `border-top` at the breakpoint).

## Code blocks + hand-rolled highlighting

Code lives in a dark `--slate` rounded panel, `overflow-x: auto`, mono ~13px. No
Prism/highlight.js — wrap tokens in semantic spans:

```css
.code { background: var(--slate); color: #E8E6DF; border-radius: var(--radius-panel);
        padding: 16px 18px; font-family: var(--mono); font-size: 13px; overflow-x: auto; }
.code .kw  { color: var(--clay); }    /* keywords */
.code .str { color: var(--olive); }   /* strings */
.code .cm  { color: var(--gray-500); }/* comments */
.code .fn  { color: #C9B98A; }        /* function names (warm tan) */
```

**Diff rendering** — a 3-column grid (line-no | mark | code) with tinted full-width
rows. Values match the gallery's `03-code-review-pr.html` verbatim:

```css
.diff-row { display: grid; grid-template-columns: 48px 18px 1fr; white-space: pre;
            font-family: var(--mono); font-size: 12.5px; }
.diff-row .ln   { color: var(--gray-500); text-align: right; padding-right: 10px; }
.diff-row .code { color: #E8E6DC; }
.diff-row.add { background: rgba(120,140,93,0.15); }   /* olive tint */
.diff-row.add .mark { color: var(--olive); }
.diff-row.del { background: rgba(176,74,63,0.15); }    /* rust tint */
.diff-row.del .mark { color: var(--rust); }
.diff-row.ctx  .code { color: #B8B6AC; }               /* unchanged context */
.diff-row.hunk .code { color: var(--gray-500); }       /* @@ -0,0 +1,58 @@ headers */
```

## Callouts, pills, badges (pure CSS)

```css
.callout { background: rgba(217,119,87,0.06); border-left: 3px solid var(--clay);
           border-radius: var(--radius-row); padding: 14px 16px; }
.pill  { border-radius: var(--radius-pill); padding: 2px 10px; font-family: var(--mono);
         font-size: 11px; background: var(--oat); }
.badge { border-radius: 6px; padding: 1px 7px; font-family: var(--mono); font-size: 11px; }
.badge.new { background: rgba(120,140,93,0.18); color: var(--olive); }
.badge.del { background: rgba(176,74,63,0.18); color: var(--rust); }
```

Tinted backgrounds use `rgba()` of an accent — don't add new tokens for them.

## Decoration is drawn, not imported

- **Timeline** = a `::before` vertical rail + absolutely-positioned dots, colored by state.
- **Checkbox tick** = a bordered square with an `::after` rotated-border tick when `.done`.
- **Progress bar** = a track div + a `width:%` fill div.
- **Diagrams/charts/icons** = hand-authored inline `<svg>` (see `svg-diagrams.md`).

## Spacing rhythm

Section gaps ~52–64px; element gaps on an 8 / 12 / 14 / 18 / 22px scale. Consistent
spacing is most of what reads as "designed".
