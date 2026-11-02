# Atlas design language

When to load: before filling `data.mjs` or touching `template.html` — layout, palette, isometric grammar, shapes by role, labels, copy rules, and the chapter recipe live here.

The reference was a "codebase as interactive isometric diagram" screenshot: khaki paper, black hatched isometric structures, a left index of components, a right panel with *What it does / How it's built / Condition*, moving dots that are data packets you can inspect, hover to read, go inside a structure to see its steps, pan/zoom. Keep that grammar.

## Layout

- **Top strip** — stats (system, model roles, chapter n/N, structures shown n/N, questions open·routed·resolved) + controls: `◂ Back`, `Next ▸` (primary), `‖ Pause / ▸ Play`, `Trace one step`, `Refit`.
- **Left index** — grouped buttons (code · name · count). Unrevealed structures dimmed with `ch N` (click → jump to that chapter). New-in-this-chapter gets a dashed outline. Ghost (not built) = dashed border.
- **Canvas** — isometric SVG, pan by drag, wheel zoom, `+/−`. Chapter rail top-left (numbered squares + title). Flow picker bottom-left on the last chapter only.
- **Right panel** — tabs *What it does / How it's built / Open questions*. Nothing selected → the chapter story (title, lede, 2–3 sentences, "New in this chapter" chips, Back/Next). Structure selected → eyebrow (code · pinned/hovering · new), name, status chip, one-sentence `one`, then `Read more` and `Steps in execution` as `<details>`. Packet selected → route + representative JSON payload.
- **Hint bar** — keys: `enter / ] next chapter · [ back · hover to read · click to pin · → go inside · ← come out · click a dot to inspect`.

## Palette and type

Paper `#E6DFBE` · ink `#17170F` · muted `#6E6B54` · rule `#B9B293` · face `#EFE9CC` · grid `#CFC8A3`; dark theme swaps to dark olive paper `#1E1D15` / pale ink `#E6DFBE`. Tokens on `:root`, redefined under `prefers-color-scheme: dark` (guarded `:not([data-theme="light"])`) and `[data-theme="dark"]`. One face: IBM Plex Mono (Google Fonts) with a monospace fallback. Key phrases use `<mark>` = ink background, paper text. Buttons: 1.5px ink border + 2px hard shadow; primary = inverted.

## Isometric grammar

- Tile 72×36; `P(gx,gy,z) = [(gx−gy)·36, (gx+gy)·18 − z]`. Structures sorted by `gx+gy+w+d` for painter's order. Hops are ground-plane polylines with diamond bend markers; route from footprint centre to centre with one bend (`xy` or `yx`); **return hops take the other bend** so request and reply dots don't overlap.
- **Shapes by role** (the user asked for "better box shapes"): `tall` = the brain (big block with a ridge line); `store` = three stacked drums (memory, ledgers, directories); `cards` = a deck of five thin slabs (tools); `slab` = wide flat hatched top (existing backend); `screen` = thin slab with an inset rectangle and text lines (surfaces: web chat, mobile, device); `gate` = box with a dark band (confirm/approval); `job` = hatched-top box (scheduled passes); `box` = everything else. Ghosts: dashed outline, no fill.
- **Labels** (the user asked for "better labelling"): a 1–2-letter code chip on the top face **and** a readable uppercase `short` name (≤14 chars) on a paper-coloured tag under the front corner of every structure. Ghost labels dashed/muted.
- New-in-chapter: pulsing dashed halo on the ground footprint (respect `prefers-reduced-motion`). Selected: top face tinted + chip inverted + label inverted.
- Packets: ink dot with paper stroke; label visible on chapters 1–N−1 (always) and on hover/selection in the last chapter; clicking pauses everything and opens the payload. Dot pauses briefly at each hop and longer at loop start.
- Inside view: steps laid diagonally (`gx 2+i·3.4, gy 2+i·0.6`, 2×2 footprint) with a single packet walking them; breadcrumb replaces the rail; `← Come back out`.

## Copy rules

Plain words from the person's side (per the glossary): "confirm card" not "HITL prompt". Structure names are nouns; `one` is a single sentence; `what` is for a non-engineer; `how` names files, tables, APIs with `<code>`; `cond` items are questions or tasks, one line each. Chapter ledes are one sentence; stories 2–3 sentences with one `<mark>` idea. Numbers carry their date and source.

## Progressive-disclosure recipe

1. You and X (2 structures, one hop each way) → 2. Knowing (context + memory) → 3. Doing (tools + backend) → 4. Asking first (gate) → 5. Learning (hooks, signals, log) → 6. Reflecting (nightly job) → 7. Many voices (characters + directory) → 8. Keeping it honest (evals, o11y) → 9. Later (ghosts) → 10. The whole system (flow picker). Adapt the nouns; keep the shape: each chapter adds ≤3 structures and one flow that only touches revealed structures.
