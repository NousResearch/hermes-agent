# Design Directions Renderer

Subagent prompt. Produce 3-6 distinct visual directions for the product's hero screen, each a separate self-contained HTML file at `.working/direction-{slug}.html` (or one combined `directions-{n}.html` if the parent's intent says side-by-side).

Each direction is a *complete visual personality* applied to the same key screen — not a palette swap. Differ on density, type weight, motion implication, brand register. Each file: 2-3 sentence rationale, near-1:1 hero screen mockup in a phone or browser frame, ideally a secondary screen, at least one state variant visible (aging row, empty state, etc).

Use real product content from the conversation. Voice/tone from `.memlog.md` applied to every visible string — no lorem. Inline CSS, system fonts, no JS or network. Document hex values in `<style>` comments per direction.

Return to the parent: file paths, one-line personality summary per direction, what hero screen was depicted. Do not dump HTML into parent context. If interactive, open each file in the browser.
