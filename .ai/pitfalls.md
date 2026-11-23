# Pitfalls & Known Behaviors

Real issues encountered during the One Dark Pro theme and integrated terminal work, recorded to prevent regressions and wasted debugging cycles.

## 1. CSS `zoom` vs. `getBoundingClientRect()` on `position: fixed` Overlays
- **Symptom:** Under browser UI scale (`document.documentElement.style.zoom`, e.g., `1.1`), `PersistentTerminal` shifted downward (~46px empty top gap) and overflowed rightward over `TerminalRail`, hiding the `+` button.
- **Root cause:** `getBoundingClientRect()` returns viewport-scaled coordinates, while `position: fixed` inside a CSS-zoomed `<html>` multiplies `top/left/width/height` by `zoom` a second time.
- **Rule:** Whenever converting `getBoundingClientRect()` into inline CSS coordinates inside the zoomed app tree, divide by `el.currentCSSZoom || parseFloat(document.documentElement.style.zoom) || 1`.

## 2. Stacking Context Trap: `KeepAlivePaneHost` (`z-0`) vs. `PersistentTerminal` (`z-4`)
- **Symptom:** `TerminalRail` declares `z-40`, yet `PersistentTerminal` (`z-index: 4`) still painted over it when the overlay overflowed rightward.
- **Root cause:** `KeepAlivePaneHost` has `position: absolute` + `z-0`, creating a stacking context that traps `TerminalRail`'s `z-40` inside `z-index: 0`, below the root-level `z-index: 4` overlay.
- **Rule:** Never rely on `z-40` inside `KeepAlivePaneHost` to beat `PersistentTerminal`; keep the overlay's right edge strictly `<= slot.right` (using `Math.floor(r.right / zoom) - left`).

## 3. Known Behavior at 125% UI Scale (Accepted — Do Not Alter)
- **Observation:** At `125%` UI scale, a small residual space can still appear in the terminal pane due to terminal height / xterm row quantization.
- **Status:** Validated and explicitly accepted as a non-issue. Do not add layout hacks to chase sub-row residual space at 125%.

## 4. Upstream xterm.js Mouse Selection Under CSS `zoom !== 1`
- **Observation:** Clicking/dragging inside xterm.js's canvas when `document.documentElement.style.zoom !== 1` can exhibit cursor-to-cell offset because xterm computes pointer offsets from zoomed `getBoundingClientRect()` against unzoomed `css.cell` metrics.
- **Status:** Known upstream xterm.js limitation. Do not patch xterm internals.

## 5. Empty Initial `replay: true` Frame on Fresh Browser PTYs
- **Symptom:** `stripBootGap` was skipped on newly opened persistent browser terminals.
- **Root cause:** `browser-terminal.ts` sets `replayNext = true` on the first binary frame after metadata, even when the server sends an empty (`""`) snapshot for a brand-new shell.
- **Rule:** Only treat a replay frame as real history (`hadReplay = true`) when `data.trim().length > 0`, and re-arm `createBootGapFilter()` on each `startSession()`.

## 6. Unlayered CSS vs. Tailwind v4 `@layer utilities`
- **Symptom:** List markers (`li::marker`) can be styled by both `@tailwindcss/typography` (`--tw-prose-bullets`/`--tw-prose-counters`) and utility classes (`prose-li:marker:text-muted-foreground/70` in `markdown-text.tsx`).
- **Rule:** `one-dark.css` is imported unlayered at the top of `styles.css`, so selectors under `:root[data-hermes-palette='one-dark']` outrank `@layer utilities`. Set both `li::marker` and `--tw-prose-bullets`/`--tw-prose-counters` for consistent coverage.

## 7. `lefthook` Pre-Commit Hook in Sandboxed Environments
- **Symptom:** Committing `.ts`/`.tsx` files triggers `lefthook` → `lint desktop app` → `scripts/hook-just` → `uv`, which fails with `Read-only file system` when attempting to write temporary files outside the writable workspace.
- **Rule:** Docs-only (`.md`) and CSS-only (`.css`) commits are skipped cleanly by `lefthook` without `--no-verify`. For `.ts`/`.tsx` commits in a read-only sandbox, only use `--no-verify` after explicit maintainer approval, and record the reason in the commit message.

## 8. `pnpm` Version Pin for `--no-install` Webapp Builds
- **Symptom:** Running `node apps/desktop/scripts/build-webapp.mjs --no-install` with a global `pnpm` version newer than `pm/lock.json` fails with `pnpm 12.15.0 is required`.
- **Rule:** Prepend Hermes's pinned tool runtimes (`~/.hermes/tools/pnpm-12.15.0-linux-x64` and `~/.hermes/tools/node-26.12.1-linux-x64/bin`) to `PATH` so `node-deps` reuses the existing `.build/webapp-workspace` install without reinstalling dependencies.
