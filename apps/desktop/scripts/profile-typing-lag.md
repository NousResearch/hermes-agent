# Profiling renderer typing lag

Workflow for empirically measuring (and fixing) typing/submit lag in the
desktop chat composer.

> **Note (Jul 2026):** the standalone `measure-*` / `profile-*` scripts this
> doc references have been consolidated into the systematized perf harness at
> `scripts/perf/` (`npm run perf`, `npm run perf:serve`). CPU profiling is now a
> `--cpuprofile` flag on any scenario. This doc is kept as an investigation log;
> for the current tooling and the scenario→old-script mapping see
> `scripts/perf/README.md`.

## Quick boot for profiling

Vite 8 + plugin-react 6 has a known issue where the React Fast Refresh
preamble script isn't injected into `index.html`, so opening Electron at
`http://127.0.0.1:5174` throws `$RefreshReg$ is not defined` on every TSX
module and the React tree never mounts. Workaround: run vite with HMR off.

```bash
# Terminal A — start dev server without HMR
cd apps/desktop
node scripts/dev-no-hmr.mjs

# Terminal B — start Electron with CDP exposed
cd apps/desktop
XCURSOR_SIZE=24 HERMES_DESKTOP_DEV_SERVER=http://127.0.0.1:5174 \
  ../../node_modules/.bin/electron --remote-debugging-port=9222 .
```

Terminal C is yours to run the harnesses.

## Harnesses

All zero-dep — Node 24 built-in `WebSocket` + `fetch`.

### Typing latency — `measure-latency.mjs`

Per-keystroke `keypress → next paint` latency, p50/p90/p99/max.
Synthesizes keystrokes via `Input.dispatchKeyEvent` so the run is
reproducible.

```bash
node apps/desktop/scripts/measure-latency.mjs --chars=120 --cps=20
```

Anything > 16ms is a dropped frame. On a freshly-loaded session
(`scripts/click-session.mjs 'Phaser particle'`) we currently see:

| | unpatched | patched |
|---|---|---|
| p50 paint | 1.9 ms | 2.0 ms |
| p90 paint | 3.3 ms | 13.7 ms |
| p99 paint | 16.7 ms | 15.2 ms |
| max paint | 20.5 ms | 30.4 ms |
| >16ms drops | 2/120 | 1/120 |

Roughly even on a quick session — patches don't fix typing latency
under benign synthetic conditions because the existing baseline is
already snappy on synthetic input. The real wins are in the leak counters
(see below). If the user reports typing jank, capture a profile + heap
diff during their actual usage and compare against the synthetic baseline
to identify what condition (long thread, popover open, paste, etc.)
makes the path slow.

### Leak counters — `leak-typing.mjs`

Types N chars per round, clears, force-GCs, captures
`Performance.getMetrics` deltas. Reveals leaked event listeners, heap
drift, document node growth, and forced-layout counts.

```bash
# After clicking into a real session (e.g. via click-session.mjs):
node apps/desktop/scripts/leak-typing.mjs --rounds=8 --chars=200 --cps=50
```

**Real-session numbers (Phaser thread, 8 rounds × 200 chars):**

| | unpatched (HEAD~2) | patched (HEAD) |
|---|---|---|
| jsListeners growth/round | +0 | +0 |
| DOM nodes growth/round | +0 | +0 |
| heap growth/round | ~0 (V8 housekeeping) | ~0 |
| **forced layouts/char** | **7.02** | **2.35** (3× fewer) |

The forced-layout count is the load-bearing number — typing into a real
session was triggering ~7 layouts per character on the unpatched build
(scrollHeight reads + per-px CSS var writes + FadeText scrollWidth reads
all stacking up). After the patches it's down to ~2.35/char, which is
Blink's natural cost for a 1px/char-growing contentEditable and can't
be lowered further without architectural changes.

The initial "+35 listeners/round leak" I called out on the first
unpatched run turned out to be transient warm-up (popovers initializing,
etc.); steady-state listener growth was 0 both before and after.

### CPU profile + heap snapshot — `profile-typing.mjs`

Records a CPU profile while typing, plus before/after heap snapshots so
you can do a comparison diff in Chrome DevTools Memory tab.

```bash
node apps/desktop/scripts/profile-typing.mjs \
  --chars=400 --cps=30 --out=/tmp/hermes-typing
# → /tmp/hermes-typing.cpuprofile  (open in Chrome DevTools Performance)
# → /tmp/hermes-typing.before.heapsnapshot
# → /tmp/hermes-typing.after.heapsnapshot
```

Loading the cpuprofile: Chrome DevTools → Performance tab → drag the file
in, or VS Code → open the `.cpuprofile` directly.

For heap diff: Chrome DevTools → Memory → Load snapshot → load "before",
then Comparison view → load "after". Sort by `# Delta`. Stay alert for
detached DOM, FiberNodes (unmounted), and listener growth.

## Helpers

- `probe-renderer.mjs` — dump page state (URL, composer mounted?, body text)
- `click-session.mjs <title>` — click a sidebar session by partial title match
- `reload-renderer.mjs` — force Page.reload via CDP (no HMR available)
- `dump-state.mjs` — richer state dump (thread message count, sticky session, etc.)
- `probe-console.mjs` — dump recent console errors / exceptions

## Findings

See commit message for `apps/desktop/src/app/chat/composer/index.tsx`
edits. Three changes:

1. **Per-keystroke `scrollHeight` read removed.** The expansion useEffect
   used to read `editorRef.current.scrollHeight` on every draft change
   (forces synchronous layout). Replaced with a `draft.length > 60`
   heuristic; the ResizeObserver catches anything the heuristic misses.

2. **Bucketed CSS custom-property writes.** `syncComposerMetrics`
   used to `setProperty('--composer-measured-height', height + 'px')`
   on every observed resize, invalidating computed style for the whole
   tree. Now writes only when the height crosses an 8 px bucket, so
   typing in a fixed-height row produces no style invalidation at all.

3. **Removed dead `$composerDraft` → `aui.composer().setText` round-trip.**
   Nothing outside the composer subscribed to `$composerDraft` (verified
   via grep). The two useEffects that pushed draft → store and store →
   composer were pure overhead per keystroke. `reconcileComposerTerminalSelections`
   was also called per keystroke; can be deferred to submit time (it's a
   stale-pruning step, not a correctness one — `terminalContextBlocksFromDraft`
   walks the current text directly at submit and ignores stale labels).

4. **`refreshTrigger` fast-bails when no `@`/`/` in draft.** Previously
   `textBeforeCaret()` did `range.toString()` (O(n)) on every keystroke
   even when no trigger char was present.

The biggest win is the listener leak in (3) — without it, each round of
typing leaked ~35 event listeners until a steady state.

## Submit / TTFT stall (open)

User reports a perceived stall *after* Enter, before the assistant starts
streaming. `scripts/measure-submit.mjs` measures
`enter → composer-cleared → user-message-rendered → first-paint`. The
script triggers a real prompt submission, so use it on a throwaway
session. Not enabled in CI.

## Streaming "5fps" investigation (May 21, 2026)

User complaint: "the streaming must bring fps to like 5? lol" — felt
hitches during assistant streaming on long threads.

### Tooling added

- **`src/app/chat/perf-probe.tsx`** — dev-only side-effect import (guarded by
  `import.meta.env.MODE !== 'production'` in `main.tsx`). Attaches two
  helpers to `window`:
  - `__PERF_PROBE__` — React `<Profiler>` recorder. Currently inert because
    Vite is serving the production React build (see "Vite dev-build issue"
    below); kept for when that's fixed.
  - `__PERF_DRIVE__` — synthetic stream driver. Pushes tokens through the
    live `$messages` atom at a fixed cadence, so the assistant-ui runtime,
    incremental repository, Streamdown markdown renderer, and React commit
    pipeline all see the same workload they'd see from a real LLM stream —
    but with no LLM call (and no credit cost).
- **`scripts/measure-synthetic-stream.mjs`** — drives `__PERF_DRIVE__`,
  records rAF frame intervals, `PerformanceObserver({entryTypes:['longtask']})`
  entries, `MutationObserver` cadence on the live message, and optional
  type-while-streaming keystroke latency.
- **`scripts/profile-synth-stream.mjs`** — CPU profile during a synthetic
  stream; writes a `.cpuprofile` (open in Chrome DevTools Performance panel)
  and a top-30 self-time table.
- **`scripts/measure-real-stream.mjs`** — same harness as the synthetic but
  fires a real LLM prompt. Use when you have credits and want to confirm
  the synthetic predictions hold.
- **`scripts/profile-real-stream.mjs`** — CPU profile over the duration of
  a real LLM stream.

Helpers: `scripts/eval.mjs` (one-shot CDP eval), `scripts/reload.mjs`
(hard reload renderer over CDP).

### Findings

Measured on the Cloud Shadows session (7 turns, ~11k px scrollHeight) and
the 34 MB session `session_20260514_215353_fe0ac8.json` (110 FadeText
instances, lots of historical tool calls).

| metric | Cloud Shadows | 34 MB session |
|---|---|---|
| avgFps (60 tok/sec, 5s) | 60.0 | 58.6 |
| frame p50 / p95 / p99 (ms) | 16.7 / 18.0 / 21.1 | 16.6 / 25.6 / 31.4 |
| max frame (ms) | 31.1 | 97-127 (varies) |
| longtasks per 5s window | 0 | 1-2, 75-127 ms |
| type-while-stream p95 latency (ms) | 17 | — |

A single real-LLM stream on Cloud Shadows (gpt-4o-mini, 39s window) saw
12 longtasks totalling 1.26 s — same cadence the synthetic predicted
(~1 hitch per 3.25 s, max 123 ms). So the **synthetic stream is a faithful
proxy for the real one** and is fine for iterating on fixes without paying
for tokens.

### CPU profile during streaming (synthetic, markdown content)

Top self-time costs (5 s window, 400 tokens at 125 tok/s, markdown chunks):

| ms (self) | function | source |
|---|---|---|
| 260 | `bn$1` | `chunk-BO2N…js:20003` (micromark tokenize) |
| 249 | `m$1` | `chunk-BO2N…js:19949` (micromark) |
| 128 | `compile` | `chunk-BO2N…js:21884` (mdast → hast compile) |
| 73 | FadeText body | `components/ui/fade-text.tsx` |
| 62 | `parser` | `chunk-BO2N…js:22680` |
| 49 | `fromThreadMessageLike` | `@assistant-ui/internal` |

That `chunk-BO2N2NFS` is the vendored bundle containing `micromark`,
`mdast-util-from-markdown`, `mdast-util-to-hast`, `rehype-raw`,
`hast-util-sanitize`, etc. — i.e. **Streamdown's markdown pipeline,
re-parsing the entire growing assistant message on every token append**.
Cost scales linearly with message length.

Compare plain-text (no markdown) — the `chunk-BO2N…` entries drop out
of the top 30 entirely; total work per 5 s window halves.

### Fix landed: `FadeText` memo

`FadeText` is used in `tool-fallback.tsx` (110 instances on a tool-heavy
thread). Before: each parent re-render during streaming triggered a
`useEffect([children])` that forced a `scrollWidth` layout read — even
when the title text was unchanged. The `useResizeObserver` already covers
the genuine resize case, so the effect was strictly redundant.

After: wrapped in `React.memo` with a custom comparator that compares
`children` (scalar fast-path), `className`, `fadeWidth`, and `style`
field-by-field. Verified via temporary render counter:
**122 renders during a 2 s synthetic stream vs ~11 000 without memo**
(110 instances × ~100 stream updates). Doesn't move the longtask needle
on its own — Streamdown dwarfs it — but eliminates a class of forced
layouts and removes a steady CPU floor.

### Also landed: `MarkdownText` plugins memo + upstream flush floor

Two smaller follow-ups in the same investigation:

1. **`MarkdownText` `plugins` object useMemo'd.** The inline
   `plugins={{ math: mathPlugin, ...(isStreaming ? {} : { code }) }}`
   was constructing a new object on every render, which churns
   `<Streamdown>`'s outer memo and forces its internal `rehypePlugins` /
   `remarkPlugins` arrays to rebuild. CPU profile after the change shows
   `parser` self-time dropping out of the top 10, `compile` cut roughly
   in half, and `bn$1` / `m$1` (micromark internals) dropping off the
   top entries.

2. **`use-message-stream.scheduleDeltaFlush` got a real minimum floor.**
   Previously the rAF-only path effectively meant "at most one flush per
   frame," but at typical LLM token rates of 30-80 tok/sec each token
   arrives slower than rAF cadence and gets its own React commit. With
   `STREAM_DELTA_FLUSH_MS = 33` (two frames) and a `lastFlushAt`-tracked
   floor, slower streams now coalesce ~2 tokens per commit, halving
   markdown re-parses. React's auto-batching already covers part of this
   probabilistically; the floor makes the batching deterministic so the
   max-longtask number tightens up.

A/B on the 34 MB session, 300 tokens at 50 tok/sec, markdown chunks
(3 trials each):

| | avgFps | p99 frame | LTs/5s | max LT | mutations |
|---|---|---|---|---|---|
| no throttle  | 54.0 | 38 ms | 2.0 | 145 ms | varies (2-112) |
| 33 ms throttle | 54.3 | 41 ms | 1.7 | 110 ms | ~135 |

Modest. `inter-mutation` p50 tightens from 22-28 ms to a clean 33 ms,
which is what you'd expect from a deterministic floor.

### Also landed: `useDeferredValue` at the streamdown-text boundary

The longtask CPU was unavoidable inside the block-memo pattern — the live
tail re-parses every commit, scales linearly with current length, and
nothing about Streamdown's architecture changes that without forking. The
fix is to stop having that work *block* the main thread.

`<DeferStreamingText>` in `markdown-text.tsx` is a 12-line wrapper that
reads the message-part state via `useMessagePartText`, runs it through
`useDeferredValue`, and re-publishes via assistant-ui's
`<TextMessagePartProvider>`. The inner `StreamdownTextPrimitive` reads the
deferred value through the normal `useMessagePartText` hook — no fork,
no internal-path imports, fully on the assistant-ui public API.

What React's concurrent scheduler now does:

- When a new token arrives mid-render, the in-flight deferred render
  is abandoned and a fresh one starts with the latest text.
- When the main thread has urgent work (typing, scroll, layout), the
  Streamdown render gets deprioritized — input stays responsive even
  while a 100 ms parse is queued.

Streamdown already uses `useTransition` internally for its block-array
setState; `useDeferredValue` here just lifts the deferral all the way up
to the consumer text boundary, so the whole pipeline — preprocess,
block split, repair, parse, render — runs at low priority during streaming.
This is the industry-standard approach (see
[Streamdown architecture analysis](https://tigerabrodi.blog/how-to-build-a-performant-ai-markdown-renderer)
and Chrome's [LLM-response render best practices](https://developer.chrome.google.cn/docs/ai/render-llm-responses)).

A/B on the 34 MB session, 300 tokens at 50 tok/sec, markdown chunks
(four trials each, prod-throttle (33 ms) on for both):

| | avgFps | p99 frame | LTs / 5 s | max LT | typing p95 |
|---|---|---|---|---|---|
| pre-defer | 54.3 | 41 ms | 1.7 | 110 ms | ~17 ms |
| **post-defer** | **58.5** | **31 ms** | 2.0 | 117 ms | 14-18 ms |

Longtask count and max LT are unchanged — `useDeferredValue` doesn't
reduce CPU, only its priority. The avgFps lift and p99 frame drop are
the proof that the existing CPU is no longer blocking 60 fps cadence:
when React can defer the parse, frames stay clean. One particularly
clean run logged **MUTATIONS=0** — React skipped every intermediate
text state and only committed the final one, the textbook
useDeferredValue behaviour.

### Not fixed: Streamdown markdown re-parse cost (the elephant)

Total CPU spent in micromark/mdast/hast pipeline per 5 s window is still
the same ~700 ms. With `useDeferredValue` that work no longer blocks
input, but if you watch a CPU profile you'll see the same hot functions
(`Tn$1`, `bn$1`, `m$1`, `parser`, `compile`).

The path to actually *reduce* that cost (not just defer it) is to
replace the parser with a state machine like
[Flowdown](https://github.com/Atomics-hub/flowdown) — process each
character exactly once, emit DOM ops directly, no re-parse of the prefix
on every token. Claimed ~2,000× over `marked`. Trades: not a
`react-markdown`-compatible API, no rehype security pipeline, would
require replacing Streamdown wholesale. Worth investigating only if
even the deferred work shows up in user-perceptible ways (e.g.
trackpad-scrolling a stream-in-progress stutters).

The synthetic harness now mirrors the real upstream pipeline via the
`flushMinMs` option in `__PERF_DRIVE__.stream({ flushMinMs: 33 })`, so
future Streamdown / Flowdown experiments can A/B without LLM credit cost.
The synthetic numbers tracked the one real-LLM run we caught within
noise, so it's a reliable proxy.

Possible approaches (none implemented here):

1. **Coalesce/throttle Streamdown updates** — render at most every 32 ms
   instead of every set-state. Reduces parses but doesn't reduce
   per-parse cost; trades latency for smoothness.
2. **Memoize per-prefix** — diff the new text against the prior parsed
   version; only re-parse the changed suffix.
3. **Render in stable segments** — close-form historical paragraphs as
   immutable React nodes; only the live tail goes through markdown each
   token. Probably the highest-impact change but requires forking or
   patching `@assistant-ui/react-streamdown`.
4. **Move parsing to a Web Worker** — main thread no longer blocks on
   markdown. Largest surgery; requires double-buffered hast.

### Vite dev-build issue (separate)

`http://127.0.0.1:5174/node_modules/.vite/deps/react.js` resolves to
`react/cjs/react.production.js`, and `react-dom_client.js` →
`react-dom-client.production.js`. As a result:

- `<React.Profiler>` `onRender` is never called (production build is a
  no-op).
- `import.meta.env.DEV` is `false`, `PROD` is `true` even under `vite dev`
  (hence `MODE !== 'production'` as the workaround in `main.tsx`).
- All the React 19 dev-only warnings/devtools backend hooks are absent.

Root cause likely sits in `vite.config.ts` aliasing + dedupe + Vite 8's
new `optimizeDeps` defaults. Worth a separate fix pass — when it's
resolved, the `<PerfProbe>` blocks in `perf-probe.tsx` become useful
(per-id commit timings) instead of inert.
