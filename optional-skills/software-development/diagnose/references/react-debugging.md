# React Debugging & Log Playbook

Load when the bug lives in a React app. Feeds Phase 1 (repro loops) and Phase 4 (probes) of `diagnose`.

## Feedback loops (Phase 1)

| Goal | Command / move |
| --- | --- |
| Reproduce via component test | `npx vitest run the-file` or `npx jest the-file` — Testing Library `screen.debug()` dumps the DOM at the failure point |
| Loop a flaky test | `for i in {1..50}; do npx vitest run the-file || break; done` |
| Reproduce the real UI | dev server + Playwright script driving the failing flow, asserting DOM/console/network |
| Capture failing request | DevTools Network → *Copy as cURL* / *Save all as HAR* → replay and redact |
| Production-accurate profile | `npm run build` + serve the bundle — dev mode is multiple times slower and adds dev-only checks; keep sourcemaps on for readable stacks |

## React DevTools (primary tool)

Browser extension → two tabs:

- **Components** — inspect the tree, view props/state/hooks, **edit** props and state live (don't mutate via the JS debugger — React state must go through setters). Settings → *Highlight updates when components render* paints every re-render — flickering regions are suspects.
- **Profiler** — record during the slow interaction. Top bars = commits; click one to see the flamegraph:
  - gray bar = component did *not* render this commit
  - green/teal = fast, yellow/orange = slow — your targets
  - Settings → *Record why each component rendered* → hover a component for "Why did this render?" (props changed / state changed / parent rendered / hooks changed)

## Re-render and effect bugs

- **`why-did-you-render`** (dev only): `npm i -D @welldone-software/why-did-you-render`, patch entry point, then `Component.whyDidYouRender = true` — console reports *why* each render happened (prop identity changes you can't see in DevTools).
- Check, in order: inline object/array/function props breaking memoization → context value not memoized → `useEffect` dep array missing/over-broad → state lifted too high (cascade renders).
- **Stale closures**: effect/callback captures old state → fix deps or use functional updates / `useRef`.
- **`Maximum update depth exceeded`**: setState inside render or an effect that always re-triggers itself — find the loop via DevTools Profiler (one component rendering every commit).
- **StrictMode**: double-invoked render/effects in dev surface impure side effects — if removing StrictMode "fixes" it, the effect is the bug.

## Browser DevTools

- Breakpoints on real source via sourcemaps; **conditional breakpoints** (`count > 3`) and **logpoints** (no code edits) over `console.log` scatter.
- `debugger;` statement when the tool UI is awkward.
- Performance panel: long tasks, layout thrashing — a slow React commit can be caused by DOM size or effects, not React itself; corroborate before optimizing.
- `performance.mark()`/`performance.measure()` around the suspect interaction → survives in production telemetry.

## Error capture

- Error boundary at the right seam → logs `error` + `componentStack` (the *component* stack you lose otherwise).
- Production: upload sourcemaps to your error tracker (Sentry etc.) so minified stacks decode — `hidden-source-map` keeps maps off the public bundle.

## Instrumentation (Phase 4)

```ts
console.debug('[DEBUG-a4f2]', { props, state });              // tagged probe — grep to remove
useEffect(() => { console.debug('[DEBUG-a4f2] effect ran', deps); });
```

## Common failure patterns

| Symptom | Likely cause | Check |
| --- | --- | --- |
| Sluggish interaction, fast network | wasted re-renders or expensive render | Profiler → wide bars + "Why did this render?" |
| Render flicker/flash | unstable keys, remount on each render | Components tab — does the instance persist? |
| `Maximum update depth exceeded` | setState in render / self-triggering effect | Profiler — same component in every commit |
| Stale data after action | stale closure in effect/handler | deps array; functional `setState` |
| SSR hydration mismatch | non-deterministic render (Date.now, random, window access) | server vs client HTML diff; `suppressHydrationWarning` is a band-aid |
| Memory climbs | leaked subscriptions/timers in effects | effect cleanup return; DevTools Memory snapshot diff |

## Log queries

```bash
# browser console — "Preserve log" on; filter: [DEBUG-a4f2]
# correlate with backend: send X-Request-Id/traceparent on fetch, grep server logs
grep '<requestId>' backend.log
```

## Docs

- https://react.dev/learn/react-developer-tools
- https://github.com/welldone-software/why-did-you-render
