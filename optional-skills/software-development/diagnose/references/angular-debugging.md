# Angular Debugging & Log Playbook

Load when the bug lives in an Angular app. Most techniques feed Phase 1 (feedback loop) and Phase 4 (instrumentation) of `diagnose`.

## Feedback loops (Phase 1)

| Goal | Command / move |
| --- | --- |
| Reproduce via unit test | `ng test --include='**/the-file.spec.ts'` (Karma) or `npx vitest run the-file` |
| Loop a flaky test | `for i in {1..50}; do npx vitest run the-file || break; done` |
| Reproduce the real UI | `ng serve` + Playwright/Cypress script that drives the failing flow and asserts DOM/console |
| Capture the failing API call | Chrome DevTools → Network → right-click request → *Copy as cURL* / *Save all as HAR* → replay/redact |
| Dev server with proxy to staging API | `ng serve --proxy-config proxy.conf.json` |
| Production-accurate repro locally | `ng build --configuration production --source-map` + serve `dist/` — dev mode exaggerates CD cost |

## Angular DevTools (primary tool)

Browser extension (Chrome/Edge/Firefox), requires Angular ≥12 running in **development mode** (`optimization:false`). Two tabs:

- **Components** — inspect the component/directive tree, view and **edit** inputs/state live. Use `$0` (the selected DOM node) as the entry point.
- **Profiler** — record while performing the slow interaction. Each bar = one change-detection cycle; taller = longer. Click a bar to see per-component times, the lifecycle hooks that ran, and **what triggered the cycle**. Export the profile and attach to the bug report.

## Console inspection APIs (dev mode)

With an element selected in DevTools (`$0`):

```js
ng.getComponent($0)          // component instance → read/mutate state
ng.getDirectives($0)         // directives on the node
ng.getInjector($0)           // resolve services: ng.getInjector($0).get(MyService)
ng.getHostElement($0)        // host element of a component/directive
ng.applyChanges($0)          // mark dirty + run CD after manual edits
ng.getDependencies($0)       // what this node injects
```

## Change-detection / performance profiling

- `ng.profiler.timeChangeDetection()` (console) → prints average CD ms per cycle. Baseline first, change second.
- **Angular track inside Chrome DevTools Performance panel**: run `ng.enableProfiling()` in the console (or call `enableProfiling()` from `@angular/core` at startup), then record in the Performance tab — get per-component CD timings correlated with browser frames. Dev mode only.
- Fixes to *verify*, not apply blindly: `OnPush` strategy, signals, `ChangeDetectorRef.detach()`, moving heavy work out of templates/getters.

## Instrumentation (Phase 4)

- **Tagged probe in templates/code**: `console.debug('[DEBUG-a4f2]', value)` — strip with one grep.
- **RxJS tap** at the suspect seam:
  ```ts
  obs$.pipe(tap({ next: v => console.debug('[DEBUG-a4f2] next', v),
                  error: e => console.debug('[DEBUG-a4f2] error', e),
                  complete: () => console.debug('[DEBUG-a4f2] complete') }))
  ```
- **HTTP boundary**: a temporary `HttpInterceptor` logging method+url+duration+status gives you the request/response ground truth.
- **Global errors**: custom `ErrorHandler` that logs `error` + component stack to console/backend before rethrowing.

## Common failure patterns

| Symptom | Likely cause | Check |
| --- | --- | --- |
| `ExpressionChangedAfterItHasBeenCheckedError` | state mutated during CD (dev-mode-only check) | move mutation to `afterNextRender`, signal update, or next tick; find which binding flips |
| UI slow / janky | CD running too often or too deep | DevTools Profiler → tallest bars; `ng.profiler.timeChangeDetection()` |
| Stale data on screen | `OnPush` + mutated (not replaced) input | new object/array reference, or signals |
| Duplicate API calls | `async` pipe on a cold observable twice, no `shareReplay` | count requests in Network; interceptor log |
| Memory grows over session | unsubscribed streams | `takeUntilDestroyed()`, DevTools Memory heap snapshots diff |
| Works locally, fails behind prod proxy | base href, CSP, missing headers | compare Network request/response vs local; HAR diff |

## Log queries

```bash
# browser console (persistent across reloads: enable "Preserve log")
# filter box:  [DEBUG-a4f2]        level: Verbose for console.debug

# end-to-end correlation: send the backend trace id on every request
# (interceptor → X-Request-Id) then grep backend logs by that id
grep '<requestId>' backend.log
```

## Docs

- https://angular.dev/best-practices/profiling-with-chrome-devtools
- https://angular.dev/tools/devtools
