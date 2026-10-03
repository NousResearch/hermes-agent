# Runtime integration: host owns truth, Rive owns illustration

Authoritative sources: https://rive.app/docs/runtimes/web/rive-parameters.md, https://rive.app/docs/runtimes/web/data-binding.md, https://rive.app/docs/runtimes/web/canvas-vs-webgl.md, https://rive.app/docs/feature-support.md. Recheck the exact version/target on adoption.

## Pick the smallest adequate renderer

For new web work, Rive currently recommends `@rive-app/webgl2`. It uses the Rive Renderer and supports vector feathering. `@rive-app/canvas` uses Canvas2D; it can be preferable for some blend-heavy content and does not have WebGL context limits. Preserve an existing proven integration unless changing it closes a real gap. Do not switch merely to make an installation list look more advanced.

- Keep runtime JS and WASM on matching exact package versions; lock dependencies. Inspect the selected package's actual types and release notes rather than copying a historical version pin.
- Canvas and WebGL2 share high-level APIs but differ in content fidelity/performance. Clockwise fill rules, overlapping paths and clipping can differ in Canvas; compare real output, not only API signatures.
- WebGL2 non-Normal blends may be expensive on its web MSAA path. Test representative assets on real target devices before asserting performance.
- Ordinary WebGL2 visuals can use `useOffscreenRenderer: true` to share a context; this does not eliminate all memory/draw cost or prove unlimited instances.
- Avoid deprecated `@rive-app/webgl` (no updates after 2.37.0 per docs). Canvas-lite removes text/layout/audio/scripting; use only when the actual asset does not require them. Canvas-single bundles WASM into JS at a payload trade-off.
- Check feature matrix cells before using scripting, semantics, fonts, components/lists or GPU Canvas. A shader authoring API is not a target-runtime availability test.

### Experimental Web GPU Canvas

Current official guidance supports GPU Canvas in `@rive-app/webgl2` 2.42.0+
(and React WebGL2 4.34.0+) with `enableGPUCanvas: true`. Canvas2D does not support
it. Keep `useOffscreenRenderer: false`: each instance needs its own WebGL context.
A GPU-enabled `RiveFile` cannot be shared across instances; additional instances
may load fresh copies, so bindings on the first file do not configure the others.
The feature is experimental and may change in a minor release. Export the
required shader targets and verify on the actual device. Sources:
https://rive.app/docs/runtimes/web/gpu-canvas.md and
https://rive.app/docs/runtimes/react/gpu-canvas.md.

## Asset loading and provenance

Record source URL/owner/license, editor/source revision, actual artboard/state-machine/ViewModel names, runtime version, required features and file dependencies. A public `.riv` download is not blanket permission for reuse. Preserve embedded font/image/audio notices; keep licensed fixture art visibly distinct from original artwork.

Use a task-local asset directory, not the entire private project as docroot. Validate approved same-origin paths or a bounded explicit URL allowlist. Fetching a document may fetch referenced assets; hosted-font/image/audio behavior requires separate policy.

For offline/private work set both:
- `enableRiveAssetCDN: false` on instances;
- local `RuntimeLoader.setWasmUrl(...)` and `RuntimeLoader.setWasmFallbackUrl(null)`.

Those are independent network channels. A failed local WASM load must not silently fetch a CDN fallback. Verify with request interception/offline tests. Do not embed private source or secrets into the bundle. Strict CSP, CORS and WASM MIME compatibility are deployment-specific checks, not universal defaults.

## Typed data-binding procedure

1. Construct the instance with explicit `artboard`, singular `stateMachine` for API 2.41.0+, canvas, load/error handlers and deliberate autoplay policy.
2. After actual load, inspect `contents`, model definitions/properties and available instances. `contents` is not the full editor hierarchy. Check exact names and types against the asset contract before driving them.
3. Either use the asset's intended default with `autoBind: true` and `viewModelInstance`, or choose a definition/instance and explicitly bind it. Do not silently mix manual/automatic instance ownership.
4. Resolve properties with their actual type (`string`, `number`, `boolean`, `color`, `enum`, trigger or nested models/lists). A null property is an asset-contract error, not optional success. Validate finite numbers, ranges, strings/enums and domain state before writes; document clamp versus reject.
5. Update the host's semantic output immediately from validated state, then drive Rive. State machine/artboard evaluation applies values to bound targets and dispatches observers. Read back actual properties and rendered consequences, not only a JS mirror object.
6. Handle nested paths and independent component instances deliberately. Global/reused instances can unintentionally couple independent visuals. Test isolation for multiple instances.
7. Remove every installed observer/listener at cleanup. Do not use an observer echo to recursively write the same value forever. Distinguish host-originated changes from asset-originated signals.

Explicit binding shape from the documented API:

```javascript
const model = r.viewModelByName(contract.viewModel);
const instance = model?.instanceByName(contract.instance);
if (!instance) throw new Error('Missing expected Rive instance');
r.bindViewModelInstance(instance);
```

Use the actual fixture/asset schema, not these placeholder contract names. `bindViewModelInstance` assigns the instance; it does not certify that bound art has evaluated/rendered. New APIs stage `setViewModelInstance`/global instances and flush with one `bind()` after assignments; avoid repeated expensive rebinding. Check the adopted version's actual types. From 2.41.0, plural `stateMachines`, direct animation playback, `onLoop`, and `onStateChange` are deprecated; preserve functioning legacy assets unless migration is needed.

## Host state and lifecycle

Keep these independent:
- **Desired semantic state:** last validated input even while loading or unavailable.
- **User intent:** explicit animation enable/pause; never overwritten by visibility restoration.
- **Motion preference:** initial and live `prefers-reduced-motion` policy. Rive does not apply this automatically. Author a boolean such as `prefersReducedMotion`, bind it to reduced/static behavior, and pass initial/live host preference changes into the file. See https://rive.app/docs/editor/accessibility/reduced-motion.md.
- **Visibility:** tab hidden, offscreen, suspended history entry.
- **Resource lifetime:** loading generation, live instance, disposed instance.

Recommended behavior:
- Render a useful native/static state before JS. Motion is optional; learning interfaces should start still unless the brief requires otherwise.
- Load asynchronously with an AbortController/generation guard. Rapid pre-load inputs resolve to the latest desired state; stale completions cannot recreate a disposed instance.
- Direct state changes remain responsive. A bounded evaluation can update a static Rive pose, then stop; do not assume one/two frames settles any arbitrary graph. Test a composed terminal-state contract or keep the native still until it does.
- `stopRendering()` suspends scheduling without necessarily changing playback state. Keep it distinct from user pause and `cleanup()`.
- Stop owned loops when hidden/offscreen. Resume only if user intent and motion policy permit. Don't restart sound automatically.
- `pagehide.persisted` suspends; terminal pagehide/unmount destroys. On `pageshow.persisted`, restore without duplicate bindings. Test actual bfcache eligibility/navigation separately from synthetic events or fresh reloads.
- Destroy idempotently; clean runtime, observers, event handlers, resize/intersection listeners and pending callbacks. A JS object becoming unreachable does not prove native/WASM/GPU resources were freed.

## Layout, typography and accessibility

Use `ResizeObserver`, resize the drawing surface and cap DPR according to measured quality/performance. `Fit.Contain` is not layout reflow. `Fit.Layout` requires layout-authored content and a version supporting it; test meaningful size changes, not just canvas aspect ratio.

CSS fonts do not automatically enter Rive. Use supported authored/runtime font assets and verify actual glyphs/weights, complex text, localization and clipping. Font licensing remains independent from runtime licensing. A successful FontFace load is not proof the intended glyphs were rendered.

Keep essential copy, labels, links, buttons, form state, focus and error messages in semantic HTML. For decorative/supporting canvas use an appropriate static description or `aria-hidden` without removing the equivalent content. Canvas-only interaction must not become the only route to an action.

Rive's current Web semantics API is **opt-in and experimental**. It requires authored semantic nodes and builds a sibling overlay. `SemanticMode.Disabled` is the default; `Enabled` can expose authored roles/values. API behavior may change without a major-version bump. Current interactive overlay elements use `tabindex=-1`; a screen reader's cursor access does not supply ordinary sighted-keyboard Tab traversal. Retain native controls and avoid duplicate announcements. Source: https://rive.app/docs/runtimes/web/semantics.md.

The current API also exposes canvas `tabIndex` and `focusOptions`; these do not automatically make every control keyboard-accessible. Check the feature matrix: Web Focus is listed from 2.43.1, while Text Input remains target-dependent.

Test actual overlay DOM and appropriate screen reader on the actual target before claiming semantic accessibility. Axe and automated role checks are useful but not certification. Use no audio autoplay; an API event is not a listening test.

## Representative proof suite

- Actual asset/WASM loads with no unexpected external requests; decoded canvas pixels are meaningful.
- Valid start/interior/end/reverse state has typed readback and the right visible geometry/color/text.
- New context/size/copy beyond the initial tutorial works; source and target remain correctly matched.
- Invalid property/type/enum, NaN/infinite/range inputs, corrupt/missing `.riv` and missing WASM fail visibly and recover only through the documented path.
- Static/no-JS and reduced states preserve meaning; changing reduction while moving settles correctly.
- Rapid load/replace/cancel/unmount does not apply stale state or leave loops/subscriptions.
- Hidden/offscreen pause, explicit user pause and restoration remain separate; actual history cache behavior is measured honestly.
- Desktop/700px/390px screenshots, actual glyph use, keyboard/focus, disclosure state and overflow pass. Device emulation is labeled as such.
- Source `.rev` and exported `.riv` are reopened/reloaded if promised. New runtime features cannot be inferred from 'file loaded'.
- Clean-input install/build/test succeeds; report exact environment, artifacts and limits. Pixel inequality alone is weaker than an expected-pose/geometry oracle.

## Native/game routes: optional, separately validated

Consult current platform docs only when a concrete project needs them: Apple, Android/Compose, Flutter, React Native/Nitro, Unity, Unreal, Defold, C++. APIs, release channels, target minimums and feature parity differ; web evidence transfers design intent, not platform readiness. See https://rive.app/docs/feature-support.md and the official documentation index.

On the audited date, several native APIs were new/beta/dev or had platform-specific semantics/GPU gaps. Prefer a project-specific version decision and real native build to copying a stale version matrix into every brief. Adding all toolchains in advance is not expertise.
