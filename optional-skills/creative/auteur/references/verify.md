# verify.md — the acceptance pipeline

A page that "looks done" in the editor is exactly what every AI ships. Auteur's output is done when it survives three independent checks: a deterministic linter (catches slop defaults in code), a screenshot journey (catches what only eyes catch), and a numeric rubric (catches what eyes forgive). Run them in this order — each is cheaper than the next.

## 1. slopscan (deterministic)

```bash
node <skill-dir>/scripts/slopscan.mjs <src-dir>
```

- Exit 1 → fix every FAIL and re-run. Fix means *redesign the element*, not rename the class.
- **Paste the linter's final `Summary:` line verbatim into your report and QA sheet** — after your LAST edit, not from an earlier run. A remembered result is not a result; re-run the command.
- A FAIL that is a genuinely deliberate, argued choice → suppress in-file with
  `/* auteur-allow: RULE_ID -- <real reason, min 10 chars> */` — the reason must reference the commit-sheet ("committed glass nav over video hero, see COMMIT-SHEET §2"). Suppressions without real reasons are themselves reported.
- WARNs: read each one; fix or consciously accept. >3 accepted warns on one page usually means the design is drifting toward the mode — reread taste.md §8.

## 2. Screenshot journey (eyes-on, mandatory)

```bash
node <skill-dir>/scripts/shoot.mjs <url> --stops 7 --breakpoints 390,768,1440 --reduced-motion
```

> **Serve it. Do not verify over `file://`.** `fetch` to a `file:` URL is blocked, so any page that
> loads a glTF, an HDRI, a JSON or a sequence manifest will silently fall back to its poster — and
> `shoot.mjs` will happily photograph a perfectly attractive page in which the entire peak never
> booted. A false PASS is worse than a FAIL, and this one is invisible unless you already know what
> the peak looks like. Any static server does; 40 lines of `node:http` is enough.

Then **open and look at every frame**. You are looking for what linters cannot see:

- text overflowing/wrapping ugly at any breakpoint (the viewport is part of the design)
- blank or half-fired scenes (reveal gated on an animation that never ran)
- scenes where the scroll-stop caught the page mid-jank
- contrast that "passes" numerically but reads muddy on the actual background
- two adjacent frames that look like the same layout family (storyboard said they must differ)
- the reduced-motion journey: is it a *watchable cut*, or a broken silent ruin?

Console errors and "possibly blank" warnings printed by shoot.mjs are FAILs until explained.

> **`--full` lies about `position: fixed` and stuck `position: sticky`.** A full-page capture
> composites them at their viewport position, so a fixed bottom nav appears in the middle of the page
> and a sticky table head appears to overlap row 1. Both look exactly like layout bugs and neither is.
> Judge fixed and sticky elements from the viewport frames only.

**Interaction smoke-test** (when the page has interactive elements — nav, forms, tabs, mute toggle): drive the real page with playwright (`playwright-cli` skill, or a short script on the same playwright install shoot.mjs uses): click every nav link (lands on the right anchor, header doesn't cover the target), open/close the mobile menu, submit the form empty (inline error appears, nothing explodes), toggle sound if present, tab through the page once (focus visible and in order, ESC closes overlays). Any interaction that throws or dead-ends is a FAIL. Static pages skip this.

## 2.5 Motion / perf / audio QA (direct register — screenshots are blind to time)

The screenshot journey (§2) proves the page looks right FROZEN; it says nothing about jank, dropped frames,
audio drift, or a WebGL context leaking on route change. For any Tier-1 scene (scroll state-machine,
audio-reactive, 2.5D composite, scrubbed video) run a MOTION pass with playwright and assert real numbers:

> **A perf number is only as honest as the three settings behind it, and they all default to flattering.**
> **Pixels:** `motionqa.mjs` renders 1440×900 **@2x = 5.2MP**, because fullscreen passes cost per pixel
> and the reviewer's laptop is retina. At DPR 1 the same scene is 1.3MP, the expensive part of the frame
> costs a quarter as much, and the gate certifies 60fps on a page that stutters on a MacBook. Expect the
> two numbers to be identical on a scene with **no** fullscreen pass (measured: 53→54, 53→53, 54→54 on
> three showcase sites) — that is the honest result, not a broken gate; DPR only bites where post-processing does.
> **GPU:** headless chromium has none, and its software raster is not a small error — the same three sites
> measured 6 / 20 / 20 fps headless against 53 / 53 / 54 headed. A headless number is a floor, never a verdict.
> **Build:** measure the **production build**, never the dev server — HMR clients and unminified bundles
> cost roughly 2× per frame. That error runs the other way (false FAILs), and a false FAIL is how a scene
> gets amputated for nothing; motionqa flags the dev servers it can recognize, but not all of them.
> Quote both in the report: "minFps 57 @4× CPU, 1440×900@2x (5.2MP), production build".

```js
// scripts/motionqa.mjs — record while scrolling the whole page, at a mid-laptop CPU tier
const cdp = await page.context().newCDPSession(page)
await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 })   // jank hides at full speed
await page.evaluate(() => { window.__lt = 0; new PerformanceObserver(l => {
  for (const e of l.getEntries()) window.__lt = Math.max(window.__lt, e.duration)
}).observe({ type:'longtask', buffered:true }) })
const minFps = await page.evaluate(() => new Promise(res => {
  let last = performance.now(), min = 999, end = last + 4000
  ;(function tick(t){ const d = t - last; last = t; if (d>0) min = Math.min(min, 1000/d)
    scrollBy(0, innerHeight/60); t < end ? requestAnimationFrame(tick) : res(min) })(performance.now())
}))
const longTask = await page.evaluate(() => window.__lt)
```

Assert (fail = fix, don't ship):
- **minFps ≥ 50** on the scrub at 4× CPU throttle — below that the scroll reads as a slideshow.
- **No long task > 50ms** during the scroll (one 200ms task IS the jank users feel).
- **WebGL context count flat across route changes** — no `Too many active WebGL contexts` in console (the #1 reported leak); swap textures, never remount.
- **prefers-reduced-motion journey renders a valid static alternative** — the §2 `--reduced-motion` frames must still tell the story (no blank canvas, no missing hero), not just "motion off".
- **Audio (if reactive):** OFF until a user gesture (`audio.paused` true on load), a visible mute control exists, and the visual is complete muted (the scroll pass above already ran silent).
- **Poster / first frame paints within LCP budget** before textures decode — no blank hero.

Console over the whole run: zero errors, zero `THREE.WebGLRenderer: Context Lost`. Report the numbers:
"motionqa: minFps 57 @4× throttle · max long-task 34ms · reduced-motion journey clean · audio gesture-gated".

## 3. Numeric rubric

| Check | Threshold | How |
|---|---|---|
| Body contrast | ≥4.5:1 (large ≥3:1, placeholders ≥4.5:1), **measured in every state, not just the default** — an error or stale view that dims its own text is the usual way this fails, and every linter reads the undimmed colour | devtools / axe on final colors, then again with each degraded state applied |
| LCP | <2.5s (throttled Fast 3G / 4× CPU) | Lighthouse |
| CLS | <0.1 | Lighthouse |
| INP | <200ms | Lighthouse / manual scroll+click |
| Hero video | ≤2MB (poster ≤300KB) | file sizes on disk |
| Sequence frames | ≤150KB each @1440w | file sizes |
| Motion budget | ≤3 scroll-pattern families; uniform-reveal check | count them honestly |
| Background lightness | within ±0.12 of the L committed in commit-sheet §2 | `chromadiff.mjs <frame> --target-l <L>` |
| House tells | ≥2 broken and named in commit-sheet §7; the built page actually breaks them | taste.md §2.5 list vs the hero frame |
| One peak | exactly one scene intensity ≥8 | storyboard vs built page |
| Adjacent-scene variety | no two neighbors share layout family | screenshot journey |
| Reduced-motion | full journey watchable, nothing blank | shoot.mjs --reduced-motion frames |
| Motion (Tier-1 scenes) | minFps ≥50 @4× throttle **at DPR 2, on the production build** · long-task ≤50ms · no WebGL leak · audio gesture-gated | scripts/motionqa.mjs (§2.5) |
| Keyboard | tab order sane, focus visible, no traps, ESC closes overlays | manual pass |
| No-JS | content readable, page navigable | disable JS, reload |
| Fallback payload | every degraded cut still carries the peak's *information*, not just a picture of it | reduced-motion / no-JS / no-WebGL, at 390 too — a callout panel hidden by a mobile breakpoint deletes the payload while the desktop screenshots look fine |

## 3.5 systemscan (system register — one page cannot show you drift)

A product fails differently from a page: every screen looks fine alone while the fourth button
variant quietly appears on screen seven. Run the cross-route gate over **every** route, not a sample:

```bash
node scripts/systemscan.mjs http://localhost:3000 --routes /,/settings,/billing,/team
```

It reads what the browser actually painted, fails a control type over its declared variant budget,
presses Tab to catch any control that paints identically focused and unfocused, and writes
`components.png` — one tile per rendered variant. Look at that sheet the way the direct register
makes you watch your own film: a tile that looks foreign is drift your eyes caught before the
numbers did. Full doctrine in `system.md`.

## 3.7 Reference diff (the last look, before you decide you're done)

Recon (`design/refs/`) decided the direction in phase 0 and then went quiet for the whole build. Bring it back for one pass: put your own screenshot beside the reference that set the bar and ask what the reference does better. This catches the class of failure every other gate is blind to — the page is fast, clean, linted, and quietly a friendlier, cheaper version of what you set out to make. Drift toward the pleasant middle is not visible from the inside; it is obvious side by side.

- Compare the same frame at the same width, and **also greyscaled** — value structure and figure/ground survive the conversion, seductive colour does not.
- Ask for the differences ranked **from cheap-and-decisive to expensive-and-marginal**, in exactly those terms. Unranked, the answer is twenty equally-weighted bullet points and you cannot tell which one is the page. Ranked, the top two are usually worth more than the rest combined, and the tail is honestly ignorable.
- Fix the top of the list, re-shoot, look again. Two rounds is normally the end of it.
- Chroma is the specific thing to check against a desaturated reference (`taste.md` §3): the model drifts toward a bluer, happier, more saturated frame even with the reference in view. That one is measurable, so measure it instead of arguing about it:

```bash
node scripts/chromadiff.mjs shots/bp1440-stop00.png design/refs/<the-reference>.png
# chromadiff: yours C 0.042 L 0.873 - reference C 0.004 L 0.874 - delta +0.039  → FAIL
```

It fails only upward: quieter than the reference is a choice, louder than the reference is the reflex. `--max-delta` moves the line when the brief genuinely calls for more colour than its reference — say so in the commit-sheet when you move it.

Then the same check for lightness, against your own sheet rather than a reference:

```bash
node scripts/shoot.mjs <url> --stops 5 --breakpoints 1440 --full --out shots
node scripts/chromadiff.mjs shots/bp1440-full.png --target-l 0.55
# chromadiff: L 0.177 (committed 0.55) - C 0.013 - 98% dark pixels  → FAIL, and that number is the tell
```

**Measure the full-page capture, not a viewport frame.** The sheet commits to the page's *mean* lightness and the first screen is never the mean — a page whose dark fields live below the fold measured 0.876 at the hero and 0.740 over the whole page, one false FAIL from the crop alone. Same caveat as `--full` in §2: use it for this measurement, judge fixed/sticky elements from the viewport frames.

**And re-read `taste.md` §2.5 with the hero frame in front of you.** Commit-sheet §7 named two house tells to break; confirm the built page actually broke them, because they creep back in during assembly — the corner labels return as "just a caption", the status bar as "just a nav". A page that quietly rebuilt all seven tells is on style, and being on *this skill's* style is the failure the whole file is about.

The output of this pass is one line in the report: what the reference did better, what you changed, what you consciously left.

## 4. Sign-off

- **direct register:** copy `templates/CINEMA-QA.md` into the project, fill every row with PASS/FAIL + evidence (metric value or screenshot filename). All-PASS ships; any FAIL loops back to its phase.
- **build register:** the rubric table above, inline in your final report.
- **system register:** the rubric table, plus systemscan's verdict quoted verbatim (variant counts per control, focus failures, one-off variants) and confirmation that you looked at `components.png`.
- Report honestly and concretely: "slopscan 0 fails / 2 accepted warns (reasons logged); 21+6 screenshots reviewed — fixed S4 headline overflow at 390; LCP 1.8s; CLS 0.02; reduced-motion cut verified." If something is unverified (e.g. no local server to measure LCP), say so explicitly rather than implying a pass.
- Optional second opinion: an outside UI critique (upstream used a separate 'impeccable' skill, not vendored; in Hermes, `vision_analyze` on the shoot.mjs screenshots works) — it measures UX heuristics auteur doesn't; disagreement between the two is signal, not noise.

## When verification keeps failing

Two full loops on the same gate → stop patching symptoms. The failure is upstream: a scene too ambitious for its asset (descend the ladder in assets.md), a palette that never worked (redo commit-sheet §2), a motion budget blown by scope creep (cut a pattern family). Fixing upstream is cheaper than a third loop downstream.

**When you cannot tell what a setting even controls, take it to an absurd value.** Set the blur to 200, the displacement to 10, the duration to 5s — and look. If nothing changes, the knob is not connected to what you think it is, and you have found the real bug instead of tuning a decoy for an hour. This is how a focus that is permanently locked to one object gets discovered: crank the blur and watch the foreground stay soft no matter what the number says.

**And when a fix keeps not working, consider deleting the element instead.** An idea that needs three rounds of rescue is usually just weak — light rays built from cones, a transition that reads as signal loss. Polishing your own bad idea burns tokens and hours on something that comes out anyway; cutting it is a decision, not a defeat.
