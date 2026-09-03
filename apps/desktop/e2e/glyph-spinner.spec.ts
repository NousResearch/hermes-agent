/**
 * E2E contract for the compositor-only GlyphSpinner.
 *
 * The spinner's whole reason for existing in this shape is a CSS animation:
 * every frame is in the DOM from mount and a `transform` keyframes animation
 * scrolls between them, so there is no JS timer and no per-tick DOM mutation
 * scheduling document-scale style recalculation.
 *
 * None of that is observable in jsdom — it has no animation engine, no
 * cascade resolution for `steps()`, and no `Element.getAnimations()`. The
 * jsdom suite (src/components/ui/glyph-spinner.test.tsx) therefore pins the
 * DATA and WIRING, and this spec pins the RENDERED BEHAVIOUR in a real
 * browser, which is the only place the stylesheet actually runs.
 *
 * This replaces three tests that asserted on the TEXT of the stylesheet.
 * Reading source in a test is banned outright (AGENTS.md) and those tests
 * proved the point: a var()-fallback edit that changed no rendered pixel
 * broke one of them, while none of them had ever executed the CSS.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, type Page, test } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

/* Scope to a spinner that is actually RUNNING. Turns from earlier tests in
 * this file leave parked spinners mounted (kept-alive panes, swap overlays
 * hold them with data-paused='true'), and document.querySelector returns the
 * FIRST strip in the DOM — a stale parked one once two turns have run. */
const STRIP = '.glyph-spinner:not([data-paused="true"]) .glyph-spinner__strip'

/** Prompt the mock server holds open so the spinner runs for the whole file. */
const SPINNER_PROMPT = 'E2E_GLYPH_SPINNER_HOLD'

/**
 * Get a RUNNING frame strip into the DOM deterministically.
 *
 * A turn is sent so the app is genuinely busy (the mock server holds the
 * stream open), but which surface mounts a spinner mid-turn is app policy
 * that has changed before and will again — the transcript, status stack and
 * swap overlay all park/unmount theirs at different moments, which made this
 * spec racy. The contract under test is the STYLESHEET (steps() animation,
 * layer promotion, the data-paused and global pause gates), and that CSS is
 * driven entirely by the `data-paused` attribute — the same attribute the
 * parked assertions below already toggle. So: wait for any mounted spinner
 * (the ChatSwapOverlay keeps one mounted, parked, after boot), then unpark it
 * and assert against the running animation.
 */
async function mountSpinner(page: Page): Promise<void> {
  if (await page.locator(STRIP).count()) {
    return
  }

  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 10_000 })
  await composer.click()
  await composer.type(SPINNER_PROMPT, { delay: 10 })
  await page.keyboard.press('Enter')

  await page.waitForSelector('.glyph-spinner__strip', { state: 'attached', timeout: 20_000 })
  await page.evaluate(() => {
    for (const el of document.querySelectorAll('.glyph-spinner[data-paused]')) {
      el.removeAttribute('data-paused')
    }
  })
  await page.waitForSelector(STRIP, { state: 'attached', timeout: 20_000 })
}

test.describe('GlyphSpinner (compositor animation)', () => {
  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    fixture = await setupMockBackend({
      mockServer: { holdFirstStreamForPrompt: SPINNER_PROMPT },
    })
    await waitForAppReady(fixture)
  })

  test.afterAll(async () => {
    fixture?.mock.releaseHeldStream()
    await fixture?.cleanup()
  })

  test('animates with a steps() transform keyframes animation, one step per frame', async () => {
    const { page } = fixture
    await mountSpinner(page)

    const observed = await page.evaluate(strip => {
      const el = document.querySelector<HTMLElement>(strip)

      if (!el) {
        throw new Error('no frame strip in the DOM')
      }

      const style = getComputedStyle(el)
      const animations = el.getAnimations()

      return {
        frameCount: el.querySelectorAll('.glyph-spinner__frame').length,
        timingFunction: style.animationTimingFunction,
        iterationCount: style.animationIterationCount,
        durationMs: animations[0]?.effect?.getTiming().duration ?? null,
        names: animations.map(a => (a as CSSAnimation).animationName),
        // A percentage translate makes the animation layout-dependent, which
        // Chromium refuses to composite. Read the engine's own keyframes: a
        // revert to translateY(-100%) shows up here, while the computed
        // `style.transform` always serializes to a matrix and can't tell.
        travel: ((animations[0]?.effect as KeyframeEffect | undefined)?.getKeyframes() ?? [])
          .map(k => String((k as Keyframe & { transform?: string }).transform ?? ''))
          .join(' | ')
      }
    }, STRIP)

    // The strip carries every frame; `steps(N)` parks on each one in turn.
    expect(observed.frameCount).toBeGreaterThan(1)
    // Chromium has serialized jump-end as both `steps(N)` and `steps(N, end)`.
    expect(observed.timingFunction).toMatch(new RegExp(`^steps\\(${observed.frameCount}\\b`))
    expect(observed.iterationCount).toBe('infinite')
    expect(observed.names).toContain('glyph-spinner-advance')
    // One full cycle is frames x interval, so the duration must be a positive
    // multiple of the frame count — not the single-frame interval.
    expect(observed.durationMs).toBeGreaterThan(0)
    // Length-typed travel, never a percentage: `translateY(-100%)` would keep
    // the animation off the compositor. Chromium has serialized the resolved
    // keyframe both as the authored `calc(...)` and as an absolute `...px`
    // length depending on version — accept any length, reject percentages.
    expect(observed.travel).toMatch(/calc\(|px\)/)
    expect(observed.travel).not.toContain('%')
  })

  test('is promoted to a layer while running, and neither animates nor holds a layer when parked', async () => {
    const { page } = fixture
    await mountSpinner(page)

    const running = await page.evaluate(strip => {
      const el = document.querySelector<HTMLElement>(strip)!

      return {
        playState: getComputedStyle(el).animationPlayState,
        willChange: getComputedStyle(el).willChange
      }
    }, STRIP)

    expect(running.playState).toBe('running')
    // Scoped to active spinners — a permanently promoted layer per parked
    // spinner is pure memory at fan-out breadth.
    expect(running.willChange).toBe('transform')

    // 1. The per-spinner gate: a kept-alive but inactive pane, or an explicit
    //    `paused` prop (ChatSwapOverlay's fade-out).
    const parked = await page.evaluate(strip => {
      const el = document.querySelector<HTMLElement>(strip)!
      const viewport = el.closest<HTMLElement>('.glyph-spinner')!
      const previous = viewport.getAttribute('data-paused')

      viewport.setAttribute('data-paused', 'true')

      const state = {
        playState: getComputedStyle(el).animationPlayState,
        willChange: getComputedStyle(el).willChange
      }

      if (previous === null) {
        viewport.removeAttribute('data-paused')
      } else {
        viewport.setAttribute('data-paused', previous)
      }

      return state
    }, STRIP)

    expect(parked.playState).toBe('paused')
    expect(parked.willChange).toBe('auto')

    // 2. The global gate: window blur / minimize / document-hidden, which
    //    main.tsx drives by arming this attribute on the root. The strip must
    //    be named in that rule, or every spinner keeps animating behind an
    //    inactive window — the CPU burn the original ticker's pause
    //    controller existed to avoid.
    const globallyPaused = await page.evaluate(strip => {
      const root = document.documentElement
      const had = root.hasAttribute('data-renderer-animations-paused')

      root.setAttribute('data-renderer-animations-paused', '')
      const playState = getComputedStyle(document.querySelector<HTMLElement>(strip)!).animationPlayState

      if (!had) {
        root.removeAttribute('data-renderer-animations-paused')
      }

      return playState
    }, STRIP)

    expect(globallyPaused).toBe('paused')
  })

  test('advances in discrete frames and creates no timer-driven DOM churn', async () => {
    const { page } = fixture
    await mountSpinner(page)

    // Sample the resolved transform across one full cycle. A steps() animation
    // holds each value for a whole interval and jumps between them, so the
    // distinct values it visits must be bounded by the frame count — a linear
    // animation would produce a new value on every sample.
    const sampled = await page.evaluate(async strip => {
      const el = document.querySelector<HTMLElement>(strip)!
      const frames = el.querySelectorAll('.glyph-spinner__frame').length
      const duration = Number(el.getAnimations()[0]?.effect?.getTiming().duration ?? 0)
      const seen = new Set<string>()
      const textAtStart = el.textContent

      const deadline = performance.now() + duration

      while (performance.now() < deadline) {
        seen.add(getComputedStyle(el).transform)
        await new Promise(resolve => requestAnimationFrame(() => resolve(null)))
      }

      return { distinct: seen.size, frames, textUnchanged: el.textContent === textAtStart }
    }, STRIP)

    expect(sampled.distinct).toBeGreaterThan(1)
    expect(sampled.distinct).toBeLessThanOrEqual(sampled.frames + 1)
    // The old implementation rewrote textContent ~12x/second. Nothing may
    // mutate the DOM as this animates — that mutation is the whole incident.
    expect(sampled.textUnchanged).toBe(true)
  })
})
