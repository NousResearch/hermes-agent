/**
 * E2E for the Jarvis product shell — the "pełny pion" the release gate blocks on.
 *
 * Every other suite under e2e/ drives the Hermes runtime surfaces. None of them
 * touches the product chrome that AI Evolution Jarvis actually ships: the shell,
 * its navigation, the Core, and the layout contract from the design doc's P0
 * acceptance criteria (§14.11 — no horizontal overflow at 390/768/1150/1440/2560
 * px, 44 px touch targets, a visible focus ring). That chrome has jsdom unit
 * tests, which cannot see a real layout: a `flex-row` that overflows at 390 px
 * passes every one of them.
 *
 * So this suite asserts the product shell in a real Electron window, against a
 * real `hermes serve` backend with a mock provider — the same chain boot.spec.ts
 * uses:
 *
 *   electron → hermes serve (python) → mock provider → renderer → JarvisShell
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 *   npm exec playwright test e2e/jarvis-shell-vertical.spec.ts --reporter=list
 */
import type { CDPSession } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

let fixture: MockBackendFixture | null = null
let cdp: CDPSession | null = null

/** Must match JARVIS_ONBOARDING_STATE_KEY / _VERSION in src/app/jarvis/onboarding-state.ts. */
const ONBOARDING_KEY_PREFIX = 'ai-evolution-jarvis-onboarding-v1'
const ONBOARDING_STEPS = ['welcome', 'profile', 'engine', 'model', 'voice', 'access', 'computer', 'connections', 'approvals']
/** Must match JARVIS_TIPS_STATE_KEY in src/app/jarvis/tips-state.ts. */
const TIPS_KEY_PREFIX = 'ai-evolution-jarvis-tips-v1'

/**
 * The widths §14.11 names. 390 is a phone-width window a user can genuinely
 * drag the desktop app down to, and it is where a row-first layout breaks first.
 */
const RESPONSIVE_WIDTHS = [390, 768, 1150, 1440, 2560] as const

/**
 * The main views on the nav rail, in render order, and the route each one
 * opens. `hash` is null for the home view: the app runs under a HashRouter, so
 * home is `#/` and asserting "contains /" would pass against any route at all.
 */
const MAIN_VIEWS = [
  { view: 'jarvis', hash: null },
  { view: 'tasks', hash: '#/cron' },
  { view: 'agents', hash: '#/agents' },
  { view: 'messaging', hash: '#/messaging' },
  { view: 'webhooks', hash: '#/webhooks' },
  { view: 'artifacts', hash: '#/artifacts' },
  { view: 'memory', hash: '#/starmap?view=list' },
  { view: 'starmap', hash: '#/starmap' },
  { view: 'tools', hash: '#/skills' },
  { view: 'connections', hash: '#/connections' },
  { view: 'insights', hash: '#/command-center' }
] as const

/**
 * Mark Jarvis onboarding as finished, then reload into the shell.
 *
 * A fresh sandbox has no onboarding state, so `shouldShowJarvisOnboarding`
 * returns true and a full-viewport overlay covers the shell — `waitForAppReady`
 * would sit there until it timed out. The overlay persists its own state on
 * mount, so by the time it is on screen localStorage holds the real scope key
 * (which depends on the active connection and profile). Rewriting whatever keys
 * are actually there beats hardcoding a scope and hoping it matches; the
 * `local::default` write is the fallback for the case where nothing was
 * persisted yet.
 */
async function completeOnboardingAndReload(): Promise<void> {
  const page = fixture!.page

  await page.waitForSelector('[data-testid="jarvis-onboarding"], [data-jarvis-shell]', {
    state: 'attached',
    timeout: 120_000
  })

  await page.evaluate(
    ({ prefix, steps, tipsPrefix }) => {
      const value = JSON.stringify({
        version: 3,
        currentStep: 'approvals',
        completedSteps: steps,
        selections: {}
      })

      const keys = Object.keys(window.localStorage).filter(key => key.startsWith(`${prefix}:`))

      for (const key of keys.length > 0 ? keys : [`${prefix}:local::default`]) {
        window.localStorage.setItem(key, value)
        // The tips dialog opens itself once after onboarding; this suite is
        // about the shell, so spend that auto-open for the same scope.
        window.localStorage.setItem(
          `${tipsPrefix}:${key.slice(prefix.length + 1)}`,
          JSON.stringify({ autoOpen: false, dismissedIds: [], version: 1 })
        )
      }
    },
    { prefix: ONBOARDING_KEY_PREFIX, steps: ONBOARDING_STEPS, tipsPrefix: TIPS_KEY_PREFIX }
  )

  await page.reload()
  await waitForAppReady(fixture!, 120_000)

  // The generic provider picker may open after the gateway connects. This
  // suite tests the Jarvis shell, so use its persistent first-run escape.
  const chooseLater = page.getByRole('button', { name: /Wybiorę dostawcę później|I'll choose a provider later/ })
  if (await chooseLater.waitFor({ state: 'visible', timeout: 45_000 }).then(() => true, () => false)) {
    await chooseLater.click()
  }

  // Loud rather than mysterious: if the scope key ever changes shape, this is
  // the assertion that says so, instead of a 120 s timeout three tests later.
  await expect(
    page.locator('[data-testid="jarvis-onboarding"]'),
    'onboarding should be dismissed — the seeded localStorage scope key may no longer match'
  ).toHaveCount(0)
}

/**
 * Resize the *viewport* rather than the OS window.
 *
 * CI runs xvfb at 1280x1024, so asking Electron for a 2560 px window is a
 * request the display server is free to clamp — and a clamped window would make
 * the widest overflow check silently re-test 1280 px. CDP metrics override is
 * what the renderer's media queries and layout actually read, and it has no
 * such ceiling.
 */
async function setViewportWidth(width: number, height = 800): Promise<void> {
  await cdp!.send('Emulation.setDeviceMetricsOverride', {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: false
  })

  await fixture!.page.waitForFunction(
    expected => window.innerWidth === expected,
    width,
    { timeout: 10_000 }
  )
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await completeOnboardingAndReload()
  cdp = await fixture.app.context().newCDPSession(fixture.page)
})

test.afterAll(async () => {
  await cdp?.detach().catch(() => undefined)
  cdp = null
  await fixture?.cleanup()
  fixture = null
})

test.describe('Jarvis product shell', () => {
  test('the shell wraps the runtime instead of replacing it', async () => {
    const page = fixture!.page

    // Both must hold at once: the product chrome is present AND the Hermes
    // runtime is still mounted inside it. A shell that rendered standalone —
    // or a runtime that escaped the shell — is the regression this catches.
    await expect(page.locator('[data-jarvis-shell]')).toBeVisible()
    await expect(page.locator('nav[data-jarvis-nav]')).toBeVisible()
    await expect(page.locator('[data-jarvis-view]')).toHaveCount(1)
    await expect(page.locator('textarea, [contenteditable="true"]').first()).toBeAttached()
  })

  test('the Core renders on the home view', async () => {
    await expect(fixture!.page.locator('[data-testid="jarvis-core"]')).toBeVisible()
  })

  test('every nav entry routes and marks itself current', async () => {
    const page = fixture!.page

    const buttons = page.locator('nav[data-jarvis-nav] [data-jarvis-nav-view]')
    await expect(buttons).toHaveCount(MAIN_VIEWS.length)

    for (const { view, hash } of MAIN_VIEWS) {
      const button = page.locator(`[data-jarvis-nav-view="${view}"]`)
      await button.click()

      // The view attribute is the shell's own state; the hash is the runtime's.
      // Asserting both is what proves the shell drives real navigation rather
      // than swapping a local tab and leaving the runtime where it was.
      await expect(page.locator(`[data-jarvis-view="${view}"]`)).toBeVisible()
      await expect(button).toHaveAttribute('aria-current', 'page')

      if (hash) {
        await expect.poll(() => page.evaluate(() => window.location.hash)).toContain(hash)
      }
    }

    await page.getByRole('button', { name: 'Historia czatu' }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
    await page.keyboard.press('Escape')

    for (const { name, view, hash } of [
      { name: 'Ustawienia', view: 'settings', hash: '#/settings' },
      { name: 'Profil', view: 'profile', hash: '#/profiles' }
    ]) {
      const button = page.locator('[data-jarvis-nav-rail]').getByRole('button', { name, exact: true })
      await button.click()
      await expect(page.locator(`[data-jarvis-view="${view}"]`)).toBeVisible()
      await expect(button).toHaveAttribute('aria-current', 'page')
      await expect.poll(() => page.evaluate(() => window.location.hash)).toContain(hash)
    }

    // Leave the shell on home so the screenshot below is comparable run to run.
    await page.locator('[data-jarvis-nav-view="jarvis"]').click()
    await expect(page.locator('[data-jarvis-view="jarvis"]')).toBeVisible()
  })

  test('the language rail switches the whole shell to Polish and back', async () => {
    const page = fixture!.page
    const language = page.locator('[data-jarvis-nav-rail] [role="radiogroup"]')

    await language.getByRole('radio', { name: 'PL' }).click()
    await expect(language.getByRole('radio', { name: 'PL' })).toHaveAttribute('aria-checked', 'true')
    await expect(page.locator('html')).toHaveAttribute('lang', 'pl')
    await expect(page.locator('nav[data-jarvis-nav]').getByRole('button', { name: 'Pulpit' })).toBeVisible()

    await language.getByRole('radio', { name: 'EN' }).click()
    await expect(language.getByRole('radio', { name: 'EN' })).toHaveAttribute('aria-checked', 'true')
    await expect(page.locator('html')).toHaveAttribute('lang', 'en')
  })

  test('keyboard focus reaches the nav and stays visible', async () => {
    const page = fixture!.page

    // Tab, rather than locator.focus(): the nav styles its ring with
    // `focus-visible`, which Chromium only applies to focus it attributes to
    // the keyboard. A programmatic focus() would leave the ring off and make
    // this test assert the opposite of what it means to.
    await page.evaluate(() => {
      const active = document.activeElement

      if (active instanceof HTMLElement) {
        active.blur()
      }
    })

    // The budget is one full tab cycle of a busy home screen (rail cards,
    // composer, status bar), not a claim about where the nav sits in it.
    let reached = false
    for (let press = 0; press < 120 && !reached; press += 1) {
      await page.keyboard.press('Tab')
      reached = await page.evaluate(() =>
        Boolean(document.activeElement?.closest('nav[data-jarvis-nav]'))
      )
    }

    expect(reached, 'the nav rail should be reachable by Tab').toBe(true)

    // §14.11 requires a visible focus ring. An `outline-none` rule with no
    // matching `focus-visible` rule is invisible in a screenshot diff but fatal
    // for keyboard users, so assert the computed outline directly.
    const outline = await page.evaluate(() => {
      const style = window.getComputedStyle(document.activeElement!)

      return { style: style.outlineStyle, width: style.outlineWidth }
    })

    expect(outline.style).not.toBe('none')
    expect(parseFloat(outline.width)).toBeGreaterThan(0)
  })

  test('nav targets meet the 44 px touch minimum', async () => {
    const heights = await fixture!.page
      .locator('nav[data-jarvis-nav] button')
      .evaluateAll(elements => elements.map(element => element.getBoundingClientRect().height))

    expect(heights.length).toBeGreaterThanOrEqual(MAIN_VIEWS.length)
    for (const height of heights) {
      expect(height).toBeGreaterThanOrEqual(44)
    }
  })

  test.describe('responsive layout', () => {
    // The override is process-wide for this page: a leaked 390 px viewport
    // would silently reshape the screenshot taken by the test after it.
    test.afterAll(async () => {
      await cdp?.send('Emulation.clearDeviceMetricsOverride').catch(() => undefined)
    })

    for (const width of RESPONSIVE_WIDTHS) {
      test(`no horizontal overflow at ${width} px`, async () => {
        await setViewportWidth(width)

        const overflow = await fixture!.page.evaluate(() => ({
          clientWidth: document.documentElement.clientWidth,
          scrollWidth: document.documentElement.scrollWidth
        }))

        // 1 px of slack absorbs sub-pixel rounding; anything larger is a real
        // element pushing the viewport wider than the window.
        expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.clientWidth + 1)
      })
    }
  })

  test('the home screen raises real pulse suggestions and remembers a dismissal', async () => {
    const page = fixture!.page
    // A fresh sandbox home has an empty USER.md and no cron jobs: the real
    // /api/pulse must turn both into suggestions.
    const knowOwner = page.locator('[data-pulse-kind="know_owner"]')
    const firstAutomation = page.locator('[data-pulse-kind="first_automation"]')

    await expect(knowOwner).toBeVisible({ timeout: 60_000 })
    await expect(firstAutomation).toBeVisible()

    // Second button in the row is "Not now".
    await knowOwner.locator('button').nth(1).click()
    await expect(knowOwner).toHaveCount(0)

    // The backend, not the renderer, holds the dismissal: it survives a reload.
    await page.reload()
    await waitForAppReady(fixture!, 120_000)
    await expect(firstAutomation).toBeVisible({ timeout: 30_000 })
    await expect(knowOwner).toHaveCount(0)
  })

  test('a guided connection setup lands in the real composer, ready to read before sending', async () => {
    const page = fixture!.page

    await page.locator('[data-jarvis-nav-view="connections"]').click()
    await expect(page.locator('[data-connection-card="google"]')).toBeVisible()

    // First button on an agent-driven card starts the guided setup.
    await page.locator('[data-connection-card="google"] button').first().click()
    await expect.poll(() => page.evaluate(() => window.location.hash)).toBe('#/')

    const composer = page.locator('textarea, [contenteditable="true"]').first()

    await expect(composer).toBeVisible()
    await expect
      .poll(async () => (await composer.inputValue().catch(() => composer.textContent())) ?? '')
      .toContain('google-workspace')
  })

  test('shell screenshot', async () => {
    await expectVisualSnapshot(fixture!.page, { name: 'jarvis-shell', app: fixture!.app })
  })
})
