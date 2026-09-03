/**
 * E2E — annotate_screen draws on the transparent overlay window.
 *
 * Full-loop proof, not a mock of the seam: the prompt triggers a scripted
 * tool call in the mock inference server, and the draw rides the REAL path —
 * agent → gateway `screen.annotate.request` → desktop renderer bridge →
 * preload IPC → Electron main → the click-through overlay window.
 *
 * The scripted target is 'screen' (whole-display coordinates) so the test
 * never depends on which other OS windows happen to be open on the machine
 * running the suite. NOTE: the overlay really appears over the test machine's
 * display for ~8 seconds (ANNOTATE_TTL_SECONDS).
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import type { ElectronApplication, Page } from '@playwright/test'

import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { ANNOTATE_TEXTS, ANNOTATE_TRIGGER, ANNOTATE_TTL_SECONDS } from './mock-server'

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture!, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

/** The overlay window's page, identified by its ?win=annotate boot URL. */
function findOverlayPage(app: ElectronApplication): Page | undefined {
  return app.windows().find(candidate => candidate.url().includes('win=annotate'))
}

/** Main-process view of the overlay window (visibility, focusability). */
function overlayWindowState(app: ElectronApplication) {
  return app.evaluate(({ BrowserWindow }) => {
    const overlay = BrowserWindow.getAllWindows().find(
      (candidate: { webContents: { getURL(): string } }) => candidate.webContents.getURL().includes('win=annotate')
    )

    if (!overlay) {
      return null
    }

    return {
      focusable: overlay.isFocusable(),
      focused: overlay.isFocused(),
      visible: overlay.isVisible()
    }
  })
}

test.describe('annotate_screen overlay', () => {
  test('a scripted tool call draws marks on the overlay, then the TTL clears them', async ({}, testInfo) => {
    const page = fixture!.page
    const app = fixture!.app

    const composer = page.locator('[contenteditable="true"]').first()
    await composer.waitFor({ state: 'visible', timeout: 10_000 })

    // waitForAppReady only proves the shell painted; on a cold first boot the
    // gateway/inference can still be coming up ("Gateway checking" in the
    // status bar), and a submit fired into that window is silently dropped.
    await page.waitForFunction(
      () => {
        const text = document.body?.textContent ?? ''

        return !text.includes('Gateway checking') && !text.includes('Gateway connecting') && !text.includes('Gateway offline')
      },
      undefined,
      { timeout: 90_000 }
    )

    const prompt = `${ANNOTATE_TRIGGER} what move should I go next?`

    await composer.click()
    await composer.type(prompt, { delay: 10 })
    await page.keyboard.press('Enter')

    // The submit landed when the prompt shows up in the transcript.
    await expect(page.getByText(prompt).last()).toBeVisible({ timeout: 30_000 })

    // The tool call spawns the overlay window mid-turn.
    await expect
      .poll(async () => Boolean(findOverlayPage(app)), {
        message: 'overlay window (?win=annotate) should spawn for the draw',
        timeout: 45_000
      })
      .toBe(true)

    const overlayPage = findOverlayPage(app)!

    // The renderer paints one <g> per shape: circle + arrow + label.
    await expect
      .poll(async () => overlayPage.locator('svg g.hermes-annotation').count(), {
        message: 'overlay should paint the three scripted shapes',
        timeout: 15_000
      })
      .toBe(3)

    await expect(overlayPage.locator('svg circle').first()).toBeVisible()
    await expect(overlayPage.getByText('E2E annotation')).toBeVisible()
    await expect(overlayPage.getByText('pawn')).toBeVisible()

    // A pointer overlay must be a pure display surface: visible, but never
    // focusable and never holding focus (clicks belong to the app below).
    const state = await overlayWindowState(app)
    expect(state?.visible).toBe(true)
    expect(state?.focusable).toBe(false)
    expect(state?.focused).toBe(false)

    // Visual receipt for the report.
    await overlayPage.screenshot({ path: testInfo.outputPath('screen-annotations-overlay.png') })

    // The tool result rode back through screen.annotate.respond and the agent
    // finished its turn.
    await expect(page.getByText(ANNOTATE_TEXTS.final)).toBeVisible({ timeout: 60_000 })

    // TTL expiry hides the overlay and clears the shapes on its own — no
    // agent action, no user action. Loose deadline: the 8s TTL armed when the
    // draw landed, well before this point.
    await expect
      .poll(async () => (await overlayWindowState(app))?.visible, {
        message: 'overlay should hide itself when the TTL expires',
        timeout: (ANNOTATE_TTL_SECONDS + 15) * 1000
      })
      .toBe(false)

    await expect
      .poll(async () => overlayPage.locator('svg g.hermes-annotation').count(), {
        message: 'expired shapes should be cleared from the overlay renderer',
        timeout: 10_000
      })
      .toBe(0)
  })
})
