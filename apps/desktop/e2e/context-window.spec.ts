/**
 * E2E coverage for `/ctxwindow` — the context-window quick access.
 *
 * Drives the REAL Electron app against the mock-backend fixture, so the whole
 * chain is exercised: composer keystrokes → slash popover → the dialog →
 * `PUT /api/config` → the on-disk `config.yaml` under the sandbox HERMES_HOME.
 *
 * The on-disk assertions are the point of this spec. A unit test can only
 * prove the renderer sent `model_context_length`; only reading the real
 * config.yaml proves the save (and, more importantly, the CLEAR) survived the
 * backend's deep-merge — the path that previously silently did nothing.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

let fixture: MockBackendFixture | null = null

/**
 * Read `model.context_length` straight out of the on-disk config.yaml.
 *
 * Hand-parsed rather than pulled through a YAML library: there is no YAML
 * parser in this package's dependency tree, and the assertion only needs the
 * one scalar under the top-level `model:` mapping. Returns `undefined` when the
 * key is absent — which is exactly the "auto-detect" state under test, so the
 * absent case must stay distinguishable from an explicit `0`.
 */
function readConfigContextLength(hermesHome: string): number | undefined {
  const raw = fs.readFileSync(path.join(hermesHome, 'config.yaml'), 'utf8')
  const lines = raw.split('\n')
  let inModel = false

  for (const line of lines) {
    if (/^model:\s*$/.test(line)) {
      inModel = true
      continue
    }

    // A non-indented, non-blank line ends the `model:` block.
    if (inModel && line.trim() && !/^\s/.test(line)) {
      inModel = false
    }

    if (!inModel) {
      continue
    }

    const match = /^\s+context_length:\s*(\S+)\s*$/.exec(line)

    if (match) {
      const parsed = Number.parseInt(match[1], 10)

      return Number.isFinite(parsed) ? parsed : undefined
    }
  }

  return undefined
}

/** Whether the `model:` section still carries a `context_length` key at all. */
function hasConfigContextLengthKey(hermesHome: string): boolean {
  return readConfigContextLength(hermesHome) !== undefined
}

/** The auto-detected figure the backend resolver reports (e.g. "256k"). */
let autoDetectedValue = ''

/** Type a slash command into the composer and submit it. */
async function runSlash(command: string): Promise<void> {
  const page = fixture!.page
  const composer = page.locator('[contenteditable="true"]').first()

  await composer.click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Backspace')
  await composer.type(command, { delay: 30 })
  // The completion popover swallows Enter to accept the highlighted row, so
  // dismiss it first — this submits exactly what the user typed.
  await page.keyboard.press('Escape')
  await page.keyboard.press('Enter')
}

function dialog() {
  return fixture!.page.locator('[role="dialog"]', { hasText: 'Context window' })
}

/** Open the dialog through the composer if it isn't already showing. */
async function ensureDialogOpen(): Promise<void> {
  if (await dialog().isVisible()) {
    return
  }

  await runSlash('/ctxwindow')
  await expect(dialog()).toBeVisible({ timeout: 20_000 })
}

test('/ctxwindow quick access persists and clears the backend override', async ({}, testInfo) => {
  test.setTimeout(240_000)
  fixture = await setupMockBackend()

  try {
    await waitForAppReady(fixture, 120_000)

    await test.step('is discoverable in the slash popover when typing /ctxw', async () => {
      const page = fixture!.page
      const composer = page.locator('[contenteditable="true"]').first()

      await composer.click()
      await page.keyboard.press('ControlOrMeta+a')
      await page.keyboard.press('Backspace')
      await composer.type('/ctxw', { delay: 60 })

      const popover = page.locator('[data-slot="composer-completion-drawer"]')
      await expect(popover).toBeVisible({ timeout: 15_000 })

      await page.screenshot({ path: testInfo.outputPath('slash-popover-ctxw.png') })

      // "As convenient to switch as /model" means the command has to show up
      // while the user is typing a prefix of it — that's the discoverability
      // half of the acceptance criteria.
      await expect(popover.locator('button', { hasText: '/ctxwindow' })).toBeVisible({ timeout: 15_000 })

      await page.keyboard.press('Escape')
      await page.keyboard.press('ControlOrMeta+a')
      await page.keyboard.press('Backspace')
    })

    await test.step('opens with the auto-detected window and an empty override', async () => {
      const page = fixture!.page

      expect(hasConfigContextLengthKey(fixture!.sandbox.hermesHome)).toBe(false)

      await runSlash('/ctxwindow')

      await expect(dialog()).toBeVisible({ timeout: 20_000 })
      await page.screenshot({ path: testInfo.outputPath('context-dialog-auto.png') })

      const override = page.locator('#context-window-override')
      await expect(override).toBeVisible()
      // No pin on disk → the field is blank and the dialog says it's on auto.
      await expect(override).toHaveValue('')
      // Whatever `get_model_context_length()` resolves for the mock route, the
      // dialog must show it as a concrete figure rather than "unknown", and must
      // say it's the value in use. The exact number is the backend resolver's
      // business (asserting a literal here would be a change-detector on the
      // provider metadata), so capture it and reuse it after the clear.
      await expect(dialog()).toContainText(/Auto-detected: \d/)
      await expect(dialog()).toContainText('Auto-detect')

      // Capture just the compact figure. The dialog text runs together without
      // separators, so anchor on the digits-plus-unit shape rather than \S+.
      autoDetectedValue = (await dialog().textContent())?.match(/Auto-detected: (\d[\d.]*[kM]?)/)?.[1] ?? ''
      expect(autoDetectedValue).toMatch(/^\d/)
      // Nothing pinned, so the window in use is the auto figure.
      await expect(dialog()).toContainText(`In use: ${autoDetectedValue} · Auto-detect`)

      // "Use auto-detect" is inert while nothing is pinned.
      await expect(page.getByRole('button', { name: 'Use auto-detect' })).toBeDisabled()
    })

    await test.step('saving an explicit value writes model.context_length to config.yaml', async () => {
      const page = fixture!.page

      await ensureDialogOpen()

      const override = page.locator('#context-window-override')
      await override.fill('128000')
      await page.screenshot({ path: testInfo.outputPath('context-dialog-filled.png') })

      await page.getByRole('button', { name: 'Save', exact: true }).click()

      // The dialog closes on a successful commit.
      await expect(dialog()).toBeHidden({ timeout: 20_000 })

      // The real assertion: the value survived the backend's config merge and
      // landed on disk under the test's HERMES_HOME.
      await expect.poll(() => readConfigContextLength(fixture!.sandbox.hermesHome), { timeout: 20_000 }).toBe(128_000)
    })

    await test.step('reopening reflects the persisted pin with no stale draft', async () => {
      const page = fixture!.page

      await runSlash('/ctxwindow')

      await expect(dialog()).toBeVisible({ timeout: 20_000 })

      const override = page.locator('#context-window-override')
      await expect(override).toHaveValue('128000', { timeout: 20_000 })
      await expect(dialog()).toContainText('128k')
      await page.screenshot({ path: testInfo.outputPath('context-dialog-pinned.png') })

      // With a pin present, the escape hatch back to auto is now live.
      await expect(page.getByRole('button', { name: 'Use auto-detect' })).toBeEnabled()
    })

    await test.step('Use auto-detect removes the key from config.yaml and returns to auto', async () => {
      const page = fixture!.page

      await ensureDialogOpen()
      expect(readConfigContextLength(fixture!.sandbox.hermesHome)).toBe(128_000)

      await page.getByRole('button', { name: 'Use auto-detect' }).click()
      await expect(dialog()).toBeHidden({ timeout: 20_000 })

      // The previously broken path: `_deep_merge` cannot delete, so the pin used
      // to be resurrected on every save and "back to auto" did nothing. The key
      // must be GONE, not merely set to 0 — an explicit 0 is a different state.
      await expect.poll(() => hasConfigContextLengthKey(fixture!.sandbox.hermesHome), { timeout: 20_000 }).toBe(false)

      // And the UI agrees: reopening shows a blank field on the auto value.
      await runSlash('/ctxwindow')
      await expect(dialog()).toBeVisible({ timeout: 20_000 })
      await expect(page.locator('#context-window-override')).toHaveValue('', { timeout: 20_000 })
      await expect(dialog()).toContainText(`Auto-detected: ${autoDetectedValue}`)
      await expect(dialog()).toContainText(`In use: ${autoDetectedValue} · Auto-detect`)
      await expect(page.getByRole('button', { name: 'Use auto-detect' })).toBeDisabled()
      await page.screenshot({ path: testInfo.outputPath('context-dialog-back-to-auto.png') })

      await page.keyboard.press('Escape')
    })
  } finally {
    await fixture.cleanup()
    fixture = null
  }
})
