/**
 * E2E visual capture and interaction verification for the Action Center panel.
 *
 * Uses the existing Electron Playwright fixtures (setupMockBackend) to
 * exercise the real built renderer with a disposable sandbox and mock
 * inference backend. Captures real UI screenshots in various states.
 *
 * Output: .inbox-work/screenshots/*.png
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test, type Page } from '@playwright/test'

import {
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const SCREENSHOTS_DIR = path.resolve(DESKTOP_ROOT, '..', '..', '.inbox-work', 'screenshots')

// ── helpers ──────────────────────────────────────────────────────────────────

/** Force the Electron window to fixed dimensions. */
async function forceSize(app: MockBackendFixture['app'], width: number, height: number): Promise<void> {
  await app.evaluate(({ BrowserWindow }, { width, height }) => {
    const win = BrowserWindow.getAllWindows()[0]
    if (win) {
      win.unmaximize()
      win.setMinimumSize(width, height)
      win.setSize(width, height, false)
      win.setBounds({ x: 0, y: 0, width, height })
    }
  }, { width, height })
}

/** Return the inbox chip locator. */
function inboxChip(page: Page) {
  return page.getByRole('button', { name: /Action Center/ }).first()
}

/**
 * Wait for the inbox panel to be visible AND connected (not showing
 * "Disconnected" or "Loading inbox…"). This asserts a successful inbox.list
 * response by checking for a terminal connected state: "All clear",
 * "Inbox not supported", "Partial read", or actual session items.
 *
 * Previous `waitForPanelVisible` only checked for "Action Center" heading text,
 * which appears regardless of connection state — screenshots were mislabeled.
 */
async function waitForInboxConnected(page: Page, timeoutMs = 30_000): Promise<void> {
  await page.waitForFunction(
    () => {
      const text = document.body.textContent ?? ''

      // Terminal connected states: inbox.list succeeded
      if (text.includes('All clear')) {return true}
      if (text.includes('Inbox not supported')) {return true}
      if (text.includes('Partial read')) {return true}
      if (text.includes('Incomplete data')) {return true}

      // Session rows visible = connected with items
      if (document.querySelector('[data-panel-row]')) {return true}

      // Still loading or disconnected — keep waiting
      return false
    },
    undefined,
    { timeout: timeoutMs },
  )
}

/** Wait for the inbox panel heading to appear (used for panel-open checks). */
function isPanelVisible(page: Page): Promise<boolean> {
  return page.locator('h2:has-text("Action Center")').isVisible()
}

/** Close the panel by pressing Escape. */
async function closePanelWithEscape(page: Page): Promise<void> {
  await page.keyboard.press('Escape')
  await page.waitForTimeout(500)
}

// ── manifest ─────────────────────────────────────────────────────────────────

interface ManifestEntry {
  dimensions: string
  fixture: string
  path: string
  state: string
}

const manifest: ManifestEntry[] = []

function record(state: string, fixtureType: string, dims: string, filePath: string): void {
  manifest.push({ dimensions: dims, fixture: fixtureType, path: filePath, state })
}

async function capture(testInfo: { outputPath: (n: string) => string }, name: string, state: string, page: Page, dims = '1220x800'): Promise<void> {
  const shotPath = testInfo.outputPath(name)
  await page.screenshot({ path: shotPath })
  record(state, 'real-backend', dims, shotPath)

  const dest = path.join(SCREENSHOTS_DIR, name)
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  fs.copyFileSync(shotPath, dest)
  record(state, 'real-backend', dims, dest)
}

// ── fixture ──────────────────────────────────────────────────────────────────

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null

  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true })
  fs.writeFileSync(
    path.join(SCREENSHOTS_DIR, 'manifest.json'),
    JSON.stringify(manifest, null, 2),
    'utf8',
  )
  console.log(`[inbox-capture] manifest written to ${path.join(SCREENSHOTS_DIR, 'manifest.json')}`)
})

// ── Visual captures ──────────────────────────────────────────────────────────

test.describe('inbox visual capture', () => {
  test('chip visible in statusbar', async ({}, testInfo) => {
    const page = fixture!.page
    const chip = inboxChip(page)
    await expect(chip).toBeVisible({ timeout: 30_000 })

    await forceSize(fixture!.app, 1220, 800)
    await page.waitForTimeout(500)
    await capture(testInfo, 'inbox-chip-default.png', 'chip-default', page)
  })

  test('panel opens — connected state', async ({}, testInfo) => {
    const page = fixture!.page
    const chip = inboxChip(page)

    await chip.click()
    await waitForInboxConnected(page)

    // Assert the panel is visible
    expect(await isPanelVisible(page)).toBe(true)

    // Assert NOT in disconnected/error state
    const disconnected = page.getByText('Disconnected')
    const gatewayUnavailable = page.getByText('Inbox gateway is unavailable')
    expect(await disconnected.count()).toBe(0)
    expect(await gatewayUnavailable.count()).toBe(0)

    // Assert connected: either "All clear", "Inbox not supported", or items
    const allClear = page.getByText('All clear')
    const unsupported = page.getByText('Inbox not supported')
    const partialRead = page.getByText('Partial read')
    const hasConnectedState = (await allClear.count()) > 0 || (await unsupported.count()) > 0 || (await partialRead.count()) > 0
    expect(hasConnectedState).toBe(true)

    await forceSize(fixture!.app, 1220, 800)
    await page.waitForTimeout(500)

    const stateLabel = (await allClear.count()) > 0 ? 'panel-allclear' : (await unsupported.count()) > 0 ? 'panel-unsupported' : 'panel-partial'
    await capture(testInfo, `inbox-${stateLabel}.png`, stateLabel, page)
  })

  test('panel — switch to Automation tab', async ({}, testInfo) => {
    const page = fixture!.page

    const autoTab = page.getByRole('button', { name: 'Automation' })
    await expect(autoTab).toBeVisible()
    await autoTab.click()
    await page.waitForTimeout(300)

    await capture(testInfo, 'inbox-panel-automation.png', 'panel-automation', page)
  })

  test('panel — search field visible', async ({}, testInfo) => {
    const page = fixture!.page

    const search = page.getByPlaceholder('Filter sessions…')
    await expect(search).toBeVisible()

    await capture(testInfo, 'inbox-panel-search.png', 'panel-search', page)
  })

  test('panel closes with Escape — assert against visible panel', async ({}, testInfo) => {
    const page = fixture!.page

    // Panel is open from previous test (Automation tab)
    expect(await isPanelVisible(page)).toBe(true)

    await closePanelWithEscape(page)
    await page.waitForTimeout(500)

    // Assert the panel overlay is gone, not just an arbitrary h2
    expect(await isPanelVisible(page)).toBe(false)

    await capture(testInfo, 'inbox-closed-escape.png', 'closed-escape', page)
  })

  test('panel reopens after close', async ({}, testInfo) => {
    const page = fixture!.page
    const chip = inboxChip(page)
    await chip.click()
    await waitForInboxConnected(page)

    expect(await isPanelVisible(page)).toBe(true)

    await capture(testInfo, 'inbox-reopened.png', 'reopened', page)
    await closePanelWithEscape(page)
  })

  test('narrow window — panel adapts', async ({}, testInfo) => {
    const page = fixture!.page

    await forceSize(fixture!.app, 600, 700)
    await page.waitForTimeout(800)

    const chip = inboxChip(page)
    await chip.click()
    await waitForInboxConnected(page)
    await page.waitForTimeout(500)

    await capture(testInfo, 'inbox-panel-narrow.png', 'panel-narrow', page, '600x700')
    await closePanelWithEscape(page)
  })

  test('restore normal size', async ({}, testInfo) => {
    const page = fixture!.page
    await forceSize(fixture!.app, 1220, 800)
    await page.waitForTimeout(500)

    await capture(testInfo, 'inbox-restored.png', 'restored', page)
  })
})

// ── Interaction checks ───────────────────────────────────────────────────────

test.describe('inbox interaction', () => {
  test('chip aria-pressed toggles', async () => {
    const page = fixture!.page
    const chip = inboxChip(page)

    await expect(chip).toHaveAttribute('aria-pressed', 'false')

    await chip.click()
    await waitForInboxConnected(page)
    await expect(chip).toHaveAttribute('aria-pressed', 'true')

    await closePanelWithEscape(page)
    await page.waitForTimeout(300)
    await expect(chip).toHaveAttribute('aria-pressed', 'false')
  })

  test('backdrop click closes overlay', async () => {
    const page = fixture!.page
    const chip = inboxChip(page)

    await chip.click()
    await waitForInboxConnected(page)
    expect(await isPanelVisible(page)).toBe(true)

    // Click the backdrop (fixed overlay behind the card).
    await page.locator('[role="presentation"]').click({ position: { x: 10, y: 10 } })
    await page.waitForTimeout(500)

    expect(await isPanelVisible(page)).toBe(false)
  })

  test('open/close cycle preserves app state — no accidental session creation', async () => {
    const page = fixture!.page
    const chip = inboxChip(page)

    // Capture pre-condition: session count via backend session list
    const preSessions = await page.evaluate(async () => {
      // @ts-expect-error -- desktop IPC bridge
      const rpc = window.hermesDesktop?.request ?? window.hermesDesktop?.rpc
      if (!rpc) {return null}
      try {
        const result = await rpc('session.list', {})
        return Array.isArray(result?.sessions) ? result.sessions.length : null
      } catch {
        return null
      }
    })

    // Open and close the inbox panel.
    await chip.click()
    await waitForInboxConnected(page)
    await closePanelWithEscape(page)
    await page.waitForTimeout(500)

    // Reopen — connected state should persist.
    await chip.click()
    await waitForInboxConnected(page)

    const allClear = page.getByText('All clear')
    const unsupported = page.getByText('Inbox not supported')
    const hasEmpty = (await allClear.count()) > 0 || (await unsupported.count()) > 0
    expect(hasEmpty).toBe(true)

    // Verify no accidental session was created through the inbox open/close cycle.
    const postSessions = await page.evaluate(async () => {
      // @ts-expect-error -- desktop IPC bridge
      const rpc = window.hermesDesktop?.request ?? window.hermesDesktop?.rpc
      if (!rpc) {return null}
      try {
        const result = await rpc('session.list', {})
        return Array.isArray(result?.sessions) ? result.sessions.length : null
      } catch {
        return null
      }
    })

    if (preSessions !== null && postSessions !== null) {
      expect(postSessions).toBe(preSessions)
    }

    await closePanelWithEscape(page)
  })
})
