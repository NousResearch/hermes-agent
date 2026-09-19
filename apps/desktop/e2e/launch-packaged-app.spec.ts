import * as fs from 'node:fs'
import * as path from 'node:path'

import {
  PACKAGED_BINARY_PATH,
  type PackagedAppFixture,
  packagedBinaryExists,
  setupPackagedApp,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import {
  FOLLOW_UP_EMPTY_TRIGGER,
  FOLLOW_UP_LONG_PROMPT,
  FOLLOW_UP_TRIGGER,
  FOLLOW_UP_TRUNCATED_TRIGGER,
  type MockServer,
  startMockServer
} from '../../../tests-js/scripts/mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'
import { expectVisualSnapshot } from './visual-snapshot'

/**
 * E2E smoke tests for the packaged Hermes desktop app.
 *
 * Launches the real packaged Electron binary (produced by `npm run pack` →
 * `electron-builder --dir`) with BOOT_FAKE=1 and full sandbox isolation
 * (credential stripping, isolated HERMES_HOME + userData, unique app name).
 *
 * Skips if the packaged binary doesn't exist — run `npm run pack` first.
 */

let fixture: PackagedAppFixture | null = null
let mock: MockServer | null = null

test.beforeAll(async () => {
  test.skip(!packagedBinaryExists(), `Built app binary not found: ${PACKAGED_BINARY_PATH}. Run 'npm run pack' first.`)

  const hermesCliOverride = process.env.HERMES_DESKTOP_HERMES

  const hermesCli = hermesCliOverride && fs.existsSync(hermesCliOverride) ? hermesCliOverride : undefined

  const followUpSource = path.resolve(import.meta.dirname, 'fixtures', 'follow-up-plugin.js')
  mock = await startMockServer()

  try {
    const prepareSandbox = async (sandbox: PackagedAppFixture['sandbox']) => {
      const followUpDir = path.join(sandbox.hermesHome, 'desktop-plugins', 'follow-up')

      fs.mkdirSync(followUpDir, { recursive: true })
      fs.copyFileSync(followUpSource, path.join(followUpDir, 'plugin.js'))
      // The agent refuses models reporting under 64K context, and the mock
      // advertises 4096 — pin a real window so real turns can run (#98173).
      writeMockProviderConfig(
        sandbox.hermesHome,
        mock!.url,
        undefined,
        undefined,
        200_000,
      )
      writeEnvFile(sandbox.hermesHome)

      const builder = await RealSessionBuilder.start(sandbox.hermesHome)

      try {
        await builder.createSession({
          title: 'Packaged Follow-up',
          turns: [FOLLOW_UP_TRIGGER]
        })
        await builder.createSession({
          title: 'Packaged Truncated Follow-up',
          turns: [FOLLOW_UP_TRUNCATED_TRIGGER]
        })
        await builder.createSession({
          title: 'Packaged Empty Follow-up',
          turns: [FOLLOW_UP_EMPTY_TRIGGER]
        })
      } finally {
        await builder.close()
      }
    }

    fixture = await setupPackagedApp({
      env: hermesCli ? { HERMES_DESKTOP_HERMES: hermesCli } : undefined,
      keepSandbox: process.env.HERMES_E2E_KEEP_SANDBOX === '1',
      prepareSandbox
    })
  } catch (error) {
    await mock.close()
    mock = null
    throw error
  }
})

test.afterAll(async () => {
  await fixture?.cleanup()
  await mock?.close()
  fixture = null
  mock = null
})

test('window opens with the Hermes title', async () => {
  const title = await fixture!.page.title()
  expect(title).toContain('Hermes')
})

test('renderer loads and shows DOM content', async () => {
  const page = fixture!.page
  await page.waitForSelector('#root', { state: 'attached', timeout: 30_000 })
  const childCount = await page.locator('#root > *').count()
  expect(childCount).toBeGreaterThan(0)
})

test('boots to the app UI, not the QueryClient error boundary (#95560)', async () => {
  const page = fixture!.page
  await page.waitForSelector('#root', { state: 'attached', timeout: 30_000 })

  // Wait until the root has real content (boot overlay fades, app paints) —
  // the error boundary also paints, so assert on its absence explicitly.
  await page.waitForFunction(
    () => (document.getElementById('root')?.textContent ?? '').trim().length > 0,
    undefined,
    { timeout: 60_000 },
  )

  const text = await page.locator('#root').textContent()
  // The #95560 crash: a duplicate @tanstack/react-query runtime made the
  // QueryClientProvider's context invisible to useQuery, so the app hit the
  // error boundary at launch. Neither the boundary headline nor the throw
  // message may appear on a healthy boot.
  expect(text).not.toContain('No QueryClient set')
  expect(text).not.toContain('Something broke in the interface')
})

test('packaged renderer activates Follow-up and renders its directive card', async () => {
  const page = fixture!.page

  await waitForAppReady(fixture!, 120_000)
  const session = page
    .getByRole('button', { name: /Packaged Follow-up$/ })
    .filter({ hasText: 'Packaged Follow-up' })
    .first()
  await session.waitFor({ state: 'visible', timeout: 30_000 })
  await session.click()

  await expect(page.getByText(FOLLOW_UP_TRIGGER)).toBeVisible({ timeout: 30_000 })
  const panel = page.getByTestId('follow-up-panel')
  await expect(panel).toBeVisible({ timeout: 30_000 })
  const rows = panel.locator('button[aria-keyshortcuts]')
  await expect(rows).toHaveCount(2)
  const openPrButton = rows.nth(1)

  await expect(openPrButton).toBeVisible()
  await expect(openPrButton).toHaveAttribute('aria-keyshortcuts', 'Alt+2')
  await expect(openPrButton).toHaveAttribute('aria-describedby', /.+/)
  await openPrButton.focus()
  await expect(openPrButton).toBeFocused()
  await page.keyboard.press('Alt+2')

  const composer = page.locator('[data-slot="composer-rich-input"]:visible').first()

  await expect(composer).toContainText('Open a PR')
})

test('packaged Follow-up exposes a safe truncated prompt', async () => {
  const page = fixture!.page
  const session = page
    .getByRole('button', { name: /Packaged Truncated Follow-up$/ })
    .filter({ hasText: 'Packaged Truncated Follow-up' })
    .first()
  await session.waitFor({ state: 'visible', timeout: 30_000 })
  await session.click()

  const panel = page.getByTestId('follow-up-panel')
  const row = panel.locator('button[aria-keyshortcuts]').first()
  await expect(row).toBeVisible()
  await expect(row).toHaveAttribute('aria-describedby', /\S+/)
  await expect(row).toContainText('shortened')

  await row.click()
  const composer = page.locator('[data-slot="composer-rich-input"]:visible').first()
  await expect(composer).toContainText(FOLLOW_UP_LONG_PROMPT)
})

test('packaged Follow-up shows an accessible empty-directive state', async () => {
  const page = fixture!.page
  const session = page
    .getByRole('button', { name: /Packaged Empty Follow-up$/ })
    .filter({ hasText: 'Packaged Empty Follow-up' })
    .first()
  await session.waitFor({ state: 'visible', timeout: 30_000 })
  await session.click()

  const sessionSurface = page
    .locator('[data-composer-surface-id]')
    .filter({ has: page.getByTestId('directive-drop-badge') })
    .last()
  const badge = sessionSurface.getByTestId('directive-drop-badge')
  await expect(badge).toBeVisible()
  await expect(badge).toHaveAttribute('role', 'note')
  await expect(badge).toHaveAttribute('aria-label', /no prompts|không có nội dung/i)
  await expect(badge).toHaveAttribute('title', /no prompts|no usable prompt|không có nội dung/i)
})

test('packaged Follow-up keeps compact rows keyboard-focusable', async () => {
  const page = fixture!.page
  const session = page
    .getByRole('button', { name: /Packaged Follow-up$/ })
    .filter({ hasText: 'Packaged Follow-up' })
    .first()
  await session.click()

  const panel = page.getByTestId('follow-up-panel')
  const rows = panel.locator('button[aria-keyshortcuts]')
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).toHaveAttribute('aria-keyshortcuts', 'Alt+1')
  await expect(rows.nth(1)).toHaveAttribute('aria-keyshortcuts', 'Alt+2')
  await rows.nth(0).focus()
  await expect(rows.nth(0)).toBeFocused()
  await expect(rows.nth(0)).toHaveAttribute('aria-describedby', /.+/)
})

test('packaged Follow-up visual snapshot uses the compact command list', async () => {
  const page = fixture!.page
  const session = page
    .getByRole('button', { name: /Packaged Follow-up$/ })
    .filter({ hasText: 'Packaged Follow-up' })
    .first()
  await session.waitFor({ state: 'visible', timeout: 30_000 })
  await session.click()

  const panel = page.getByTestId('follow-up-panel')
  await expect(panel).toBeVisible({ timeout: 30_000 })
  await expectVisualSnapshot(page, { name: 'packaged-follow-up-compact-list', app: fixture!.app })
})

test('HUD composer remains fully inside the transparent window', async () => {
  const hudPagePromise = fixture!.app.waitForEvent('window')

  await fixture!.page.evaluate(() =>
    (
      window as typeof window & {
        hermesDesktop?: { hud?: { open: (options: { sessionId: null }) => Promise<void> } }
      }
    ).hermesDesktop?.hud?.open({ sessionId: null })
  )

  const hudPage = await hudPagePromise
  await hudPage.waitForSelector('[data-slot="composer-rich-input"]', { state: 'visible' })

  const geometry = await hudPage.evaluate(() => {
    const dock = document.querySelector<HTMLElement>('[data-slot="composer-dock"]')
    const input = document.querySelector<HTMLElement>('[data-slot="composer-rich-input"]')

    if (!dock || !input) {
      throw new Error('HUD composer did not render')
    }

    const dockRect = dock.getBoundingClientRect()
    const inputRect = input.getBoundingClientRect()

    return {
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      dockLeft: dockRect.left,
      dockRight: dockRect.right,
      dockTop: dockRect.top,
      dockBottom: dockRect.bottom,
      inputLeft: inputRect.left,
      inputRight: inputRect.right,
      inputTop: inputRect.top,
      inputBottom: inputRect.bottom,
      // The bug class this guards: a build-time CSS optimization folding the
      // dock's identity `translate` override into `transform`, leaving
      // Tailwind's standalone `translate: -50%` live and shifting the dock
      // half a window off-screen. Surface the computed value so a failure
      // says WHY the dock moved, not just that it did.
      dockTranslate: getComputedStyle(dock).translate
    }
  })

  // Horizontal containment — the composer shifted half a window left when the
  // standalone `translate: -50%` survived optimization (#82214, #82233).
  expect(geometry.dockLeft).toBeGreaterThanOrEqual(0)
  expect(geometry.inputLeft).toBeGreaterThanOrEqual(0)
  expect(geometry.dockRight).toBeLessThanOrEqual(geometry.viewportWidth)
  expect(geometry.inputRight).toBeLessThanOrEqual(geometry.viewportWidth)

  // Vertical containment — the toolbar/transcript clipping reported on
  // Windows (#82203) and macOS (#82214) is the same "composer escapes the
  // window" class on the other axis.
  expect(geometry.dockTop).toBeGreaterThanOrEqual(0)
  expect(geometry.inputTop).toBeGreaterThanOrEqual(0)
  expect(geometry.dockBottom).toBeLessThanOrEqual(geometry.viewportHeight)
  expect(geometry.inputBottom).toBeLessThanOrEqual(geometry.viewportHeight)

  // The dock's centering translate must be fully neutralized. Any live
  // percentage translate means the HUD override lost to the app's centering.
  // (Computed `translate` keeps percentages as-is, so this is assertable;
  // computed `transform` resolves to a matrix and is covered by the
  // geometric containment checks above.)
  expect(geometry.dockTranslate ?? 'none').not.toContain('%')

  await hudPage.close()
})

test('boot progress overlay fades out or shows error state', async () => {
  const page = fixture!.page
  await page.waitForFunction(
    () => {
      const root = document.getElementById('root')

      if (!root) {
        return false
      }

      const text = root.textContent ?? ''

      // Error path: boot failure overlay renders an error message.
      if (text.includes('error') || text.includes('Error') || text.includes('failed')) {
        return true
      }

      // Success path: overlay disappears and the app renders. If there's
      // no "boot" / "starting" / "installing" text visible, boot has
      // completed (either to the main UI or to onboarding).
      const bootIndicators = ['starting', 'resolving', 'spawning', 'waiting', 'installing']
      const lower = text.toLowerCase()

      return !bootIndicators.some(word => lower.includes(word))
    },
    undefined,
    { timeout: 60_000 }
  )
})

test('can capture a screenshot for the CI artifact', async () => {
  if (!fixture) {
    test.skip(true, 'Previous test failed — no app running')

    return
  }

  // Visual snapshot — won't fail on diff, just logs + generates diff image
  await expectVisualSnapshot(fixture!.page, { name: 'packaged-app-booted', timeout: 10_000, app: fixture!.app })
})
