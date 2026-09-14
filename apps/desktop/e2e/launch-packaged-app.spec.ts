import fs from 'node:fs'
import path from 'node:path'

import { expect, test } from './test'

import {
  PACKAGED_BINARY_PATH,
  type PackagedAppFixture,
  packagedBinaryExists,
  setupPackagedApp,
} from './fixtures'
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

test.beforeAll(async () => {
  test.skip(
    !packagedBinaryExists(),
    `Built app binary not found: ${PACKAGED_BINARY_PATH}. Run 'npm run pack' first.`,
  )

  fixture = await setupPackagedApp()
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
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

test('Desktop IPC and the loopback controller expose the same resource and event identities', async () => {
  const page = fixture!.page
  const ipcSnapshot = await page.evaluate(async () =>
    (window as typeof window & {
      hermesDesktop?: {
        workstationBrowser?: {
          resources: () => Promise<unknown>
        }
      }
    }).hermesDesktop?.workstationBrowser?.resources()
  ) as {
    schema_version: number
    runtime: string
    resources: Array<{ resource_id: string; resource_type: string; state: Record<string, unknown> }>
  } | undefined

  expect(ipcSnapshot?.schema_version).toBe(1)
  expect(ipcSnapshot?.runtime).toBe('electron-chromium')

  const controlPath = path.join(fixture!.sandbox.root, 'workstation', 'Runtime', 'browser-control.json')
  await expect.poll(() => fs.existsSync(controlPath), { timeout: 15_000 }).toBe(true)
  const control = JSON.parse(fs.readFileSync(controlPath, 'utf8')) as { url: string; token: string }
  const response = await fetch(`${control.url}/resources`, {
    headers: { Authorization: `Bearer ${control.token}` }
  })
  expect(response.ok).toBe(true)
  const controllerSnapshot = await response.json() as {
    success: boolean
    schema_version: number
    runtime: string
    resources: Array<{ resource_id: string; resource_type: string; state: Record<string, unknown> }>
  }

  expect(controllerSnapshot.success).toBe(true)
  expect(controllerSnapshot.schema_version).toBe(ipcSnapshot?.schema_version)
  expect(controllerSnapshot.runtime).toBe(ipcSnapshot?.runtime)
  expect(controllerSnapshot.resources.map(resource => [resource.resource_type, resource.resource_id]))
    .toEqual(ipcSnapshot?.resources.map(resource => [resource.resource_type, resource.resource_id]))

  const ipcEvents = await page.evaluate(async () =>
    (window as typeof window & {
      hermesDesktop?: {
        workstationBrowser?: {
          events: (taskId?: string | null, limit?: number) => Promise<unknown>
        }
      }
    }).hermesDesktop?.workstationBrowser?.events(null, 200)
  ) as {
    schema_version: number
    runtime: string
    task_id: string | null
    events: Array<{ event_id: string; task_id: string; session_id: string; timestamp: string }>
  } | undefined

  expect(ipcEvents?.schema_version).toBe(1)
  expect(ipcEvents?.runtime).toBe('electron-chromium')

  const eventsResponse = await fetch(`${control.url}/events?limit=200`, {
    headers: { Authorization: `Bearer ${control.token}` }
  })
  expect(eventsResponse.ok).toBe(true)
  const controllerEvents = await eventsResponse.json() as {
    success: boolean
    schema_version: number
    runtime: string
    task_id: string | null
    events: Array<{ event_id: string; task_id: string; session_id: string; timestamp: string }>
  }

  expect(controllerEvents.success).toBe(true)
  expect(controllerEvents.schema_version).toBe(ipcEvents?.schema_version)
  expect(controllerEvents.runtime).toBe(ipcEvents?.runtime)
  expect(controllerEvents.task_id).toBe(ipcEvents?.task_id)
  expect(controllerEvents.events).toEqual(ipcEvents?.events)
})

test('HUD composer remains fully inside the transparent window', async () => {
  const hudPagePromise = fixture!.app.waitForEvent('window')

  await fixture!.page.evaluate(() =>
    (window as typeof window & {
      hermesDesktop?: { hud?: { open: (options: { sessionId: null }) => Promise<void> } }
    }).hermesDesktop?.hud?.open({ sessionId: null })
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
      dockTranslate: getComputedStyle(dock).translate,
    }
  })

  // Native window scaling can leave CSS geometry a fraction of a pixel past
  // the integer viewport edge. Keep the tolerance below any meaningful
  // off-screen regression while avoiding false failures from subpixel layout.
  const containmentTolerance = 1

  // Horizontal containment — the composer shifted half a window left when the
  // standalone `translate: -50%` survived optimization (#82214, #82233).
  expect(geometry.dockLeft).toBeGreaterThanOrEqual(0)
  expect(geometry.inputLeft).toBeGreaterThanOrEqual(0)
  expect(geometry.dockRight).toBeLessThanOrEqual(geometry.viewportWidth + containmentTolerance)
  expect(geometry.inputRight).toBeLessThanOrEqual(geometry.viewportWidth + containmentTolerance)

  // Vertical containment — the toolbar/transcript clipping reported on
  // Windows (#82203) and macOS (#82214) is the same "composer escapes the
  // window" class on the other axis.
  expect(geometry.dockTop).toBeGreaterThanOrEqual(0)
  expect(geometry.inputTop).toBeGreaterThanOrEqual(0)
  expect(geometry.dockBottom).toBeLessThanOrEqual(geometry.viewportHeight + containmentTolerance)
  expect(geometry.inputBottom).toBeLessThanOrEqual(geometry.viewportHeight + containmentTolerance)

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

      return !bootIndicators.some((word) => lower.includes(word))
    },
    undefined,
    { timeout: 60_000 },
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
