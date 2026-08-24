/**
 * Regression coverage for an attached image in a durable session. The gateway
 * persists the turn, the builder exits, and desktop renders it from SessionDB
 * for the first time — the "quit and relaunch" case, where the transcript used
 * to come back as vision-enrichment prose instead of a thumbnail.
 *
 * The fixture pins `image_input_mode: native` because that is the majority
 * routing path (any vision-capable model) and the one where a text-only
 * persist override is silently dropped. The image also sits behind directory
 * and file names containing spaces, mirroring the macOS composer's
 * `~/Library/Application Support/...` staging path.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { type MockServer, startMockServer } from '../../../tests-js/scripts/mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { type ElectronApplication, expect, type Page, test } from './test'

// The builder-provided title now labels the sidebar row directly (seeded
// sessions no longer fall back to the first-user-message preview).
const SESSION_TITLE = 'E2E attached image session'
const CAPTION = 'E2E attached image must survive a relaunch'
const IMAGE_DIR = 'Application Support/e2e shots'
const IMAGE_NAME = 'e2e capture.png'
const NATIVE_IMAGE_CONFIG = 'agent:\n  image_input_mode: native'

/** A 160x100 framed magenta block — small, but visible in the screenshots. */
const PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAKAAAABkCAIAAACO1KzYAAAA30lEQVR42u3dwQ2AIBAAQTAWAx1iBXYI7diCuWhEMvP2dZsj+CL30hLr2oxAYARGYARGYARGYIERGIH53n7nozpOk5rQqIcNdkQjMAIjMNPeomP3N54V+5exwY5oBEZgBEZgBEZggREYgREYgREYgQVGYARGYARGYARGYIERGIERGIERGIEFRmAERmAERmAEFhiBERiBERiBERiBBUZgBEZgBEZgBBYYgREYgREYgRFYYCMQGIERmPSj94Njb9ligxEYgRFYYJaQe2mmYIMRGIERGIERGIEFRmAERmDedAFtjAtAGWDnoAAAAABJRU5ErkJggg=='

interface SeededFixture {
  app: ElectronApplication
  mock: MockServer
  page: Page
  sandbox: Sandbox
  cleanup: () => Promise<void>
}

function writeImage(sandbox: Sandbox): string {
  const dir = path.join(sandbox.root, IMAGE_DIR)
  fs.mkdirSync(dir, { recursive: true })

  const imagePath = path.join(dir, IMAGE_NAME)
  fs.writeFileSync(imagePath, Buffer.from(PNG_BASE64, 'base64'))

  return imagePath
}

async function setupSeededDesktop(): Promise<SeededFixture> {
  const mock = await startMockServer()
  const sandbox = createSandbox('image-attachment')
  writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, NATIVE_IMAGE_CONFIG)
  writeEnvFile(sandbox.hermesHome)

  const builder = await RealSessionBuilder.start(sandbox.hermesHome)

  try {
    await builder.createSession({
      title: SESSION_TITLE,
      turns: [{ images: [writeImage(sandbox)], text: CAPTION }],
    })
  } finally {
    await builder.close()
  }

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  return {
    app,
    mock,
    page,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    },
  }
}

function sessionRow(page: Page) {
  return page.locator('[data-slot="sidebar"] button').filter({ hasText: SESSION_TITLE }).first()
}

// Inactive tabs stay mounted under a data-pane-hidden ancestor. Match the
// renderer's keep-alive visibility policy instead of relying on DOM order.
const SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'

function activeViewportText(surfaceSelector: string): string {
  const surfaces = document.querySelectorAll(surfaceSelector)

  return surfaces[surfaces.length - 1]?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? ''
}

async function openSeededSession(page: Page): Promise<void> {
  const row = sessionRow(page)
  await row.waitFor({ state: 'visible', timeout: 60_000 })
  await row.click()
  await page.waitForFunction(
    ([expected, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const text = surfaces[surfaces.length - 1]?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? ''

      return text.includes(expected)
    },
    [CAPTION, SURFACE] as [string, string],
    { timeout: 30_000 },
  )
}

/**
 * The sidebar "+" opens a NEW TAB beside the current chat instead of replacing
 * it, so the seeded session stays mounted in its own surface. Assert the new
 * surface is empty rather than waiting for the old caption to leave the page.
 */
async function openNewSession(page: Page): Promise<void> {
  await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()
  await page.waitForFunction(
    ([expected, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const text = surfaces[surfaces.length - 1]?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? ''

      return surfaces.length > 0 && !text.includes(expected)
    },
    [CAPTION, SURFACE] as [string, string],
    { timeout: 15_000 },
  )
}

async function transcriptText(page: Page): Promise<string> {
  return page.evaluate(activeViewportText, SURFACE)
}

async function assertRendersThumbnail(page: Page, label: string): Promise<void> {
  const thumbnail = page.locator('[data-slot="aui_directive-image"] img')
  await expect(thumbnail, `${label}: the attachment should render as an image`).toHaveCount(1)
  await expect(thumbnail, `${label}: the thumbnail should resolve off disk`).toHaveAttribute('src', /^data:image\//)

  const text = await transcriptText(page)
  expect(text, `${label}: the caption should survive alongside the image`).toContain(CAPTION)
  // A broken ref falls back to a chip whose label leaks the path, and a
  // flattened multimodal turn leaves the agent's placeholder behind.
  expect(text, `${label}: the raw image path should not leak into the transcript`).not.toContain(IMAGE_NAME)
  expect(text, `${label}: the image directive should not render literally`).not.toContain('@image:')
  expect(text, `${label}: the flattening placeholder should not render`).not.toContain('[screenshot]')
}

test.describe('attached image resume', () => {
  let fixture: SeededFixture | null = null

  test.afterEach(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  test('renders a persisted attachment as a thumbnail on first open and after a cold reload', async ({}, testInfo) => {
    // Seeding through the real gateway plus two full app boots does not fit the
    // default per-test budget on a cold runner.
    test.slow()

    fixture = await setupSeededDesktop()
    await waitForAppReady(fixture, 120_000)

    // The sidebar labels a seeded session by its title. Whatever the label
    // source, an attachment directive must never leak into it as a file path.
    const row = sessionRow(fixture.page)
    await row.waitFor({ state: 'visible', timeout: 60_000 })

    const label = (await row.textContent())?.trim() ?? ''
    expect(label.startsWith(SESSION_TITLE), `sidebar label should open with the title: ${label}`).toBe(true)
    expect(label, `sidebar label should not leak the image path: ${label}`).not.toContain(IMAGE_NAME)
    expect(label, `sidebar label should not render the directive: ${label}`).not.toContain('@image:')

    await openSeededSession(fixture.page)
    await assertRendersThumbnail(fixture.page, 'first open')
    await fixture.page.screenshot({ path: testInfo.outputPath('attachment-first-open.png') })

    // A reload drops every cached attachment ref, so the transcript has to come
    // back from the persisted turn alone.
    await fixture.page.reload()
    await waitForAppReady(fixture, 120_000)
    await openNewSession(fixture.page)

    await openSeededSession(fixture.page)
    await assertRendersThumbnail(fixture.page, 'cold reload')
    await fixture.page.screenshot({ path: testInfo.outputPath('attachment-cold-reload.png') })
  })
})
