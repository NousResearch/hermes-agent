/**
 * R3 acceptance: a Codex-native assistant row persisted through SessionDB is
 * replayed by the real headless backend/gateway into an isolated built Electron
 * renderer with reasoning, commentary, canonical final, and tool boundaries intact.
 *
 * Limitation: this proves SessionDB -> backend/gateway -> Electron hydration.
 * It does not prove provider -> persistence ingestion; the one native row is
 * deliberately seeded through SessionDB after the real gateway-created session
 * is closed.
 */

import type { Locator } from '@playwright/test'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { type MockServer, startMockServer } from '../../../tests-js/scripts/mock-server'
import { RealSessionBuilder, seedCodexCommentarySession, withBackendPythonEnv } from './real-session-builder'
import { type ElectronApplication, expect, type Page, test } from './test'

const SESSION_TITLE = 'E2E commentary hydration fixture'
const SUMMARY = 'Inspecting the transcript'
const COMMENTARY = 'I’m checking the persisted turn now.'
const CANONICAL_FINAL = 'Persisted canonical answer.'
const ACTIVE_SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'

interface SeededFixture {
  app: ElectronApplication
  cleanup: () => Promise<void>
  mock: MockServer
  mockUrl: string
  page: Page
  sandbox: Sandbox
  sessionId: string
}

async function setupSeededDesktop(): Promise<SeededFixture> {
  const mock = await startMockServer()
  const sandbox = createSandbox('codex-commentary-hydration')
  let app: ElectronApplication | undefined

  try {
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)

    const builder = await RealSessionBuilder.start(sandbox.hermesHome)
    let sessionId: string

    try {
      const session = await builder.createSession({ title: SESSION_TITLE, turns: [SESSION_TITLE] })
      sessionId = session.sessionId
    } finally {
      await builder.close()
    }

    // The real gateway/AIAgent chain has created and closed the durable session.
    // Add only the sanitized native replay row through the real SessionDB API.
    await seedCodexCommentarySession(sandbox.hermesHome, sessionId)

    const launched = await launchDesktop(withBackendPythonEnv(buildAppEnv(sandbox)))
    app = launched.app

    return {
      app,
      mock,
      mockUrl: mock.url,
      page: launched.page,
      sandbox,
      sessionId,
      cleanup: async () => {
        await app?.close().catch(() => undefined)
        await mock.close()
        sandbox.cleanup()
      }
    }
  } catch (error) {
    await app?.close().catch(() => undefined)
    await mock.close().catch(() => undefined)
    sandbox.cleanup()
    throw error
  }
}

function activeSurface(page: Page): Locator {
  return page.locator(ACTIVE_SURFACE).last()
}

function sessionRow(page: Page): Locator {
  return page.locator('[data-slot="sidebar"] button').filter({ hasText: SESSION_TITLE }).first()
}

async function assertBackendDisplaySidecars(page: Page, sessionId: string): Promise<void> {
  const serialized = await page.evaluate(async id => {
    const desktop = (window as unknown as { hermesDesktop: { api<T>(request: { path: string }): Promise<T> } })
      .hermesDesktop

    return desktop.api<{ messages: Array<Record<string, unknown>> }>({
      path: `/api/sessions/${encodeURIComponent(id)}/messages?limit=500&order=latest`
    })
  }, sessionId)

  const text = JSON.stringify(serialized)
  expect(text).not.toContain('codex_reasoning_items')
  expect(text).not.toContain('codex_message_items')
  expect(text).not.toContain('E2E_ANALYSIS_SENTINEL')
  expect(text).not.toContain('E2E_ENCRYPTED_SENTINEL')
  const persistedRow = serialized.messages.find(message => Array.isArray(message.codex_display_items))
  expect(persistedRow, 'REST history should include a safe typed display projection').toBeTruthy()
}

async function openSeededSession(page: Page): Promise<void> {
  const row = sessionRow(page)
  await row.waitFor({ state: 'visible', timeout: 60_000 })
  await row.click()
  await expect(
    activeSurface(page).locator('.aui-md').filter({ hasText: COMMENTARY }),
    'the active keep-alive chat surface should hydrate the seeded commentary'
  ).toHaveCount(1, { timeout: 30_000 })
}

async function openNewSession(page: Page): Promise<void> {
  const button = page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first()
  await button.waitFor({ state: 'visible', timeout: 15_000 })
  await button.click()
  await expect(
    activeSurface(page).locator('.aui-md').filter({ hasText: COMMENTARY }),
    'a distinct new session should become the active keep-alive surface'
  ).toHaveCount(0, { timeout: 15_000 })
}

async function assertHydrationBoundaries(page: Page, label: string): Promise<void> {
  const surface = activeSurface(page)
  const thought = surface.locator('[data-slot="aui_thinking-disclosure"]:visible')
  await expect(thought, `${label}: exactly one Thought disclosure should be visible`).toHaveCount(1)
  await expect(thought, `${label}: Thought disclosure should be visible`).toBeVisible()

  const toggle = thought.locator('button[aria-expanded]').first()

  if ((await toggle.getAttribute('aria-expanded')) !== 'true') {
    await toggle.click()
  }

  await expect(toggle, `${label}: completed Thought should expand`).toHaveAttribute('aria-expanded', 'true')

  const reasoning = thought.locator('[data-slot="aui_reasoning-text"]:visible')
  await expect(reasoning, `${label}: exactly one native reasoning summary row should render`).toHaveCount(1)
  await expect(reasoning, `${label}: reasoning summary text must remain exact`).toHaveText(SUMMARY)
  await expect(thought, `${label}: Thought contains the summary`).toContainText(SUMMARY)
  await expect(thought, `${label}: commentary must not be folded into Thought`).not.toContainText(COMMENTARY)

  const commentary = surface.locator('.aui-md:visible').filter({ hasText: COMMENTARY })
  await expect(
    commentary,
    `${label}: commentary should occur once in an ordinary assistant markdown block`
  ).toHaveCount(1)
  await expect(commentary, `${label}: commentary text must remain exact`).toHaveText(COMMENTARY)
  expect(
    await commentary.evaluate(node => node.closest('[data-slot="aui_thinking-disclosure"]') === null),
    `${label}: commentary markdown must not have a Thought ancestor`
  ).toBe(true)

  const canonicalFinal = surface.locator('.aui-md:visible').filter({ hasText: CANONICAL_FINAL })
  await expect(canonicalFinal, `${label}: canonical final should occur once`).toHaveCount(1)
  await expect(canonicalFinal, `${label}: canonical final text must remain exact`).toHaveText(CANONICAL_FINAL)
  expect(
    await canonicalFinal.evaluate(node => node.closest('[data-slot="aui_thinking-disclosure"]') === null),
    `${label}: canonical final markdown must not have a Thought ancestor`
  ).toBe(true)

  const tool = surface.locator('[data-slot="tool-block"]:visible')
  await expect(tool, `${label}: exactly one visible tool row should follow the canonical final`).toHaveCount(1)

  const thoughtHandle = await thought.elementHandle()
  const commentaryHandle = await commentary.elementHandle()
  const canonicalFinalHandle = await canonicalFinal.elementHandle()
  const toolHandle = await tool.elementHandle()
  expect(
    thoughtHandle && commentaryHandle && canonicalFinalHandle && toolHandle,
    `${label}: ordered elements must exist`
  ).toBeTruthy()

  const thoughtBeforeCommentary = await thoughtHandle!.evaluate(
    (node, following) => Boolean(node.compareDocumentPosition(following as Node) & Node.DOCUMENT_POSITION_FOLLOWING),
    commentaryHandle!
  )

  const commentaryBeforeCanonicalFinal = await commentaryHandle!.evaluate(
    (node, following) => Boolean(node.compareDocumentPosition(following as Node) & Node.DOCUMENT_POSITION_FOLLOWING),
    canonicalFinalHandle!
  )

  const canonicalFinalBeforeTool = await canonicalFinalHandle!.evaluate(
    (node, following) => Boolean(node.compareDocumentPosition(following as Node) & Node.DOCUMENT_POSITION_FOLLOWING),
    toolHandle!
  )

  expect(thoughtBeforeCommentary, `${label}: Thought must precede commentary in the DOM`).toBe(true)
  expect(commentaryBeforeCanonicalFinal, `${label}: commentary must precede the canonical final in the DOM`).toBe(true)
  expect(canonicalFinalBeforeTool, `${label}: canonical final must precede the tool in the DOM`).toBe(true)

  const thoughtBox = await thought.boundingBox()
  const commentaryBox = await commentary.boundingBox()
  const canonicalFinalBox = await canonicalFinal.boundingBox()
  const toolBox = await tool.boundingBox()
  expect(
    thoughtBox && commentaryBox && canonicalFinalBox && toolBox,
    `${label}: ordered elements must have layout boxes`
  ).toBeTruthy()
  expect(thoughtBox!.y + thoughtBox!.height, `${label}: Thought and commentary must not overlap`).toBeLessThanOrEqual(
    commentaryBox!.y + 0.5
  )
  expect(
    commentaryBox!.y + commentaryBox!.height,
    `${label}: commentary and canonical final must not overlap`
  ).toBeLessThanOrEqual(canonicalFinalBox!.y + 0.5)
  expect(
    canonicalFinalBox!.y + canonicalFinalBox!.height,
    `${label}: canonical final and tool must not overlap`
  ).toBeLessThanOrEqual(toolBox!.y + 0.5)
}

test.describe('Codex commentary persisted hydration replay', () => {
  let fixture: SeededFixture | null = null

  test.afterEach(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  // eslint-disable-next-line no-empty-pattern -- Playwright requires an object-destructured fixtures parameter.
  test('preserves reasoning, commentary, final, and tool boundaries on first open and cold reload', async ({}, testInfo) => {
    // This is acceptance coverage for the existing production hydration change,
    // not a TDD RED run. Real gateway seeding plus Electron boot needs a cold-run budget.
    test.slow()
    test.setTimeout(300_000)

    fixture = await setupSeededDesktop()
    await waitForAppReady(fixture, 120_000)
    await assertBackendDisplaySidecars(fixture.page, fixture.sessionId)

    await openSeededSession(fixture.page)
    await assertHydrationBoundaries(fixture.page, 'first open')
    await fixture.page.screenshot({ path: testInfo.outputPath('codex-commentary-first-open.png'), fullPage: false })

    await fixture.page.reload()
    await waitForAppReady(fixture, 120_000)
    await openNewSession(fixture.page)
    await openSeededSession(fixture.page)
    await assertHydrationBoundaries(fixture.page, 'cold reload')
    await fixture.page.screenshot({ path: testInfo.outputPath('codex-commentary-cold-reload.png'), fullPage: false })
  })
})
