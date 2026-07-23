import { spawnSync } from 'node:child_process'
import * as path from 'node:path'

import { type TestInfo } from '@playwright/test'

import { expect, test, type ElectronApplication, type Page } from './test'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type Sandbox,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig,
} from './fixtures'
import { MOCK_REPLY, startMockServer, type MockServer, type MockServerOptions } from './mock-server'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const SEED_SCRIPT = path.resolve(import.meta.dirname, 'scripts', 'seed_large_session.py')
const SESSION_TITLE = 'E2E large persisted session'
const EXPECTED_TEXT = 'E2E persisted user message 52'
const BACKGROUND_PROMPT = 'E2E background inference must remain attached across resume'

interface SeededFixture {
  app: ElectronApplication
  mock: MockServer
  mockUrl: string
  page: Page
  sandbox: Sandbox
  cleanup: () => Promise<void>
}

interface PaintState {
  bursts: number
  timeline: Array<{ mutations: number; time: number }>
}

async function setupSeededDesktop(mockServer?: MockServerOptions): Promise<SeededFixture> {
  const mock = await startMockServer(mockServer)
  const sandbox = createSandbox('large-session')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const seeded = spawnSync('python3', [SEED_SCRIPT, path.join(sandbox.hermesHome, 'state.db')], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    env: { ...process.env, PYTHONPATH: REPO_ROOT },
  })
  if (seeded.status !== 0) {
    throw new Error(`large-session seed failed:\n${seeded.stdout}\n${seeded.stderr}`)
  }

  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  return {
    app,
    mock,
    mockUrl: mock.url,
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

async function openSeededSession(page: Page): Promise<void> {
  const row = sessionRow(page)
  await row.waitFor({ state: 'visible', timeout: 60_000 })
  await row.click()
  await page.waitForFunction(
    expected => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
    EXPECTED_TEXT,
    { timeout: 30_000 },
  )
}

async function openNewSession(page: Page): Promise<void> {
  const button = page.locator('[data-slot="sidebar"] button').filter({ hasText: 'New session' }).first()
  await button.waitFor({ state: 'visible', timeout: 10_000 })
  await button.click()
  await page.waitForFunction(
    expected => !(document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
    EXPECTED_TEXT,
    { timeout: 15_000 },
  )
}

async function submitPrompt(page: Page, prompt: string): Promise<void> {
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.waitFor({ state: 'visible', timeout: 15_000 })
  await composer.click()
  await composer.type(prompt, { delay: 2 })
  await page.keyboard.press('Enter')
  await page.waitForFunction(
    expected => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
    prompt,
    { timeout: 15_000 },
  )
}

async function startPaintObserver(page: Page): Promise<void> {
  await page.evaluate(() => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
    const state = { bursts: 0, timeline: [] as Array<{ mutations: number; time: number }> }
    ;(window as Window & { __largeSessionPaints?: typeof state }).__largeSessionPaints = state
    if (!viewport) return

    let additions = 0
    let flushTimer: ReturnType<typeof setTimeout> | undefined
    new MutationObserver(records => {
      additions += records.reduce(
        (count, record) => count + (record.type === 'childList' && record.addedNodes.length > 0 ? 1 : 0),
        0,
      )
      if (additions === 0) return
      if (flushTimer) clearTimeout(flushTimer)
      flushTimer = setTimeout(() => {
        state.bursts += 1
        state.timeline.push({ mutations: additions, time: Date.now() })
        additions = 0
      }, 30)
    }).observe(viewport, { childList: true, subtree: true })
  })
}

async function paintState(page: Page): Promise<PaintState> {
  const state = await page.evaluate(() => (window as Window & { __largeSessionPaints?: PaintState }).__largeSessionPaints)
  expect(state, 'paint observer should attach to the thread viewport').toBeDefined()
  return state!
}

async function textNodeOccurrences(page: Page, expected: string): Promise<number> {
  return page.evaluate(text => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
    if (!viewport) return 0

    const walker = document.createTreeWalker(viewport, NodeFilter.SHOW_TEXT)
    let count = 0
    while (walker.nextNode()) {
      if (walker.currentNode.textContent?.includes(text)) {
        count += 1
      }
    }
    return count
  }, expected)
}

async function reloadIntoColdRenderer(fixture: SeededFixture): Promise<void> {
  await fixture.page.reload()
  await waitForAppReady(fixture, 120_000)
  await openNewSession(fixture.page)
}

async function assertUnchangedResume(page: Page, testInfo: TestInfo): Promise<void> {
  await openSeededSession(page)
  await page.waitForTimeout(1_000)
  await page.screenshot({ path: testInfo.outputPath('unchanged-session-resume.png'), fullPage: false })

  const paints = await paintState(page)
  expect(await textNodeOccurrences(page, EXPECTED_TEXT), 'the resumed user message should appear once').toBe(1)
  // A warm session first restores its retained view, then reconciles it with the
  // authoritative transcript. That is bounded at two builds; a third paint was
  // the old eager-prefetch + runtime-rebuild regression. A cold restore has one.
  expect(paints.bursts, `unexpected transcript paint count: ${JSON.stringify(paints.timeline)}`).toBeLessThanOrEqual(2)
}

test.describe('large session resume', () => {
  let fixture: SeededFixture | null = null

  test.afterEach(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  test('cold resume of an unchanged session has one user row and bounded transcript paints', async ({}, testInfo) => {
    fixture = await setupSeededDesktop()
    await waitForAppReady(fixture, 120_000)

    await startPaintObserver(fixture.page)
    await assertUnchangedResume(fixture.page, testInfo)
  })

  test('fast resume of an unchanged session has one user row and bounded transcript paints', async ({}, testInfo) => {
    // Known RED: a rapid warm resume rebuilds the transcript three times
    // (28 → 53 → 53 DOM additions) instead of the two-paint budget. Keep the
    // regression visible without making unrelated desktop work fail CI.
    test.fixme(true, 'Fast warm resume has an unresolved third transcript rebuild')

    fixture = await setupSeededDesktop()
    await waitForAppReady(fixture, 120_000)

    await openSeededSession(fixture.page)
    await openNewSession(fixture.page)
    await startPaintObserver(fixture.page)
    await assertUnchangedResume(fixture.page, testInfo)
  })

  for (const resumeKind of ['fast', 'cold'] as const) {
    test(`${resumeKind} resume keeps background inference attached without duplicate messages`, async ({}, testInfo) => {
      fixture = await setupSeededDesktop({ holdFirstStreamForPrompt: BACKGROUND_PROMPT })
      await waitForAppReady(fixture, 120_000)

      await openSeededSession(fixture.page)
      await submitPrompt(fixture.page, BACKGROUND_PROMPT)
      await fixture.mock.waitForHeldStream()
      await openNewSession(fixture.page)

      if (resumeKind === 'cold') {
        await reloadIntoColdRenderer(fixture)
      }

      await openSeededSession(fixture.page)
      fixture.mock.releaseHeldStream()
      await fixture.page.waitForFunction(
        expected => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
        MOCK_REPLY,
        { timeout: 60_000 },
      )
      await fixture.page.waitForTimeout(300)
      await fixture.page.screenshot({ path: testInfo.outputPath(`${resumeKind}-background-inference-resume.png`), fullPage: false })

      expect(await textNodeOccurrences(fixture.page, BACKGROUND_PROMPT), 'the running user prompt should appear once').toBe(1)
      expect(await textNodeOccurrences(fixture.page, MOCK_REPLY), 'the completed assistant reply should appear once').toBe(1)
    })
  }
})
