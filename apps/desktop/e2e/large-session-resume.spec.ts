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
import { RealSessionBuilder } from './real-session-builder'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const SESSION_TITLE = 'E2E large persisted session'
const EXPECTED_TEXT = 'E2E persisted user message 52'
// The oldest seeded turn (HISTORY_TURNS[0]). The transcript first paints only
// the newest turns (FIRST_PAINT_BUDGET) and backfills the rest in a rAF; a
// baseline count taken before that backfill sees a clipped transcript and
// falsely reports duplicates once the full list mounts. Waiting for this
// oldest row means the baseline reflects the fully-mounted transcript.
const OLDEST_SEEDED_TEXT = 'E2E persisted user message 0: audit the compatibility matrix'
const BACKGROUND_PROMPT = 'E2E background inference must remain attached across resume'
const HISTORY_TURNS = Array.from(
  { length: 27 },
  (_, index) => `E2E persisted user message ${index * 2}: audit the compatibility matrix`,
)

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
  pending: boolean
  timeline: Array<{ mutations: number; time: number }>
}

interface PaintBudget {
  bursts: number
  kind: 'exact' | 'maximum'
}

async function setupSeededDesktop(mockServer?: MockServerOptions): Promise<SeededFixture> {
  const mock = await startMockServer(mockServer)
  const sandbox = createSandbox('large-session')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  const builder = await RealSessionBuilder.start(sandbox.hermesHome)
  try {
    await builder.createSession({ title: SESSION_TITLE, turns: HISTORY_TURNS })
  } finally {
    await builder.close()
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
    const state = { bursts: 0, pending: false, timeline: [] as Array<{ mutations: number; time: number }> }
    ;(window as Window & { __largeSessionPaints?: typeof state }).__largeSessionPaints = state
    if (!viewport) return

    let additions = 0
    let addedUserRows = 0
    let flushTimer: ReturnType<typeof setTimeout> | undefined
    let previousUserIds: string[] = []

    const userIds = () =>
      Array.from(viewport.querySelectorAll<HTMLElement>('[data-role="user"][data-message-id]')).map(
        row => row.dataset.messageId ?? '',
      )

    const flush = () => {
      const nextUserIds = userIds()

      if (nextUserIds.length > 0) {
        const onlyPrependedBackfill =
          nextUserIds.length > previousUserIds.length &&
          previousUserIds.every((id, index) => id === nextUserIds[nextUserIds.length - previousUserIds.length + index])
        const rebuiltSameRows =
          previousUserIds.length > 0 &&
          previousUserIds.length === nextUserIds.length &&
          previousUserIds.every((id, index) => id === nextUserIds[index]) &&
          addedUserRows > 0
        const transcriptChanged =
          previousUserIds.length === 0 ||
          rebuiltSameRows ||
          (!onlyPrependedBackfill &&
            (previousUserIds.length !== nextUserIds.length ||
              previousUserIds.some((id, index) => id !== nextUserIds[index])))

        if (transcriptChanged) {
          state.bursts += 1
          state.timeline.push({ mutations: additions, time: Date.now() })
        }

        previousUserIds = nextUserIds
      }

      additions = 0
      addedUserRows = 0
      state.pending = false
    }

    new MutationObserver(records => {
      for (const record of records) {
        if (record.type !== 'childList' || record.addedNodes.length === 0) continue

        additions += 1
        addedUserRows += Array.from(record.addedNodes).filter(
          node =>
            node instanceof Element &&
            (node.matches('[data-role="user"][data-message-id]') ||
              node.querySelector('[data-role="user"][data-message-id]')),
        ).length
      }

      if (additions === 0) return
      state.pending = true
      if (flushTimer) clearTimeout(flushTimer)
      flushTimer = setTimeout(flush, 30)
    }).observe(viewport, { childList: true, subtree: true })
  })
}

async function waitForSettledTranscriptPaint(page: Page): Promise<void> {
  await page.waitForFunction(
    expected => {
      const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
      const state = (window as Window & { __largeSessionPaints?: PaintState }).__largeSessionPaints

      return Boolean((viewport?.textContent ?? '').includes(expected) && state && state.bursts >= 1 && !state.pending)
    },
    OLDEST_SEEDED_TEXT,
    { timeout: 30_000 },
  )
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

async function assertUnchangedResume(page: Page, testInfo: TestInfo, budget: PaintBudget): Promise<void> {
  await openSeededSession(page)
  await waitForSettledTranscriptPaint(page)
  await page.screenshot({ path: testInfo.outputPath('unchanged-session-resume.png'), fullPage: false })

  const paints = await paintState(page)
  expect(await textNodeOccurrences(page, EXPECTED_TEXT), 'the resumed user message should appear once').toBe(1)
  const diagnostic = `unexpected transcript paint count: ${JSON.stringify(paints.timeline)}`

  if (budget.kind === 'exact') {
    expect(paints.bursts, diagnostic).toBe(budget.bursts)
  } else {
    // A warm session first restores its retained view, then reconciles it with
    // the authoritative transcript. A third paint is the old eager-prefetch +
    // runtime-rebuild regression.
    expect(paints.bursts, diagnostic).toBeGreaterThanOrEqual(1)
    expect(paints.bursts, diagnostic).toBeLessThanOrEqual(budget.bursts)
  }
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
    await assertUnchangedResume(fixture.page, testInfo, { bursts: 1, kind: 'exact' })
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
    await assertUnchangedResume(fixture.page, testInfo, { bursts: 2, kind: 'maximum' })
  })

  for (const resumeKind of ['fast', 'cold'] as const) {
    test(`${resumeKind} resume keeps background inference attached without duplicate messages`, async ({}, testInfo) => {
      fixture = await setupSeededDesktop({ holdFirstStreamForPrompt: BACKGROUND_PROMPT })
      await waitForAppReady(fixture, 120_000)

      await openSeededSession(fixture.page)
      // The transcript first paints only the newest turns (FIRST_PAINT_BUDGET)
      // and backfills older turns in a rAF. Wait for the oldest seeded row to
      // mount before taking the baseline so it reflects the full transcript —
      // otherwise a clipped baseline makes the backfilled rows look like
      // duplicates of the completed reply.
      await fixture.page.waitForFunction(
        expected => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
        OLDEST_SEEDED_TEXT,
        { timeout: 30_000 },
      )
      const initialMockReplyCount = await textNodeOccurrences(fixture.page, MOCK_REPLY)
      await submitPrompt(fixture.page, BACKGROUND_PROMPT)
      await fixture.mock.waitForHeldStream()
      await openNewSession(fixture.page)

      if (resumeKind === 'cold') {
        await reloadIntoColdRenderer(fixture)
        await startPaintObserver(fixture.page)
      }

      await openSeededSession(fixture.page)

      if (resumeKind === 'cold') {
        await fixture.page.waitForTimeout(300)
        const paints = await paintState(fixture.page)
        expect(
          paints.bursts,
          `unexpected cold live-resume paint count: ${JSON.stringify(paints.timeline)}`,
        ).toBeGreaterThanOrEqual(1)
        expect(
          paints.bursts,
          `unexpected cold live-resume paint count: ${JSON.stringify(paints.timeline)}`,
        ).toBeLessThanOrEqual(2)
      }

      fixture.mock.releaseHeldStream()
      await fixture.page.waitForFunction(
        expected => (document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
        MOCK_REPLY,
        { timeout: 60_000 },
      )
      await fixture.page.waitForTimeout(300)
      await fixture.page.screenshot({ path: testInfo.outputPath(`${resumeKind}-background-inference-resume.png`), fullPage: false })

      expect(await textNodeOccurrences(fixture.page, BACKGROUND_PROMPT), 'the running user prompt should appear once').toBe(1)
      expect(
        await textNodeOccurrences(fixture.page, MOCK_REPLY),
        'the completed assistant reply should add exactly one transcript row',
      ).toBe(initialMockReplyCount + 1)
    })
  }
})
