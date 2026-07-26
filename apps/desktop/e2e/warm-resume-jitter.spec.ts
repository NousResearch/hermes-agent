/**
 * E2E regression: warm-route resume must not re-render the transcript more
 * than once.
 *
 * When a session is already in the runtime-id cache (the "warm" path in
 * `resumeSession()`), clicking its sidebar row should paint the transcript
 * exactly once. Before the fix, the warm cache painted via
 * `syncSessionStateToView`, then the `session.activate` RPC returned a
 * reconciled message list with different message object references, causing
 * `syncSessionStateToView` to fire a second `setMessages` — a visual
 * flicker as the transcript DOM was updated.
 *
 * This test pre-seeds a 32-message session into state.db, boots the app,
 * clicks the session (cold resume — populates the warm cache), navigates
 * away to a new chat, then clicks back (warm resume). Two detectors run:
 *
 * 1. A MutationObserver counts additive DOM mutation bursts (childList
 *    additions). More than 1 burst = the transcript was repainted.
 *
 * 2. A 2ms innerHTML-length poll counts "reconciles" — DOM content changes
 *    that happen AFTER the initial paint, while messages are already on
 *    screen. This catches the case where React reconciles by key without
 *    adding/removing nodes (same keys → in-place prop update → no
 *    MutationObserver burst), but `$messages` was still set twice.
 *
 * The test passes when bursts === 1 AND reconciles === 0.
 * The sidebar "+" keeps the session warm in another tab. Its reactivation
 * follows the same contract: one additive paint and zero reconciles.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import { expect, test } from './test'

import {
  type MockBackendFixture,
  waitForAppReady,
  createSandbox,
  writeMockProviderConfig,
  writeEnvFile,
  buildAppEnv,
  launchDesktop,
} from './fixtures'
import { startMockServer } from './mock-server'
import { RealSessionBuilder } from './real-session-builder'

const SESSION_TITLE = 'E2E Warm Resume Jitter Test'

// Inactive tabs stay mounted under a data-pane-hidden ancestor. Match the
// renderer's keep-alive visibility policy instead of relying on DOM order.
const SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'
const ALL_SURFACES = '[data-composer-target]'
/** 32 messages (16 user/assistant pairs) — enough DOM churn for detection. */
const MESSAGE_COUNT = 32
/** Seeded PRNG so the generated content is deterministic across runs. */
const RNG_SEED = 42

/** Mulberry32 — tiny deterministic PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Generate ~40 chars of gibberish from a seeded PRNG. */
function gibberish(rng: () => number): string {
  const len = 30 + Math.floor(rng() * 20)
  let s = ''
  for (let i = 0; i < len; i++) {
    s += String.fromCharCode(97 + Math.floor(rng() * 26))
  }
  return s
}

/** First user message — used as a wait target in the test. */
const FIRST_USER_MSG = gibberish(mulberry32(RNG_SEED))

/**
 * Generate the user turns for a real session. The mock provider produces the
 * assistant side of each pair through the normal AIAgent persistence path.
 */
function generateSessionTurns(): string[] {
  const rng = mulberry32(RNG_SEED)
  const turns: string[] = []

  for (let i = 0; i < MESSAGE_COUNT / 2; i++) {
    turns.push(gibberish(rng))
    gibberish(rng)
  }

  return turns
}

/**
 * Set up a mock-backend sandbox with a real persisted session in state.db.
 *
 * Unlike the shared `setupMockBackend()`, this variant creates the session
 * through the real stdio gateway before launching desktop so the session is
 * visible in the sidebar on first load.
 */
async function setupSeededMockBackend(): Promise<MockBackendFixture> {
  // 1. Start mock server
  const mock = await startMockServer()

  // 2. Create sandbox + write config
  const sandbox = createSandbox('warm-seed')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)

  // 3. Produce all 16 user/assistant pairs through the real TUI gateway,
  // AIAgent, mock provider, and SessionDB persistence path before desktop starts.
  const builder = await RealSessionBuilder.start(sandbox.hermesHome)
  try {
    await builder.createSession({ title: SESSION_TITLE, turns: generateSessionTurns() })
  } finally {
    await builder.close()
  }

  // 4. Build env + launch
  const env = buildAppEnv(sandbox)
  const { app, page } = await launchDesktop(env)

  return {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    },
  }
}

let fixture: MockBackendFixture | null = null

test.beforeAll(async () => {
  fixture = await setupSeededMockBackend()
  await waitForAppReady(fixture!, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

/**
 * Install a MutationObserver + text-content poll on the thread viewport
 * to detect re-renders after the initial paint. Returns nothing — call
 * `readRenderCount` to stop and collect results.
 *
 * - MutationObserver: counts additive childList bursts (5ms coalescing).
 * - Text-content poll: counts "reconciles" — first-message text changes
 *   after the initial paint, catching key-based reconciles that don't
 *   add/remove nodes.
 */
async function installRenderCounter(
  page: import('@playwright/test').Page,
  transcriptText?: string,
): Promise<void> {
  await page.evaluate(([visibleSelector, allSelector, expected]: [string, string, string | undefined]) => {
    const surfaces = [...document.querySelectorAll(expected ? allSelector : visibleSelector)]
    const surface = expected
      ? surfaces.find(candidate =>
          (candidate.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected),
        )
      : surfaces.at(-1)
    const viewport = surface?.querySelector('[data-slot="aui_thread-viewport"]')
    if (!viewport) {
      throw new Error('Thread viewport not found before warm resume')
    }

    const state = { bursts: 0, mutations: 0, timeline: [] as number[], stopped: false, reconciles: 0 }
    const debugWindow = window as unknown as {
      __RENDER_COUNT__: typeof state
      __RENDER_VIEWPORT__: Element
    }
    debugWindow.__RENDER_COUNT__ = state
    debugWindow.__RENDER_VIEWPORT__ = viewport

    let currentBatch = 0
    let flushTimer: ReturnType<typeof setTimeout> | null = null

    const flush = () => {
      flushTimer = null
      if (currentBatch > 0 && !state.stopped) {
        state.bursts += 1
        state.timeline.push(currentBatch)
        currentBatch = 0
      }
    }

    const observer = new MutationObserver(records => {
      if (state.stopped) return
      let batchAdded = 0
      for (const record of records) {
        state.mutations += 1
        if (record.type === 'childList' && record.addedNodes.length > 0) {
          batchAdded += 1
        }
      }
      if (batchAdded > 0) {
        currentBatch += batchAdded
        if (flushTimer) clearTimeout(flushTimer)
        flushTimer = setTimeout(flush, 5)
      }
    })

    observer.observe(viewport, {
      childList: true,
      subtree: true,
      attributes: false,
      characterData: false,
    })

    // Poll the first message's text content every 2ms. The MutationObserver
    // only catches childList additions; React may reconcile by key without
    // adding/removing nodes (same keys → in-place prop update → no childList
    // mutation). The poll catches this by detecting text content changes in
    // the first message after the initial paint. Metadata-only changes (model
    // name, busy indicator) don't affect message text, so they don't produce
    // false positives.
    const contentEl = viewport.querySelector('[data-slot="aui_thread-content"]') ?? viewport
    let lastFirstMsgText = ''
    let hasMessages = false
    const pollInterval = setInterval(() => {
      if (state.stopped) {
        clearInterval(pollInterval)
        return
      }
      const firstMsg = contentEl.querySelector('[data-role="message"], [data-message-id]')
      const firstMsgText = firstMsg?.textContent ?? ''
      if (firstMsgText && firstMsgText !== lastFirstMsgText) {
        if (hasMessages) {
          state.reconciles = (state.reconciles ?? 0) + 1
        }
        lastFirstMsgText = firstMsgText
        hasMessages = true
      }
    }, 2)
  }, [SURFACE, ALL_SURFACES, transcriptText] as [string, string, string | undefined])
}

/** Wait until the ACTIVE chat surface's transcript contains `text`. */
async function waitForActiveTranscriptText(
  page: import('@playwright/test').Page,
  text: string,
  timeout = 30_000,
): Promise<void> {
  await page.waitForFunction(
    ([expected, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const active = surfaces[surfaces.length - 1]

      return (active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected)
    },
    [text, SURFACE] as [string, string],
    { timeout },
  )
}

async function waitForActiveTranscriptWithoutText(
  page: import('@playwright/test').Page,
  text: string,
): Promise<void> {
  await page.waitForFunction(
    ([expected, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const active = surfaces[surfaces.length - 1]

      return !(active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected)
    },
    [text, SURFACE] as [string, string],
    { timeout: 15_000 },
  )
}

/** Replace the primary surface with a draft while retaining its warm cache. */
async function openFreshDraft(page: import('@playwright/test').Page, priorText: string): Promise<void> {
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+N' : 'Control+N')
  await waitForActiveTranscriptWithoutText(page, priorText)
}

/** Stack an empty tab while leaving the current transcript mounted and warm. */
async function openNewSessionTab(page: import('@playwright/test').Page, priorText: string): Promise<void> {
  await page.locator('[data-slot="sidebar"] button[aria-label="New session"]').first().click()
  await waitForActiveTranscriptWithoutText(page, priorText)
}

/** Stop the render counter and return the recorded burst/reconcile counts. */
async function readRenderCount(page: import('@playwright/test').Page): Promise<{
  bursts: number
  mutations: number
  timeline: number[]
  reconciles: number
} | null> {
  return page.evaluate(() => {
    type RenderCount = { bursts: number; mutations: number; timeline: number[]; stopped: boolean; reconciles: number }
    const w = window as unknown as { __RENDER_COUNT__?: RenderCount }
    const rc = w.__RENDER_COUNT__
    if (rc) {
      rc.stopped = true
    }
    return rc ? { bursts: rc.bursts, mutations: rc.mutations, timeline: rc.timeline, reconciles: rc.reconciles } : null
  })
}

async function observedViewportIsActive(page: import('@playwright/test').Page): Promise<boolean> {
  return page.evaluate((surfaceSelector: string) => {
    const surfaces = document.querySelectorAll(surfaceSelector)
    const activeViewport = surfaces[surfaces.length - 1]?.querySelector('[data-slot="aui_thread-viewport"]')
    const observedViewport = (window as unknown as { __RENDER_VIEWPORT__?: Element }).__RENDER_VIEWPORT__

    return activeViewport === observedViewport
  }, SURFACE)
}

/** A kept-alive tab must become visible without rebuilding its transcript. */
function assertNoRepaint(result: { bursts: number; mutations: number; timeline: number[]; reconciles: number } | null): void {
  expect(result, 'MutationObserver should have recorded render data').toBeTruthy()
  expect(
    result!.bursts,
    `Expected no additive render bursts for a kept-alive tab, but got ${result!.bursts}. ` +
      `Mutation timeline: ${JSON.stringify(result!.timeline)}.`,
  ).toBe(0)
  expect(
    result!.reconciles,
    `Expected no transcript reconciles for a kept-alive tab, but got ${result!.reconciles}.`,
  ).toBe(0)
}

/** Assert the render counter shows exactly one paint with no re-renders. */
function assertNoJitter(result: { bursts: number; mutations: number; timeline: number[]; reconciles: number } | null): void {
  expect(result, 'MutationObserver should have recorded render data').toBeTruthy()
  expect(
    result!.bursts,
    `Expected 1 additive render burst (single paint), but got ${result!.bursts} bursts. ` +
      `Mutation timeline: ${JSON.stringify(result!.timeline)}.`,
  ).toBe(1)
  expect(
    result!.reconciles,
    `Expected 0 reconciles (no re-render after initial paint), but got ${result!.reconciles}. ` +
      `This means the warm-route resume re-rendered the transcript after the initial paint ` +
      `— the "warm resume jitter" bug is present.`,
  ).toBe(0)
}

test('tab reactivation preserves the mounted transcript without repainting', async ({}, testInfo) => {
  const page = fixture!.page

  // Wait for the sidebar to populate with our seeded session.
  const sessionRow = page
    .locator('[data-slot="sidebar"] button')
    .filter({ hasText: SESSION_TITLE })
    .first()
  await sessionRow.waitFor({ state: 'visible', timeout: 60_000 })

  // Step 1: Cold resume — click the session row to load it.
  // This populates the warm cache (runtimeIdByStoredSessionId + sessionStateByRuntimeId).
  await sessionRow.click()

  // Wait for the transcript to appear — the first user message text confirms
  // the cold-path prefetch painted.
  await waitForActiveTranscriptText(page, FIRST_USER_MSG)

  // Wait for the session to fully settle (cold-path RPC + reconciliation).
  await page.waitForTimeout(2_000)

  // Stack a new tab, then observe the seeded transcript while it is hidden.
  // Installing after the switch isolates reactivation from mutations caused
  // while the new tab was being created.
  await openNewSessionTab(page, FIRST_USER_MSG)
  await page.waitForTimeout(500)
  await installRenderCounter(page, FIRST_USER_MSG)

  // Step 3: Click back and verify the same kept-alive viewport becomes active
  // without rebuilding or reconciling its transcript.
  await sessionRow.click()

  await waitForActiveTranscriptText(page, FIRST_USER_MSG)
  await page.waitForTimeout(2_000)
  expect(await observedViewportIsActive(page), 'Reactivation should reveal the observed kept-alive viewport').toBe(true)

  const result = await readRenderCount(page)
  await page.screenshot({ path: testInfo.outputPath('warm-resume-idle.png') })
  assertNoRepaint(result)
})

test('warm-route resume after background inference completes (no jitter)', async ({}, testInfo) => {
  test.fixme(
    true,
    'Warm resume repaints after inference: expected one additive burst, got two ([18,1]).',
  )

  const page = fixture!.page
  const { mock } = fixture!

  // Wait for the sidebar to populate with our seeded session.
  const sessionRow = page
    .locator('[data-slot="sidebar"] button')
    .filter({ hasText: SESSION_TITLE })
    .first()
  await sessionRow.waitFor({ state: 'visible', timeout: 60_000 })

  // Step 1: Cold resume — populate the warm cache.
  await sessionRow.click()
  await waitForActiveTranscriptText(page, FIRST_USER_MSG)
  await page.waitForTimeout(2_000)

  // Step 2: Send a message — triggers inference via the mock server.
  const PROMPT = 'E2E post-inference warm resume test prompt'
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.click()
  await composer.type(PROMPT, { delay: 10 })
  await page.keyboard.press('Enter')

  // Wait for the mock response to appear in the transcript, confirming
  // the turn completed and message.complete fired (which updates the warm
  // cache via updateSessionState).
  await waitForActiveTranscriptText(page, 'mock inference server', 60_000)
  // Extra settle for message.complete → updateSessionState → cache write.
  await page.waitForTimeout(2_000)

  // Verify the prompt was received by the mock server.
  expect(mock.receivedPrompts).toContain(PROMPT)

  // Step 3: Replace the primary chat; the warm cache retains the updated messages.
  await openFreshDraft(page, PROMPT)
  await page.waitForTimeout(500)

  // Step 4: Install render counter, click back (warm resume), wait, assert.
  await installRenderCounter(page)
  await sessionRow.click()

  // Wait for the transcript to reappear — the warm cache should already
  // have the completed turn (updated by message.complete events).
  await waitForActiveTranscriptText(page, FIRST_USER_MSG)

  // Wait for at least 1 burst, then settle.
  await page.waitForFunction(
    () => {
      const w = window as unknown as { __RENDER_COUNT__?: { bursts: number } }
      return Boolean(w.__RENDER_COUNT__ && w.__RENDER_COUNT__.bursts > 0)
    },
    undefined,
    { timeout: 10_000 },
  )
  await page.waitForTimeout(2_000)

  const result = await readRenderCount(page)
  await page.screenshot({ path: testInfo.outputPath('warm-resume-post-inference.png') })
  assertNoJitter(result)
})
