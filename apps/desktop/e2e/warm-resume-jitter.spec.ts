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
async function installRenderCounter(page: import('@playwright/test').Page): Promise<void> {
  await page.evaluate(() => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
    if (!viewport) {
      throw new Error('Thread viewport not found before warm resume')
    }

    const state = { bursts: 0, mutations: 0, timeline: [] as number[], stopped: false, reconciles: 0 }
    ;(window as unknown as { __RENDER_COUNT__: typeof state }).__RENDER_COUNT__ = state

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
  })
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

test('warm-route resume paints transcript exactly once (no jitter)', async ({}, testInfo) => {
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
  await page.waitForFunction(
    (text: string) =>
      document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent?.includes(text) ??
      false,
    FIRST_USER_MSG,
    { timeout: 30_000 },
  )

  // Wait for the session to fully settle (cold-path RPC + reconciliation).
  await page.waitForTimeout(2_000)

  // Step 2: Navigate away to a new chat — this does NOT evict the warm cache.
  const newSessionButton = page
    .locator('[data-slot="sidebar"] button[aria-label="New session"]')
    .first()
  await newSessionButton.click()

  // Wait for the new-chat empty state.
  await page.waitForFunction(
    (firstMsg: string) => {
      const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
      if (!viewport) return false
      const text = viewport.textContent ?? ''
      return !text.includes(firstMsg)
    },
    FIRST_USER_MSG,
    { timeout: 15_000 },
  )

  await page.waitForTimeout(500)

  // Step 3: Install render counter, click back (warm resume), wait, assert.
  await installRenderCounter(page)
  await sessionRow.click()

  await page.waitForFunction(
    (text: string) =>
      document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent?.includes(text) ??
      false,
    FIRST_USER_MSG,
    { timeout: 30_000 },
  )

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
  await page.screenshot({ path: testInfo.outputPath('warm-resume-idle.png') })
  assertNoJitter(result)
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
  await page.waitForFunction(
    (text: string) =>
      document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent?.includes(text) ??
      false,
    FIRST_USER_MSG,
    { timeout: 30_000 },
  )
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
  await page.waitForFunction(
    () => {
      const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
      return viewport?.textContent?.includes('mock inference server') ?? false
    },
    undefined,
    { timeout: 60_000 },
  )
  // Extra settle for message.complete → updateSessionState → cache write.
  await page.waitForTimeout(2_000)

  // Verify the prompt was received by the mock server.
  expect(mock.receivedPrompts).toContain(PROMPT)

  // Step 3: Navigate away — the warm cache retains the updated messages.
  const newSessionButton = page
    .locator('[data-slot="sidebar"] button[aria-label="New session"]')
    .first()
  await newSessionButton.click()
  await page.waitForFunction(
    (prompt: string) => {
      const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
      if (!viewport) return false
      return !(viewport.textContent ?? '').includes(prompt)
    },
    PROMPT,
    { timeout: 15_000 },
  )
  await page.waitForTimeout(500)

  // Step 4: Install render counter, click back (warm resume), wait, assert.
  await installRenderCounter(page)
  await sessionRow.click()

  // Wait for the transcript to reappear — the warm cache should already
  // have the completed turn (updated by message.complete events).
  await page.waitForFunction(
    (text: string) =>
      document.querySelector('[data-slot="aui_thread-viewport"]')?.textContent?.includes(text) ??
      false,
    FIRST_USER_MSG,
    { timeout: 30_000 },
  )

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
