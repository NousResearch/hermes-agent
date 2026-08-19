/**
 * E2E regression: warm-route resume must keep the settled viewport anchored
 * across the resume-final authoritative transcript publication.
 *
 * When a session is already in the runtime-id cache (the "warm" path in
 * `resumeSession()`), clicking its sidebar row paints the cached transcript
 * and settles at the bottom. `session.activate` then returns a persisted
 * transcript that may legitimately differ from the cache (e.g. after
 * background inference), and `syncSessionStateToView` publishes it a second
 * time. That second authoritative publication is ALLOWED — suppressing it
 * would reintroduce missing completed turns. What is forbidden is exposing
 * the resize: the same session key and a non-empty transcript mean the
 * Thread's explicit settle loop does not re-arm, so the height change lands
 * at the stale scroll offset and use-stick-to-bottom only corrects after a
 * ResizeObserver + requestAnimationFrame round trip (the recorded
 * long-context warm-switch flicker).
 *
 * This test pre-seeds a 32-message session into state.db, boots the app,
 * clicks the session (cold resume — populates the warm cache), runs a real
 * mock-backed inference turn, navigates away to a draft, then clicks back
 * (warm resume). A test-only observer on the real thread viewport records,
 * on every animation frame, scrollTop/scrollHeight/clientHeight, the
 * distance from the bottom, the first/last visible message ordinals, and a
 * transcript-mutation ordinal. Once the first paint has settled (target
 * transcript visible, viewport within the bottom tolerance, mutations
 * quiet), the test allows any further authoritative publication but fails
 * if the viewport ever leaves the bottom tolerance or the visible anchor
 * falls back to an older message. No assertion requires exactly one burst
 * or zero reconciles.
 *
 * The tab-reactivation control follows its own contract: a kept-alive tab
 * becomes visible without repainting (zero bursts, zero reconciles).
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
  launchDesktop
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
const AUTHORITY_ONLY_TEXT = Array.from(
  { length: 8 },
  (_, index) => `E2E delayed persisted authority row ${index + 1}`
).join('\n')
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
    }
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
async function installRenderCounter(page: import('@playwright/test').Page, transcriptText?: string): Promise<void> {
  await page.evaluate(
    ([visibleSelector, allSelector, expected]: [string, string, string | undefined]) => {
      const surfaces = [...document.querySelectorAll(expected ? allSelector : visibleSelector)]
      const surface = expected
        ? surfaces.find(candidate =>
            (candidate.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected)
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
        characterData: false
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
    },
    [SURFACE, ALL_SURFACES, transcriptText] as [string, string, string | undefined]
  )
}

/** Wait until the ACTIVE chat surface's transcript contains `text`. */
async function waitForActiveTranscriptText(
  page: import('@playwright/test').Page,
  text: string,
  timeout = 30_000
): Promise<void> {
  await page.waitForFunction(
    ([expected, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const active = surfaces[surfaces.length - 1]

      return (active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected)
    },
    [text, SURFACE] as [string, string],
    { timeout }
  )
}

async function waitForActiveTranscriptWithoutText(page: import('@playwright/test').Page, text: string): Promise<void> {
  await page.waitForFunction(
    ([expected, surfaceSelector]: [string, string]) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const active = surfaces[surfaces.length - 1]

      return !(active?.querySelector('[data-slot="aui_thread-viewport"]')?.textContent ?? '').includes(expected)
    },
    [text, SURFACE] as [string, string],
    { timeout: 15_000 }
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
function assertNoRepaint(
  result: { bursts: number; mutations: number; timeline: number[]; reconciles: number } | null
): void {
  expect(result, 'MutationObserver should have recorded render data').toBeTruthy()
  expect(
    result!.bursts,
    `Expected no additive render bursts for a kept-alive tab, but got ${result!.bursts}. ` +
      `Mutation timeline: ${JSON.stringify(result!.timeline)}.`
  ).toBe(0)
  expect(
    result!.reconciles,
    `Expected no transcript reconciles for a kept-alive tab, but got ${result!.reconciles}.`
  ).toBe(0)
}

type MainAuthorityGate = {
  hit: boolean
  release: () => void
  released: boolean
}

/**
 * Make the next real session-messages response deterministically differ from
 * the warm cache, but hold it in Electron main until the cached first paint is
 * settled. The renderer still uses the production IPC → REST → reconcile path;
 * only the response's release time and one appended SessionMessage are test
 * inputs. The original IPC handler is restored as soon as the target request
 * is captured.
 */
async function armDelayedPersistedAuthority(app: MockBackendFixture['app']): Promise<void> {
  await app.evaluate(({ ipcMain }, marker) => {
    type InvokeHandler = (...args: unknown[]) => unknown
    const handlers = (ipcMain as unknown as { _invokeHandlers?: Map<string, InvokeHandler> })._invokeHandlers
    const original = handlers?.get('hermes:api')
    if (!handlers || !original) {
      throw new Error('Electron hermes:api invoke handler is unavailable')
    }

    const mainGlobal = globalThis as typeof globalThis & { __E2E_AUTHORITY_GATE__?: MainAuthorityGate }
    if (mainGlobal.__E2E_AUTHORITY_GATE__) {
      throw new Error('Delayed persisted-authority gate is already installed')
    }

    let releaseResponse!: () => void
    const held = new Promise<void>(resolve => {
      releaseResponse = resolve
    })
    const gate: MainAuthorityGate = {
      hit: false,
      released: false,
      release: () => {
        if (!gate.released) {
          gate.released = true
          releaseResponse()
        }
      }
    }
    mainGlobal.__E2E_AUTHORITY_GATE__ = gate

    handlers.set('hermes:api', async (...args: unknown[]) => {
      const result = await original(...args)
      const request = args[1] as { method?: string; path?: string } | undefined
      const isTargetRead =
        (!request?.method || request.method === 'GET') &&
        /^\/api\/sessions\/[^/]+\/messages(?:\?|$)/.test(request?.path ?? '')

      if (!gate.hit && isTargetRead) {
        gate.hit = true
        handlers.set('hermes:api', original)
        await held

        const response = result as { messages?: unknown[]; session_id?: string }
        if (!Array.isArray(response?.messages)) {
          throw new Error('Target session-messages response has no messages array')
        }

        const maxId = response.messages.reduce<number>((current, row) => {
          const record = row && typeof row === 'object' ? (row as { id?: unknown; row_id?: unknown }) : null
          const candidate = Number(record?.id ?? record?.row_id)
          return Number.isFinite(candidate) ? Math.max(current, candidate) : current
        }, 0)

        return {
          ...response,
          messages: [
            ...response.messages,
            {
              content: marker,
              id: maxId + 1000,
              role: 'user',
              timestamp: Date.now() / 1000
            }
          ]
        }
      }

      return result
    })
  }, AUTHORITY_ONLY_TEXT)
}

async function persistedAuthorityRequestIsWaiting(app: MockBackendFixture['app']): Promise<boolean> {
  return app.evaluate(() => {
    const mainGlobal = globalThis as typeof globalThis & { __E2E_AUTHORITY_GATE__?: MainAuthorityGate }
    return Boolean(mainGlobal.__E2E_AUTHORITY_GATE__?.hit && !mainGlobal.__E2E_AUTHORITY_GATE__?.released)
  })
}

async function releasePersistedAuthority(app: MockBackendFixture['app']): Promise<void> {
  await app.evaluate(() => {
    const mainGlobal = globalThis as typeof globalThis & { __E2E_AUTHORITY_GATE__?: MainAuthorityGate }
    const gate = mainGlobal.__E2E_AUTHORITY_GATE__
    if (!gate?.hit) {
      throw new Error('Persisted-authority request was not waiting at release time')
    }
    gate.release()
  })
}

// ─── Viewport-anchor observer (RED for the resume-final publication) ────────
//
// The warm-route resume legitimately publishes the persisted transcript a
// second time when the cached paint and the stored authority differ. This
// observer therefore does NOT count publications. It records, on every
// animation frame, the thread viewport's scroll metrics, the first/last
// visible message ordinals, and a transcript-mutation ordinal fed by a
// MutationObserver. Once the target first paint has settled (transcript
// visible, viewport at the bottom, mutations quiet), any distance-from-bottom
// excursion beyond a small pixel tolerance — or a visible-anchor fallback to
// an older message — is the defect: the resume-final publication resized the
// transcript under a stale scroll offset and use-stick-to-bottom could only
// correct it after a ResizeObserver + requestAnimationFrame round trip.
//
// Only numbers/ordinals are recorded — never message text.

/** Bottom tolerance for "settled at the bottom" (subpixel quantization is 0.5px). */
const SETTLE_BOTTOM_TOLERANCE_PX = 2
/** Small explicit tolerance for post-settle distance-from-bottom excursions. */
const EXCURSION_TOLERANCE_PX = 4
/** Consecutive bottom frames required before the first paint counts as settled. */
const SETTLE_FRAMES = 4
/** Consecutive mutation-quiet frames required (first paint/backfill must be done). */
const SETTLE_QUIET_FRAMES = 3
/** Safety cap on the in-page frame buffer. */
const MAX_RECORDED_FRAMES = 4000

interface ViewportFrameSample {
  f: number
  t: number
  st: number
  sh: number
  ch: number
  dfb: number
  ord: number
  lastOrd: number
  mutOrd: number
  msgs: number
  following: string
}

interface ViewportMutationSample {
  t: number
  ord: number
  added: number
}

interface ViewportAnchorReport {
  settled: boolean
  settledFrame: number
  settledT: number
  settledOrd: number
  settledLastOrd: number
  settledMsgs: number
  settledMutOrd: number
  maxExcursionPx: number
  maxExcursionFrame: number
  maxExcursionT: number
  anchorFallbacks: number[]
  frames: ViewportFrameSample[]
  mutations: ViewportMutationSample[]
}

/**
 * Install a test-only per-frame recorder on the active draft's real thread
 * scroll viewport. Primary-route navigation reuses this viewport when the
 * warm session is selected, so binding while the draft is empty sees both the
 * cached first paint and the resume-final publication. Returns nothing — call
 * `readViewportAnchorReport` to stop and collect.
 */
async function installViewportAnchorObserver(page: import('@playwright/test').Page): Promise<void> {
  await page.evaluate(
    ([visibleSelector, settleTol, excursionTol, settleFrames, quietFrames, maxFrames]: [
      string,
      number,
      number,
      number,
      number,
      number
    ]) => {
      const surfaces = [...document.querySelectorAll(visibleSelector)]
      const surface = surfaces.at(-1)
      const viewport = surface?.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')
      if (!viewport) {
        throw new Error('Active draft thread viewport not found before warm resume')
      }
      const content = viewport.querySelector('[data-slot="aui_thread-content"]') ?? viewport
      const MESSAGE_ROOTS = '[data-role="user"], [data-role="assistant"], [data-role="system"]'

      const state = {
        startedAt: performance.now(),
        frames: [] as ViewportFrameSample[],
        mutations: [] as ViewportMutationSample[],
        mutOrd: 0,
        settled: false,
        settledFrame: -1,
        settledT: 0,
        settledOrd: -1,
        settledLastOrd: -1,
        settledMsgs: 0,
        settledMutOrd: 0,
        bottomStreak: 0,
        quietStreak: 0,
        prevMutOrd: 0,
        maxExcursionPx: 0,
        maxExcursionFrame: -1,
        maxExcursionT: 0,
        anchorFallbacks: [] as number[],
        stopped: false
      }
      const debugWindow = window as unknown as {
        __ANCHOR_OBSERVER__: typeof state
        __ANCHOR_VIEWPORT__: Element
      }
      debugWindow.__ANCHOR_OBSERVER__ = state
      debugWindow.__ANCHOR_VIEWPORT__ = viewport

      const observer = new MutationObserver(records => {
        if (state.stopped) return
        let added = 0
        for (const record of records) {
          if (record.type === 'childList' && record.addedNodes.length > 0) {
            added += record.addedNodes.length
          }
        }
        state.mutOrd += 1
        state.mutations.push({ t: performance.now() - state.startedAt, ord: state.mutOrd, added })
      })
      observer.observe(viewport, { childList: true, subtree: true, attributes: false, characterData: false })

      let frameNo = 0

      const tick = () => {
        if (state.stopped) return
        frameNo += 1
        const t = performance.now() - state.startedAt
        const st = viewport.scrollTop
        const sh = viewport.scrollHeight
        const ch = viewport.clientHeight
        const dfb = Math.max(0, sh - ch - st)

        // Only the ACTIVE pane can settle — a kept-alive pane keeps a
        // preserved layout box while hidden, so its old bottom position must
        // not count as "settled" before the warm click.
        const activePane = !viewport.closest('[data-pane-hidden]')

        const messages = [...content.querySelectorAll(MESSAGE_ROOTS)]
        const vTop = viewport.getBoundingClientRect().top
        const vBottom = vTop + ch
        let ord = -1
        let lastOrd = -1
        for (let i = 0; i < messages.length; i++) {
          const r = messages[i].getBoundingClientRect()
          if (r.bottom >= vTop && r.top <= vBottom) {
            if (ord < 0) ord = i
            lastOrd = i
          }
        }

        const sample: ViewportFrameSample = {
          f: frameNo,
          t,
          st,
          sh,
          ch,
          dfb,
          ord,
          lastOrd,
          mutOrd: state.mutOrd,
          msgs: messages.length,
          following: viewport.getAttribute('data-following') ?? ''
        }
        state.frames.push(sample)
        if (state.frames.length > maxFrames) state.frames.shift()

        if (!state.settled) {
          const bottom = activePane && sample.msgs > 0 && sample.dfb <= settleTol
          const quiet = sample.mutOrd === state.prevMutOrd
          state.prevMutOrd = sample.mutOrd
          state.bottomStreak = bottom ? state.bottomStreak + 1 : 0
          state.quietStreak = bottom && quiet ? state.quietStreak + 1 : 0
          if (state.bottomStreak >= settleFrames && state.quietStreak >= quietFrames) {
            state.settled = true
            state.settledFrame = sample.f
            state.settledT = sample.t
            state.settledOrd = sample.ord
            state.settledLastOrd = sample.lastOrd
            state.settledMsgs = sample.msgs
            state.settledMutOrd = sample.mutOrd
          }
        } else {
          if (sample.dfb > excursionTol && sample.dfb > state.maxExcursionPx) {
            state.maxExcursionPx = sample.dfb
            state.maxExcursionFrame = sample.f
            state.maxExcursionT = sample.t
          }
          if (state.settledOrd >= 0 && sample.ord >= 0 && sample.ord < state.settledOrd - 1) {
            state.anchorFallbacks.push(sample.f)
          }
          if (state.settledLastOrd >= 0 && sample.lastOrd >= 0 && sample.lastOrd < state.settledLastOrd - 1) {
            state.anchorFallbacks.push(sample.f)
          }
        }

        requestAnimationFrame(tick)
      }

      requestAnimationFrame(tick)
    },
    [
      SURFACE,
      SETTLE_BOTTOM_TOLERANCE_PX,
      EXCURSION_TOLERANCE_PX,
      SETTLE_FRAMES,
      SETTLE_QUIET_FRAMES,
      MAX_RECORDED_FRAMES
    ] as [string, number, number, number, number, number]
  )
}

/** Stop the anchor observer and return its full frame/mutation report. */
async function readViewportAnchorReport(page: import('@playwright/test').Page): Promise<ViewportAnchorReport | null> {
  return page.evaluate(() => {
    type AnchorState = {
      stopped: boolean
      settled: boolean
      settledFrame: number
      settledT: number
      settledOrd: number
      settledLastOrd: number
      settledMsgs: number
      settledMutOrd: number
      maxExcursionPx: number
      maxExcursionFrame: number
      maxExcursionT: number
      anchorFallbacks: number[]
      frames: ViewportFrameSample[]
      mutations: ViewportMutationSample[]
    }
    const w = window as unknown as { __ANCHOR_OBSERVER__?: AnchorState }
    const s = w.__ANCHOR_OBSERVER__
    if (s) {
      s.stopped = true
      return {
        settled: s.settled,
        settledFrame: s.settledFrame,
        settledT: s.settledT,
        settledOrd: s.settledOrd,
        settledLastOrd: s.settledLastOrd,
        settledMsgs: s.settledMsgs,
        settledMutOrd: s.settledMutOrd,
        maxExcursionPx: s.maxExcursionPx,
        maxExcursionFrame: s.maxExcursionFrame,
        maxExcursionT: s.maxExcursionT,
        anchorFallbacks: [...s.anchorFallbacks],
        frames: s.frames,
        mutations: s.mutations
      }
    }
    return null
  })
}

/**
 * Assert the warm-route contract: a resume-final authoritative publication
 * is allowed, but after the first paint has settled the viewport must never
 * leave the bottom tolerance and the visible anchor must never fall back to
 * an older message. The failure message carries the compact frame timeline
 * and the maximum offset so the failure is attributable to the target
 * behavior rather than to the harness.
 */
function assertSettledViewportStaysAnchored(report: ViewportAnchorReport | null): void {
  expect(report, 'Viewport-anchor observer should have recorded frame data').toBeTruthy()
  expect(
    report!.settled,
    'Viewport never reached first-settled (active transcript + consecutive bottom frames with quiet mutations). ' +
      'Observation window too short, target never activated, or the kept-alive binding was wrong.'
  ).toBe(true)

  const settled = report!
  const ok = settled.maxExcursionPx <= EXCURSION_TOLERANCE_PX && settled.anchorFallbacks.length === 0

  if (!ok) {
    const postSettle = settled.frames.filter(f => f.f >= settled.settledFrame)
    const dfbSeq = postSettle.map(f => f.dfb)
    const seq =
      dfbSeq.length <= 200
        ? dfbSeq.join(',')
        : `${dfbSeq.slice(0, 80).join(',')},…(${dfbSeq.length - 160} frames)…,${dfbSeq.slice(-80).join(',')}`
    const excursionSample = settled.frames.find(f => f.f === settled.maxExcursionFrame)
    const postSettleMutations = settled.mutations.filter(m => m.t >= settled.settledT)
    const excursionLine = excursionSample ? `  excursion frame: ${JSON.stringify(excursionSample)}\n` : ''
    const range =
      postSettle.length > 0
        ? `frames ${postSettle[0].f}..${postSettle[postSettle.length - 1].f}`
        : 'no post-settle frames'
    expect(
      ok,
      `Warm-route resume-final publication was allowed, but it exposed a viewport excursion after first-settled:\n` +
        `  first-settled: frame ${settled.settledFrame} (t=+${settled.settledT}ms), visible message ordinals ` +
        `(${settled.settledOrd}, ${settled.settledLastOrd}), ${settled.settledMsgs} messages, ` +
        `mutation ordinal ${settled.settledMutOrd}\n` +
        `  max distance-from-bottom excursion: ${settled.maxExcursionPx}px at frame ${settled.maxExcursionFrame} ` +
        `(t=+${settled.maxExcursionT}ms), tolerance ${EXCURSION_TOLERANCE_PX}px\n` +
        excursionLine +
        `  anchor fallbacks (frame indices): ${settled.anchorFallbacks.length ? settled.anchorFallbacks.join(',') : 'none'}\n` +
        `  post-settle DOM publications: ${postSettleMutations.length} ` +
        `(${postSettleMutations.map(m => `+${m.added}@${m.t}ms`).join(', ') || 'none'})\n` +
        `  distance-from-bottom timeline (px, ${range}): ${seq}\n` +
        `  The persisted authority published a non-equivalent transcript after the first paint settled, and the ` +
        `viewport sat at the stale scroll offset until use-stick-to-bottom's ResizeObserver/RAF correction — ` +
        `the long-context warm-switch flicker.`
    ).toBe(true)
  }
}

test('tab reactivation preserves the mounted transcript without repainting', async ({}, testInfo) => {
  const page = fixture!.page

  // Wait for the sidebar to populate with our seeded session.
  const sessionRow = page.locator('[data-slot="sidebar"] button').filter({ hasText: SESSION_TITLE }).first()
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

test('warm-route resume may republish the persisted authority, but the settled viewport must stay anchored', async ({}, testInfo) => {
  const page = fixture!.page
  const { mock } = fixture!

  // Wait for the sidebar to populate with our seeded session.
  const sessionRow = page.locator('[data-slot="sidebar"] button').filter({ hasText: SESSION_TITLE }).first()
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

  // Step 4: Install the per-frame viewport-anchor observer on the empty
  // primary-route viewport and hold a deliberately non-equivalent persisted
  // response until the warm cached first paint has settled.
  await armDelayedPersistedAuthority(fixture!.app)
  await installViewportAnchorObserver(page)
  await sessionRow.click()

  // Wait for the transcript to reappear — the warm cache should already
  // have the completed turn (updated by message.complete events).
  await waitForActiveTranscriptText(page, FIRST_USER_MSG)

  // Wait for the first paint to settle (rAF-stable bottom + quiet mutations),
  // then keep observing for a bounded window so the resume-final
  // authoritative publication has time to land.
  await page.waitForFunction(
    () => {
      const w = window as unknown as { __ANCHOR_OBSERVER__?: { settled: boolean } }
      return Boolean(w.__ANCHOR_OBSERVER__ && w.__ANCHOR_OBSERVER__.settled)
    },
    undefined,
    { timeout: 15_000 }
  )

  await expect
    .poll(() => persistedAuthorityRequestIsWaiting(fixture!.app), {
      message: 'The real persisted transcript read should be waiting behind the deterministic gate',
      timeout: 10_000
    })
    .toBe(true)
  await releasePersistedAuthority(fixture!.app)
  await waitForActiveTranscriptText(page, AUTHORITY_ONLY_TEXT, 15_000)
  await page.waitForTimeout(500)

  // Harness sanity: primary-route navigation must have reused the observed viewport.
  expect(
    await page.evaluate((surfaceSelector: string) => {
      const surfaces = document.querySelectorAll(surfaceSelector)
      const activeViewport = surfaces[surfaces.length - 1]?.querySelector('[data-slot="aui_thread-viewport"]')
      const observedViewport = (window as unknown as { __ANCHOR_VIEWPORT__?: Element }).__ANCHOR_VIEWPORT__
      return activeViewport === observedViewport
    }, SURFACE),
    'Primary-route navigation should reuse the observed viewport (harness sanity, not the RED invariant)'
  ).toBe(true)

  const report = await readViewportAnchorReport(page)
  await page.screenshot({ path: testInfo.outputPath('warm-resume-post-inference.png') })
  assertSettledViewportStaysAnchored(report)
})
