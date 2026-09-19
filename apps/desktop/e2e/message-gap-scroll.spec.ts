import type { Locator } from '@playwright/test'
import { createServer, type ViteDevServer } from 'vite'

import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'
import { RealSessionBuilder } from './real-session-builder'
import { type ElectronApplication, expect, type Page, test } from './test'

declare global {
  interface Window {
    messageGapHarness: {
      collapseGap: () => void
      completeBackfill: () => void
      messages: () => Array<{ contentLength: number; id: string; role: string; status: string }>
      pagination: () => { hasMoreBefore: boolean; nextOffset: number }
      routeState: () => {
        loadingSession: boolean
        mismatch: boolean
        paneId: string
        routedId: string
        selectedId: string
        showChatBar: boolean
      }
      reset: () => void
    }
  }
}

let server: ViteDevServer
let harnessUrl = ''

const ACTIVE_ELECTRON_SURFACE = '[data-composer-target]:not([data-pane-hidden] [data-composer-target])'
const REAL_SESSION_TITLE = 'E2E transcript gap persisted replay'
const REAL_BACKFILLED_PROMPT = 'E2E persisted gap history 1'
const REAL_RECENT_PROMPT = 'E2E persisted gap history 120'
const REAL_COMPACTION_DRAFT = 'draft survives real compaction'
const REAL_COMPACTION_SELECTION_DRAFT = 'selected draft survives delayed real compaction'
const REAL_RUNNING_PROMPT = 'E2E_MESSAGE_GAP_RUNNING_RESUME'
const REAL_QUEUED_PROMPT = 'E2E_MESSAGE_GAP_QUEUED_AFTER_RUNNING_RESUME'
const COMPRESSION_REQUEST_MARKER = 'You are a summarization agent creating a context checkpoint.'

const realHistoryTurns = Array.from({ length: 121 }, (_, index) =>
  index === 0 ? REAL_SESSION_TITLE : `E2E persisted gap history ${index}`
)

interface RestTranscriptMessage {
  content: unknown
  id: number
  role: string
  timestamp: number
}

interface RestTranscriptPage {
  messages: RestTranscriptMessage[]
  pagination: { limit: number; offset: number; order: 'latest' | 'oldest'; returned: number }
  session_id: string
}

interface RealReplayFixture extends MockBackendFixture {
  sessionId: string
}

interface RealDomSample {
  activeIsComposer: boolean
  assistantRootCount: number
  blankAssistantRootCount: number
  composer: { draft: string; identity: string | null; selectionEnd: number | null; selectionStart: number | null }
  focusEvents: Array<{ kind: string; slot: string | null }>
  following: string | null
  mountedUserIds: string[]
  mountedUserRows: Array<{ id: string; identity: string | null; text: string }>
  scroll: { clientHeight: number; distanceFromBottom: number; scrollHeight: number; scrollTop: number }
  sentinel: { identity: string | null; present: boolean }
  sessionAnchor: string | null
  surfaceId: string | null
}

async function setupRealReplayDesktop({
  mockOptions,
  turns
}: {
  mockOptions?: Parameters<typeof startMockServer>[0]
  turns: readonly string[]
}): Promise<RealReplayFixture> {
  const mock = await startMockServer(mockOptions)
  const sandbox = createSandbox('message-gap-replay')
  let app: ElectronApplication | undefined

  try {
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)

    const builder = await RealSessionBuilder.start(sandbox.hermesHome)
    let sessionId: string

    try {
      const session = await builder.createSession({ title: turns[0], turns })
      sessionId = session.sessionId
    } finally {
      await builder.close()
    }

    const launched = await launchDesktop(buildAppEnv(sandbox))
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

function recordGatewayFrames(): void {
  const sentFrames: string[] = []
  const receivedFrames: string[] = []
  const observedSockets = new WeakSet<WebSocket>()
  Object.defineProperty(window, '__messageGapSentFrames', {
    configurable: true,
    value: sentFrames
  })
  Object.defineProperty(window, '__messageGapReceivedFrames', {
    configurable: true,
    value: receivedFrames
  })

  const originalSend = WebSocket.prototype.send
  WebSocket.prototype.send = function send(data) {
    ;(
      window as typeof window & {
        __messageGapGatewaySocket?: WebSocket
      }
    ).__messageGapGatewaySocket = this

    if (!observedSockets.has(this)) {
      observedSockets.add(this)
      this.addEventListener('message', event => {
        if (typeof event.data === 'string') {
          receivedFrames.push(event.data)
        }
      })
    }

    if (typeof data === 'string') {
      sentFrames.push(data)
    }

    return originalSend.call(this, data)
  }
}

async function installGatewayFrameRecorder(page: Page): Promise<void> {
  await page.addInitScript(recordGatewayFrames)

  await page.reload()
}

async function submitQueuedPrompt(page: Page, sessionId: string, text: string): Promise<unknown> {
  return page.evaluate(
    ({ queuedText, runtimeId }) =>
      new Promise<unknown>((resolve, reject) => {
        const socket = (
          window as typeof window & {
            __messageGapGatewaySocket?: WebSocket
          }
        ).__messageGapGatewaySocket

        if (!socket || socket.readyState !== WebSocket.OPEN) {
          reject(new Error('isolated Desktop gateway socket is not open'))

          return
        }

        const id = `message-gap-queued-${Date.now()}`
        const timeout = window.setTimeout(() => {
          socket.removeEventListener('message', onMessage)
          reject(new Error('queued prompt submission did not receive a gateway response'))
        }, 10_000)
        const onMessage = (event: MessageEvent) => {
          if (typeof event.data !== 'string') {
            return
          }

          try {
            const response = JSON.parse(event.data) as { error?: unknown; id?: unknown; result?: unknown }

            if (response.id !== id) {
              return
            }

            window.clearTimeout(timeout)
            socket.removeEventListener('message', onMessage)

            if (response.error) {
              reject(new Error(`queued prompt submission failed: ${JSON.stringify(response.error)}`))
            } else {
              resolve(response.result)
            }
          } catch {
            // Ignore unrelated non-JSON gateway frames.
          }
        }

        socket.addEventListener('message', onMessage)
        socket.send(
          JSON.stringify({
            id,
            jsonrpc: '2.0',
            method: 'prompt.submit',
            params: { queued: true, session_id: runtimeId, text: queuedText }
          })
        )
      }),
    { queuedText: text, runtimeId: sessionId }
  )
}

function activeElectronSurface(page: Page) {
  return page.locator(ACTIVE_ELECTRON_SURFACE).last()
}

function activeElectronViewport(page: Page) {
  return activeElectronSurface(page).locator('[data-slot="aui_thread-viewport"]')
}

async function openRealSession(page: Page, expectedRecentText: string): Promise<void> {
  const row = page.locator('[data-slot="sidebar"] button').filter({ hasText: REAL_SESSION_TITLE }).first()
  await row.waitFor({ state: 'visible', timeout: 60_000 })
  await row.click()
  await expect(activeElectronViewport(page)).toContainText(expectedRecentText, { timeout: 30_000 })
}

async function fetchTranscriptPage(page: Page, sessionId: string, offset: number): Promise<RestTranscriptPage> {
  return page.evaluate(
    async ({ id, pageOffset }) => {
      const desktop = (
        window as unknown as {
          hermesDesktop: { api<T>(request: { path: string }): Promise<T> }
        }
      ).hermesDesktop

      return desktop.api<RestTranscriptPage>({
        path: `/api/sessions/${encodeURIComponent(id)}/messages?limit=120&offset=${pageOffset}&order=latest&include_compacted=true`
      })
    },
    { id: sessionId, pageOffset: offset }
  )
}

async function sampleRealDom(page: Page): Promise<RealDomSample> {
  return activeElectronSurface(page).evaluate(surface => {
    const harnessWindow = window as typeof window & {
      __messageGapFocusEvents?: Array<{ kind: string; slot: string | null }>
      __messageGapNodeCounter?: number
    }
    const identify = (node: HTMLElement | null): string | null => {
      if (!node) {
        return null
      }
      if (!node.dataset.messageGapIdentity) {
        harnessWindow.__messageGapNodeCounter = (harnessWindow.__messageGapNodeCounter ?? 0) + 1
        node.dataset.messageGapIdentity = `real-node-${harnessWindow.__messageGapNodeCounter}`
      }
      return node.dataset.messageGapIdentity
    }
    const viewport = surface.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    const composer = surface.querySelector<HTMLElement>('[data-slot="composer-rich-input"]')
    const sentinel = [...surface.querySelectorAll<HTMLButtonElement>('button')].find(button =>
      (button.textContent ?? '').includes('Show earlier messages')
    )
    const assistantRoots = [...surface.querySelectorAll<HTMLElement>('[data-slot="aui_assistant-message-root"]')]
    const selection = window.getSelection()
    const selectionInside = Boolean(composer && selection?.anchorNode && composer.contains(selection.anchorNode))

    return {
      activeIsComposer: document.activeElement === composer,
      assistantRootCount: assistantRoots.length,
      blankAssistantRootCount: assistantRoots.filter(
        root => root.getBoundingClientRect().height === 0 && !(root.textContent ?? '').trim()
      ).length,
      composer: {
        draft: composer?.textContent ?? '',
        identity: identify(composer),
        selectionEnd: selectionInside ? (selection?.focusOffset ?? null) : null,
        selectionStart: selectionInside ? (selection?.anchorOffset ?? null) : null
      },
      focusEvents: [...(harnessWindow.__messageGapFocusEvents ?? [])],
      following: viewport.dataset.following ?? null,
      mountedUserIds: [
        ...surface.querySelectorAll<HTMLElement>('[data-slot="aui_user-message-root"][data-message-id]')
      ].map(root => root.dataset.messageId ?? ''),
      mountedUserRows: [
        ...surface.querySelectorAll<HTMLElement>('[data-slot="aui_user-message-root"][data-message-id]')
      ].map(root => ({
        id: root.dataset.messageId ?? '',
        identity: identify(root),
        text: (root.textContent ?? '').trim()
      })),
      scroll: {
        clientHeight: viewport.clientHeight,
        distanceFromBottom: viewport.scrollHeight - viewport.scrollTop,
        scrollHeight: viewport.scrollHeight,
        scrollTop: viewport.scrollTop
      },
      sentinel: { identity: identify(sentinel ?? null), present: Boolean(sentinel) },
      sessionAnchor: surface.getAttribute('data-session-anchor'),
      surfaceId: surface.getAttribute('data-composer-surface-id')
    }
  })
}

async function focusDraftAndRecord(
  page: Page,
  draft: string,
  selectionStart: number,
  selectionEnd = selectionStart
): Promise<void> {
  const composer = activeElectronSurface(page).locator('[data-slot="composer-rich-input"]')
  await composer.fill(draft)
  await composer.evaluate(
    (element, selectionRange) => {
      const editor = element as HTMLElement
      const text = editor.firstChild
      const selection = window.getSelection()
      const range = document.createRange()

      editor.focus()
      if (text && selection) {
        range.setStart(text, selectionRange.start)
        range.setEnd(text, selectionRange.end)
        selection.removeAllRanges()
        selection.addRange(range)
      }

      const events: Array<{ kind: string; slot: string | null }> = []
      ;(
        window as typeof window & {
          __messageGapFocusEvents?: Array<{ kind: string; slot: string | null }>
        }
      ).__messageGapFocusEvents = events

      for (const kind of ['blur', 'focusout', 'focusin'] as const) {
        document.addEventListener(
          kind,
          event => {
            const target = event.target instanceof HTMLElement ? event.target : null
            events.push({ kind, slot: target?.dataset.slot ?? null })
          },
          true
        )
      }
    },
    { end: selectionEnd, start: selectionStart }
  )
}

test.beforeAll(async () => {
  server = await createServer({
    root: process.cwd(),
    server: { host: '127.0.0.1', port: 0 }
  })
  await server.listen()
  const address = server.httpServer?.address()

  if (!address || typeof address === 'string') {
    throw new Error('message-gap harness Vite server did not expose a TCP port')
  }

  harnessUrl = `http://127.0.0.1:${address.port}/e2e/message-gap-harness.html`
})

test.afterAll(async () => {
  await server.close()
})

type GeometrySample = {
  activeTop: number
  blankShellCount: number
  clientHeight: number
  distanceFromBottom: number
  following: string | null
  roots: Array<{
    contentVisibility: string
    display: string
    height: number
    marginBottom: string
    marginTop: string
    minHeight: string
    reservedPlaceholderHeight: string
    text: string
    top: number
    visibility: string
  }>
  scaffoldGap: number
  scrollHeight: number
  scrollTop: number
  sourceMessages: Array<{ contentLength: number; id: string; role: string; status: string }>
  viewportOverflowAnchor: string
}

type ContinuitySample = {
  activeElement: { identity: string | null; slot: string | null }
  composer: {
    draft: string
    identity: string | null
    selectionFocusOffset: number | null
    selectionOffset: number | null
  }
  focusEvents: Array<{ kind: string; slot: string | null }>
  mountedAssistantTexts: string[]
  mountedUserIds: string[]
  mountedUserRows: Array<{ id: string; identity: string | null }>
  pagination: { hasMoreBefore: boolean; nextOffset: number }
  routeState: {
    loadingSession: boolean
    mismatch: boolean
    paneId: string
    routedId: string
    selectedId: string
    showChatBar: boolean
  }
  sentinel: { identity: string | null; present: boolean; top: number | null }
  scroll: { clientHeight: number; distanceFromBottom: number; scrollHeight: number; scrollTop: number }
}

const sampleGeometry = async (page: Page): Promise<GeometrySample> =>
  page.evaluate(() => {
    const viewport = document.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    const roots = [...document.querySelectorAll<HTMLElement>('[data-slot="aui_assistant-message-root"]')]

    const blankShellCount = roots.filter(
      root => root.getBoundingClientRect().height === 0 && !(root.textContent ?? '').trim()
    ).length

    const scaffoldRoots = roots.filter(root =>
      root.querySelector('[data-slot="aui_thinking-disclosure"], [data-slot="tool-block"]')
    )

    const before = scaffoldRoots.at(-2)!
    const active = scaffoldRoots.at(-1)!
    const beforeRect = before.getBoundingClientRect()
    const activeRect = active.getBoundingClientRect()
    const viewportStyle = getComputedStyle(viewport)

    return {
      activeTop: activeRect.top,
      blankShellCount,
      clientHeight: viewport.clientHeight,
      distanceFromBottom: viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop,
      following: viewport.dataset.following ?? null,
      roots: roots.map(root => {
        const rect = root.getBoundingClientRect()
        const style = getComputedStyle(root)

        return {
          contentVisibility: style.contentVisibility,
          display: style.display,
          height: rect.height,
          marginBottom: style.marginBottom,
          marginTop: style.marginTop,
          minHeight: style.minHeight,
          reservedPlaceholderHeight: style.containIntrinsicSize,
          text: (root.textContent ?? '').trim().replaceAll(/\s+/g, ' ').slice(0, 120),
          top: rect.top,
          visibility: style.visibility
        }
      }),
      scaffoldGap: activeRect.top - beforeRect.bottom,
      scrollHeight: viewport.scrollHeight,
      scrollTop: viewport.scrollTop,
      sourceMessages: window.messageGapHarness.messages(),
      viewportOverflowAnchor: viewportStyle.overflowAnchor
    }
  })

const collapseAndSampleFrames = async (page: Page): Promise<GeometrySample[]> => {
  await page.evaluate(() => window.messageGapHarness.collapseGap())
  const samples: GeometrySample[] = []

  for (let frame = 0; frame < 12; frame += 1) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
    samples.push(await sampleGeometry(page))
  }

  return samples
}

const sampleContinuity = async (page: Page): Promise<ContinuitySample> =>
  page.evaluate(() => {
    const harnessWindow = window as typeof window & {
      __focusEvents?: Array<{ kind: string; slot: string | null }>
      __nodeIdentity?: number
    }

    const identify = (node: HTMLElement | null): string | null => {
      if (!node) {
        return null
      }

      if (!node.dataset.harnessIdentity) {
        harnessWindow.__nodeIdentity = (harnessWindow.__nodeIdentity ?? 0) + 1
        node.dataset.harnessIdentity = `node-${harnessWindow.__nodeIdentity}`
      }

      return node.dataset.harnessIdentity
    }

    const composer = document.querySelector<HTMLElement>('[data-slot="composer-rich-input"]')
    const active = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const viewport = document.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    const mountedUserRows = [
      ...document.querySelectorAll<HTMLElement>('[data-slot="aui_user-message-root"][data-message-id]')
    ]

    const sentinel = [...document.querySelectorAll<HTMLButtonElement>('button')].find(button =>
      (button.textContent ?? '').includes('Show earlier messages')
    )

    const selection = window.getSelection()

    const selectionOffset =
      composer && selection?.anchorNode && composer.contains(selection.anchorNode) ? selection.anchorOffset : null

    return {
      activeElement: { identity: identify(active), slot: active?.dataset.slot ?? null },
      composer: {
        draft: composer?.textContent ?? '',
        identity: identify(composer),
        selectionFocusOffset:
          composer && selection?.focusNode && composer.contains(selection.focusNode) ? selection.focusOffset : null,
        selectionOffset
      },
      focusEvents: [...(harnessWindow.__focusEvents ?? [])],
      mountedAssistantTexts: [
        ...document.querySelectorAll<HTMLElement>('[data-slot="aui_assistant-message-root"]')
      ].map(root => (root.textContent ?? '').trim().replaceAll(/\s+/g, ' ').slice(0, 100)),
      mountedUserIds: mountedUserRows.map(root => root.dataset.messageId ?? ''),
      mountedUserRows: mountedUserRows.map(root => ({
        id: root.dataset.messageId ?? '',
        identity: identify(root)
      })),
      pagination: window.messageGapHarness.pagination(),
      routeState: window.messageGapHarness.routeState(),
      sentinel: {
        identity: identify(sentinel ?? null),
        present: Boolean(sentinel),
        top: sentinel?.getBoundingClientRect().top ?? null
      },
      scroll: {
        clientHeight: viewport.clientHeight,
        distanceFromBottom: viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop,
        scrollHeight: viewport.scrollHeight,
        scrollTop: viewport.scrollTop
      }
    }
  })

test('a stale empty assistant shell owns no geometry or scroll displacement', async ({ page }, testInfo) => {
  await page.goto(`${harnessUrl}?mode=geometry`)
  await page.waitForSelector('[data-slot="aui_assistant-message-root"]')
  await page.waitForTimeout(250)

  const viewport = page.locator('[data-slot="aui_thread-viewport"]')
  await viewport.evaluate(element => {
    element.scrollTop = element.scrollHeight
    element.dispatchEvent(new Event('scroll'))
  })
  await expect(viewport).toHaveAttribute('data-following', 'true')

  const before = await sampleGeometry(page)
  const frames = await collapseAndSampleFrames(page)
  const after = frames.at(-1)!
  const proof = { before, frames }

  console.log('MESSAGE_GAP_GEOMETRY', JSON.stringify(proof))
  await testInfo.attach('message-gap-geometry.json', {
    body: JSON.stringify(proof, null, 2),
    contentType: 'application/json'
  })

  const scrollHeights = [before, ...frames].map(sample => sample.scrollHeight)
  const scrollTops = [before, ...frames].map(sample => sample.scrollTop)
  const activeTops = [before, ...frames].map(sample => sample.activeTop)

  // Exact symptom: a logical blank assistant boundary must not become a DOM
  // flex item between scaffolding rows. If it does, the generic bubble margins
  // reserve a full turn gap and hydration later removes that geometry.
  expect.soft(before.blankShellCount, JSON.stringify(proof, null, 2)).toBe(0)
  expect.soft(before.scaffoldGap).toBeLessThanOrEqual(5)
  expect.soft(Math.max(...scrollHeights) - Math.min(...scrollHeights)).toBeLessThanOrEqual(1)
  expect.soft(Math.max(...scrollTops) - Math.min(...scrollTops)).toBeLessThanOrEqual(1)
  expect.soft(Math.max(...activeTops) - Math.min(...activeTops)).toBeLessThanOrEqual(1)

  // Controls: the real completed/active reasoning rows remain present, the
  // bottom-follow owner stays locked, and ordinary prose supplies enough geometry
  // for this to be a real scrolling transcript rather than a zero-layout mock.
  expect(after.roots.filter(root => /Thought|Thinking/.test(root.text))).toHaveLength(2)
  expect(after.following).toBe('true')
  expect(after.scrollHeight).toBeGreaterThan(after.clientHeight)
  expect(before.viewportOverflowAnchor).toBe('none')
})

test('bottom-follow remains locked when the same logical shell disappears', async ({ page }, testInfo) => {
  await page.goto(`${harnessUrl}?mode=geometry`)
  await page.waitForSelector('[data-slot="aui_thread-viewport"]')
  await page.waitForTimeout(250)

  const viewport = page.locator('[data-slot="aui_thread-viewport"]')
  await viewport.evaluate(element => {
    element.scrollTop = element.scrollHeight
    element.dispatchEvent(new Event('scroll'))
  })
  await expect(viewport).toHaveAttribute('data-following', 'true')

  const before = await sampleGeometry(page)
  const frames = await collapseAndSampleFrames(page)
  const after = frames.at(-1)!

  await testInfo.attach('message-gap-bottom-follow.json', {
    body: JSON.stringify({ before, frames }, null, 2),
    contentType: 'application/json'
  })

  expect(before.blankShellCount).toBe(0)
  expect(after.following).toBe('true')
  expect(Math.abs(after.distanceFromBottom)).toBeLessThanOrEqual(1)
  expect(Math.abs(after.scrollHeight - before.scrollHeight)).toBeLessThanOrEqual(1)
})

test('compaction reconciliation retains loaded history, backfill truth, and the focused composer', async ({
  page
}, testInfo) => {
  await page.goto(`${harnessUrl}?mode=continuity`)
  await page.waitForSelector('[data-slot="composer-rich-input"]')
  const showEarlier = page.getByRole('button', { name: /Show earlier messages/i })

  await expect(showEarlier).toBeVisible()
  const firstWindow = await sampleContinuity(page)

  for (let step = 0; step < 3; step += 1) {
    await showEarlier.click()
    await page.waitForTimeout(150)
  }

  const editor = page.locator('[data-slot="composer-rich-input"]')
  const draft = 'draft survives compaction'

  await editor.fill(draft)
  await editor.evaluate((element, offset) => {
    const editorElement = element as HTMLElement
    const text = editorElement.firstChild
    const selection = window.getSelection()
    const range = document.createRange()

    editorElement.focus()

    if (text && selection) {
      range.setStart(text, offset)
      range.collapse(true)
      selection.removeAllRanges()
      selection.addRange(range)
    }
  }, 7)
  await page.evaluate(() => {
    const harnessWindow = window as typeof window & {
      __focusEvents?: Array<{ kind: string; slot: string | null }>
    }

    const events: Array<{ kind: string; slot: string | null }> = []

    harnessWindow.__focusEvents = events

    for (const kind of ['blur', 'focusout', 'focusin'] as const) {
      document.addEventListener(
        kind,
        event => {
          const target = event.target instanceof HTMLElement ? event.target : null

          events.push({ kind, slot: target?.dataset.slot ?? null })
        },
        true
      )
    }
  })

  const before = await sampleContinuity(page)

  await page.evaluate(() => window.messageGapHarness.collapseGap())

  for (let frame = 0; frame < 12; frame += 1) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
  }

  const after = await sampleContinuity(page)
  const proof = { after, before, firstWindow }

  console.log('MESSAGE_GAP_CONTINUITY', JSON.stringify(proof))
  await testInfo.attach('message-gap-continuity.json', {
    body: JSON.stringify(proof, null, 2),
    contentType: 'application/json'
  })

  expect(before.mountedUserIds.length).toBeGreaterThan(firstWindow.mountedUserIds.length)
  expect(before.pagination).toEqual({ hasMoreBefore: true, nextOffset: 120 })
  expect(after.pagination).toEqual(before.pagination)
  expect(after.sentinel.present).toBe(true)
  expect(after.sentinel.identity).toBe(before.sentinel.identity)
  expect(before.routeState).toMatchObject({
    loadingSession: false,
    mismatch: false,
    paneId: 'workspace',
    routedId: 'stored-gap-before-compaction',
    selectedId: 'stored-gap-before-compaction',
    showChatBar: true
  })
  expect(after.routeState).toMatchObject({
    loadingSession: false,
    mismatch: false,
    paneId: 'workspace',
    routedId: 'stored-gap-before-compaction',
    selectedId: 'stored-gap-after-compaction',
    showChatBar: true
  })

  for (const messageId of before.mountedUserIds) {
    expect(after.mountedUserIds, `lost mounted row ${messageId}`).toContain(messageId)
  }

  expect(before.activeElement.identity).toBe(before.composer.identity)
  expect(before.composer).toMatchObject({ draft, selectionOffset: 7 })
  expect(after.activeElement.identity).toBe(before.composer.identity)
  expect(after.composer).toEqual(before.composer)
  expect(after.focusEvents.filter(event => event.kind === 'blur' || event.kind === 'focusout')).toEqual([])
})

test('immediate reader scroll supersedes stale pagination geometry before the debounce', async ({ page }, testInfo) => {
  await page.goto(`${harnessUrl}?mode=continuity`)
  await page.waitForSelector('[data-slot="composer-rich-input"]')

  const viewport = page.locator('[data-slot="aui_thread-viewport"]')
  await viewport.evaluate(
    element =>
      new Promise<void>((resolve, reject) => {
        const content = element.querySelector<HTMLElement>('[data-slot="aui_thread-content"]')

        if (!content) {
          reject(new Error('missing transcript content while waiting for initial layout'))
          return
        }

        let quietTimer = 0
        const deadline = window.setTimeout(() => {
          observer.disconnect()
          window.clearTimeout(quietTimer)
          reject(new Error('initial transcript layout did not become quiet'))
        }, 10_000)
        const finish = () => {
          observer.disconnect()
          window.clearTimeout(deadline)
          resolve()
        }
        const armQuietWindow = () => {
          window.clearTimeout(quietTimer)
          quietTimer = window.setTimeout(finish, 750)
        }
        const observer = new ResizeObserver(armQuietWindow)

        observer.observe(content)
        armQuietWindow()
      })
  )
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await viewport.evaluate(element => {
      element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight - 240)
      element.dispatchEvent(new Event('scroll'))
    })
    await page.waitForTimeout(120)
  }
  await expect(viewport).toHaveAttribute('data-following', 'false')

  const editor = page.locator('[data-slot="composer-rich-input"]')
  const draft = 'reader intent preserves this selected draft'
  await editor.fill(draft)
  await editor.evaluate(element => {
    const editorElement = element as HTMLElement
    const text = editorElement.firstChild
    const selection = window.getSelection()

    editorElement.focus()
    if (!text || !selection) {
      throw new Error('composer selection target is unavailable')
    }

    const range = document.createRange()
    range.setStart(text, 3)
    range.setEnd(text, 9)
    selection.removeAllRanges()
    selection.addRange(range)
  })
  await page.evaluate(() => {
    const harnessWindow = window as typeof window & {
      __focusEvents?: Array<{ kind: string; slot: string | null }>
    }
    const events: Array<{ kind: string; slot: string | null }> = []

    harnessWindow.__focusEvents = events
    for (const kind of ['blur', 'focusout', 'focusin'] as const) {
      document.addEventListener(
        kind,
        event => {
          const target = event.target instanceof HTMLElement ? event.target : null

          events.push({ kind, slot: target?.dataset.slot ?? null })
        },
        true
      )
    }
  })

  const before = await sampleContinuity(page)
  const moved = await viewport.evaluate(element => {
    element.scrollTop = Math.max(0, element.scrollTop - 180)
    element.dispatchEvent(new Event('scroll'))
    const sample = {
      distanceFromBottom: element.scrollHeight - element.clientHeight - element.scrollTop,
      scrollHeight: element.scrollHeight,
      scrollTop: element.scrollTop
    }

    window.messageGapHarness.completeBackfill()
    return sample
  })

  for (let frame = 0; frame < 12; frame += 1) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
  }
  await page.waitForTimeout(120)
  const after = await sampleContinuity(page)
  const proof = { after, before, moved }

  console.log(
    'MESSAGE_GAP_IMMEDIATE_SCROLL',
    JSON.stringify({
      after: { pagination: after.pagination, scroll: after.scroll, sentinel: after.sentinel },
      before: { pagination: before.pagination, scroll: before.scroll, sentinel: before.sentinel },
      moved
    })
  )
  await testInfo.attach('message-gap-immediate-scroll.json', {
    body: JSON.stringify(proof, null, 2),
    contentType: 'application/json'
  })

  expect(before.scroll.distanceFromBottom).toBeGreaterThanOrEqual(239)
  expect(moved.distanceFromBottom - before.scroll.distanceFromBottom).toBeGreaterThanOrEqual(179)
  expect(before.sentinel.present).toBe(true)
  expect(after.sentinel.present).toBe(true)
  expect(after.sentinel.identity).toBe(before.sentinel.identity)
  expect(after.pagination).toEqual({ hasMoreBefore: false, nextOffset: 120 })
  expect(after.mountedUserIds.length).toBeGreaterThanOrEqual(before.mountedUserIds.length)
  expect(new Set(after.mountedUserIds).size).toBe(after.mountedUserIds.length)
  for (const messageId of before.mountedUserIds) {
    expect(after.mountedUserIds, `lost mounted row ${messageId}`).toContain(messageId)
  }
  const afterRowsById = new Map(after.mountedUserRows.map(row => [row.id, row]))

  for (const row of before.mountedUserRows) {
    expect(afterRowsById.get(row.id)?.identity, `remounted row ${row.id}`).toBe(row.identity)
  }

  expect(before.composer).toMatchObject({
    draft,
    selectionFocusOffset: 9,
    selectionOffset: 3
  })
  expect(after.activeElement.identity).toBe(before.composer.identity)
  expect(after.composer).toEqual(before.composer)
  expect(after.focusEvents.filter(event => event.kind === 'blur' || event.kind === 'focusout')).toEqual([])
  expect(Math.abs(after.scroll.distanceFromBottom - moved.distanceFromBottom)).toBeLessThanOrEqual(2)
})

test('wheel, keyboard, pointer, and touch intent supersede stale pagination geometry synchronously', async ({
  page
}, testInfo) => {
  test.setTimeout(180_000)
  const intentCases = [
    { event: 'wheel', label: 'wheel' },
    { event: 'keydown', label: 'keyboard' },
    { event: 'pointerdown', label: 'pointer' },
    { event: 'touchstart', label: 'touch' }
  ] as const
  const proofs: Array<{
    after: ContinuitySample['scroll']
    before: ContinuitySample['scroll']
    intent: (typeof intentCases)[number]['label']
    moved: { distanceFromBottom: number; scrollHeight: number; scrollTop: number }
  }> = []

  for (const intentCase of intentCases) {
    await page.goto(`${harnessUrl}?mode=continuity`)
    await page.waitForSelector('[data-slot="composer-rich-input"]')
    const viewport = page.locator('[data-slot="aui_thread-viewport"]')

    await viewport.evaluate(
      element =>
        new Promise<void>((resolve, reject) => {
          const content = element.querySelector<HTMLElement>('[data-slot="aui_thread-content"]')

          if (!content) {
            reject(new Error('missing transcript content while waiting for initial layout'))
            return
          }

          let quietTimer = 0
          const deadline = window.setTimeout(() => {
            observer.disconnect()
            window.clearTimeout(quietTimer)
            reject(new Error('initial transcript layout did not become quiet'))
          }, 10_000)
          const finish = () => {
            observer.disconnect()
            window.clearTimeout(deadline)
            resolve()
          }
          const armQuietWindow = () => {
            window.clearTimeout(quietTimer)
            quietTimer = window.setTimeout(finish, 750)
          }
          const observer = new ResizeObserver(armQuietWindow)

          observer.observe(content)
          armQuietWindow()
        })
    )
    let positioned = false
    for (let attempt = 0; attempt < 4; attempt += 1) {
      const distanceFromBottom = await viewport.evaluate(element => {
        element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight - 240)
        element.dispatchEvent(new Event('scroll'))

        return element.scrollHeight - element.clientHeight - element.scrollTop
      })
      await page.waitForTimeout(120)
      const retainedDistance = await viewport.evaluate(
        element => element.scrollHeight - element.clientHeight - element.scrollTop
      )

      if (distanceFromBottom >= 239 && retainedDistance >= 239) {
        positioned = true
        break
      }
    }
    expect(positioned, `${intentCase.label} baseline did not settle`).toBe(true)

    const before = await sampleContinuity(page)
    const moved = await viewport.evaluate((element, eventType) => {
      element.scrollTop = Math.max(0, element.scrollTop - 180)
      const event =
        eventType === 'wheel'
          ? new WheelEvent('wheel', { bubbles: true, deltaY: -180 })
          : eventType === 'keydown'
            ? new KeyboardEvent('keydown', { bubbles: true, key: 'PageUp' })
            : eventType === 'pointerdown'
              ? new PointerEvent('pointerdown', { bubbles: true })
              : new Event('touchstart', { bubbles: true })

      element.dispatchEvent(event)
      const sample = {
        distanceFromBottom: element.scrollHeight - element.clientHeight - element.scrollTop,
        scrollHeight: element.scrollHeight,
        scrollTop: element.scrollTop
      }

      window.messageGapHarness.completeBackfill()
      return sample
    }, intentCase.event)

    for (let frame = 0; frame < 12; frame += 1) {
      await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
    }
    await page.waitForTimeout(120)
    const after = await sampleContinuity(page)

    proofs.push({ after: after.scroll, before: before.scroll, intent: intentCase.label, moved })
    expect(before.scroll.distanceFromBottom, intentCase.label).toBeGreaterThanOrEqual(239)
    expect(moved.distanceFromBottom - before.scroll.distanceFromBottom, intentCase.label).toBeGreaterThanOrEqual(179)
    expect(after.pagination, intentCase.label).toEqual({ hasMoreBefore: false, nextOffset: 120 })
    expect(after.sentinel.identity, intentCase.label).toBe(before.sentinel.identity)
    expect(Math.abs(after.scroll.distanceFromBottom - moved.distanceFromBottom), intentCase.label).toBeLessThanOrEqual(
      2
    )
  }

  console.log('MESSAGE_GAP_IMMEDIATE_SCROLL_INTENTS', JSON.stringify(proofs))
  await testInfo.attach('message-gap-immediate-scroll-intents.json', {
    body: JSON.stringify(proofs, null, 2),
    contentType: 'application/json'
  })
})

// Playwright requires fixture callbacks to destructure their first argument.
// eslint-disable-next-line no-empty-pattern
test('real SessionDB keyboard reader intent follows transcript focus without capturing composer keys', async ({}, testInfo) => {
  test.setTimeout(180_000)
  const fixture = await setupRealReplayDesktop({
    turns: realHistoryTurns
  })

  try {
    await waitForAppReady(fixture)
    await openRealSession(fixture.page, REAL_RECENT_PROMPT)

    const surface = activeElectronSurface(fixture.page)
    const viewport = activeElectronViewport(fixture.page)
    const composer = surface.locator('[data-slot="composer-rich-input"]')
    const draft = 'keyboard focus boundary draft'
    await composer.fill(draft)
    await composer.evaluate(element => {
      const text = element.firstChild
      const selection = window.getSelection()
      const range = document.createRange()

      element.focus()
      if (text && selection) {
        range.setStart(text, 2)
        range.setEnd(text, 11)
        selection.removeAllRanges()
        selection.addRange(range)
      }
    })
    await viewport.evaluate(element => {
      const observedWindow = window as typeof window & {
        __messageGapKeyboardIntents?: number
      }
      observedWindow.__messageGapKeyboardIntents = 0
      const observer = new MutationObserver(records => {
        for (const record of records) {
          if (record.attributeName === 'data-reader-intent' && record.oldValue !== 'true') {
            observedWindow.__messageGapKeyboardIntents = (observedWindow.__messageGapKeyboardIntents ?? 0) + 1
          }
        }
      })
      observer.observe(element, {
        attributeFilter: ['data-reader-intent'],
        attributeOldValue: true,
        attributes: true
      })
    })

    const intentCount = () =>
      fixture.page.evaluate(
        () =>
          (
            window as typeof window & {
              __messageGapKeyboardIntents?: number
            }
          ).__messageGapKeyboardIntents ?? 0
      )
    const scrollTop = () => viewport.evaluate(element => element.scrollTop)
    const distanceFromBottom = () =>
      viewport.evaluate(element => element.scrollHeight - element.clientHeight - element.scrollTop)
    type KeyboardProbeWindow = typeof window & {
      __messageGapChildKeyboard?: {
        blurs: Record<string, number>
        clicks: Record<string, number>
        explicitScrollBy: number
      }
    }
    await viewport.evaluate(element => {
      const probeWindow = window as KeyboardProbeWindow
      const nativeScrollBy = element.scrollBy.bind(element)

      probeWindow.__messageGapChildKeyboard = {
        blurs: {},
        clicks: {},
        explicitScrollBy: 0
      }
      element.scrollBy = ((leftOrOptions?: number | ScrollToOptions, top?: number) => {
        probeWindow.__messageGapChildKeyboard!.explicitScrollBy += 1
        if (typeof leftOrOptions === 'number') {
          nativeScrollBy(leftOrOptions, top ?? 0)
        } else {
          nativeScrollBy(leftOrOptions)
        }
      }) as typeof element.scrollBy
    })
    const probeValue = (bucket: 'blurs' | 'clicks', label: string) =>
      fixture.page.evaluate(
        ({ bucket: useBucket, label: useLabel }) =>
          (window as KeyboardProbeWindow).__messageGapChildKeyboard?.[useBucket][useLabel] ?? 0,
        { bucket, label }
      )
    const explicitScrollByCount = () =>
      fixture.page.evaluate(() => (window as KeyboardProbeWindow).__messageGapChildKeyboard?.explicitScrollBy ?? 0)
    const armClickProbe = (control: Locator, label: string) =>
      control.evaluate((element, useLabel) => {
        const probe = (window as KeyboardProbeWindow).__messageGapChildKeyboard!

        probe.blurs[useLabel] = 0
        probe.clicks[useLabel] = 0
        element.addEventListener('blur', () => {
          probe.blurs[useLabel] += 1
        })
        element.addEventListener(
          'click',
          event => {
            probe.clicks[useLabel] += 1
            event.preventDefault()
            event.stopImmediatePropagation()
          },
          { capture: true }
        )
      }, label)
    const pressChildKeyWithoutReaderOwnership = async (control: Locator, key: string) => {
      await control.focus()
      await expect.poll(() => control.evaluate((element: Element) => document.activeElement === element)).toBe(true)
      const beforeIntent = await intentCount()
      const beforeScrollBy = await explicitScrollByCount()
      const beforeScrollTop = await scrollTop()

      await fixture.page.keyboard.press(key)

      expect(await intentCount(), key).toBe(beforeIntent)
      expect(await explicitScrollByCount(), key).toBe(beforeScrollBy)
      expect(Math.abs((await scrollTop()) - beforeScrollTop), key).toBeLessThanOrEqual(1)
      expect(await control.evaluate((element: Element) => document.activeElement === element), key).toBe(true)
    }
    const proveNativeButtonSpace = async (control: Locator, label: string) => {
      await expect(control).toBeAttached()
      await armClickProbe(control, label)
      await pressChildKeyWithoutReaderOwnership(control, 'Space')
      await expect.poll(() => probeValue('clicks', label), { timeout: 5_000 }).toBe(1)
      expect(await probeValue('blurs', label)).toBe(0)
    }
    const pressAndProve = async (key: string, direction: 'down' | 'end' | 'home' | 'up') => {
      const beforeIntent = await intentCount()
      const beforeScrollTop = await scrollTop()

      await fixture.page.keyboard.press(key)
      await expect.poll(intentCount).toBeGreaterThan(beforeIntent)
      if (direction === 'up') {
        await expect.poll(scrollTop).toBeLessThan(beforeScrollTop)
      } else if (direction === 'down') {
        await expect.poll(scrollTop).toBeGreaterThan(beforeScrollTop)
      } else if (direction === 'home') {
        await expect.poll(scrollTop).toBeLessThanOrEqual(1)
      } else {
        await expect.poll(distanceFromBottom).toBeLessThanOrEqual(2)
      }

      const proof = {
        afterIntent: await intentCount(),
        afterScrollTop: await scrollTop(),
        beforeIntent,
        beforeScrollTop,
        key
      }
      await expect(viewport).not.toHaveAttribute('data-reader-intent', 'true', {
        timeout: 3_000
      })

      return proof
    }

    await expect
      .poll(() =>
        viewport.evaluate(async element => {
          const sample = () => ({
            distanceFromBottom: element.scrollHeight - element.clientHeight - element.scrollTop,
            scrollHeight: element.scrollHeight,
            scrollTop: element.scrollTop
          })
          let previous = sample()

          if (
            previous.scrollHeight <= element.clientHeight ||
            previous.scrollTop <= 0 ||
            previous.distanceFromBottom > 2
          ) {
            return false
          }
          for (let frame = 0; frame < 8; frame += 1) {
            await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
            const current = sample()

            if (
              Math.abs(current.scrollHeight - previous.scrollHeight) > 1 ||
              Math.abs(current.scrollTop - previous.scrollTop) > 1 ||
              current.distanceFromBottom > 2
            ) {
              return false
            }
            previous = current
          }

          return true
        })
      )
      .toBe(true)
    const initialGeometry = await viewport.evaluate(element => ({
      clientHeight: element.clientHeight,
      distanceFromBottom: element.scrollHeight - element.clientHeight - element.scrollTop,
      scrollHeight: element.scrollHeight,
      scrollTop: element.scrollTop
    }))
    expect(initialGeometry.scrollTop).toBeGreaterThan(0)

    const composerScrollTop = await scrollTop()
    const composerIntentCount = await intentCount()
    await fixture.page.keyboard.press('ArrowUp')
    expect(await intentCount()).toBe(composerIntentCount)
    expect(await scrollTop()).toBe(composerScrollTop)
    await expect(composer).toContainText(draft)

    await expect(viewport).toHaveAttribute('tabindex', '0')
    await viewport.focus()
    await expect.poll(() => viewport.evaluate(element => document.activeElement === element)).toBe(true)

    const proofs = []
    proofs.push(await pressAndProve('PageUp', 'up'))
    proofs.push(await pressAndProve('ArrowUp', 'up'))
    proofs.push(await pressAndProve('Home', 'home'))
    proofs.push(await pressAndProve('ArrowDown', 'down'))
    await fixture.page.keyboard.press('Home')
    await expect.poll(scrollTop).toBeLessThanOrEqual(1)
    await expect(viewport).not.toHaveAttribute('data-reader-intent', 'true', {
      timeout: 3_000
    })
    proofs.push(await pressAndProve('PageDown', 'down'))
    await fixture.page.keyboard.press('Home')
    await expect.poll(scrollTop).toBeLessThanOrEqual(1)
    await expect(viewport).not.toHaveAttribute('data-reader-intent', 'true', {
      timeout: 3_000
    })
    proofs.push(await pressAndProve('Space', 'down'))
    proofs.push(await pressAndProve('Shift+Space', 'up'))
    proofs.push(await pressAndProve('End', 'end'))

    const assistantActions = surface.locator('[data-slot="aui_msg-actions"]').last()
    const copyAction = assistantActions.getByRole('button', { exact: true, name: 'Copy' })
    const retryAction = assistantActions.getByRole('button', { exact: true, name: 'Refresh' })
    const editAction = viewport
      .locator('[data-slot="aui_user-message-root"]')
      .last()
      .getByRole('button', { exact: true, name: 'Edit message' })

    await proveNativeButtonSpace(copyAction, 'copy')
    await proveNativeButtonSpace(retryAction, 'retry')
    await proveNativeButtonSpace(editAction, 'other-message-control')

    await viewport.evaluate(element => {
      const host = document.createElement('div')
      const link = document.createElement('a')
      const checkbox = document.createElement('input')
      const input = document.createElement('input')
      const editable = document.createElement('div')

      host.dataset.messageGapKeyboardProbes = 'true'
      Object.assign(host.style, {
        height: '1px',
        left: '-10000px',
        position: 'fixed',
        top: '0',
        width: '1px'
      })
      link.dataset.messageGapKeyboardProbe = 'link'
      link.href = '#message-gap-keyboard-link'
      link.textContent = 'keyboard link'
      checkbox.dataset.messageGapKeyboardProbe = 'checkbox'
      checkbox.type = 'checkbox'
      input.dataset.messageGapKeyboardProbe = 'input'
      input.type = 'text'
      input.value = 'reader'
      editable.contentEditable = 'true'
      editable.dataset.messageGapKeyboardProbe = 'editable'
      editable.textContent = 'editable'
      host.append(link, checkbox, input, editable)
      element.append(host)
    })

    const keyboardProbe = (name: string) => viewport.locator(`[data-message-gap-keyboard-probe="${name}"]`)
    const link = keyboardProbe('link')
    const checkbox = keyboardProbe('checkbox')
    const input = keyboardProbe('input')
    const editable = keyboardProbe('editable')

    await armClickProbe(link, 'link')
    await pressChildKeyWithoutReaderOwnership(link, 'Space')
    expect(await probeValue('clicks', 'link')).toBe(0)
    await fixture.page.keyboard.press('Enter')
    await expect.poll(() => probeValue('clicks', 'link')).toBe(1)
    expect(await probeValue('blurs', 'link')).toBe(0)

    await pressChildKeyWithoutReaderOwnership(checkbox, 'Space')
    expect(await checkbox.evaluate(element => (element as HTMLInputElement).checked)).toBe(true)

    await input.evaluate(element => {
      const field = element as HTMLInputElement

      field.focus()
      field.setSelectionRange(2, 4)
    })
    await pressChildKeyWithoutReaderOwnership(input, 'Space')
    expect(
      await input.evaluate(element => {
        const field = element as HTMLInputElement

        return { end: field.selectionEnd, start: field.selectionStart, value: field.value }
      })
    ).toEqual({ end: 3, start: 3, value: 're er' })

    await editable.evaluate(element => {
      const text = element.firstChild
      const selection = window.getSelection()
      const range = document.createRange()

      element.focus()
      if (text && selection) {
        range.setStart(text, 2)
        range.setEnd(text, 4)
        selection.removeAllRanges()
        selection.addRange(range)
      }
    })
    await pressChildKeyWithoutReaderOwnership(editable, 'Space')
    expect(
      await editable.evaluate(element => {
        const selection = window.getSelection()

        return {
          anchorOffset: selection?.anchorOffset ?? null,
          focusOffset: selection?.focusOffset ?? null,
          text: element.textContent
        }
      })
    ).toEqual({ anchorOffset: 3, focusOffset: 3, text: 'ed able' })

    await input.evaluate(element => {
      const field = element as HTMLInputElement

      field.value = 'reader keys stay with the field'
      field.focus()
      field.setSelectionRange(8, 8)
    })
    for (const childKey of ['ArrowDown', 'ArrowUp', 'End', 'Home', 'PageDown', 'PageUp']) {
      await pressChildKeyWithoutReaderOwnership(input, childKey)
    }

    await expect(composer).toContainText(draft)
    console.log('MESSAGE_GAP_REAL_KEYBOARD_INTENT', JSON.stringify({ initialGeometry, proofs }))
    await testInfo.attach('message-gap-real-keyboard-intent.json', {
      body: JSON.stringify({ initialGeometry, proofs }, null, 2),
      contentType: 'application/json'
    })
  } finally {
    await fixture.cleanup()
  }
})

// Playwright requires fixture callbacks to destructure their first argument.
// eslint-disable-next-line no-empty-pattern
test('real SessionDB replay preserves REST backfill geometry and composer focus through compaction', async ({}, testInfo) => {
  test.setTimeout(300_000)
  const fixture = await setupRealReplayDesktop({
    mockOptions: { holdFirstCompletionContaining: COMPRESSION_REQUEST_MARKER },
    turns: realHistoryTurns
  })
  try {
    await installGatewayFrameRecorder(fixture.page)
    await waitForAppReady(fixture)

    const tailPage = await fetchTranscriptPage(fixture.page, fixture.sessionId, 0)
    const olderPage = await fetchTranscriptPage(fixture.page, fixture.sessionId, 120)
    const backfilledIndex = olderPage.messages.findIndex(
      message => message.role === 'user' && message.content === REAL_BACKFILLED_PROMPT
    )
    const backfilledMessage = olderPage.messages[backfilledIndex]

    expect(tailPage.session_id).toBe(fixture.sessionId)
    expect(tailPage.pagination).toEqual({ limit: 120, offset: 0, order: 'latest', returned: 120 })
    expect(olderPage.pagination).toEqual({ limit: 120, offset: 120, order: 'latest', returned: 120 })
    expect(backfilledIndex).toBeGreaterThanOrEqual(0)
    expect(backfilledMessage.id).toBeGreaterThan(0)

    const authoritativeRowIds = [...tailPage.messages, ...olderPage.messages].map(message => message.id)
    expect(new Set(authoritativeRowIds).size).toBe(authoritativeRowIds.length)
    const expectedBackfilledRendererId = `${backfilledMessage.timestamp}-${backfilledIndex}-user`

    await openRealSession(fixture.page, REAL_RECENT_PROMPT)
    const viewport = activeElectronViewport(fixture.page)
    const showEarlier = activeElectronSurface(fixture.page).getByRole('button', {
      name: /Show earlier messages/i
    })

    await expect(showEarlier).toBeVisible({ timeout: 30_000 })
    await showEarlier.evaluate(button => {
      const probeWindow = window as typeof window & {
        __messageGapShowEarlierSpaceClicks?: number
      }

      probeWindow.__messageGapShowEarlierSpaceClicks = 0
      button.addEventListener(
        'click',
        event => {
          probeWindow.__messageGapShowEarlierSpaceClicks = (probeWindow.__messageGapShowEarlierSpaceClicks ?? 0) + 1
          event.preventDefault()
          event.stopImmediatePropagation()
        },
        { capture: true, once: true }
      )
    })
    await showEarlier.focus()
    const beforeShowEarlierSpace = await viewport.evaluate(element => element.scrollTop)
    await fixture.page.keyboard.press('Space')
    await expect
      .poll(
        () =>
          fixture.page.evaluate(
            () =>
              (
                window as typeof window & {
                  __messageGapShowEarlierSpaceClicks?: number
                }
              ).__messageGapShowEarlierSpaceClicks ?? 0
          ),
        { timeout: 5_000 }
      )
      .toBe(1)
    expect(
      Math.abs((await viewport.evaluate(element => element.scrollTop)) - beforeShowEarlierSpace)
    ).toBeLessThanOrEqual(1)
    await expect(viewport).not.toHaveAttribute('data-reader-intent', 'true')
    await expect
      .poll(() =>
        viewport
          .locator('[data-slot="aui_user-message-root"]')
          .evaluateAll(
            (roots, expectedText) => roots.filter(root => (root.textContent ?? '').trim() === expectedText).length,
            REAL_BACKFILLED_PROMPT
          )
      )
      .toBe(0)
    await viewport.evaluate(
      element =>
        new Promise<void>((resolve, reject) => {
          const content = element.querySelector<HTMLElement>('[data-slot="aui_thread-content"]')

          if (!content) {
            reject(new Error('missing transcript content while waiting for initial layout'))
            return
          }

          let quietTimer = 0
          const deadline = window.setTimeout(() => {
            observer.disconnect()
            element.removeEventListener('scroll', armQuietWindow)
            window.clearTimeout(quietTimer)
            reject(new Error('initial transcript layout did not become quiet'))
          }, 10_000)
          const finish = () => {
            observer.disconnect()
            element.removeEventListener('scroll', armQuietWindow)
            window.clearTimeout(deadline)
            resolve()
          }
          const armQuietWindow = () => {
            window.clearTimeout(quietTimer)
            quietTimer = window.setTimeout(finish, 100)
          }
          const observer = new ResizeObserver(armQuietWindow)

          observer.observe(content)
          element.addEventListener('scroll', armQuietWindow, { passive: true })
          armQuietWindow()
        })
    )
    await viewport.hover()
    await fixture.page.mouse.wheel(0, -240)
    await expect(viewport).toHaveAttribute('data-following', 'false')
    const beforeBackfillState = await sampleRealDom(fixture.page)
    const beforeBackfillScroll = await viewport.evaluate(element => {
      const diagnosticWindow = window as typeof window & {
        __messageGapBackfillFrames?: Array<{
          distanceFromBottom: number
          scrollHeight: number
          scrollTop: number
        }>
      }
      const frames: Array<{ distanceFromBottom: number; scrollHeight: number; scrollTop: number }> = []
      const showEarlier = [...element.querySelectorAll<HTMLButtonElement>('button')].find(button =>
        /Show earlier messages/i.test(button.textContent ?? '')
      )

      if (!showEarlier) {
        throw new Error('missing Show earlier control in active transcript viewport')
      }

      diagnosticWindow.__messageGapBackfillFrames = frames
      let remaining = 120

      const sample = () => {
        frames.push({
          distanceFromBottom: element.scrollHeight - element.scrollTop,
          scrollHeight: element.scrollHeight,
          scrollTop: element.scrollTop
        })

        remaining -= 1
        if (remaining > 0) {
          requestAnimationFrame(sample)
        }
      }

      const scroll = {
        clientHeight: element.clientHeight,
        distanceFromBottom: element.scrollHeight - element.scrollTop,
        scrollHeight: element.scrollHeight,
        scrollTop: element.scrollTop
      }
      requestAnimationFrame(sample)
      showEarlier.click()

      return scroll
    })
    const beforeBackfill = { ...beforeBackfillState, scroll: beforeBackfillScroll }
    expect(beforeBackfill.scroll.distanceFromBottom - beforeBackfill.scroll.clientHeight).toBeGreaterThanOrEqual(239)
    await expect(
      activeElectronSurface(fixture.page).locator(`[data-message-id="${expectedBackfilledRendererId}"]`)
    ).toContainText(REAL_BACKFILLED_PROMPT, { timeout: 30_000 })
    await fixture.page.waitForTimeout(250)
    const afterBackfill = await sampleRealDom(fixture.page)
    const backfillFrames = await fixture.page.evaluate(
      () =>
        (
          window as typeof window & {
            __messageGapBackfillFrames?: Array<{
              distanceFromBottom: number
              scrollHeight: number
              scrollTop: number
            }>
          }
        ).__messageGapBackfillFrames ?? []
    )

    expect(beforeBackfill.sentinel.present).toBe(true)
    expect(afterBackfill.sentinel.present).toBe(true)
    expect(afterBackfill.sentinel.identity).toBe(beforeBackfill.sentinel.identity)
    expect(afterBackfill.following).toBe('false')
    expect(afterBackfill.mountedUserIds.length).toBeGreaterThan(beforeBackfill.mountedUserIds.length)
    expect(new Set(afterBackfill.mountedUserIds).size).toBe(afterBackfill.mountedUserIds.length)
    const afterBackfillRowsById = new Map(afterBackfill.mountedUserRows.map(row => [row.id, row]))
    const backfillRowIdentityFailures = beforeBackfill.mountedUserRows.flatMap(row => {
      const afterRow = afterBackfillRowsById.get(row.id)

      return afterRow?.identity === row.identity
        ? []
        : [
            {
              afterIdentity: afterRow?.identity ?? null,
              beforeIdentity: row.identity,
              id: row.id,
              missing: !afterRow
            }
          ]
    })
    expect(backfillRowIdentityFailures).toEqual([])
    expect(afterBackfill.blankAssistantRootCount).toBe(0)
    expect(
      Math.abs(afterBackfill.scroll.distanceFromBottom - beforeBackfill.scroll.distanceFromBottom),
      JSON.stringify({
        afterBackfill: afterBackfill.scroll,
        beforeBackfill: beforeBackfill.scroll,
        backfillFrames
      })
    ).toBeLessThanOrEqual(2)

    const composer = activeElectronSurface(fixture.page).locator('[data-slot="composer-rich-input"]')
    await composer.fill('/compress preserve transcript-gap replay')
    await activeElectronSurface(fixture.page).getByRole('button', { name: 'Send', exact: true }).click()
    await fixture.mock.waitForHeldCompletion()
    const compressionStatus = fixture.page.getByText(/^compressing context for: preserve transcript-gap replay$/i)
    await expect(compressionStatus).toBeVisible()

    await focusDraftAndRecord(fixture.page, REAL_COMPACTION_DRAFT, 7)
    const beforeCompaction = await sampleRealDom(fixture.page)
    fixture.mock.releaseHeldStream()
    await expect(compressionStatus).toHaveCount(0, { timeout: 60_000 })
    const compactionFrames: RealDomSample[] = []
    for (let frame = 0; frame < 12; frame += 1) {
      await fixture.page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
      compactionFrames.push(await sampleRealDom(fixture.page))
    }
    await expect.poll(async () => (await sampleRealDom(fixture.page)).sentinel.present, { timeout: 10_000 }).toBe(false)
    const afterCompaction = await sampleRealDom(fixture.page)

    expect(beforeCompaction.activeIsComposer).toBe(true)
    expect(beforeCompaction.composer).toMatchObject({
      draft: REAL_COMPACTION_DRAFT,
      selectionEnd: 7,
      selectionStart: 7
    })
    expect(afterCompaction.activeIsComposer).toBe(true)
    expect(afterCompaction.composer).toEqual(beforeCompaction.composer)
    expect(afterCompaction.surfaceId).toBe(beforeCompaction.surfaceId)
    expect(afterCompaction.sessionAnchor).toBe(beforeCompaction.sessionAnchor)
    // Compression's active projection contributes the two oldest copied rows
    // that were not in the two loaded REST pages. Once those are grafted onto
    // the retained 240 rows, the transcript is complete and the sentinel must
    // retire rather than offer an empty/overlapping page.
    expect(beforeCompaction.sentinel.present).toBe(true)
    for (const frame of compactionFrames.filter(frame => frame.sentinel.present)) {
      expect(frame.sentinel.identity).toBe(beforeCompaction.sentinel.identity)
    }
    expect(afterCompaction.sentinel.present).toBe(false)
    expect(afterCompaction.blankAssistantRootCount).toBe(0)
    expect(afterCompaction.focusEvents.filter(event => event.kind === 'blur' || event.kind === 'focusout')).toEqual([])
    const afterCompactionRowsById = new Map(afterCompaction.mountedUserRows.map(row => [row.id, row]))
    const compactionRowIdentityFailures = beforeCompaction.mountedUserRows.flatMap(row => {
      const afterRow = afterCompactionRowsById.get(row.id)

      return afterRow?.identity === row.identity
        ? []
        : [
            {
              afterIdentity: afterRow?.identity ?? null,
              beforeIdentity: row.identity,
              id: row.id,
              sameTextRows: afterCompaction.mountedUserRows
                .filter(candidate => candidate.text === row.text)
                .map(candidate => ({ id: candidate.id, identity: candidate.identity })),
              text: row.text
            }
          ]
    })
    expect(compactionRowIdentityFailures).toEqual([])
    const compactionScrollProof = {
      after: afterCompaction.scroll,
      before: beforeCompaction.scroll,
      frames: compactionFrames.map(frame => frame.scroll)
    }
    for (const frame of compactionFrames) {
      expect(
        Math.abs(frame.scroll.distanceFromBottom - beforeCompaction.scroll.distanceFromBottom),
        JSON.stringify(compactionScrollProof)
      ).toBeLessThanOrEqual(2)
    }
    expect(
      Math.abs(afterCompaction.scroll.distanceFromBottom - beforeCompaction.scroll.distanceFromBottom),
      JSON.stringify(compactionScrollProof)
    ).toBeLessThanOrEqual(2)
    const afterCompactionUserTexts = afterCompaction.mountedUserRows.map(row => row.text)
    expect(new Set(afterCompactionUserTexts).size).toBe(realHistoryTurns.length)
    expect(afterCompactionUserTexts).toHaveLength(realHistoryTurns.length)
    expect(afterCompactionUserTexts).toContain(REAL_SESSION_TITLE)

    const afterCompactionRestPages = await Promise.all([
      fetchTranscriptPage(fixture.page, fixture.sessionId, 0),
      fetchTranscriptPage(fixture.page, fixture.sessionId, 120),
      fetchTranscriptPage(fixture.page, fixture.sessionId, 240)
    ])
    const afterCompactionRestMessages = afterCompactionRestPages.flatMap(page => page.messages)
    const afterCompactionRestUserTexts = afterCompactionRestMessages
      .filter(message => message.role === 'user' && typeof message.content === 'string')
      .map(message => message.content as string)
    const adoptedBackfilledRow = afterCompactionRestMessages.find(
      message => message.role === 'user' && message.content === REAL_BACKFILLED_PROMPT
    )

    expect(afterCompactionRestPages.map(page => page.pagination.returned)).toEqual([120, 120, 3])
    expect(new Set(afterCompactionRestMessages.map(message => message.id)).size).toBe(
      afterCompactionRestMessages.length
    )
    expect([...afterCompactionRestUserTexts].sort()).toEqual([...afterCompactionUserTexts].sort())
    expect(adoptedBackfilledRow?.id).toBeGreaterThan(0)
    expect(adoptedBackfilledRow?.id).not.toBe(backfilledMessage.id)

    const sentFrames = await fixture.page.evaluate(
      () =>
        (
          window as typeof window & {
            __messageGapSentFrames?: string[]
          }
        ).__messageGapSentFrames ?? []
    )
    const sentRequests = sentFrames.flatMap(frame => {
      try {
        return [JSON.parse(frame) as { method?: string; params?: Record<string, unknown> }]
      } catch {
        return []
      }
    })
    const resumeRequest = sentRequests.find(request => request.method === 'session.resume')
    expect(resumeRequest).toBeDefined()
    expect(resumeRequest?.params).not.toHaveProperty('messages')

    const proof = {
      afterBackfill,
      afterCompaction,
      afterCompactionRest: afterCompactionRestPages.map(page => page.pagination),
      adoptedBackfilledRowId: adoptedBackfilledRow?.id,
      authoritativeBackfilledRowId: backfilledMessage.id,
      beforeBackfill,
      beforeCompaction,
      compactionFrames,
      expectedBackfilledRendererId,
      rest: { older: olderPage.pagination, tail: tailPage.pagination },
      resumeRequest,
      sessionId: fixture.sessionId
    }
    console.log('MESSAGE_GAP_REAL_REPLAY', JSON.stringify(proof))
    await testInfo.attach('message-gap-real-replay.json', {
      body: JSON.stringify(proof, null, 2),
      contentType: 'application/json'
    })
  } finally {
    await fixture.cleanup()
  }
})

// Playwright requires fixture callbacks to destructure their first argument.
// eslint-disable-next-line no-empty-pattern
test('real SessionDB delayed compaction preserves a non-collapsed composer selection', async ({}, testInfo) => {
  test.setTimeout(240_000)
  const fixture = await setupRealReplayDesktop({
    mockOptions: { holdFirstCompletionContaining: COMPRESSION_REQUEST_MARKER },
    turns: realHistoryTurns
  })

  try {
    await waitForAppReady(fixture)
    await openRealSession(fixture.page, REAL_RECENT_PROMPT)

    const surface = activeElectronSurface(fixture.page)
    const composer = surface.locator('[data-slot="composer-rich-input"]')
    await composer.fill('/compress preserve non-collapsed composer selection')
    await surface.getByRole('button', { name: 'Send', exact: true }).click()
    await fixture.mock.waitForHeldCompletion()

    const compressionStatus = fixture.page.getByText(
      /^compressing context for: preserve non-collapsed composer selection$/i
    )
    await expect(compressionStatus).toBeVisible()

    await focusDraftAndRecord(fixture.page, REAL_COMPACTION_SELECTION_DRAFT, 9, 24)
    const before = await sampleRealDom(fixture.page)

    expect(before.activeIsComposer).toBe(true)
    expect(before.composer).toMatchObject({
      draft: REAL_COMPACTION_SELECTION_DRAFT,
      selectionEnd: 24,
      selectionStart: 9
    })

    fixture.mock.releaseHeldStream()
    await expect(compressionStatus).toHaveCount(0, { timeout: 60_000 })
    for (let frame = 0; frame < 12; frame += 1) {
      await fixture.page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
    }
    const after = await sampleRealDom(fixture.page)

    expect(after.activeIsComposer).toBe(true)
    expect(after.composer).toEqual(before.composer)
    expect(after.composer.selectionStart).toBe(9)
    expect(after.composer.selectionEnd).toBe(24)
    expect(after.surfaceId).toBe(before.surfaceId)
    expect(after.sessionAnchor).toBe(before.sessionAnchor)
    expect(after.focusEvents.filter(event => event.kind === 'blur' || event.kind === 'focusout')).toEqual([])

    const proof = { after, before, sessionId: fixture.sessionId }
    console.log('MESSAGE_GAP_REAL_COMPACTION_SELECTION', JSON.stringify(proof))
    await testInfo.attach('message-gap-real-compaction-selection.json', {
      body: JSON.stringify(proof, null, 2),
      contentType: 'application/json'
    })
  } finally {
    await fixture.cleanup()
  }
})

// Playwright requires fixture callbacks to destructure their first argument.
// eslint-disable-next-line no-empty-pattern
test('running SessionDB replay adopts one assistant stream after renderer resume', async ({}, testInfo) => {
  test.setTimeout(180_000)
  const fixture = await setupRealReplayDesktop({
    mockOptions: { holdFirstStreamForPrompt: REAL_RUNNING_PROMPT },
    turns: [REAL_SESSION_TITLE, 'E2E warm history 1']
  })

  try {
    await installGatewayFrameRecorder(fixture.page)
    await waitForAppReady(fixture)
    await openRealSession(fixture.page, 'E2E warm history 1')

    const surface = activeElectronSurface(fixture.page)
    const composer = surface.locator('[data-slot="composer-rich-input"]')
    await composer.fill(REAL_RUNNING_PROMPT)
    await surface.getByRole('button', { name: 'Send', exact: true }).click()
    await fixture.mock.waitForHeldStream()
    await expect(activeElectronViewport(fixture.page)).toContainText('Hello')

    await fixture.page.reload()
    await waitForAppReady(fixture)
    await openRealSession(fixture.page, 'E2E warm history 1')
    const resumedBeforeRelease = await sampleRealDom(fixture.page)

    await activeElectronViewport(fixture.page).evaluate(element => {
      element.scrollTop = element.scrollHeight
      element.dispatchEvent(new Event('scroll'))
    })
    fixture.mock.releaseHeldStream()
    await expect(
      activeElectronSurface(fixture.page).locator('[data-slot="aui_assistant-message-root"]').last()
    ).toContainText(MOCK_REPLY, { timeout: 60_000 })

    const frames: RealDomSample[] = []
    for (let frame = 0; frame < 12; frame += 1) {
      await fixture.page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => resolve())))
      frames.push(await sampleRealDom(fixture.page))
    }

    const after = frames.at(-1)!
    expect(after.blankAssistantRootCount).toBe(0)
    expect(new Set(frames.map(frame => frame.assistantRootCount)).size).toBe(1)
    expect(after.assistantRootCount).toBe(3)
    for (const frame of frames) {
      expect(Math.abs(frame.scroll.distanceFromBottom - frame.scroll.clientHeight)).toBeLessThanOrEqual(2)
    }

    const persisted = await fetchTranscriptPage(fixture.page, fixture.sessionId, 0)
    const persistedPromptIndex = persisted.messages.findIndex(
      message => message.role === 'user' && message.content === REAL_RUNNING_PROMPT
    )
    expect(persistedPromptIndex).toBeGreaterThanOrEqual(0)
    expect(persisted.messages[persistedPromptIndex].id).toBeGreaterThan(0)
    expect(persisted.messages[persistedPromptIndex + 1]).toMatchObject({ role: 'assistant', content: MOCK_REPLY })

    const proof = {
      after,
      frames,
      persistedAssistantRowId: persisted.messages[persistedPromptIndex + 1].id,
      persistedUserRowId: persisted.messages[persistedPromptIndex].id,
      resumedBeforeRelease,
      sessionId: fixture.sessionId
    }
    console.log('MESSAGE_GAP_RUNNING_REPLAY', JSON.stringify(proof))
    await testInfo.attach('message-gap-running-replay.json', {
      body: JSON.stringify(proof, null, 2),
      contentType: 'application/json'
    })
  } finally {
    await fixture.cleanup()
  }
})

// Playwright requires fixture callbacks to destructure their first argument.
// eslint-disable-next-line no-empty-pattern
test('queued running SessionDB replay keeps one authoritative assistant stream across resume', async ({}, testInfo) => {
  test.setTimeout(240_000)
  const fixture = await setupRealReplayDesktop({
    mockOptions: { holdFirstStreamForPrompt: REAL_RUNNING_PROMPT },
    turns: [REAL_SESSION_TITLE, 'E2E queued-resume warm history']
  })

  try {
    await installGatewayFrameRecorder(fixture.page)
    await waitForAppReady(fixture)
    await openRealSession(fixture.page, 'E2E queued-resume warm history')

    const surface = activeElectronSurface(fixture.page)
    const viewport = activeElectronViewport(fixture.page)
    const composer = surface.locator('[data-slot="composer-rich-input"]')
    await composer.fill(REAL_RUNNING_PROMPT)
    await surface.getByRole('button', { name: 'Send', exact: true }).click()
    await fixture.mock.waitForHeldStream()
    await expect(viewport).toContainText('Hello')

    const runtimeId = await fixture.page.evaluate(marker => {
      const frames =
        (
          window as typeof window & {
            __messageGapSentFrames?: string[]
          }
        ).__messageGapSentFrames ?? []

      for (const frame of [...frames].reverse()) {
        try {
          const request = JSON.parse(frame) as {
            method?: string
            params?: { session_id?: unknown; text?: unknown }
          }

          if (
            request.method === 'prompt.submit' &&
            request.params?.text === marker &&
            typeof request.params.session_id === 'string'
          ) {
            return request.params.session_id
          }
        } catch {
          // Ignore unrelated non-JSON frames.
        }
      }

      throw new Error('active prompt.submit frame did not expose its runtime id')
    }, REAL_RUNNING_PROMPT)

    const queuedResult = await submitQueuedPrompt(fixture.page, runtimeId, REAL_QUEUED_PROMPT)
    expect(queuedResult).toMatchObject({ status: 'queued' })

    await fixture.app.context().addInitScript(recordGatewayFrames)
    const ownerRoute = await fixture.page.evaluate(async () => {
      const desktop = (
        window as unknown as {
          hermesDesktop?: {
            getConnection: () => Promise<{ connectionId?: string; profile?: string }>
          }
        }
      ).hermesDesktop
      const connection = await desktop?.getConnection()

      if (!connection) {
        throw new Error('isolated Desktop connection is unavailable')
      }

      const profile = connection.profile?.trim() || 'default'

      return {
        connectionId: connection.connectionId?.trim() || 'local',
        profile,
        targetProfile: profile
      }
    })
    const secondaryPagePromise = fixture.app.waitForEvent('window')
    const openResult = await fixture.page.evaluate(
      ({ route, storedSessionId }) => {
        const desktop = (
          window as unknown as {
            hermesDesktop?: {
              openSessionWindow: (sessionId: string, options: { ownerRoute: typeof route }) => Promise<{ ok: boolean }>
            }
          }
        ).hermesDesktop

        return desktop?.openSessionWindow(storedSessionId, { ownerRoute: route })
      },
      { route: ownerRoute, storedSessionId: fixture.sessionId }
    )
    expect(openResult).toEqual({ ok: true })

    const resumedPage = await secondaryPagePromise
    await waitForAppReady({ ...fixture, page: resumedPage })
    await expect(activeElectronViewport(resumedPage)).toContainText('E2E queued-resume warm history', {
      timeout: 60_000
    })
    let resumeResult: unknown = null
    await expect
      .poll(
        async () => {
          resumeResult = await resumedPage.evaluate(() => {
            const recordedWindow = window as typeof window & {
              __messageGapReceivedFrames?: string[]
              __messageGapSentFrames?: string[]
            }
            const requests = (recordedWindow.__messageGapSentFrames ?? []).flatMap(frame => {
              try {
                return [JSON.parse(frame) as { id?: unknown; method?: string }]
              } catch {
                return []
              }
            })
            const resumeId = requests.findLast(request => request.method === 'session.resume')?.id

            if (resumeId === undefined) {
              return null
            }

            for (const frame of recordedWindow.__messageGapReceivedFrames ?? []) {
              try {
                const response = JSON.parse(frame) as { id?: unknown; result?: unknown }

                if (response.id === resumeId) {
                  return response.result ?? null
                }
              } catch {
                // Ignore unrelated non-JSON frames.
              }
            }

            return null
          })

          return resumeResult
        },
        { timeout: 10_000 }
      )
      .not.toBeNull()
    expect(resumeResult).toMatchObject({
      queued: { user: REAL_QUEUED_PROMPT },
      running: true
    })

    await expect(activeElectronViewport(resumedPage)).toContainText(REAL_QUEUED_PROMPT)
    const transportOrderingBeforeDisconnect = {
      aPageClosed: fixture.page.isClosed(),
      aSocketState: await fixture.page.evaluate(
        () =>
          (
            window as typeof window & {
              __messageGapGatewaySocket?: WebSocket
            }
          ).__messageGapGatewaySocket?.readyState ?? null
      ),
      bCompleteCount: await resumedPage.evaluate(() => {
        const frames =
          (
            window as typeof window & {
              __messageGapReceivedFrames?: string[]
            }
          ).__messageGapReceivedFrames ?? []

        return frames.reduce((count, frame) => {
          try {
            const parsed = JSON.parse(frame) as { params?: { type?: unknown } }

            return count + (parsed.params?.type === 'message.complete' ? 1 : 0)
          } catch {
            return count
          }
        }, 0)
      }),
      bSocketState: await resumedPage.evaluate(
        () =>
          (
            window as typeof window & {
              __messageGapGatewaySocket?: WebSocket
            }
          ).__messageGapGatewaySocket?.readyState ?? null
      ),
      activeProviderPromptCount: fixture.mock.receivedPrompts.filter(prompt => prompt.includes(REAL_RUNNING_PROMPT))
        .length,
      queuedProviderPromptCount: fixture.mock.receivedPrompts.filter(prompt => prompt.includes(REAL_QUEUED_PROMPT))
        .length
    }

    expect(transportOrderingBeforeDisconnect).toEqual({
      aPageClosed: false,
      aSocketState: 1,
      bCompleteCount: 0,
      bSocketState: 1,
      activeProviderPromptCount: 1,
      queuedProviderPromptCount: 0
    })
    await fixture.page.close()
    expect(fixture.page.isClosed()).toBe(true)
    await expect
      .poll(() =>
        resumedPage.evaluate(
          () =>
            (
              window as typeof window & {
                __messageGapGatewaySocket?: WebSocket
              }
            ).__messageGapGatewaySocket?.readyState ?? null
        )
      )
      .toBe(1)

    const resumedBeforeRelease = await sampleRealDom(resumedPage)

    expect(resumedBeforeRelease.blankAssistantRootCount).toBe(0)
    expect(resumedBeforeRelease.assistantRootCount).toBe(3)

    fixture.mock.releaseHeldStream()
    await expect
      .poll(() => fixture.mock.receivedPrompts.filter(prompt => prompt.includes(REAL_QUEUED_PROMPT)).length, {
        timeout: 60_000
      })
      .toBe(1)
    await expect
      .poll(
        () =>
          resumedPage.evaluate(() => {
            const frames =
              (
                window as typeof window & {
                  __messageGapReceivedFrames?: string[]
                }
              ).__messageGapReceivedFrames ?? []

            return frames.reduce((count, frame) => {
              try {
                const parsed = JSON.parse(frame) as { params?: { type?: unknown } }

                return count + (parsed.params?.type === 'message.complete' ? 1 : 0)
              } catch {
                return count
              }
            }, 0)
          }),
        { timeout: 60_000 }
      )
      .toBe(2)
    await expect(activeElectronSurface(resumedPage).locator('[data-slot="aui_assistant-message-root"]')).toHaveCount(
      4,
      {
        timeout: 60_000
      }
    )
    await expect(
      activeElectronSurface(resumedPage).locator('[data-slot="aui_assistant-message-root"]').last()
    ).toContainText(MOCK_REPLY, { timeout: 60_000 })

    const tail = await activeElectronSurface(resumedPage).evaluate(surfaceElement =>
      [
        ...surfaceElement.querySelectorAll<HTMLElement>(
          '[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"]'
        )
      ]
        .map(root => {
          const messageId = root.dataset.messageId ?? ''
          const role = root.dataset.slot === 'aui_user-message-root' ? 'user' : 'assistant'
          const content =
            role === 'assistant' ? root.querySelector<HTMLElement>('[data-slot="aui_assistant-message-content"]') : root

          return {
            id: messageId,
            role,
            text: (content?.textContent ?? '').trim().replaceAll(/\s+/g, ' ')
          }
        })
        .slice(-4)
    )

    expect(tail.map(row => ({ role: row.role, text: row.text }))).toEqual([
      { role: 'user', text: REAL_RUNNING_PROMPT },
      { role: 'assistant', text: MOCK_REPLY },
      { role: 'user', text: REAL_QUEUED_PROMPT },
      { role: 'assistant', text: MOCK_REPLY }
    ])
    expect(tail.filter(row => row.role === 'assistant' && row.text !== MOCK_REPLY)).toEqual([])

    const persisted = await fetchTranscriptPage(resumedPage, fixture.sessionId, 0)
    const activeIndex = persisted.messages.findIndex(
      message => message.role === 'user' && message.content === REAL_RUNNING_PROMPT
    )
    const queuedIndex = persisted.messages.findIndex(
      message => message.role === 'user' && message.content === REAL_QUEUED_PROMPT
    )

    expect(activeIndex).toBeGreaterThanOrEqual(0)
    expect(queuedIndex).toBe(activeIndex + 2)
    expect(persisted.messages[activeIndex + 1]).toMatchObject({ role: 'assistant', content: MOCK_REPLY })
    expect(persisted.messages[queuedIndex + 1]).toMatchObject({ role: 'assistant', content: MOCK_REPLY })

    const proof = {
      ownerRoute,
      persisted: persisted.messages.slice(activeIndex, queuedIndex + 2),
      resumeResult,
      resumedBeforeRelease,
      runtimeId,
      tail,
      transportOrderingBeforeDisconnect
    }
    console.log('MESSAGE_GAP_QUEUED_RUNNING_REPLAY', JSON.stringify(proof))
    await testInfo.attach('message-gap-queued-running-replay.json', {
      body: JSON.stringify(proof, null, 2),
      contentType: 'application/json'
    })
  } finally {
    await fixture.cleanup()
  }
})
