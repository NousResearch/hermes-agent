import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import { $browserCurrentState, $browserDriveCommand, openBrowserRail, resetBrowserRegistryForTests } from '@/store/browser'
import { $rightRailActiveTabId, PREVIEW_PANE_ID, RIGHT_RAIL_BROWSER_TAB_ID, RIGHT_RAIL_PREVIEW_TAB_ID } from '@/store/layout'
import { $paneOpen } from '@/store/panes'
import {
  $previewTarget,
  clearSessionPreviewRegistry,
  type PreviewTarget,
  registerSessionPreview
} from '@/store/preview'
import { $activeSessionId, $currentCwd, $messages, $selectedStoredSessionId } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { usePreviewRouting } from './use-preview-routing'

function assistantMessage(id: string, text: string): ChatMessage {
  return {
    id,
    parts: [assistantTextPart(text)],
    role: 'assistant'
  }
}

function previewTarget(source: string): PreviewTarget {
  const isUrl = /^https?:\/\//i.test(source)

  return {
    kind: isUrl ? 'url' : 'file',
    label: source,
    path: isUrl ? undefined : source,
    previewKind: isUrl ? undefined : 'html',
    source,
    url: isUrl ? source : `file://${source}`
  }
}

let handleEvent: (event: RpcEvent) => void = () => undefined

function PreviewRoutingHarness({
  activeSessionId = 'session-1',
  onEvent,
  requestGateway = vi.fn(),
  routedSessionId = 'session-1',
  selectedStoredSessionId = null
}: {
  activeSessionId?: string | null
  onEvent: (handler: (event: RpcEvent) => void) => void
  requestGateway?: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  routedSessionId?: string | null
  selectedStoredSessionId?: string | null
}) {
  const activeSessionIdRef = useRef<string | null>(activeSessionId)

  const routing = usePreviewRouting({
    activeSessionIdRef,
    baseHandleGatewayEvent: vi.fn(),
    currentCwd: '/work',
    currentView: 'chat',
    requestGateway,
    routedSessionId,
    selectedStoredSessionId
  })

  useEffect(() => {
    onEvent(routing.handleDesktopGatewayEvent)
  }, [onEvent, routing.handleDesktopGatewayEvent])

  return null
}

describe('usePreviewRouting', () => {
  beforeEach(() => {
    resetBrowserRegistryForTests()
    $activeSessionId.set('session-1')
    $selectedStoredSessionId.set(null)
    $currentCwd.set('/work')
    $messages.set([])
    $previewTarget.set(null)
    window.localStorage.clear()
    clearSessionPreviewRegistry()
    handleEvent = () => undefined

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        normalizePreviewTarget: vi.fn(async (target: string) => previewTarget(target))
      }
    })
  })

  afterEach(() => {
    cleanup()
    resetBrowserRegistryForTests()
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $messages.set([])
    $previewTarget.set(null)
    window.localStorage.clear()
    clearSessionPreviewRegistry()
    vi.restoreAllMocks()
  })

  it('opens the active session preview from the registry', async () => {
    const target = previewTarget('/work/demo.html')

    registerSessionPreview('session-1', target, 'tool-result')
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
      />
    )

    await waitFor(() => {
      expect($previewTarget.get()).toEqual({ ...target, renderMode: 'preview' })
    })
  })

  it('does not infer previews from assistant prose', async () => {
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
      />
    )

    act(() => {
      $messages.set([
        assistantMessage('a1', 'Preview: http://localhost:5173/'),
        assistantMessage('a2', 'Open /work/demo.html')
      ])
    })

    expect($previewTarget.get()).toBeNull()
    expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
  })

  it('does not auto-open a preview from tool results', async () => {
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
      />
    )

    act(() =>
      handleEvent({
        payload: { inline_diff: '\u001b[38;2;218;165;32ma/preview-demo.html -> b/preview-demo.html\u001b[0m\n' },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )
    act(() => handleEvent({ payload: { path: './dist/index.html' }, session_id: 'session-1', type: 'tool.complete' }))

    expect($previewTarget.get()).toBeNull()
    expect(JSON.parse(window.localStorage.getItem('hermes.desktop.sessionPreviews.v1') ?? '{}')).toEqual({})
  })

  it('routes active browser drive events to the browser rail', () => {
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
      />
    )

    act(() =>
      handleEvent({
        payload: { action: 'navigate', title: 'Example', url: 'https://example.com/' },
        session_id: 'session-1',
        type: 'browser.drive'
      })
    )

    expect($browserCurrentState.get()).toMatchObject({ title: 'Example', url: 'https://example.com/' })
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_BROWSER_TAB_ID)
  })

  it('routes browser DOM action payloads to the active browser session', () => {
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
      />
    )

    act(() =>
      handleEvent({
        payload: { action: 'act', domAction: { index: 3, kind: 'type', text: 'hello' }, requestId: 'act-1' },
        session_id: 'session-1',
        type: 'browser.drive'
      })
    )

    expect($browserDriveCommand.get()).toMatchObject({
      action: 'act',
      domAction: { index: 3, kind: 'type', text: 'hello' },
      requestId: 'act-1',
      sessionId: 'session-1'
    })
  })

  it('answers active browser snapshot requests with the live Electron webview result', async () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    const snapshot = {
      capturedAt: 123,
      elements: [],
      headings: [],
      ok: true,
      sessionId: 'session-1',
      tables: [],
      text: '客户消息',
      title: '抖音',
      url: 'https://www.douyin.com/'
    }
    window.hermesDesktop.browser = {
      ...window.hermesDesktop.browser,
      navigate: vi.fn().mockResolvedValue(undefined),
      open: vi.fn().mockResolvedValue(undefined),
      snapshot: vi.fn().mockResolvedValue(snapshot)
    }
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
        requestGateway={requestGateway}
      />
    )

    act(() =>
      handleEvent({
        payload: { operation: 'snapshot', payload: {}, request_id: 'browser-1' },
        session_id: 'session-1',
        type: 'browser.request'
      })
    )

    await waitFor(() => expect(requestGateway).toHaveBeenCalled())
    const [, response] = requestGateway.mock.calls[0]
    expect(requestGateway.mock.calls[0][0]).toBe('browser.respond')
    expect(response.request_id).toBe('browser-1')
    expect(JSON.parse(response.text)).toEqual({ ok: true, snapshot })
  })

  it('rejects browser requests for an inactive session without touching its webview', async () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    const snapshot = vi.fn()
    window.hermesDesktop.browser = {
      ...window.hermesDesktop.browser,
      navigate: vi.fn().mockResolvedValue(undefined),
      open: vi.fn().mockResolvedValue(undefined),
      snapshot
    }
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
        requestGateway={requestGateway}
      />
    )

    act(() =>
      handleEvent({
        payload: { operation: 'snapshot', payload: {}, request_id: 'browser-2' },
        session_id: 'session-2',
        type: 'browser.request'
      })
    )

    await waitFor(() => expect(requestGateway).toHaveBeenCalled())
    expect(snapshot).not.toHaveBeenCalled()
    const response = JSON.parse(requestGateway.mock.calls[0][1].text)
    expect(response.ok).toBe(false)
    expect(response.error).toContain('active Desktop session')
  })

  it('does not answer navigation with a snapshot from the previous URL', async () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    const oldSnapshot = {
      capturedAt: 123,
      elements: [],
      headings: [],
      ok: true,
      sessionId: 'session-1',
      tables: [],
      text: 'old page',
      title: 'Old',
      url: 'https://old.example/'
    }
    const newSnapshot = {
      ...oldSnapshot,
      capturedAt: 456,
      text: 'new page',
      title: 'New',
      url: 'https://new.example/'
    }
    const snapshot = vi.fn().mockResolvedValueOnce(oldSnapshot).mockResolvedValue(newSnapshot)
    window.hermesDesktop.browser = {
      ...window.hermesDesktop.browser,
      getState: vi.fn().mockResolvedValue({ loading: false, url: 'https://old.example/' }),
      navigate: vi.fn().mockResolvedValue(undefined),
      open: vi.fn().mockResolvedValue(undefined),
      snapshot
    }
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
        requestGateway={requestGateway}
      />
    )

    act(() =>
      handleEvent({
        payload: {
          operation: 'navigate',
          payload: { url: 'https://new.example/' },
          request_id: 'browser-nav'
        },
        session_id: 'session-1',
        type: 'browser.request'
      })
    )

    await waitFor(() => expect(requestGateway).toHaveBeenCalled(), { timeout: 3_000 })
    const response = JSON.parse(requestGateway.mock.calls[0][1].text)
    expect(snapshot).toHaveBeenCalledTimes(2)
    expect(response.snapshot.url).toBe('https://new.example/')
  })

  it('closes the browser rail when the active session has no browser record', async () => {
    openBrowserRail('example.com', 'session-1')
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)

    render(
      <PreviewRoutingHarness
        activeSessionId="session-2"
        onEvent={handler => {
          handleEvent = handler
        }}
        routedSessionId="session-2"
      />
    )

    await waitFor(() => {
      expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(false)
    })
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_PREVIEW_TAB_ID)
  })
})
