import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import {
  $filePreviewTabs,
  $previewTarget,
  clearSessionPreviewRegistry,
  type PreviewTarget,
  registerSessionPreview,
  setCurrentSessionPreviewTarget
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
  onEvent,
  routedSessionId = 'session-1'
}: {
  onEvent: (handler: (event: RpcEvent) => void) => void
  routedSessionId?: string
}) {
  const activeSessionIdRef = useRef<string | null>(routedSessionId)
  activeSessionIdRef.current = routedSessionId

  const routing = usePreviewRouting({
    activeSessionIdRef,
    baseHandleGatewayEvent: vi.fn(),
    currentCwd: '/work',
    currentView: 'chat',
    requestGateway: vi.fn(),
    routedSessionId,
    selectedStoredSessionId: null
  })

  useEffect(() => {
    onEvent(routing.handleDesktopGatewayEvent)
  }, [onEvent, routing.handleDesktopGatewayEvent])

  return null
}

describe('usePreviewRouting', () => {
  beforeEach(() => {
    $currentCwd.set('/work')
    $messages.set([])
    $previewTarget.set(null)
    $activeSessionId.set('session-1')
    $selectedStoredSessionId.set(null)
    clearSessionPreviewRegistry()
    handleEvent = () => undefined
    window.localStorage.clear()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        normalizePreviewTarget: vi.fn(async (target: string) => previewTarget(target))
      }
    })
  })

  afterEach(() => {
    cleanup()
    $messages.set([])
    $previewTarget.set(null)
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $filePreviewTabs.set([])
    vi.restoreAllMocks()
    clearSessionPreviewRegistry()
    window.localStorage.clear()
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

  it('dismisses a file preview tab when switching to another conversation', async () => {
    const file = previewTarget('/work/attachment.png')

    // Opened while session-1 is active → tagged with session-1.
    setCurrentSessionPreviewTarget(file, 'manual')
    expect($filePreviewTabs.get()).toHaveLength(1)

    const { rerender } = render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
        routedSessionId="session-1"
      />
    )

    // The file belongs to session-1, so it survives while that conversation is active.
    expect($filePreviewTabs.get()).toHaveLength(1)

    // Switching conversations moves both the active session and the route.
    act(() => {
      $activeSessionId.set('session-2')
    })
    rerender(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
        routedSessionId="session-2"
      />
    )

    await waitFor(() => {
      expect($filePreviewTabs.get()).toEqual([])
    })
  })

  it('keeps a tab scoped to the routed session when the runtime session store diverges', async () => {
    const file = previewTarget('/work/attachment.png')

    // Opened while session-1 is active → tagged with session-1.
    setCurrentSessionPreviewTarget(file, 'manual')
    expect($filePreviewTabs.get()[0]?.sessionId).toBe('session-1')

    // A distinct live preview for session-1 lets us observe that the effect
    // routes to session-1 (not the diverged runtime session-2).
    const livePreview = previewTarget('/work/live.html')
    registerSessionPreview('session-1', livePreview, 'tool-result')

    // The runtime `$activeSessionId` store advances to session-2, but the hook
    // is still routing to session-1 (routedSessionId). Scoping the sync to a
    // re-derived current id (which reads the session-2 store) would wrongly drop
    // the session-1 tab we are still viewing; scoping to the routed id keeps it.
    act(() => {
      $activeSessionId.set('session-2')
    })
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
        routedSessionId="session-1"
      />
    )

    // The effect ran and routed to session-1's live preview...
    await waitFor(() => {
      expect($previewTarget.get()).toEqual({ ...livePreview, renderMode: 'preview' })
    })
    // ...and the session-1 file tab was kept, not dropped by the stale runtime id.
    expect($filePreviewTabs.get()).toHaveLength(1)
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
    expect(window.localStorage.getItem('hermes.desktop.sessionPreviews.v1')).toBeNull()
  })
})
