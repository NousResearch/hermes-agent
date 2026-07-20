import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import { $fileBrowserOpen, setFileBrowserOpen } from '@/store/layout'
import {
  $previewTarget,
  clearSessionPreviewRegistry,
  type PreviewTarget,
  registerSessionPreview,
  setPreviewTarget
} from '@/store/preview'
import { $currentCwd, $messages } from '@/store/session'
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

// Mirrors controller.tsx's synchronous `$previewTarget` reveal listener. The
// hook under test is the session-restore path that must preserve the user's
// persisted side visibility after that listener runs.
function bindPreviewReveal() {
  return $previewTarget.listen(target => {
    if (target) {
      setFileBrowserOpen(true)
    }
  })
}

let handleEvent: (event: RpcEvent) => void = () => undefined
let unbindReveal: (() => void) | null = null

function PreviewRoutingHarness() {
  const activeSessionIdRef = useRef<string | null>('session-1')

  const routing = usePreviewRouting({
    activeSessionIdRef,
    baseHandleGatewayEvent: vi.fn(),
    currentCwd: '/work',
    currentView: 'chat',
    requestGateway: vi.fn(),
    routedSessionId: 'session-1',
    selectedStoredSessionId: null
  })

  useEffect(() => {
    handleEvent = routing.handleDesktopGatewayEvent
  }, [routing.handleDesktopGatewayEvent])

  return null
}

function renderPreviewRouting() {
  return render(<PreviewRoutingHarness />)
}

describe('usePreviewRouting', () => {
  beforeEach(() => {
    $currentCwd.set('/work')
    $messages.set([])
    $previewTarget.set(null)
    setFileBrowserOpen(false)
    clearSessionPreviewRegistry()
    handleEvent = () => undefined
    unbindReveal = null
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
    unbindReveal?.()
    unbindReveal = null
    $messages.set([])
    $previewTarget.set(null)
    setFileBrowserOpen(false)
    vi.restoreAllMocks()
    clearSessionPreviewRegistry()
    window.localStorage.clear()
  })

  it('opens the active session preview from the registry', async () => {
    const target = previewTarget('/work/demo.html')

    registerSessionPreview('session-1', target, 'tool-result')
    renderPreviewRouting()

    await waitFor(() => {
      expect($previewTarget.get()).toEqual({ ...target, renderMode: 'preview' })
    })
  })

  it('restores a session preview without expanding a user-collapsed right rail', async () => {
    const target = previewTarget('/work/demo.html')

    unbindReveal = bindPreviewReveal()
    setFileBrowserOpen(false)
    registerSessionPreview('session-1', target, 'tool-result')
    renderPreviewRouting()

    await waitFor(() => {
      expect($previewTarget.get()).toEqual({ ...target, renderMode: 'preview' })
    })
    expect($fileBrowserOpen.get()).toBe(false)
  })

  it('keeps an already-open right rail open while restoring a session preview', async () => {
    const target = previewTarget('/work/demo.html')

    unbindReveal = bindPreviewReveal()
    setFileBrowserOpen(true)
    registerSessionPreview('session-1', target, 'tool-result')
    renderPreviewRouting()

    await waitFor(() => {
      expect($previewTarget.get()).toEqual({ ...target, renderMode: 'preview' })
    })
    expect($fileBrowserOpen.get()).toBe(true)
  })

  it('still expands a collapsed rail for an explicit preview target', () => {
    const target = previewTarget('/work/demo.html')

    unbindReveal = bindPreviewReveal()
    setFileBrowserOpen(false)
    setPreviewTarget(target)

    expect($fileBrowserOpen.get()).toBe(true)
  })

  it('does not infer previews from assistant prose', async () => {
    renderPreviewRouting()

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
    renderPreviewRouting()

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
