import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import {
  $previewTarget,
  clearSessionPreviewRegistry,
  type PreviewTarget,
  registerSessionPreview
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

let handleEvent: (event: RpcEvent) => void = () => undefined

function PreviewRoutingHarness({ onEvent }: { onEvent: (handler: (event: RpcEvent) => void) => void }) {
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
    onEvent(routing.handleDesktopGatewayEvent)
  }, [onEvent, routing.handleDesktopGatewayEvent])

  return null
}

describe('usePreviewRouting', () => {
  beforeEach(() => {
    $currentCwd.set('/work')
    $messages.set([])
    $previewTarget.set(null)
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

  it('opens the preview pane on a preview.open event for the active session', async () => {
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
      />
    )

    act(() =>
      handleEvent({
        payload: { url: 'https://www.cnn.com', label: 'CNN' },
        session_id: 'session-1',
        type: 'preview.open'
      })
    )

    await waitFor(() => {
      expect($previewTarget.get()).toMatchObject({ kind: 'url', label: 'CNN', url: 'https://www.cnn.com' })
    })
  })

  it('ignores a preview.open event for a background session', async () => {
    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
      />
    )

    act(() =>
      handleEvent({ payload: { url: 'https://www.cnn.com' }, session_id: 'other-session', type: 'preview.open' })
    )

    // Give any (wrongly) scheduled async open a tick to resolve before asserting.
    await Promise.resolve()
    expect($previewTarget.get()).toBeNull()
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
