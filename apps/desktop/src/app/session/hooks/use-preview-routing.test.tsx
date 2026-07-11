import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import {
  $filePreviewTabs,
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
    expect(window.localStorage.getItem('hermes.desktop.sessionPreviews.v1')).toBeNull()
  })

  it('opens an explicitly requested read_file path as an editor tab with cwd semantics', async () => {
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() =>
      handleEvent({
        payload: {
          args: { path: './notes.txt', preview: true },
          name: 'read_file',
          tool_id: 'tool-preview-1'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )

    await waitFor(() => expect($filePreviewTabs.get()).toHaveLength(1))
    expect(window.hermesDesktop.normalizePreviewTarget).toHaveBeenCalledWith('./notes.txt', '/work')
    expect($filePreviewTabs.get()[0]?.target).toMatchObject({ renderMode: 'source', source: './notes.txt' })
  })

  it('deduplicates explicit preview requests by tool id', async () => {
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    const event: RpcEvent = {
      payload: { args: { path: 'notes.txt', preview: true }, name: 'read_file', tool_id: 'same-tool' },
      session_id: 'session-1',
      type: 'tool.complete'
    }

    act(() => {
      handleEvent(event)
      handleEvent(event)
    })

    await waitFor(() => expect($filePreviewTabs.get()).toHaveLength(1))
    expect(window.hermesDesktop.normalizePreviewTarget).toHaveBeenCalledTimes(1)
  })

  it('ignores ordinary reads, other tools, other sessions, and empty paths', () => {
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() => {
      handleEvent({
        payload: { args: { path: 'ordinary.txt' }, name: 'read_file', tool_id: 'ordinary' },
        session_id: 'session-1',
        type: 'tool.complete'
      })
      handleEvent({
        payload: { args: { path: 'other.txt', preview: true }, name: 'write_file', tool_id: 'other-tool' },
        session_id: 'session-1',
        type: 'tool.complete'
      })
      handleEvent({
        payload: { args: { path: 'other-session.txt', preview: true }, name: 'read_file', tool_id: 'other-session' },
        session_id: 'session-2',
        type: 'tool.complete'
      })
      handleEvent({
        payload: { args: { path: '  ', preview: true }, name: 'read_file', tool_id: 'empty' },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    })

    expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
    expect($filePreviewTabs.get()).toEqual([])
  })

  it('contains preview normalization failures without opening a surface', async () => {
    vi.mocked(window.hermesDesktop.normalizePreviewTarget).mockRejectedValueOnce(new Error('unavailable'))
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() =>
      handleEvent({
        payload: { args: { path: 'notes.txt', preview: true }, name: 'read_file', tool_id: 'failed-normalize' },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )

    await waitFor(() => expect(window.hermesDesktop.normalizePreviewTarget).toHaveBeenCalled())
    expect($filePreviewTabs.get()).toEqual([])
    expect($previewTarget.get()).toBeNull()
  })
})
