import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import {
  $filePreviewTabs,
  $previewTarget,
  $webPreviewTabs,
  clearSessionPreviewRegistry,
  type PreviewTarget,
  registerSessionPreview,
  setSessionPreviewTarget
} from '@/store/preview'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $currentCwd, $messages } from '@/store/session'
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
let baseHandleGatewayEvent = vi.fn()

interface PreviewRoutingHarnessProps {
  activeSessionId?: string | null
  currentView?: string
  onEvent: (handler: (event: RpcEvent) => void) => void
  routedSessionId?: string | null
  selectedStoredSessionId?: string | null
}

function PreviewRoutingHarness({
  activeSessionId = 'session-1',
  currentView = 'chat',
  onEvent,
  routedSessionId = 'session-1',
  selectedStoredSessionId = null
}: PreviewRoutingHarnessProps) {
  const activeSessionIdRef = useRef<string | null>(activeSessionId)

  const routing = usePreviewRouting({
    activeSessionIdRef,
    baseHandleGatewayEvent,
    currentCwd: '/work',
    currentView,
    requestGateway: vi.fn(),
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
    $activeGatewayProfile.set('default')
    $connection.set(null)
    $currentCwd.set('/work')
    $messages.set([])
    $previewTarget.set(null)
    window.localStorage.clear()
    clearSessionPreviewRegistry()
    handleEvent = () => undefined
    baseHandleGatewayEvent = vi.fn()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: vi.fn(async () => ({ binary: false, byteSize: 12, content: 'remote text', language: 'text' })),
        normalizePreviewTarget: vi.fn(async (target: string) => previewTarget(target))
      }
    })
  })

  afterEach(() => {
    $activeGatewayProfile.set('default')
    $connection.set(null)
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

  it('does not duplicate a registered URL that already has a browser tab', async () => {
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() => setSessionPreviewTarget('session-1', previewTarget('https://example.com/app'), 'manual'))

    await waitFor(() => expect($webPreviewTabs.get()).toHaveLength(1))
    expect($previewTarget.get()).toBeNull()
  })

  it('does not restore a stale active-session preview on the new-chat route', async () => {
    const target = previewTarget('/work/stale.html')
    registerSessionPreview('session-1', target, 'tool-result')

    render(
      <PreviewRoutingHarness
        onEvent={handler => {
          handleEvent = handler
        }}
        routedSessionId={null}
      />
    )

    await waitFor(() => expect($previewTarget.get()).toBeNull())
  })

  it('ignores explicit preview completions when no chat session is routed', () => {
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} routedSessionId={null} />)

    act(() =>
      handleEvent({
        payload: {
          args: { path: './stale.txt', preview: true },
          name: 'read_file',
          tool_id: 'stale-preview'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )

    expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
    expect($filePreviewTabs.get()).toEqual([])
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
          preview_success: true,
          tool_id: 'tool-preview-1'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )

    await waitFor(() => expect($filePreviewTabs.get()).toHaveLength(1))
    expect(baseHandleGatewayEvent).not.toHaveBeenCalled()
    expect(window.hermesDesktop.normalizePreviewTarget).toHaveBeenCalledWith('./notes.txt', '/work')
    expect($filePreviewTabs.get()[0]?.target).toMatchObject({ renderMode: 'source', source: './notes.txt' })
  })

  it('does not open a failed or denied explicit preview read', async () => {
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() =>
      handleEvent({
        payload: {
          args: { path: '~/.hermes/auth.json', preview: true },
          name: 'read_file',
          preview_success: false,
          result: { error: 'Access denied' },
          tool_id: 'denied-preview'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )

    expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
    expect($filePreviewTabs.get()).toEqual([])
  })

  it('normalizes explicit preview paths against the active remote connection', async () => {
    $connection.set({ baseUrl: 'https://remote.example', mode: 'remote', profile: 'remote-dev' } as never)
    $currentCwd.set('/srv/project')
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() =>
      handleEvent({
        payload: {
          args: { path: './remote.txt', preview: true },
          name: 'read_file',
          preview_success: true,
          tool_id: 'remote-preview'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )

    await waitFor(() => expect($filePreviewTabs.get()).toHaveLength(1))
    expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
    expect($filePreviewTabs.get()[0]?.target).toMatchObject({
      path: '/srv/project/remote.txt',
      renderMode: 'source',
      source: './remote.txt'
    })
  })

  it('deduplicates explicit preview requests by tool id', async () => {
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    const event: RpcEvent = {
      payload: {
        args: { path: 'notes.txt', preview: true },
        name: 'read_file',
        preview_success: true,
        tool_id: 'same-tool'
      },
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

  it('does not open a preview when normalization finishes after leaving the session', async () => {
    let resolveNormalize: ((target: PreviewTarget) => void) | undefined
    vi.mocked(window.hermesDesktop.normalizePreviewTarget).mockReturnValueOnce(
      new Promise(resolve => {
        resolveNormalize = resolve
      })
    )
    const view = render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() =>
      handleEvent({
        payload: {
          args: { path: 'late.txt', preview: true },
          name: 'read_file',
          preview_success: true,
          tool_id: 'late-tool'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )
    view.rerender(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} routedSessionId={null} />)

    await act(async () => resolveNormalize?.(previewTarget('/work/late.txt')))

    expect($filePreviewTabs.get()).toEqual([])
  })

  it('does not open a preview when normalization finishes after a profile switch', async () => {
    let resolveNormalize: ((target: PreviewTarget) => void) | undefined

    vi.mocked(window.hermesDesktop.normalizePreviewTarget).mockReturnValueOnce(
      new Promise(resolve => {
        resolveNormalize = resolve
      })
    )
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() =>
      handleEvent({
        payload: {
          args: { path: 'late-profile.txt', preview: true },
          name: 'read_file',
          preview_success: true,
          tool_id: 'late-profile-tool'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )
    act(() => $activeGatewayProfile.set('other-profile'))

    await act(async () => resolveNormalize?.(previewTarget('/work/late-profile.txt')))

    expect($filePreviewTabs.get()).toEqual([])
  })

  it('bounds remembered preview tool ids', async () => {
    render(<PreviewRoutingHarness onEvent={handler => (handleEvent = handler)} />)

    act(() => {
      for (let index = 0; index <= 512; index += 1) {
        handleEvent({
          payload: {
            args: { path: `notes-${index}.txt`, preview: true },
            name: 'read_file',
            preview_success: true,
            tool_id: `bounded-${index}`
          },
          session_id: 'session-1',
          type: 'tool.complete'
        })
      }
    })

    await waitFor(() => expect(window.hermesDesktop.normalizePreviewTarget).toHaveBeenCalledTimes(513))
    act(() =>
      handleEvent({
        payload: {
          args: { path: 'notes-0.txt', preview: true },
          name: 'read_file',
          preview_success: true,
          tool_id: 'bounded-0'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )
    await waitFor(() => expect(window.hermesDesktop.normalizePreviewTarget).toHaveBeenCalledTimes(514))
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
        payload: {
          args: { path: 'notes.txt', preview: true },
          name: 'read_file',
          preview_success: true,
          tool_id: 'failed-normalize'
        },
        session_id: 'session-1',
        type: 'tool.complete'
      })
    )

    await waitFor(() => expect(window.hermesDesktop.normalizePreviewTarget).toHaveBeenCalled())
    expect($filePreviewTabs.get()).toEqual([])
    expect($previewTarget.get()).toBeNull()
  })
})
