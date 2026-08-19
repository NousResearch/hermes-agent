import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import { $previewTabs, $previewTarget, closeRightRail, type PreviewTarget } from '@/store/preview'
import { $activeSessionId, $currentCwd, $messages, $selectedStoredSessionId } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { usePreviewRouting } from './use-preview-routing'

const RUNTIME_SESSION_ID = '20260727_140707_edec2d'

function assistantMessage(id: string, text: string): ChatMessage {
  return { id, parts: [assistantTextPart(text)], role: 'assistant' }
}

function fileTarget(path: string): PreviewTarget {
  return { kind: 'file', label: path, path, previewKind: 'html', source: path, url: `file://${path}` }
}

let handleEvent: (event: RpcEvent) => void = () => undefined

function Harness() {
  const routing = usePreviewRouting({
    baseHandleGatewayEvent: vi.fn(),
    currentCwd: '/work',
    requestGateway: vi.fn()
  })

  useEffect(() => {
    handleEvent = routing.handleDesktopGatewayEvent
  }, [routing.handleDesktopGatewayEvent])

  return null
}

async function emitPreviewOpen(url = '/tmp/artifact-test.html', sessionId = RUNTIME_SESSION_ID) {
  await act(async () => {
    handleEvent({
      payload: { label: 'hi bestie', url },
      session_id: sessionId,
      type: 'preview.open'
    } as unknown as RpcEvent)
  })
}

async function emitPreviewClose(url?: string, sessionId = RUNTIME_SESSION_ID) {
  await act(async () => {
    handleEvent({
      payload: url === undefined ? {} : { url },
      session_id: sessionId,
      type: 'preview.close'
    } as unknown as RpcEvent)
  })
}

describe('preview routing', () => {
  beforeEach(() => {
    // A live session always has a runtime id; only the STORED id lags.
    $activeSessionId.set(RUNTIME_SESSION_ID)
    $currentCwd.set('/work')
    $messages.set([])
    closeRightRail()
    window.localStorage.clear()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { normalizePreviewTarget: vi.fn(async (target: string) => fileTarget(target)) }
    })
  })

  afterEach(() => {
    cleanup()
    $messages.set([])
    closeRightRail()
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    window.localStorage.clear()
    vi.restoreAllMocks()
  })

  describe('open_preview', () => {
    // The rail used to hold a session-keyed singleton alongside its tabs, written
    // under one session-id rule and reconciled under another. A live session with
    // no stored id yet resolved to '' on the write side, so the target was set and
    // then immediately reconciled back to null — the pane flashed and vanished.
    it('opens a tab for a session that has no stored id yet', async () => {
      $selectedStoredSessionId.set(null)
      render(<Harness />)

      await emitPreviewOpen()

      await waitFor(() => expect($previewTarget.get()?.path).toBe('/tmp/artifact-test.html'))
      expect($previewTabs.get()).toHaveLength(1)
    })

    it('keeps the tab when the stored session id arrives afterwards', async () => {
      $selectedStoredSessionId.set(null)
      render(<Harness />)

      await emitPreviewOpen()
      await waitFor(() => expect($previewTarget.get()?.path).toBe('/tmp/artifact-test.html'))

      await act(async () => {
        $selectedStoredSessionId.set('stored-1')
      })

      expect($previewTarget.get()?.path).toBe('/tmp/artifact-test.html')
    })

    it('ignores an open from a session that is not the one on screen', async () => {
      render(<Harness />)

      await act(async () => {
        handleEvent({
          payload: { url: '/tmp/other.html' },
          session_id: 'some-other-session',
          type: 'preview.open'
        } as unknown as RpcEvent)
      })

      expect($previewTabs.get()).toHaveLength(0)
    })

    // The turn that calls open_preview is often a TILE's session while focus
    // sits on main (the user asked, then clicked elsewhere). On-screen is the
    // bar — gating on focus made an explicit "open reddit" silently vanish.
    it('honors an open from an open tile session even when main holds focus', async () => {
      const { $sessionTiles } = await import('@/store/session-states')
      const tiles = $sessionTiles.get()

      $sessionTiles.set([{ dir: 'right', runtimeId: 'tile-runtime', storedSessionId: 'stored-tile' }])
      render(<Harness />)

      try {
        await emitPreviewOpen('/tmp/from-tile.html', 'tile-runtime')

        await waitFor(() => expect($previewTarget.get()?.path).toBe('/tmp/from-tile.html'))
      } finally {
        $sessionTiles.set(tiles)
      }
    })

    it('opens a second target as its own tab rather than replacing the first', async () => {
      render(<Harness />)

      await emitPreviewOpen('/tmp/one.html')
      await waitFor(() => expect($previewTabs.get()).toHaveLength(1))

      await emitPreviewOpen('/tmp/two.html')
      await waitFor(() => expect($previewTabs.get()).toHaveLength(2))

      expect($previewTarget.get()?.path).toBe('/tmp/two.html')
    })

    it('re-fronts the existing tab when the same target opens twice', async () => {
      render(<Harness />)

      await emitPreviewOpen('/tmp/one.html')
      await emitPreviewOpen('/tmp/one.html')

      await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
    })

    it('renders a tool-opened html file rather than showing its source', async () => {
      render(<Harness />)

      await emitPreviewOpen()

      await waitFor(() => expect($previewTarget.get()?.renderMode).toBe('preview'))
    })

    // Offer, don't hijack: only an explicit open_preview call opens the rail.
    it('does not infer a preview from assistant prose', async () => {
      render(<Harness />)

      await act(async () => {
        $messages.set([
          assistantMessage('a1', 'Preview: http://localhost:5173/'),
          assistantMessage('a2', 'Open /work/demo.html')
        ])
      })

      expect($previewTabs.get()).toHaveLength(0)
      expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
    })

    it('does not open a preview off the back of a tool result', async () => {
      render(<Harness />)

      await act(async () => {
        handleEvent({
          payload: { inline_diff: 'a/preview-demo.html -> b/preview-demo.html\n' },
          session_id: RUNTIME_SESSION_ID,
          type: 'tool.complete'
        } as unknown as RpcEvent)
        handleEvent({
          payload: { path: './dist/index.html' },
          session_id: RUNTIME_SESSION_ID,
          type: 'tool.complete'
        } as unknown as RpcEvent)
      })

      expect($previewTabs.get()).toHaveLength(0)
    })
  })

  describe('close_preview', () => {
    it('closes the whole pane when no url is given', async () => {
      render(<Harness />)

      await emitPreviewOpen('/tmp/one.html')
      await emitPreviewOpen('/tmp/two.html')
      await waitFor(() => expect($previewTabs.get()).toHaveLength(2))

      await emitPreviewClose('')

      expect($previewTabs.get()).toHaveLength(0)
      expect($previewTarget.get()).toBeNull()
    })

    it('closes only the matching tab when a url is given', async () => {
      render(<Harness />)

      await emitPreviewOpen('/tmp/keep.html')
      await emitPreviewOpen('/tmp/drop.html')
      await waitFor(() => expect($previewTabs.get()).toHaveLength(2))

      await emitPreviewClose('/tmp/drop.html')

      await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
      expect($previewTarget.get()?.path).toBe('/tmp/keep.html')
    })

    it('ignores a close from a session that is not the one on screen', async () => {
      render(<Harness />)

      await emitPreviewOpen('/tmp/stay.html')
      await waitFor(() => expect($previewTabs.get()).toHaveLength(1))

      await emitPreviewClose('/tmp/stay.html', 'some-other-session')

      expect($previewTabs.get()).toHaveLength(1)
    })

    it('honors a close from an open tile session even when main holds focus', async () => {
      const { $sessionTiles } = await import('@/store/session-states')
      const tiles = $sessionTiles.get()

      $sessionTiles.set([{ dir: 'right', runtimeId: 'tile-runtime', storedSessionId: 'stored-tile' }])
      render(<Harness />)

      try {
        await emitPreviewOpen('/tmp/from-tile.html', 'tile-runtime')
        await waitFor(() => expect($previewTabs.get()).toHaveLength(1))

        await emitPreviewClose('/tmp/from-tile.html', 'tile-runtime')

        await waitFor(() => expect($previewTabs.get()).toHaveLength(0))
      } finally {
        $sessionTiles.set(tiles)
      }
    })

    it('is a no-op when nothing is open', async () => {
      render(<Harness />)

      await emitPreviewClose()

      expect($previewTabs.get()).toHaveLength(0)
    })
  })
})
