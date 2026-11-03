import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerPreviewPageReader } from '@/app/chat/right-rail/preview-reader'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $browserPages,
  $previewTabs,
  $previewTarget,
  $visiblePreviewTabs,
  closeRightRail,
  markBrowserTabPopped,
  noteBrowserPage,
  openPreview,
  type PreviewTarget
} from '@/store/preview'
import { $activeSessionId, $currentCwd, $messages, $selectedStoredSessionId, setSessions } from '@/store/session'
import {
  $sessionStates,
  $sessionTiles,
  dropSessionState,
  publishSessionState,
  recordSessionEventScope
} from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, type ServerRequestContext } from './use-message-stream/gateway-event/server-requests'
import { usePreviewRouting } from './use-preview-routing'

vi.mock('@/app/chat/right-rail/preview-popout-bridge', () => ({
  hasLivePreviewSurface: () => true
}))

const RUNTIME_SESSION_ID = '20260727_140707_edec2d'

function assistantMessage(id: string, text: string): ChatMessage {
  return { id, parts: [assistantTextPart(text)], role: 'assistant' }
}

function fileTarget(path: string): PreviewTarget {
  return { kind: 'file', label: path, path, previewKind: 'html', source: path, url: `file://${path}` }
}

function urlTarget(url: string): PreviewTarget {
  return { kind: 'url', label: url, source: url, url }
}

let handleEvent: (event: GatewayEvent) => void = () => undefined

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
    } as unknown as GatewayEvent)
  })
}

async function emitPreviewClose(url?: string, sessionId = RUNTIME_SESSION_ID) {
  await act(async () => {
    handleEvent({
      payload: url === undefined ? {} : { url },
      session_id: sessionId,
      type: 'preview.close'
    } as unknown as GatewayEvent)
  })
}

describe('preview routing', () => {
  beforeEach(() => {
    // A live session always has a runtime id; only the STORED id lags.
    $activeSessionId.set(RUNTIME_SESSION_ID)
    $currentCwd.set('/work')
    $messages.set([])
    $browserPages.set({})
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
    $browserPages.set({})
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

    it('opens an owned tab for a compression-rotated runtime of the visible conversation', async () => {
      $activeSessionId.set('stored-root')
      setSessions([{ id: 'stored-tip', _lineage_root_id: 'stored-root', _lineage_ids: ['stored-root'] } as SessionInfo])
      publishSessionState('rotated-runtime', createClientSessionState('stored-tip'))
      render(<Harness />)

      try {
        await emitPreviewOpen('/tmp/rotated.html', 'rotated-runtime')

        await waitFor(() =>
          expect($previewTabs.get().map(tab => [tab.target.path, tab.sessionId])).toEqual([
            ['/tmp/rotated.html', 'stored-tip']
          ])
        )
      } finally {
        $sessionStates.set({})
        setSessions([])
      }
    })

    it.each(['main stored', 'tile lineage'])(
      'opens a visible owned tab from a %s id and reads it through its runtime',
      async surface => {
        const runtimeId = surface === 'main stored' ? 'main-runtime' : 'tile-runtime'
        const storedId = surface === 'main stored' ? 'stored-main' : 'stored-tip'
        const addressedId = surface === 'main stored' ? storedId : 'stored-root'
        $activeSessionId.set('main-runtime')
        $selectedStoredSessionId.set('stored-main')
        publishSessionState('main-runtime', createClientSessionState('stored-main'))
        publishSessionState(runtimeId, createClientSessionState(storedId))
        setSessions([
          { id: 'stored-tip', _lineage_root_id: 'stored-root', _lineage_ids: ['stored-root'] } as SessionInfo
        ])

        if (surface === 'tile lineage') {
          $sessionTiles.set([{ dir: 'right', runtimeId, storedSessionId: storedId }])
          $layoutTree.set(group([`session-tile:${storedId}`], { active: `session-tile:${storedId}`, id: 'tile-group' }))
          noteActiveTreeGroup('tile-group')
        }

        render(<Harness />)

        let unbind = () => {}

        try {
          await emitPreviewOpen('/tmp/alias.html', addressedId)
          await waitFor(() => expect($previewTabs.get().map(tab => tab.sessionId)).toEqual([storedId]))
          const tab = $previewTabs.get()[0]!
          expect($visiblePreviewTabs.get().map(visible => visible.id)).toContain(tab.id)
          unbind = registerPreviewPageReader(tab.id, async () => ({
            text: 'ALIAS_PAGE',
            title: 'alias',
            url: 'file:///tmp/alias.html'
          }))
          const respond = vi.fn()

          const deps = {
            activeSessionIdRef: { current: 'main-runtime' },
            sessionInterrupted: () => false,
            sessionStateByRuntimeIdRef: { current: new Map(Object.entries($sessionStates.get())) },
            updateSessionState: vi.fn(),
            upsertToolCall: vi.fn()
          } as ServerRequestContext['deps']

          handleServerRequest(
            {
              id: 'alias-read',
              method: 'preview.read',
              params: { session_id: runtimeId },
              profile: 'default',
              respond,
              fail: vi.fn()
            },
            deps,
            'main-runtime'
          )
          await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
          expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ text: 'ALIAS_PAGE' })

          // The same alias must also resolve to that owner on the read side.
          respond.mockClear()
          handleServerRequest(
            {
              id: 'alias-read-2',
              method: 'preview.read',
              params: { session_id: addressedId },
              profile: 'default',
              respond,
              fail: vi.fn()
            },
            deps,
            'main-runtime'
          )
          await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
          expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ text: 'ALIAS_PAGE' })
        } finally {
          unbind()
          noteActiveTreeGroup(null)
          $layoutTree.set(null)
          $sessionTiles.set([])
          $sessionStates.set({})
          setSessions([])
        }
      }
    )

    it('owns a lineage tip without a runtime-state binding to the visible stored conversation', async () => {
      $activeSessionId.set('visible-stored-root')
      $selectedStoredSessionId.set('visible-stored-root')
      setSessions([
        { id: 'visible-stored-tip', _lineage_root_id: 'visible-stored-root' } as SessionInfo
      ])
      render(<Harness />)

      let unbind = () => {}

      try {
        await emitPreviewOpen('/tmp/lineage-without-runtime.html', 'visible-stored-tip')
        await waitFor(() => expect($previewTabs.get().map(tab => tab.sessionId)).toEqual(['visible-stored-root']))
        const tab = $previewTabs.get()[0]!
        expect($visiblePreviewTabs.get().map(visible => visible.id)).toContain(tab.id)
        unbind = registerPreviewPageReader(tab.id, async () => ({
          text: 'STORED_LINEAGE_PAGE', title: 'lineage', url: 'file:///tmp/lineage-without-runtime.html'
        }))
        const respond = vi.fn()

        const deps = {
          activeSessionIdRef: { current: 'visible-stored-root' },
          sessionInterrupted: () => false,
          sessionStateByRuntimeIdRef: { current: new Map() },
          updateSessionState: vi.fn(), upsertToolCall: vi.fn()
        } as ServerRequestContext['deps']

        handleServerRequest({
          id: 'stored-lineage-read', method: 'preview.read',
          params: { session_id: 'visible-stored-tip' }, profile: 'default', respond, fail: vi.fn()
        }, deps, 'visible-stored-root')
        await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
        expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ text: 'STORED_LINEAGE_PAGE' })
      } finally {
        unbind()
        setSessions([])
      }
    })

    it.each(['owner', 'foreign'])('uses authoritative %s source rather than a conflicting runtime ledger', async source => {
      const runtimeId = 'durable-source-runtime'
      const storedId = 'durable-source-stored'
      const owner = { connectionId: 'durable-source-A', profile: 'default' }
      const foreign = { connectionId: 'stale-source-B', profile: 'default' }
      const requestSource = source === 'owner' ? owner : foreign
      $activeSessionId.set('unrelated-main')
      $selectedStoredSessionId.set('unrelated-stored')
      publishSessionState(runtimeId, createClientSessionState(storedId))
      recordSessionEventScope({ ...foreign, session_id: runtimeId })
      $sessionTiles.set([{ dir: 'right', runtimeId, storedSessionId: storedId, ownerRoute: owner }])
      $layoutTree.set(group([`session-tile:${storedId}`], { active: `session-tile:${storedId}`, id: 'durable-group' }))
      noteActiveTreeGroup('durable-group')
      render(<Harness />)

      let unbind = () => {}

      try {
        await act(async () => {
          handleEvent({ ...requestSource, type: 'preview.open', session_id: runtimeId, payload: { url: '/tmp/durable.html' } })
        })

        if (source === 'owner') {
          await waitFor(() => expect($previewTabs.get().map(tab => tab.sessionId)).toEqual([storedId]))
          const tab = $previewTabs.get()[0]!
          expect($visiblePreviewTabs.get().map(visible => visible.id)).toContain(tab.id)
          unbind = registerPreviewPageReader(tab.id, async () => ({
            text: 'DURABLE_SOURCE_PAGE', title: 'durable', url: 'file:///tmp/durable.html'
          }))
        } else {
          expect($previewTabs.get()).toHaveLength(0)
        }

        const respond = vi.fn()
        const decline = vi.fn()

        const deps = {
          activeSessionIdRef: { current: 'unrelated-main' },
          sessionInterrupted: () => false,
          sessionStateByRuntimeIdRef: { current: new Map(Object.entries($sessionStates.get())) },
          updateSessionState: vi.fn(), upsertToolCall: vi.fn()
        } as ServerRequestContext['deps']

        handleServerRequest({
          ...requestSource, id: 'durable-source-read', method: 'preview.read',
          params: { session_id: runtimeId }, respond, decline, fail: vi.fn()
        }, deps, 'unrelated-main')

        if (source === 'owner') {
          await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
          expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ text: 'DURABLE_SOURCE_PAGE' })
          expect(decline).not.toHaveBeenCalled()
        } else {
          expect(respond).not.toHaveBeenCalled()
          expect(decline).toHaveBeenCalledTimes(1)
        }
      } finally {
        unbind()
        noteActiveTreeGroup(null)
        $layoutTree.set(null)
        $sessionTiles.set([])
        dropSessionState(runtimeId)
        $sessionStates.set({})
      }
    })

    it.each([
      ['main', 'profile'],
      ['main', 'connection'],
      ['main', 'untagged'],
      ['tile', 'profile'],
      ['tile', 'connection'],
      ['tile', 'untagged']
    ])('rejects foreign %s alias events from a different %s source', async (surface, mismatch) => {
      const storedId = 'stored-shared'
      const visibleRuntime = `${surface}-source-visible`
      const foreignRuntime = `${surface}-source-foreign-${mismatch}`
      const owner = { connectionId: 'source-visible', profile: 'default' }

      const foreign =
        mismatch === 'profile'
          ? { connectionId: owner.connectionId, profile: 'foreign-profile' }
          : { connectionId: 'source-foreign', profile: owner.profile }

      const source = mismatch === 'untagged' ? {} : foreign
      $activeSessionId.set(surface === 'main' ? visibleRuntime : 'unrelated-main')
      publishSessionState(visibleRuntime, createClientSessionState(storedId))
      publishSessionState(foreignRuntime, createClientSessionState(storedId))
      recordSessionEventScope({ ...owner, session_id: visibleRuntime })
      recordSessionEventScope({ ...foreign, session_id: foreignRuntime })

      if (surface === 'tile') {
        $sessionTiles.set([{ dir: 'right', runtimeId: visibleRuntime, storedSessionId: storedId, ownerRoute: owner }])
      }

      render(<Harness />)

      try {
        await act(async () => {
          handleEvent({
            ...source,
            type: 'preview.open',
            session_id: foreignRuntime,
            payload: { url: '/tmp/foreign.html' }
          })
        })
        expect($previewTabs.get()).toHaveLength(0)
        expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()

        await act(async () => {
          handleEvent({
            ...owner,
            type: 'preview.open',
            session_id: visibleRuntime,
            payload: { url: '/tmp/visible.html' }
          })
        })
        await waitFor(() => expect($previewTabs.get()).toHaveLength(1))
        await act(async () => {
          handleEvent({ ...source, type: 'preview.close', session_id: foreignRuntime, payload: {} })
        })
        expect($previewTabs.get()).toHaveLength(1)
      } finally {
        dropSessionState(visibleRuntime)
        dropSessionState(foreignRuntime)
        $sessionStates.set({})
        $sessionTiles.set([])
      }
    })

    it('does not let an unowned stored alias bypass the visible runtime source', async () => {
      $activeSessionId.set('source-bound-main')
      publishSessionState('source-bound-main', createClientSessionState('unowned-stored'))
      publishSessionState('unowned-foreign-runtime', createClientSessionState('unowned-stored'))
      recordSessionEventScope({
        connectionId: 'source-bound',
        profile: 'visible-profile',
        session_id: 'source-bound-main'
      })
      render(<Harness />)

      try {
        await act(async () => {
          handleEvent({
            profile: 'default',
            type: 'preview.open',
            session_id: 'unowned-foreign-runtime',
            payload: { url: '/tmp/foreign.html' }
          })
        })
        expect($previewTabs.get()).toHaveLength(0)
        expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
      } finally {
        dropSessionState('source-bound-main')
        $sessionStates.set({})
      }
    })

    it('ignores an open from a session that is not the one on screen', async () => {
      render(<Harness />)

      await act(async () => {
        handleEvent({
          payload: { url: '/tmp/other.html' },
          session_id: 'some-other-session',
          type: 'preview.open'
        } as unknown as GatewayEvent)
      })

      expect($previewTabs.get()).toHaveLength(0)
    })

    it('does not treat the selected stored id alone as a visible runtime binding', async () => {
      $selectedStoredSessionId.set('hidden-stored')
      publishSessionState('hidden-runtime', createClientSessionState('hidden-stored'))
      render(<Harness />)

      try {
        await emitPreviewOpen('/tmp/hidden.html', 'hidden-runtime')

        expect($previewTabs.get()).toHaveLength(0)
        expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
      } finally {
        $sessionStates.set({})
      }
    })

    it.each(['main', 'tile'])('rejects a branch sibling of the visible %s conversation', async surface => {
      const { $sessionTiles } = await import('@/store/session-states')
      const tiles = $sessionTiles.get()
      setSessions([
        { id: 'branch-a', _lineage_root_id: 'shared-root' } as SessionInfo,
        { id: 'branch-b', _lineage_root_id: 'shared-root' } as SessionInfo
      ])
      publishSessionState('branch-b-runtime', createClientSessionState('branch-b'))

      if (surface === 'main') {
        $activeSessionId.set('branch-a')
      } else {
        $sessionTiles.set([{ dir: 'right', runtimeId: 'branch-a-runtime', storedSessionId: 'branch-a' }])
      }

      render(<Harness />)

      try {
        await emitPreviewOpen('/tmp/sibling.html', 'branch-b-runtime')

        expect($previewTabs.get()).toHaveLength(0)
        expect(window.hermesDesktop.normalizePreviewTarget).not.toHaveBeenCalled()
      } finally {
        $sessionTiles.set(tiles)
        $sessionStates.set({})
        setSessions([])
      }
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

        // Honoured, and owned by the tile's session (#73890): it shows in
        // that session's drawer, not in whichever one holds focus.
        await waitFor(() =>
          expect($previewTabs.get().map(tab => [tab.target.path, tab.sessionId])).toEqual([
            ['/tmp/from-tile.html', 'stored-tile']
          ])
        )
      } finally {
        $sessionTiles.set(tiles)
      }
    })

    it('opens an owned tab when a visible tile is addressed by its stored id', async () => {
      const { $sessionTiles } = await import('@/store/session-states')
      const tiles = $sessionTiles.get()

      $sessionTiles.set([{ dir: 'right', runtimeId: 'tile-runtime', storedSessionId: 'stored-tile' }])
      render(<Harness />)

      try {
        await emitPreviewOpen('/tmp/from-tile.html', 'stored-tile')

        await waitFor(() =>
          expect($previewTabs.get().map(tab => [tab.target.path, tab.sessionId])).toEqual([
            ['/tmp/from-tile.html', 'stored-tile']
          ])
        )
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
        } as unknown as GatewayEvent)
        handleEvent({
          payload: { path: './dist/index.html' },
          session_id: RUNTIME_SESSION_ID,
          type: 'tool.complete'
        } as unknown as GatewayEvent)
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

    it('closes a Browser tab by the page it navigated to', async () => {
      render(<Harness />)
      openPreview(urlTarget('https://example.com/start'))
      const tabId = $previewTabs.get()[0].id

      noteBrowserPage(tabId, {
        title: 'Dashboard',
        url: 'https://example.com/dashboard'
      })

      await emitPreviewClose('https://example.com/dashboard')

      expect($previewTabs.get()).toHaveLength(0)
      expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).toBeNull()
    })

    it('does not remove a popped Browser tab through the persisted-url fallback', async () => {
      render(<Harness />)
      openPreview(urlTarget('https://example.com/start'))
      const tabId = $previewTabs.get()[0].id

      noteBrowserPage(tabId, {
        title: 'Dashboard',
        url: 'https://example.com/dashboard'
      })
      markBrowserTabPopped(tabId, true)

      try {
        await emitPreviewClose('https://example.com/start')

        expect($previewTabs.get().map(tab => tab.id)).toEqual([tabId])
      } finally {
        markBrowserTabPopped(tabId, false)
      }
    })

    it('closes only the tile session tabs when its agent closes without a url', async () => {
      const { $sessionTiles } = await import('@/store/session-states')
      const tiles = $sessionTiles.get()

      $selectedStoredSessionId.set('stored-main')
      $sessionTiles.set([{ dir: 'right', runtimeId: 'tile-runtime', storedSessionId: 'stored-tile' }])
      render(<Harness />)

      try {
        await emitPreviewOpen('/tmp/main.html')
        await emitPreviewOpen('/tmp/from-tile.html', 'tile-runtime')
        await waitFor(() => expect($previewTabs.get()).toHaveLength(2))

        await emitPreviewClose('', 'tile-runtime')

        expect($previewTabs.get().map(tab => [tab.target.path, tab.sessionId])).toEqual([
          ['/tmp/main.html', 'stored-main']
        ])
        expect($previewTarget.get()?.path).toBe('/tmp/main.html')
      } finally {
        $sessionTiles.set(tiles)
      }
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
  })
})
