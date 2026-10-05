import { afterEach, expect, it, vi } from 'vitest'

import { markActiveComposer, onComposerInsertRequest, requestComposerInsert } from '@/app/chat/composer/focus'
import { activePreviewNav, registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, activateTreePane, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $previewTabs, closeRightRail, openPreview } from '@/store/preview'
import { setActiveSessionId, setSelectedStoredSessionId, setSessions } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles, dropSessionState, publishSessionState } from '@/store/session-states'
import { $toursEnabled } from '@/store/tours'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, type ServerRequestContext } from './server-requests'

const bindings = new Map<string, ReturnType<typeof createClientSessionState>>()

const deps = {
  activeSessionIdRef: { current: 'runtime-a' },
  sessionInterrupted: () => false,
  sessionStateByRuntimeIdRef: { current: bindings },
  updateSessionState: () => undefined,
  upsertToolCall: () => undefined
} as unknown as ServerRequestContext['deps']

let unbind: (() => void)[] = []

function request(method: string, sessionId: string, replayed = false, action = 'back') {
  const respond = vi.fn()
  const decline = vi.fn()
  const fail = vi.fn()

  const handled = handleServerRequest(
    {
      decline,
      fail,
      id: `request-${method}-${sessionId}`,
      method,
      params: { action, session_id: sessionId },
      profile: 'default',
      replayed,
      respond
    },
    deps,
    deps.activeSessionIdRef.current
  )

  return { decline, fail, handled, respond }
}

afterEach(() => {
  unbind.forEach(stop => stop())
  unbind = []

  for (const id of bindings.keys()) {
    dropSessionState(id)
  }

  bindings.clear()
  noteActiveTreeGroup(null)
  $layoutTree.set(null)
  $sessionTiles.set([])
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  closeRightRail()
  $toursEnabled.set(true)
  deps.activeSessionIdRef.current = 'runtime-a'
  markActiveComposer('main')
  document.body.replaceChildren()
})

it('allows only the focused conversation to drive its own live preview, including a rotated tile runtime', async () => {
  setSessions([
    { id: 'stored-a', _lineage_root_id: 'shared-root' } as SessionInfo,
    { id: 'stored-b', _lineage_root_id: 'shared-root' } as SessionInfo
  ])
  setActiveSessionId('runtime-a')
  setSelectedStoredSessionId('stored-a')
  $sessionTiles.set([{ runtimeId: 'runtime-b', storedSessionId: 'stored-b' }])

  for (const [runtime, stored] of [
    ['runtime-a', 'stored-a'],
    ['runtime-b', 'stored-b'],
    ['runtime-b2', 'stored-b']
  ]) {
    const state = createClientSessionState(stored)
    bindings.set(runtime, state)
    publishSessionState(runtime, state)
  }

  $layoutTree.set(group(['workspace', 'session-tile:stored-b'], { active: 'workspace', id: 'chat' }))
  noteActiveTreeGroup('chat')

  const backA = vi.fn()
  const backB = vi.fn()
  openPreview({ kind: 'url', label: 'A', source: 'https://a.example', url: 'https://a.example' }, 'stored-a')
  const a = $previewTabs.get().at(-1)!
  openPreview({ kind: 'url', label: 'B', source: 'https://b.example', url: 'https://b.example' }, 'stored-b')
  const b = $previewTabs.get().at(-1)!
  unbind = [
    registerPreviewNav(a.id, { back: backA, forward: () => {}, reload: () => {} }),
    registerPreviewNav(b.id, { back: backB, forward: () => {}, reload: () => {} })
  ]
  expect(activePreviewNav({ profile: 'default', runtimeId: 'runtime-b', sessionId: 'stored-b' })).not.toBeNull()

  for (const [pane, runtime, expected] of [
    ['session-tile:stored-b', 'runtime-a', 'refuse'],
    ['session-tile:stored-b', 'runtime-b', 'b'],
    ['session-tile:stored-b', 'runtime-b2', 'b'],
    ['workspace', 'runtime-b', 'refuse'],
    ['workspace', 'runtime-a', 'a'],
    ['session-tile:stored-b', 'runtime-foreign', 'decline']
  ] as const) {
    activateTreePane('chat', pane)
    expect($focusedStoredSessionId.get()).toBe(pane === 'workspace' ? 'stored-a' : 'stored-b')
    const beforeA = backA.mock.calls.length
    const beforeB = backB.mock.calls.length
    const { decline, handled, respond } = request('preview.act', runtime)
    expect(handled).toBe(true)

    if (expected === 'decline') {
      expect(decline).toHaveBeenCalledTimes(1)
      expect(respond).not.toHaveBeenCalled()
    } else {
      await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
      const result = JSON.parse(respond.mock.calls[0][0].value)
      expect(result.success).toBe(expected !== 'refuse')
      expect(decline).not.toHaveBeenCalled()
    }

    expect(backA).toHaveBeenCalledTimes(beforeA + (expected === 'a' ? 1 : 0))
    expect(backB).toHaveBeenCalledTimes(beforeB + (expected === 'b' ? 1 : 0))
  }
})

it('rechecks foreground on a reconnect replay and lets the focused tile run its tour', async () => {
  setActiveSessionId('runtime-a')
  setSelectedStoredSessionId('stored-a')
  $sessionTiles.set([{ runtimeId: 'runtime-b', storedSessionId: 'stored-b' }])
  $layoutTree.set(group(['workspace', 'session-tile:stored-b'], { active: 'session-tile:stored-b', id: 'chat' }))
  noteActiveTreeGroup('chat')
  const backA = vi.fn()
  openPreview({ kind: 'url', label: 'A', source: 'https://a.example', url: 'https://a.example' }, 'stored-a')
  unbind.push(registerPreviewNav($previewTabs.get().at(-1)!.id, { back: backA, forward: () => {}, reload: () => {} }))

  // Reconnect delivers A's replay while primary binding is still absent. By
  // the timer turn A is hosted again, but B is still the chat in front.
  deps.activeSessionIdRef.current = null
  const replay = request('preview.act', 'runtime-a', true)
  expect(replay.respond).not.toHaveBeenCalled()
  expect(replay.decline).not.toHaveBeenCalled()
  deps.activeSessionIdRef.current = 'runtime-a'
  await vi.waitFor(() => expect(replay.respond).toHaveBeenCalledTimes(1))
  expect(JSON.parse(replay.respond.mock.calls[0][0].value)).toMatchObject({ success: false })
  expect(backA).not.toHaveBeenCalled()

  $toursEnabled.set(true)
  const backgroundTour = request('tour', 'runtime-a', false, 'stop')
  expect(JSON.parse(backgroundTour.respond.mock.calls[0][0].value)).toMatchObject({ success: false })
  const focusedTour = request('tour', 'runtime-b', false, 'stop')
  await vi.waitFor(() => expect(focusedTour.respond).toHaveBeenCalledTimes(1))
  expect(JSON.parse(focusedTour.respond.mock.calls[0][0].value)).toMatchObject({ action: 'stop', success: true })

  // Annotation flush in PreviewPane goes through requestComposerInsert. An
  // inactive primary composer is still mounted, but the visible tile receives
  // the text destined for the focused chat, not the primary route.
  const main = document.createElement('div')
  main.dataset.paneHidden = ''
  const primaryComposer = document.createElement('div')
  primaryComposer.dataset.composerTarget = 'main'
  main.append(primaryComposer)
  const tileComposer = document.createElement('div')
  tileComposer.dataset.composerTarget = 'tile:stored-b'
  document.body.append(main, tileComposer)
  markActiveComposer('main')
  const inserted: string[] = []
  const stop = onComposerInsertRequest(detail => inserted.push(detail.target))

  try {
    requestComposerInsert('A note from the preview', { mode: 'block' })
    await vi.waitFor(() => expect(inserted).toEqual(['tile:stored-b']))
  } finally {
    stop()
  }
})
