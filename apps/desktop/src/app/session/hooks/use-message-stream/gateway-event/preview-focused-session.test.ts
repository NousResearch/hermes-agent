import { afterEach, expect, it, vi } from 'vitest'

import { activePreviewNav, registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { registerPreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'
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

function request(
  method: string,
  sessionId: string,
  replayed = false,
  action = 'back',
  params: Record<string, unknown> = {}
) {
  const respond = vi.fn()
  const decline = vi.fn()
  const fail = vi.fn()

  const handled = handleServerRequest(
    {
      decline,
      fail,
      id: `request-${method}-${sessionId}`,
      method,
      params: { ...params, action, session_id: sessionId },
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
  vi.restoreAllMocks()
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
    ['session-tile:stored-b', 'runtime-a', 'decline'],
    ['session-tile:stored-b', 'runtime-b', 'b'],
    ['session-tile:stored-b', 'runtime-b2', 'b'],
    ['workspace', 'runtime-b', 'decline'],
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
      expect(result.success).toBe(true)
      expect(decline).not.toHaveBeenCalled()
    }

    expect(backA).toHaveBeenCalledTimes(beforeA + (expected === 'a' ? 1 : 0))
    expect(backB).toHaveBeenCalledTimes(beforeB + (expected === 'b' ? 1 : 0))
  }

  activateTreePane('chat', 'workspace')
  $toursEnabled.set(true)
  document.body.innerHTML = '<button id="save-b">Save B</button>'
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
    bottom: 40,
    height: 40,
    left: 0,
    right: 100,
    top: 0,
    width: 100,
    x: 0,
    y: 0,
    toJSON: () => ({})
  })
  // All action arguments and DOM contents here are fixed test fixtures.
  // The runner is the Electron boundary. Execute the real injected act/tour
  // scripts against a fixture DOM; do not mock the engines or their results.
  const runA = vi.fn(async (code: string) => new Function('return ' + code)())
  const runB = vi.fn(async (code: string) => new Function('return ' + code)())
  unbind.push(registerPreviewScriptRunner(a.id, runA), registerPreviewScriptRunner(b.id, runB))
  const bystanderAct = request('preview.act', 'runtime-b', false, 'elements')
  const bystanderTour = request('tour', 'runtime-b', false, 'targets', { surface: 'preview' })

  for (const bystander of [bystanderAct, bystanderTour]) {
    expect(bystander.decline).toHaveBeenCalledTimes(1)
    expect(bystander.respond).not.toHaveBeenCalled()
  }

  expect(runA).not.toHaveBeenCalled()
  expect(runB).not.toHaveBeenCalled()
  // Model the second renderer's foreground state, delivering the same request
  // after the first renderer declined. This is not a live Electron window test.
  activateTreePane('chat', 'session-tile:stored-b')
  const inventory = request('preview.act', 'runtime-b', false, 'elements')
  await vi.waitFor(() => expect(inventory.respond).toHaveBeenCalledTimes(1))
  expect(JSON.parse(inventory.respond.mock.calls[0][0].value)).toMatchObject({
    success: true,
    elements: [expect.objectContaining({ label: 'Save B', selector: '#save-b' })]
  })
  const targets = request('tour', 'runtime-b', false, 'targets', { surface: 'preview' })
  await vi.waitFor(() => expect(targets.respond).toHaveBeenCalledTimes(1))
  expect(JSON.parse(targets.respond.mock.calls[0][0].value)).toMatchObject({
    success: true,
    targets: expect.arrayContaining([expect.objectContaining({ selector: '#save-b' })])
  })

  try {
    const highlight = request('tour', 'runtime-b', false, 'show', {
      selector: '#save-b',
      surface: 'preview',
      title: 'Focused page',
      text: 'Save B'
    })

    await vi.waitFor(() => expect(highlight.respond).toHaveBeenCalledTimes(1))
    expect(JSON.parse(highlight.respond.mock.calls[0][0].value)).toMatchObject({ action: 'show', success: true })
    await vi.waitFor(() => expect(document.querySelector('.driver-popover-title')?.textContent).toBe('Focused page'))
    expect(runA).not.toHaveBeenCalled()
    expect(runB).toHaveBeenCalledTimes(3)
  } finally {
    const stop = request('tour', 'runtime-b', false, 'stop', { surface: 'preview' })
    await vi.waitFor(() => expect(stop.respond).toHaveBeenCalledTimes(1))
    expect(JSON.parse(stop.respond.mock.calls[0][0].value)).toMatchObject({ action: 'stop', success: true })
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
  await vi.waitFor(() => expect(replay.decline).toHaveBeenCalledTimes(1))
  expect(replay.respond).not.toHaveBeenCalled()
  expect(backA).not.toHaveBeenCalled()

  $toursEnabled.set(true)
  const backgroundTour = request('tour', 'runtime-a', false, 'stop')
  expect(backgroundTour.decline).toHaveBeenCalledTimes(1)
  expect(backgroundTour.respond).not.toHaveBeenCalled()
  const focusedTour = request('tour', 'runtime-b', false, 'stop')
  await vi.waitFor(() => expect(focusedTour.respond).toHaveBeenCalledTimes(1))
  expect(JSON.parse(focusedTour.respond.mock.calls[0][0].value)).toMatchObject({ action: 'stop', success: true })

  // Even a disabled bystander must decline, not settle the owner's request.
  $toursEnabled.set(false)
  const disabledBackgroundTour = request('tour', 'runtime-a', false, 'stop')
  expect(disabledBackgroundTour.decline).toHaveBeenCalledTimes(1)
  expect(disabledBackgroundTour.respond).not.toHaveBeenCalled()
  const disabledFocusedTour = request('tour', 'runtime-b', false, 'stop')
  expect(JSON.parse(disabledFocusedTour.respond.mock.calls[0][0].value)).toMatchObject({ success: false })
})
