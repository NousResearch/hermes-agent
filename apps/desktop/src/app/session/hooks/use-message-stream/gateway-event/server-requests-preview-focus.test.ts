import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { registerPreviewPageReader } from '@/app/chat/right-rail/preview-reader'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $previewTabs,
  closeRightRail,
  closeRightRailTab,
  openPreview,
  rekeyPreviewTabsSession,
  setPreviewTabPinned
} from '@/store/preview'
import { setActiveSessionId, setSelectedStoredSessionId, setSessions } from '@/store/session'
import { $sessionTiles, dropSessionState, publishSessionState } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, type ServerRequestContext } from './server-requests'

const bridge = vi.hoisted(() => ({
  live: vi.fn(() => true),
  act: vi.fn(async (_action: unknown, _owner?: unknown): Promise<unknown> => null),
  read: vi.fn(async () => null)
}))

vi.mock('@/app/chat/right-rail/preview-popout-bridge', () => ({
  hasLivePreviewSurface: bridge.live,
  requestPopoutPreviewAct: bridge.act,
  requestPopoutPreviewRead: bridge.read
}))

const deps: ServerRequestContext['deps'] = {
  activeSessionIdRef: { current: 'rt-a' },
  sessionInterrupted: () => false,
  sessionStateByRuntimeIdRef: { current: new Map() },
  updateSessionState: (_id, update) => update(createClientSessionState(null)),
  upsertToolCall: () => undefined
}

const stops: (() => void)[] = []
const runtimes = ['rt-a', 'rt-b', 'rt-rotated', 'rt-sibling', 'rt-other']

const focus = (pane: string, chrome = false) => {
  $layoutTree.set(
    group([...new Set(['workspace', 'session-tile:stored-b', 'session-tile:dangling', 'browser', pane])], {
      active: pane,
      id: 'tabs'
    })
  )
  noteActiveTreeGroup('tabs')

  if (chrome) {
    $layoutTree.set(group(['workspace', pane, 'browser'], { active: 'browser', id: 'tabs' }))
  }
}

const bind = (runtime: string, stored: string | null) => {
  const state = createClientSessionState(stored)
  deps.sessionStateByRuntimeIdRef.current.set(runtime, state)
  publishSessionState(runtime, state)
}

const deliver = (
  runtime: string,
  method = 'preview.act',
  replayed = false,
  primary = deps.activeSessionIdRef.current
) => {
  const respond = vi.fn()
  const decline = vi.fn()
  handleServerRequest(
    {
      id: `request-${runtime}`,
      method,
      params: { action: 'back', session_id: runtime },
      profile: 'default',
      replayed,
      respond,
      decline,
      fail: vi.fn()
    },
    deps,
    primary
  )

  return { respond, decline, runtime }
}

const declined = (request: ReturnType<typeof deliver>) => {
  expect(request.decline).toHaveBeenCalledTimes(1)
  expect(request.respond).not.toHaveBeenCalled()
}

const answered = async (request: ReturnType<typeof deliver>) => {
  await vi.waitFor(() => expect(request.respond, `requester ${request.runtime}`).toHaveBeenCalledTimes(1))
  expect(request.decline).not.toHaveBeenCalled()

  return JSON.parse(request.respond.mock.calls[0]![0].value)
}

const page = (stored: string | null, runtime?: string) => {
  openPreview(
    {
      kind: 'url',
      label: 'page',
      source: `https://${runtime ?? stored}.example`,
      url: `https://${runtime ?? stored}.example`
    },
    stored,
    runtime
  )
  const tab = $previewTabs.get().at(-1)!
  const back = vi.fn()
  stops.push(registerPreviewNav(tab.id, { back, forward: () => {}, reload: () => {} }))
  stops.push(
    registerPreviewPageReader(tab.id, async () => ({ text: 'owned page', title: 'page', url: tab.target.url }))
  )

  return back
}

beforeEach(() => {
  closeRightRail()
  bridge.live.mockReturnValue(true)
  bridge.act.mockReset().mockResolvedValue(null)
  setActiveSessionId('rt-a')
  setSelectedStoredSessionId('stored-a')
  deps.activeSessionIdRef.current = 'rt-a'
  bind('rt-a', 'stored-a')
  bind('rt-b', 'stored-b')
  $sessionTiles.set([{ runtimeId: 'rt-b', storedSessionId: 'stored-b' } as never])
  focus('workspace')
})
afterEach(() => {
  stops.splice(0).forEach(stop => stop())
  noteActiveTreeGroup(null)
  $layoutTree.set(null)
  $sessionTiles.set([])
  runtimes.forEach(dropSessionState)
  deps.sessionStateByRuntimeIdRef.current.clear()
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  closeRightRail()
})

it('lets only the retained foreground conversation drive its requester-owned browser tabs', async () => {
  const backA = page('stored-a')
  const backB = page('stored-b') // manually opened
  focus('session-tile:stored-b', true) // normal stacked tab, then browser chrome
  expect(await answered(deliver('rt-b'))).toMatchObject({ acted: 'back', success: true })
  expect(backB).toHaveBeenCalledTimes(1)
  expect(backA).not.toHaveBeenCalled()
  expect(await answered(deliver(''))).toMatchObject({ success: false })
  expect(backB).toHaveBeenCalledTimes(1) // unscoped requests must not inherit foreground authority
  expect(await answered(deliver('stored-b'))).toMatchObject({ success: true })
  expect(backB).toHaveBeenCalledTimes(2) // stored identity matches the observed binding
  deps.sessionStateByRuntimeIdRef.current.delete('rt-b')
  expect(await answered(deliver('rt-b'))).toMatchObject({ success: true })
  expect(backB).toHaveBeenCalledTimes(3) // exact runtime needs no cached stored binding
  bind('rt-b', 'stored-b')
  backB.mockClear()
  expect(await answered(deliver('rt-b'))).toMatchObject({ success: true })
  declined(deliver('rt-a')) // hosted primary must not win the fanout race

  focus('workspace')
  declined(deliver('rt-b')) // hosted background tile declines, but still reads
  expect(await answered(deliver('rt-b', 'preview.read'))).toMatchObject({ text: 'owned page' })
  expect(await answered(deliver('rt-a'))).toMatchObject({ success: true })
  expect(backA).toHaveBeenCalledTimes(1)

  // Rekey the manually stored-owned tab before publishing the new lineage tip.
  rekeyPreviewTabsSession('stored-b', 'stored-tip')
  expect($previewTabs.get().find(tab => tab.target.url === 'https://stored-b.example')?.sessionId).toBe('stored-tip')
  setSessions([{ id: 'stored-tip', _lineage_root_id: 'stored-b', _lineage_ids: ['stored-b'] } as SessionInfo])
  bind('rt-b', 'stored-tip')
  bind('rt-rotated', 'stored-tip')
  $sessionTiles.set([{ runtimeId: 'rt-b', storedSessionId: 'stored-b' } as never])
  focus('session-tile:stored-b')
  backB.mockClear()
  expect(await answered(deliver('rt-rotated'))).toMatchObject({ success: true })
  expect(backB).toHaveBeenCalledTimes(1)
  closeRightRailTab($previewTabs.get().find(tab => tab.target.url === 'https://stored-b.example')!.id)
  const agentBack = page(null, 'rt-rotated') // agent-opened pending tab belongs to requester
  expect(await answered(deliver('rt-rotated'))).toMatchObject({ success: true })
  expect(agentBack).toHaveBeenCalledTimes(1)
  focus('workspace')
  expect(await answered(deliver('rt-rotated', 'preview.read'))).toMatchObject({ text: 'owned page' })

  setSessions([
    { id: 'stored-b', _lineage_root_id: 'shared-root' } as SessionInfo,
    { id: 'sibling', _lineage_root_id: 'shared-root' } as SessionInfo
  ])
  bind('rt-sibling', 'sibling')
  const siblingBack = page(null, 'rt-sibling')
  $sessionTiles.set([{ runtimeId: 'rt-b', storedSessionId: 'shared-root' } as never])
  focus('session-tile:shared-root')
  deps.sessionStateByRuntimeIdRef.current.delete('rt-b')
  declined(deliver('rt-sibling')) // a retained shared root is not an observed runtime binding
  expect(siblingBack).not.toHaveBeenCalled()
  expect(backB).toHaveBeenCalledTimes(1)
  expect(agentBack).toHaveBeenCalledTimes(1)
  bind('rt-b', 'stored-b')
  $sessionTiles.set([{ runtimeId: 'rt-b', storedSessionId: 'shared-root' } as never])
  focus('session-tile:shared-root')
  declined(deliver('rt-sibling')) // the observed branch binding must also isolate siblings
  $sessionTiles.set([{ runtimeId: 'rt-b', storedSessionId: 'stored-b' } as never])
  focus('session-tile:stored-b')
  declined(deliver('rt-sibling'))
  $sessionTiles.set([{ runtimeId: null, storedSessionId: 'stored-b' } as never])
  declined(deliver('rt-b')) // unbound tile cannot inherit the primary runtime
  declined(deliver('rt-a'))
  focus('session-tile:dangling')
  declined(deliver('rt-a'))

  // Foreground eligibility does not grant another profile's tabs or pins.
  setPreviewTabPinned($previewTabs.get()[0]!.id, true)
  $sessionTiles.set([
    { ownerProfile: 'other-profile', runtimeId: 'rt-other', storedSessionId: 'stored-other' } as never
  ])
  bind('rt-other', 'stored-other')
  focus('session-tile:stored-other')
  expect(await answered(deliver('rt-other'))).toMatchObject({
    success: false,
    error: expect.stringContaining('No live page')
  })
  expect(backA).toHaveBeenCalledTimes(1)
  expect(backB).toHaveBeenCalledTimes(1)
})

it('revalidates foreground eligibility on replay and before delayed local browser execution', async () => {
  const backA = page('stored-a')
  const backB = page('stored-b')
  focus('workspace')
  // Real lazy engine import creates an async boundary even with a local pane.
  const importing = deliver('rt-a')
  focus('session-tile:stored-b')
  await vi.waitFor(() => expect(importing.decline).toHaveBeenCalledTimes(1))
  expect(importing.respond).not.toHaveBeenCalled()
  expect(backA).not.toHaveBeenCalled()

  focus('workspace')
  bridge.live.mockReturnValue(false)
  let finish!: (result: unknown) => void
  bridge.act.mockImplementation(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  const pending = deliver('rt-a')
  expect(bridge.act).toHaveBeenCalledWith(expect.objectContaining({ kind: 'back' }), {
    profile: 'default',
    runtimeId: 'rt-a',
    sessionId: 'stored-a'
  })
  focus('session-tile:stored-b')
  finish(null)
  await vi.waitFor(() => expect(pending.decline).toHaveBeenCalledTimes(1))
  expect(pending.respond).not.toHaveBeenCalled()
  expect(backA).not.toHaveBeenCalled()

  // Resume binds a primary between replay and retry, but a different tile owns focus.
  bridge.live.mockReturnValue(true)
  $sessionTiles.set([])
  focus('workspace')
  deps.activeSessionIdRef.current = null
  const replay = deliver('rt-a', 'preview.act', true, null)
  expect(replay.decline).not.toHaveBeenCalled()
  deps.activeSessionIdRef.current = 'rt-a'
  $sessionTiles.set([{ runtimeId: 'rt-b', storedSessionId: 'stored-b' } as never])
  focus('session-tile:stored-b')
  await vi.waitFor(() => expect(replay.decline).toHaveBeenCalledTimes(1))
  expect(replay.respond).not.toHaveBeenCalled()
  expect(backA).not.toHaveBeenCalled()
  expect(backB).not.toHaveBeenCalled()

  // Non-browser replay keeps the baseline strict host check, not shown-only claims.
  $sessionTiles.set([])
  focus('workspace')
  deps.activeSessionIdRef.current = null
  setSelectedStoredSessionId(null)
  const windowReplay = deliver('stored-only', 'window.read', true, null)
  expect(windowReplay.decline).not.toHaveBeenCalled()
  setSelectedStoredSessionId('stored-only')
  await vi.waitFor(() => expect(windowReplay.decline).toHaveBeenCalledTimes(1))
  expect(windowReplay.respond).not.toHaveBeenCalled()
  setSelectedStoredSessionId('stored-a')

  // A primary draft/resume uses the passed ref, not a stale global runtime.
  focus('workspace')
  setActiveSessionId('stale-primary')
  deps.activeSessionIdRef.current = null
  const resumed = deliver('rt-a', 'preview.act', true, null)
  deps.activeSessionIdRef.current = 'rt-a'
  expect(await answered(resumed)).toMatchObject({ success: true })
  expect(backA).toHaveBeenCalledTimes(1)

  // Forward only the focused tile's requester owner to the Browser renderer.
  $sessionTiles.set([{ runtimeId: 'rt-b', storedSessionId: 'stored-b' } as never])
  focus('session-tile:stored-b')
  bridge.live.mockReturnValue(false)
  bridge.act.mockResolvedValue({ success: true, acted: 'back' })
  expect(await answered(deliver('rt-b'))).toMatchObject({ success: true })
  expect(bridge.act).toHaveBeenLastCalledWith(expect.objectContaining({ kind: 'back' }), {
    profile: 'default',
    runtimeId: 'rt-b',
    sessionId: 'stored-b'
  })
  expect(backB).not.toHaveBeenCalled()
})
