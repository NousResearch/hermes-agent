import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { watchSessionTiles } from '@/app/chat/session-tile'
import { findGroupOfPane, group } from '@/components/pane-shell/tree/model'
import * as tree from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { registry } from '@/contrib/registry'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import * as session from '@/store/session'
import * as states from '@/store/session-states'
import { openSessionInNewWindow } from '@/store/windows'

import { host } from './index'

// Only the backend/window boundaries are replaced; SDK routing, session stores,
// native tile registration, layout adoption and hydration checks are production.
vi.mock(import('@/store/gateway'), async importOriginal => ({
  ...(await importOriginal()),
  activeGatewayConnectionId: () => 'local',
  openGatewayForAgent: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock(import('@/store/windows'), async importOriginal => ({
  ...(await importOriginal()),
  canOpenSessionWindow: () => true,
  openSessionInNewWindow: vi.fn()
}))

beforeAll(() => {
  const dispose = registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'Chat'
  })

  tree.watchContributedPanes()
  watchSessionTiles()

  return dispose
})

function resetState() {
  for (const tile of states.$sessionTiles.get()) {
    states.discardSessionTile(tile.storedSessionId)
  }

  states.clearAllSessionStates()
  setWorkspaceScope('sessions')
  session.$sessions.set([])
  session.$selectedStoredSessionId.set(null)
  session.$activeSessionId.set(null)
  session.$messages.set([])
  tree.$layoutTree.set(null)
  tree.$activeTreeGroup.set(null)
  tree.$activePresetId.set('default')
  window.localStorage.clear()
  window.location.hash = '#/c/main-chat'
  vi.clearAllMocks()
}

beforeEach(resetState)
afterEach(resetState)

function setupMain() {
  tree.declareDefaultTree(group(['workspace'], { active: 'workspace', id: 'main' }))
  session.$selectedStoredSessionId.set('main-chat')
  session.$activeSessionId.set('main-runtime')
}

it('splits at each requested edge without replacing main or duplicating/relocating an open conversation', async () => {
  const route = {
    connectionId: 'remote-a',
    mode: 'remote' as const,
    profile: 'writer',
    targetProfile: 'backend-writer'
  }

  const originalProfile = $activeGatewayProfile.get()

  for (const splitDirection of [undefined, 'right', 'left', 'top', 'bottom'] as const) {
    resetState()
    setupMain()
    await host.openSession('side-chat', { intent: 'split', route, splitDirection })

    const layout = tree.$layoutTree.get()!
    const side = findGroupOfPane(layout, 'session-tile:side-chat')
    const main = findGroupOfPane(layout, 'workspace')!
    expect(side).not.toBeNull()
    expect(side!.id).not.toBe(main.id)
    expect(layout.type).toBe('split')

    if (layout.type !== 'split') {
      throw new Error('Expected a native split')
    }

    const direction = splitDirection ?? 'right'
    expect(layout.orientation).toBe(direction === 'left' || direction === 'right' ? 'row' : 'column')
    expect(layout.children.map(child => child.id)).toEqual(
      direction === 'left' || direction === 'top' ? [side!.id, main.id] : [main.id, side!.id]
    )
    expect(states.sessionTileOwnerRoute('side-chat')).toEqual(route)
    expect(states.openTileGatewayScopes()).toContain('conn:remote-a::writer')
    expect(session.$selectedStoredSessionId.get()).toBe('main-chat')
    expect(session.$activeSessionId.get()).toBe('main-runtime')
    expect(window.location.hash).toBe('#/c/main-chat')
    expect(openSessionInNewWindow).not.toHaveBeenCalled()
    expect($activeGatewayProfile.get()).toBe(originalProfile)

    const tiles = states.$sessionTiles.get()
    await host.openSession('side-chat', { intent: 'split', splitDirection: 'left' })
    expect(states.$sessionTiles.get()).toEqual(tiles)
    expect(tree.$activeTreeGroup.get()).toBe(side!.id)
    expect(states.$focusedStoredSessionId.get()).toBe('side-chat')
    expect(findGroupOfPane(tree.$layoutTree.get()!, 'session-tile:side-chat')!.id).toBe(side!.id)

    await host.openSession('main-chat', { intent: 'split' })
    expect(states.$sessionTiles.get()).toEqual(tiles)
    expect(states.$focusedStoredSessionId.get()).toBe('main-chat')
  }
})

it('awaits the split tile runtime through the existing hydration seam, not the healthy main runtime', async () => {
  setupMain()
  let finishResume!: (runtimeId: string) => void

  const resumeTile = vi.fn(
    () =>
      new Promise<string>(resolve => {
        finishResume = resolve
      })
  )

  states.setSessionTileDelegate({
    archiveSession: vi.fn(),
    branchSession: vi.fn(),
    deleteSession: vi.fn(),
    executeSlash: vi.fn(),
    interruptSession: vi.fn(),
    resumeTile,
    submitToSession: vi.fn(),
    updateSession: vi.fn()
  })
  let hydrated = false

  const opening = host
    .openSession('side-chat', { awaitHydration: true, intent: 'split', profile: 'writer' })
    .then(() => {
      hydrated = true
    })

  await vi.waitFor(() => expect(resumeTile).toHaveBeenCalledWith('side-chat', { refreshTranscript: true }))
  expect(states.sessionTileOwnerRoute('side-chat')).toEqual({ connectionId: 'local', mode: 'local', profile: 'writer' })
  finishResume('side-runtime')
  await Promise.resolve()
  await Promise.resolve()
  expect(hydrated).toBe(false)
  // Deliver the gateway's binding via the real state seam. Main stays intact.
  states.publishSessionState('side-runtime', createClientSessionState('side-chat'))
  states.patchSessionTile('side-chat', { runtimeId: 'side-runtime' })
  await opening
  expect(hydrated).toBe(true)
  expect(session.$activeSessionId.get()).toBe('main-runtime')
  expect(session.$selectedStoredSessionId.get()).toBe('main-chat')
  expect(states.$sessionTiles.get().find(tile => tile.storedSessionId === 'side-chat')?.runtimeId).toBe('side-runtime')
  expect(window.location.hash).toBe('#/c/main-chat')
})
