import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { paneMirror } from '@/app/chat/pane-mirror'
import { useSessionActions } from '@/app/session/hooks/use-session-actions'
import type { ClientSessionState } from '@/app/types'
import * as model from '@/components/pane-shell/tree/model'
import * as tree from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import type { DesktopProfileRoute } from '@/global'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearSessionDraft, freshDraftScope } from '@/store/composer'
import { $defaultProfileRoute, setDefaultProfile } from '@/store/default-profile'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile, $newChatConnectionId, $newChatProfile, $newChatRoute } from '@/store/profile'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import * as session from '@/store/session'
import * as states from '@/store/session-states'

import { $workspaceIsPage } from '../routes'

import { openNewSessionTab } from './new-session-tab'

vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn(async () => undefined),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  activeGatewayConnectionId: vi.fn(() => 'old-host'),
  requestGatewayForAgent: vi.fn(),
  retainGatewayForAgent: vi.fn(async () => () => undefined)
}))

beforeAll(() => {
  const disposeWorkspace = registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'Chat'
  })

  tree.watchContributedPanes()
  paneMirror({
    source: states.$sessionTiles,
    key: tile => tile.storedSessionId,
    prefix: 'session-tile',
    dir: () => 'center',
    minWidth: '20rem',
    title: id => id,
    render: () => null,
    close: states.closeSessionTile
  })()

  return disposeWorkspace
})

const OLD = { connectionId: 'old-host', profile: 'old-profile' }
const NEW = { connectionId: 'new-host', profile: 'new-profile' }

/** The real hook, with the window's runtime cache backed by the real store. */
function mountPlus() {
  const ref = <T,>(current: T) => ({ current })

  const updateSessionState = (id: string, updater: (state: ClientSessionState) => ClientSessionState) => {
    const next = updater(states.$sessionStates.get()[id] ?? createClientSessionState())
    states.$sessionStates.set({ ...states.$sessionStates.get(), [id]: next })

    return next
  }

  const { result } = renderHook(() =>
    useSessionActions({
      activeSessionId: 'existing-runtime',
      activeSessionIdRef: ref<string | null>('existing-runtime'),
      busyRef: ref(false),
      creatingSessionRef: ref(false),
      ensureSessionState: () => createClientSessionState(),
      getRouteToken: () => 'route',
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn(),
      requestGateway: vi.fn(async () => ({}) as never),
      resetViewSync: vi.fn(),
      routedSessionId: null,
      runtimeIdByStoredSessionIdRef: ref(new Map()),
      selectedStoredSessionId: null,
      selectedStoredSessionIdRef: ref<string | null>(null),
      sessionStateByRuntimeIdRef: ref(new Map()),
      syncSessionStateToView: vi.fn(),
      updateSessionState
    })
  )

  return () => openNewSessionTab(result.current.openNewSessionTile)
}

/** Each session.create answers only when the test releases it. */
function holdCreates() {
  const releases: (() => void)[] = []
  let n = 0

  vi.mocked(requestGatewayForAgent).mockImplementation(() => {
    const id = ++n

    return new Promise(resolve => {
      releases.push(() => resolve({ session_id: `rt-${id}`, stored_session_id: `draft-${id}`, info: {} }))
    })
  })

  return releases
}

const creates = () => vi.mocked(requestGatewayForAgent).mock.calls.filter(call => call[2] === 'session.create')
const flush = () => act(async () => new Promise(resolve => setTimeout(resolve, 0)))

function reset() {
  states.$sessionTiles.get().forEach(tile => states.discardSessionTile(tile.storedSessionId))
  states.$sessionStates.set({})
  tree.$layoutTree.set(null)
  tree.$activeTreeGroup.set(null)
  session._resetSessionOwnerHintsForTests()
  session.$selectedStoredSessionId.set(null)
  session.setActiveSessionId(null)
  session.setSessions([])
  session.setConnection(null)
  clearSessionDraft(freshDraftScope())
  $workspaceIsPage.set(false)
  $defaultProfileRoute.set(null)
  $newChatRoute.set(null)
  $newChatProfile.set(null)
  $newChatConnectionId.set(null)
  window.localStorage.clear()
}

beforeEach(async () => {
  reset()
  tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'main' }))
  $activeGatewayProfile.set('old-profile')
  $projectScope.set(ALL_PROJECTS)
  window.hermesDesktop = { profile: { setDefault: async (route: DesktopProfileRoute) => route } } as never
  vi.mocked(requestGatewayForAgent).mockReset()
  await setDefaultProfile(OLD)
  // Main shows a loaded conversation, so the "+" has no blank main to front.
  session.$selectedStoredSessionId.set('loaded-chat')
  session.setActiveSessionId('existing-runtime')
})

afterEach(() => {
  cleanup()
  reset()
  vi.clearAllMocks()
})

describe('the registered New session "+" action (#145594)', () => {
  it('fronts a blank main instead of creating a session', async () => {
    session.$selectedStoredSessionId.set(null)
    session.setActiveSessionId(null)
    const plus = mountPlus()

    plus()
    plus()
    plus()
    await flush()

    expect(creates()).toHaveLength(0)
  })

  it('keeps a still-pending create single-flight: repeated clicks land on that one draft', async () => {
    const releases = holdCreates()
    const plus = mountPlus()

    plus()
    await flush()
    plus()
    plus()
    await flush()
    await act(async () => releases.forEach(release => release()))
    await flush()

    expect(creates()).toHaveLength(1)
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['draft-1'])

    // Settled: the draft is now a real blank tab, and "+" keeps fronting it.
    plus()
    await flush()
    expect(creates()).toHaveLength(1)
  })

  it('a failed create does not latch the "+" shut', async () => {
    vi.mocked(requestGatewayForAgent).mockRejectedValueOnce(new Error('gateway down'))
    const plus = mountPlus()

    plus()
    await flush()
    vi.mocked(requestGatewayForAgent).mockResolvedValue({ session_id: 'rt', stored_session_id: 'draft', info: {} })
    plus()
    await flush()

    expect(creates()).toHaveLength(2)
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['draft'])
  })

  it('opens under a newly selected default instead of reusing a blank draft of the old owner', async () => {
    vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({
      session_id: 'rt-old',
      stored_session_id: 'old-draft',
      info: {}
    })
    const plus = mountPlus()

    plus()
    await flush()
    expect(creates()[0].slice(0, 2)).toEqual([OLD.connectionId, OLD.profile])

    await act(() => setDefaultProfile(NEW))
    vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({
      session_id: 'rt-new',
      stored_session_id: 'new-draft',
      info: {}
    })
    plus()
    await flush()

    expect(creates()).toHaveLength(2)
    expect(creates()[1].slice(0, 2)).toEqual([NEW.connectionId, NEW.profile])
    // The old owner's draft is left intact, not re-homed.
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['old-draft', 'new-draft'])

    // Same owner as the default again: the blank draft is fronted, not duplicated.
    plus()
    await flush()
    expect(creates()).toHaveLength(2)
  })
})
