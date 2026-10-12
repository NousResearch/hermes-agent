import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterAll, afterEach, beforeEach, expect, it, vi } from 'vitest'

import { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, watchContributedPanes } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { createClientSessionState } from '@/lib/chat-runtime'
import { onComposerDraftSyncRequest, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { $interfaceMode } from '@/store/interface-mode'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions, setSessionOwnerHint } from '@/store/session'
import { $sessionStates, $sessionTiles, closeSessionTile, foregroundSessionScopes } from '@/store/session-states'
import { $openWorkspaceSessionInNewTab } from '@/store/workspace-tab-action'
import { makeSessionInfo } from '@/test/session-info'

import { openSession } from '../open-session'

import { keepWorkspaceSessionOpen, openWorkspaceSessionInNewTab } from './keep-session-open'
import { watchSessionTiles } from './session-tile'

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const disposeWorkspace = registry.register({
  id: 'workspace',
  area: 'panes',
  title: 'Chat',
  data: { placement: 'main' },
  render: () => null
})

watchContributedPanes()
watchSessionTiles()
afterAll(disposeWorkspace)

const owner = { connectionId: 'remote-work', profile: 'research', mode: 'remote' as const }
const navigate = vi.fn()
const requestGateway = vi.fn(async () => ({}))
const state = createClientSessionState('kept-chat')
let startFresh: () => unknown

function Harness() {
  const activeSessionIdRef = useRef<string | null>('runtime-kept')
  const selectedStoredSessionIdRef = useRef<string | null>('kept-chat')
  const busyRef = useRef(true)
  const creatingSessionRef = useRef(false)
  const runtimeIdByStoredSessionIdRef = useRef(new Map([['kept-chat', 'runtime-kept']]))
  const sessionStateByRuntimeIdRef = useRef(new Map([['runtime-kept', state]]))

  const actions = useSessionActions({
    activeSessionId: 'runtime-kept',
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: () => state,
    getRouteToken: () => 'kept-route',
    getRoutedStoredSessionId: () => null,
    navigate,
    requestGateway: async <T,>() => (await requestGateway()) as T,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: 'kept-chat',
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => state
  })

  useEffect(() => {
    startFresh = () => actions.startFreshSessionDraft({ replaceRoute: true })
    $openWorkspaceSessionInNewTab.set(() => {
      openWorkspaceSessionInNewTab(startFresh)
    })

    return () => $openWorkspaceSessionInNewTab.set(null)
  }, [actions])

  return null
}

beforeEach(() => {
  $interfaceMode.set('simple')
  vi.clearAllMocks()
  navigate.mockClear()
  requestGateway.mockClear()
  $activeGatewayProfile.set('research')
  $layoutTree.set(group(['workspace'], { id: 'main' }))
  $sessionTiles.set([])
  $sessions.set([makeSessionInfo({ id: 'kept-chat', profile: 'research', connection_id: 'remote-work' })])
  setSessionOwnerHint('kept-chat', owner)
  $selectedStoredSessionId.set('kept-chat')
  $activeSessionId.set('runtime-kept')
  $sessionStates.set({ 'runtime-kept': { ...state, busy: true, awaitingResponse: true } })
})

afterEach(() => {
  $interfaceMode.set('advanced')
  $openWorkspaceSessionInNewTab.set(null)
  cleanup()
  $sessionTiles.set([])
  $sessionStates.set({})
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  $layoutTree.set(null)
  $activeGatewayProfile.set('default')
})

it('hands a running chat to one retained tab through the real fresh-draft reset, keeping its owner and draft', () => {
  const flush = onComposerDraftSyncRequest(({ mode, target }) => {
    if (mode === 'flush' && target === 'main') {
      stashSessionDraft('kept-chat', 'Unsent follow-up', [])
    }
  })

  render(<Harness />)

  try {
    act(() => openSession('kept-chat', navigate, 'tab'))

    expect(navigate).toHaveBeenCalledWith('/', { replace: true })
    expect($selectedStoredSessionId.get()).toBeNull()
    expect($activeSessionId.get()).toBeNull()
    expect($sessionTiles.get()).toEqual([
      expect.objectContaining({
        storedSessionId: 'kept-chat',
        runtimeId: 'runtime-kept',
        ownerRoute: owner,
        workspaceMode: 'sessions',
        dir: 'center',
        anchor: 'workspace'
      })
    ])
    expect($sessionStates.get()['runtime-kept']).toMatchObject({ busy: true, awaitingResponse: true })
    expect(foregroundSessionScopes()).toContain('conn:remote-work::research')
    expect(takeSessionDraft('kept-chat').text).toBe('Unsent follow-up')
    expect(requestGateway).not.toHaveBeenCalled()
    expect($layoutTree.get()).toMatchObject({ active: 'session-tile:kept-chat' })

    // Browsing another chat homes main; selecting the retained chat fronts its
    // existing tab through the real registry/layout path without a new route.
    $selectedStoredSessionId.set('another-chat')
    const browse = vi.fn()
    openSession('kept-chat', browse)
    expect(browse).not.toHaveBeenCalled()
    expect($layoutTree.get()).toMatchObject({ active: 'session-tile:kept-chat' })

    $selectedStoredSessionId.set(null)
    act(() => expect(keepWorkspaceSessionOpen(startFresh)).toBe(false))
    expect($sessionTiles.get()).toHaveLength(1)
    closeSessionTile('kept-chat')
    expect($sessionTiles.get()).toHaveLength(0)
    expect($sessions.get()[0].id).toBe('kept-chat')
    expect(takeSessionDraft('kept-chat').text).toBe('Unsent follow-up')
  } finally {
    flush()
  }
})

it('leaves an unbound draft in place instead of minting a backend session to keep it open', () => {
  $activeSessionId.set(null)
  const reset = vi.fn()
  expect(keepWorkspaceSessionOpen(reset)).toBe(false)
  expect(reset).not.toHaveBeenCalled()
  expect($sessionTiles.get()).toHaveLength(0)
})

it('reports a declined or failed explicit tab handoff', () => {
  $activeSessionId.set(null)
  const reset = vi.fn()
  expect(openWorkspaceSessionInNewTab(reset)).toBe(false)
  expect(notify).toHaveBeenCalledWith({
    kind: 'error',
    title: 'Open in new tab',
    message: 'Couldn’t open this chat in a new tab. Try again.'
  })
  expect(reset).not.toHaveBeenCalled()
  $activeSessionId.set('runtime-kept')
  const error = new Error('Could not detach chat')
  expect(
    openWorkspaceSessionInNewTab(() => {
      throw error
    })
  ).toBe(false)
  expect(notifyError).toHaveBeenCalledWith(error, 'Open in new tab')
})
