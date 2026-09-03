import { registryBackendScopeKey } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect, useRef } from 'react'
import type { NavigateFunction } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { resolveSessionRpcOwner } from '@/app/contrib/wiring-routing'
import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { noteActiveTreeGroup, revealTreePane } from '@/components/pane-shell/tree/store'
import {
  deleteSession,
  getAllSessionMessages,
  getLatestSessionMessages,
  getSession,
  type ProfileScope,
  type SessionInfo,
  type SessionResumeResponse,
  setSessionArchived
} from '@/hermes'
import * as chatMessages from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { clearSessionDraft, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { requestGatewayForAgent, requestGatewayForProfile } from '@/store/gateway'
import { $pinnedSessionIds } from '@/store/layout'
import { $activeGatewayProfile, $newChatProfile, $newChatRoute, $profiles, ensureGatewayProfile } from '@/store/profile'
import { $projectScope, $projectTree, ALL_PROJECTS } from '@/store/projects'
import {
  $activeSessionId,
  $activeSessionStoredIdRotation,
  $cronSessions,
  $currentCwd,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $messages,
  $messagingSessions,
  $newChatWorkspaceTarget,
  $resumeFailedSessionId,
  $selectedStoredSessionId,
  $sessions,
  $turnStartedAt,
  getSessionOwnerHint,
  knownSessionOwner,
  sessionMatchesStoredId,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setAwaitingResponse,
  setBusy,
  setConnection,
  setCronSessions,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setMessages,
  setMessagingSessions,
  setNewChatWorkspaceTarget,
  setResumeFailedSessionId,
  setSelectedStoredSessionId,
  setSessions,
  setTurnStartedAt
} from '@/store/session'
import { $removedSessionIds, $sessionMutationsInFlight } from '@/store/session-removal'
import { requestForSessionProfile, type SessionProfileRoute } from '@/store/session-request-router'
import { $sessionTiles, sessionTileOwnerRoute } from '@/store/session-states'
import { $sessionSeenCounts, $unreadFinishedMarkers } from '@/store/session-unread'
import { clearTranscriptTail, transcriptTailState } from '@/store/transcript-tail'
import { loadTranscriptTail, saveTranscriptTail } from '@/store/transcript-tail-cache'
import * as transcriptTailCache from '@/store/transcript-tail-cache'

import sessionResumeActiveTurn from '../../../../../../tests/fixtures/session-resume-active-turn.json'
import { deferred } from '../../../test/deferred'
import { NEW_CHAT_ROUTE, sessionRoute } from '../../routes'
import type { ClientSessionState } from '../../types'

import { applyRewindOptimistic } from './use-prompt-actions/rewind'
import { clearSingleFlightSessionResumeState } from './use-prompt-actions/single-flight-resume'
import { useSessionActions } from './use-session-actions'
import * as authoritativeMessageReconciliation from './use-session-actions/authoritative-message-reconciliation'
import { SESSION_OPEN_MARKS } from './use-session-actions/session-open-marks'
import { runSessionOpenPerfFixture, setSessionOpenPerfFixtureRunner } from './use-session-actions/session-open-perf-fixture'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSession: vi.fn(),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  retainGatewayForAgent: vi.fn(async () => () => undefined)
}))

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  noteActiveTreeGroup: vi.fn(),
  revealTreePane: vi.fn()
}))

const RUNTIME_SESSION_ID = 'rt-new-001'

type HarnessHandle = Pick<
  ReturnType<typeof useSessionActions>,
  | 'archiveSession'
  | 'branchStoredSession'
  | 'createBackendSessionForSend'
  | 'openNewSessionTile'
  | 'removeSession'
  | 'selectSidebarItem'
  | 'startFreshSessionDraft'
>

function storedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'stored',
    tool_call_count: 0,
    ...overrides
  }
}

function Harness({
  activeSessionId = null,
  activeSessionIdRef: activeSessionIdRefOverride,
  navigate = vi.fn(),
  onReady,
  requestGateway,
  selectedStoredSessionId = null,
  selectedStoredSessionIdRef: selectedStoredSessionIdRefOverride
}: {
  activeSessionId?: null | string
  activeSessionIdRef?: MutableRefObject<null | string>
  navigate?: ReturnType<typeof vi.fn>
  onReady: (handle: HarnessHandle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  selectedStoredSessionId?: null | string
  selectedStoredSessionIdRef?: MutableRefObject<null | string>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId,
    activeSessionIdRef: activeSessionIdRefOverride ?? ref(activeSessionId),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: navigate as never,
    requestGateway,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId,
    selectedStoredSessionIdRef: selectedStoredSessionIdRefOverride ?? ref(selectedStoredSessionId),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

describe('desktop branch creation idempotency', () => {
  afterEach(() => {
    cleanup()
    setSessions([])
    vi.clearAllMocks()
  })

  it('coalesces duplicate stored-session branch attempts onto one backend child', async () => {
    const createReady = deferred<{ session_id: string; stored_session_id: string }>()

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        return createReady.promise as never
      }

      return {} as never
    })

    let actions: HarnessHandle | null = null

    setSessions([storedSession({ id: 'parent', message_count: 2, title: 'Parent' })])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [
        { content: 'question', role: 'user', timestamp: 1 },
        { content: 'answer', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'parent'
    } as never)

    render(<Harness onReady={value => (actions = value)} requestGateway={requestGateway} />)
    await waitFor(() => expect(actions).not.toBeNull())

    let first!: Promise<boolean>
    let second!: Promise<boolean>

    act(() => {
      first = actions!.branchStoredSession('parent')
      second = actions!.branchStoredSession('parent')
    })

    await waitFor(() =>
      expect(requestGateway.mock.calls.filter(([method]) => method === 'session.create')).toHaveLength(1)
    )

    await act(async () => {
      createReady.resolve({ session_id: 'runtime-branch', stored_session_id: 'stored-branch' })
      await expect(Promise.all([first, second])).resolves.toEqual([true, true])
    })

    expect(requestGateway.mock.calls.filter(([method]) => method === 'session.create')).toHaveLength(1)
    expect(requestGateway).toHaveBeenCalledWith(
      'session.create',
      expect.objectContaining({
        messages: [
          { content: 'question', role: 'user' },
          { content: 'answer', role: 'assistant' }
        ],
        parent_session_id: 'parent',
        source: 'desktop'
      })
    )
    expect($sessions.get().filter(session => session.id === 'stored-branch')).toHaveLength(1)
  })

  it('does not coalesce two same-id parents that live on different connections', async () => {
    // Two backends each expose a session called `parent`. They are different
    // conversations, so a route-blind flight key would collapse both branch
    // actions onto ONE create and hand the second caller the other backend's
    // child. Both creates are held open so the second call sees the first's
    // flight still in the map — that is the only state the key guards.
    const routedCreate = vi.mocked(requestGatewayForAgent)
    const pandoraCreate = deferred<{ session_id: string; stored_session_id: string }>()
    const otherCreate = deferred<{ session_id: string; stored_session_id: string }>()

    routedCreate.mockImplementation((async (connectionId: string, _profile: string, method: string) => {
      if (method !== 'session.create') {
        return {} as never
      }

      return connectionId === 'pandora' ? pandoraCreate.promise : otherCreate.promise
    }) as never)

    let actions: HarnessHandle | null = null

    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'question', role: 'user', timestamp: 1 }],
      session_id: 'parent'
    } as never)

    render(<Harness onReady={value => (actions = value)} requestGateway={vi.fn(async () => ({}) as never)} />)
    await waitFor(() => expect(actions).not.toBeNull())

    // Same stored id, one owner at a time in the row cache — the branch resolves
    // its owner from the row, so this is how the two owners reach forkBranch.
    setSessions([storedSession({ connection_id: 'pandora', id: 'parent', message_count: 2, profile: 'default' })])

    let first!: Promise<boolean>
    let second!: Promise<boolean>

    await act(async () => {
      first = actions!.branchStoredSession('parent')
      await waitFor(() => expect(routedCreate).toHaveBeenCalled())
    })

    setSessions([storedSession({ connection_id: 'other-box', id: 'parent', message_count: 2, profile: 'default' })])

    await act(async () => {
      second = actions!.branchStoredSession('parent')
      await waitFor(() =>
        expect(routedCreate.mock.calls.filter(([, , method]) => method === 'session.create')).toHaveLength(2)
      )
    })

    await act(async () => {
      pandoraCreate.resolve({ session_id: 'rt-pandora', stored_session_id: 'stored-pandora' })
      otherCreate.resolve({ session_id: 'rt-other', stored_session_id: 'stored-other-box' })
      await expect(Promise.all([first, second])).resolves.toEqual([true, true])
    })

    const creates = routedCreate.mock.calls.filter(([, , method]) => method === 'session.create')

    expect(creates.map(([connectionId]) => connectionId)).toEqual(['pandora', 'other-box'])
    // Two distinct children, not one child claimed twice.
    expect($sessions.get().filter(session => session.id === 'stored-pandora')).toHaveLength(1)
    expect($sessions.get().filter(session => session.id === 'stored-other-box')).toHaveLength(1)
  })
})

describe('connection-qualified session deletion', () => {
  afterEach(() => {
    cleanup()
    setSessions([])
    vi.clearAllMocks()
  })

  it('deletes a registry session through its captured connection owner', async () => {
    const requestGateway = vi.fn().mockResolvedValue({})
    let actions: HarnessHandle | null = null

    setSessions([
      storedSession({
        connection_id: 'source-a',
        id: 'shared-session',
        profile: 'worker'
      })
    ])
    vi.mocked(deleteSession).mockResolvedValue({ ok: true })
    vi.mocked(requestGatewayForAgent).mockResolvedValue({} as never)

    render(
      <Harness
        activeSessionId="runtime-shared"
        onReady={value => {
          actions = value
        }}
        requestGateway={requestGateway}
        selectedStoredSessionId="shared-session"
      />
    )
    await waitFor(() => expect(actions).not.toBeNull())

    await act(async () => {
      await actions?.removeSession('shared-session')
    })

    expect(deleteSession).toHaveBeenCalledWith('shared-session', {
      connectionId: 'source-a',
      profile: 'worker'
    })
    expect(requestGatewayForAgent).toHaveBeenCalledWith('source-a', 'worker', 'session.close', {
      session_id: 'runtime-shared'
    })
    expect(requestGateway).not.toHaveBeenCalledWith('session.close', expect.anything())
  })

  it('tears down the selected session from synchronous refs when render state is stale', async () => {
    const navigate = vi.fn()
    const requestGateway = vi.fn().mockResolvedValue({})
    const activeSessionIdRef: MutableRefObject<null | string> = { current: 'runtime-shared' }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: 'shared-session' }
    let actions: HarnessHandle | null = null

    setSessions([storedSession({ connection_id: 'source-a', id: 'shared-session', profile: 'worker' })])
    vi.mocked(deleteSession).mockResolvedValue({ ok: true })
    vi.mocked(requestGatewayForAgent).mockResolvedValue({} as never)

    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        navigate={navigate}
        onReady={value => {
          actions = value
        }}
        requestGateway={requestGateway}
        selectedStoredSessionId={null}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )
    await waitFor(() => expect(actions).not.toBeNull())

    await act(async () => {
      await actions?.removeSession('shared-session')
    })

    expect(navigate).toHaveBeenCalledWith(NEW_CHAT_ROUTE, { replace: true })
    expect(requestGatewayForAgent).toHaveBeenCalledWith('source-a', 'worker', 'session.close', {
      session_id: 'runtime-shared'
    })
    expect(selectedStoredSessionIdRef.current).toBeNull()
    expect(activeSessionIdRef.current).toBeNull()
  })
})

function StoredIdRotationHarness({
  activeSessionIdRef,
  getRoutedStoredSessionId,
  navigate,
  selectedStoredSessionIdRef
}: {
  activeSessionIdRef: MutableRefObject<string | null>
  getRoutedStoredSessionId: () => null | string
  navigate: (to: string, options?: { replace?: boolean }) => void
  selectedStoredSessionIdRef: MutableRefObject<string | null>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  useSessionActions({
    activeSessionId: activeSessionIdRef.current,
    activeSessionIdRef,
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId,
    navigate: navigate as never,
    requestGateway: async () => ({}) as never,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: selectedStoredSessionIdRef.current,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  return null
}

describe('active stored-session id rotation routing', () => {
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setActiveSessionStoredIdRotation(null)
    setSelectedStoredSessionId(null)
    vi.restoreAllMocks()
  })

  it('follows a rotation while the same conversation still owns the foreground route', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'runtime-A' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'stored-A' }
    const navigate = vi.fn()

    setSelectedStoredSessionId('stored-A')
    render(
      <StoredIdRotationHarness
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => 'stored-A'}
        navigate={navigate}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    act(() => {
      setActiveSessionStoredIdRotation({
        nextStoredSessionId: 'stored-A-next',
        previousStoredSessionId: 'stored-A',
        runtimeSessionId: 'runtime-A'
      })
    })

    await waitFor(() => expect(selectedStoredSessionIdRef.current).toBe('stored-A-next'))
    expect($selectedStoredSessionId.get()).toBe('stored-A-next')
    expect(navigate).toHaveBeenCalledWith(sessionRoute('stored-A-next'), { replace: true })
    expect($activeSessionStoredIdRotation.get()).toBeNull()
  })

  it('keeps draft on the previous tip when the new tip row is not loaded yet', async () => {
    const tipBefore = 'tip-root'
    const tipAfter = 'tip-new-unloaded'
    const runtimeSessionId = 'runtime-gap'
    const activeSessionIdRef: MutableRefObject<string | null> = { current: runtimeSessionId }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: tipBefore }
    const navigate = vi.fn()

    setSessions([])
    stashSessionDraft(tipBefore, 'typed during gap', [])
    setSelectedStoredSessionId(tipBefore)
    setActiveSessionId(runtimeSessionId)

    render(
      <StoredIdRotationHarness
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => tipBefore}
        navigate={navigate}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    act(() => {
      setActiveSessionStoredIdRotation({
        nextStoredSessionId: tipAfter,
        previousStoredSessionId: tipBefore,
        runtimeSessionId
      })
    })

    await waitFor(() => expect($selectedStoredSessionId.get()).toBe(tipAfter))
    expect(takeSessionDraft(tipBefore).text).toBe('typed during gap')
    expect(takeSessionDraft(tipAfter).text).toBe('')

    clearSessionDraft(tipBefore)
    clearSessionDraft(tipAfter)
    setActiveSessionId(null)
  })

  it('parks an in-progress composer draft on the lineage root across tip rotation', async () => {
    // Desktop draft must stay on the durable composer key (lineage root), not
    // move onto the fresh tip — ChatBar scopes drafts via resolveComposerSessionKey.
    const tipBefore = '20260720_062637_ad96b3'
    const tipAfter = '20260720_071049_a28905'
    const runtimeSessionId = 'runtime-desktop-thinking'
    const activeSessionIdRef: MutableRefObject<string | null> = { current: runtimeSessionId }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: tipBefore }
    const navigate = vi.fn()
    const typedWhileThinking = 'follow up I am still typing during thinking'

    setSessions([storedSession({ id: tipAfter, message_count: 2, _lineage_root_id: tipBefore })])
    stashSessionDraft(tipBefore, typedWhileThinking, [])
    setSelectedStoredSessionId(tipBefore)
    setActiveSessionId(runtimeSessionId)

    render(
      <StoredIdRotationHarness
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => tipBefore}
        navigate={navigate}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    act(() => {
      setActiveSessionStoredIdRotation({
        nextStoredSessionId: tipAfter,
        previousStoredSessionId: tipBefore,
        runtimeSessionId
      })
    })

    await waitFor(() => expect($selectedStoredSessionId.get()).toBe(tipAfter))
    // Durable key remains the lineage root — same scope ChatBar will keep using.
    expect(takeSessionDraft(tipBefore).text).toBe(typedWhileThinking)
    expect(takeSessionDraft(tipAfter).text).toBe('')

    clearSessionDraft(tipBefore)
    clearSessionDraft(tipAfter)
    setActiveSessionId(null)
    setSessions([])
  })

  it('does not overwrite a newer route intent before its resume effect has synchronized selection', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'runtime-A' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'stored-A' }
    const navigate = vi.fn()

    setSelectedStoredSessionId('stored-A')
    render(
      <StoredIdRotationHarness
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => 'stored-C'}
        navigate={navigate}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    act(() => {
      setActiveSessionStoredIdRotation({
        nextStoredSessionId: 'stored-A-next',
        previousStoredSessionId: 'stored-A',
        runtimeSessionId: 'runtime-A'
      })
    })

    await waitFor(() => expect($activeSessionStoredIdRotation.get()).toBeNull())
    expect(selectedStoredSessionIdRef.current).toBe('stored-A')
    expect($selectedStoredSessionId.get()).toBe('stored-A')
    expect(navigate).not.toHaveBeenCalled()
  })

  it('does not let the previous runtime jump back after selection already moved', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'runtime-A' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'stored-C' }
    const navigate = vi.fn()

    setSelectedStoredSessionId('stored-C')
    render(
      <StoredIdRotationHarness
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => 'stored-C'}
        navigate={navigate}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    act(() => {
      setActiveSessionStoredIdRotation({
        nextStoredSessionId: 'stored-A-next',
        previousStoredSessionId: 'stored-A',
        runtimeSessionId: 'runtime-A'
      })
    })

    await waitFor(() => expect($activeSessionStoredIdRotation.get()).toBeNull())
    expect(selectedStoredSessionIdRef.current).toBe('stored-C')
    expect($selectedStoredSessionId.get()).toBe('stored-C')
    expect(navigate).not.toHaveBeenCalled()
  })

  it('updates the underlying selection without navigating out of an overlay or page', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'runtime-A' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'stored-A' }
    const navigate = vi.fn()

    setSelectedStoredSessionId('stored-A')
    render(
      <StoredIdRotationHarness
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => null}
        navigate={navigate}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    act(() => {
      setActiveSessionStoredIdRotation({
        nextStoredSessionId: 'stored-A-next',
        previousStoredSessionId: 'stored-A',
        runtimeSessionId: 'runtime-A'
      })
    })

    await waitFor(() => expect(selectedStoredSessionIdRef.current).toBe('stored-A-next'))
    expect($selectedStoredSessionId.get()).toBe('stored-A-next')
    expect(navigate).not.toHaveBeenCalled()
  })
})

async function createWith(
  profileSetup: () => void,
  beforeCreate?: (handle: HarnessHandle) => Promise<void> | void
): Promise<Record<string, unknown> | undefined> {
  let createParams: Record<string, unknown> | undefined

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.create') {
      createParams = params

      return { session_id: RUNTIME_SESSION_ID, stored_session_id: null } as never
    }

    return {} as never
  })

  setCurrentCwd('')
  setNewChatWorkspaceTarget(undefined)
  profileSetup()

  let handle: HarnessHandle | null = null
  render(<Harness onReady={h => (handle = h)} requestGateway={requestGateway} />)
  await waitFor(() => expect(handle).not.toBeNull())

  if (beforeCreate) {
    await act(async () => {
      await beforeCreate(handle!)
    })
  }

  await act(async () => {
    await handle!.createBackendSessionForSend()
  })

  return createParams
}

describe('startFreshSessionDraft', () => {
  afterEach(() => cleanup())

  it('can reset machine-bound session state without closing the current overlay route', async () => {
    const navigate = vi.fn()
    const requestGateway = vi.fn(async () => ({}) as never)
    let handle: HarnessHandle | null = null

    render(<Harness navigate={navigate} onReady={value => (handle = value)} requestGateway={requestGateway} />)
    await waitFor(() => expect(handle).not.toBeNull())

    act(() => handle!.startFreshSessionDraft({ preserveRoute: true, workspaceTarget: null }))

    expect(navigate).not.toHaveBeenCalled()
    expect($currentCwd.get()).toBe('')
    expect($newChatWorkspaceTarget.get()).toBeNull()
  })

  it('fronts the workspace without closing a terminal that is merely behind a tab', async () => {
    // Regression: a persisted terminal takeover kept the terminal fronted
    // after New Session / ⌘N. The fix is to reveal the workspace — NOT to
    // clear the takeover atom. That atom is the terminal's open/closed state
    // in every layout: clearing it here closed a terminal sitting in its own
    // zone (Default / Terminal deck / Quad), and persisted a `false` that left
    // the Focus tab unable to mount its workspace after a restart. Behind a
    // tab the terminal is hidden, not closed.
    const navigate = vi.fn()
    const requestGateway = vi.fn(async () => ({}) as never)
    let handle: HarnessHandle | null = null

    setTerminalTakeover(true)
    expect($terminalTakeover.get()).toBe(true)

    render(<Harness navigate={navigate} onReady={value => (handle = value)} requestGateway={requestGateway} />)
    await waitFor(() => expect(handle).not.toBeNull())

    act(() => handle!.startFreshSessionDraft({ preserveRoute: true, workspaceTarget: null }))

    expect(revealTreePane).toHaveBeenCalledWith('workspace')
    expect($terminalTakeover.get()).toBe(true)
  })
})

describe('createBackendSessionForSend profile routing', () => {
  afterEach(() => {
    cleanup()
    $newChatProfile.set(null)
    $newChatRoute.set(null)
    $activeGatewayProfile.set('default')
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    $currentCwd.set('')
    $currentFastMode.set(false)
    $currentModel.set('')
    $currentProvider.set('')
    setCurrentModelSource('')
    $currentReasoningEffort.set('')
    setNewChatWorkspaceTarget(undefined)
    vi.restoreAllMocks()
  })

  it('routes a plain new chat (no explicit profile) to the live gateway profile', async () => {
    // The "rubberband to default" bug: the top New Session button clears
    // $newChatProfile to null. In global-remote mode one backend serves every
    // profile, so an omitted `profile` lands the chat on the launch (default)
    // profile. The session must instead carry the active gateway profile.
    const params = await createWith(() => {
      $activeGatewayProfile.set('coder')
      $newChatProfile.set(null)
    })

    expect(params).toMatchObject({ profile: 'coder' })
  })

  it('honours an explicit per-profile "+" selection', async () => {
    const params = await createWith(() => {
      $activeGatewayProfile.set('coder')
      $newChatProfile.set('analyst')
    })

    expect(params).toMatchObject({ profile: 'analyst' })
  })

  it('passes the default profile for single-profile users (backend resolves it to launch)', async () => {
    const params = await createWith(() => {
      $activeGatewayProfile.set('default')
      $newChatProfile.set(null)
    })

    expect(params).toMatchObject({ profile: 'default' })
  })

  it('tags new desktop chats as desktop sessions', async () => {
    const params = await createWith(() => {})

    expect(params).toMatchObject({ source: 'desktop' })
  })

  // Regression (Settings → Model doesn't stick): a stale composer selection
  // must not be shipped as a per-session override on a NEW chat.
  //
  // Saving Settings → Model while a session is live deliberately leaves
  // $currentModel painted with the LIVE agent's model (applySavedMainModel
  // keeps the live session authoritative) and only flips the source to
  // 'default'. If session.create still sent that value, every new chat was
  // pinned to the old model and the backend never resolved model.default —
  // the user-visible "my default won't change" bug.
  it('omits a default-sourced selection so the backend resolves model.default', async () => {
    const params = await createWith(() => {
      // What the composer holds after Settings saved a new default while a
      // chat was open: the previous session's model, marked default-sourced.
      setCurrentModel('openai/gpt-5.6-sol')
      setCurrentProvider('openai-codex')
      setCurrentModelSource('default')
    })

    expect(params).not.toHaveProperty('model')
    expect(params).not.toHaveProperty('provider')
  })

  it('still sends an explicit manual pick as a per-session override', async () => {
    const params = await createWith(() => {
      setCurrentModel('anthropic/claude-opus-5')
      setCurrentProvider('anthropic')
      setCurrentModelSource('manual')
    })

    expect(params).toMatchObject({
      model: 'anthropic/claude-opus-5',
      provider: 'anthropic'
    })
  })

  // An unset source is the first-run/cleared state — nothing the user picked,
  // so it must not pin the session either.
  it('omits the model when no selection source is recorded', async () => {
    const params = await createWith(() => {
      setCurrentModel('openai/gpt-5.6-sol')
      setCurrentProvider('openai-codex')
      setCurrentModelSource('')
    })

    expect(params).not.toHaveProperty('model')
    expect(params).not.toHaveProperty('provider')
  })

  // Effort and fast mode are independent of the model-override decision.
  it('keeps sending reasoning effort even when the model is omitted', async () => {
    const params = await createWith(() => {
      setCurrentModel('openai/gpt-5.6-sol')
      setCurrentProvider('openai-codex')
      setCurrentModelSource('default')
      setCurrentReasoningEffort('high')
    })

    expect(params).not.toHaveProperty('model')
    expect(params).toMatchObject({ reasoning_effort: 'high' })
  })

  it('passes the current workspace cwd into session.create', async () => {
    const params = await createWith(() => {
      $currentCwd.set('/remote/worktree')
    })

    expect(params).toMatchObject({ cwd: '/remote/worktree' })
  })

  it('keeps a route-aware New Chat pinned when foreground activation changes before Send', async () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'default',
      targetProfile: 'backend-default'
    }

    const ambientRequest = vi.fn(async () => ({}) as never)
    vi.mocked(requestGatewayForAgent).mockResolvedValueOnce({
      session_id: RUNTIME_SESSION_ID,
      stored_session_id: null
    } as never)

    $newChatProfile.set(route.profile)
    $newChatRoute.set({ ...route })
    $activeGatewayProfile.set('other-connection-profile')

    let handle: HarnessHandle | null = null
    render(<Harness onReady={value => (handle = value)} requestGateway={ambientRequest} />)
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      await handle!.createBackendSessionForSend()
    })

    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'source-a',
      'default',
      'session.create',
      expect.objectContaining({ profile: 'backend-default', source: 'desktop' })
    )
    expect(ambientRequest).not.toHaveBeenCalledWith('session.create', expect.anything())
  })

  it('freezes the visible selector state before profile readiness and sends fast: false explicitly', async () => {
    const profileReady = deferred<void>()
    vi.mocked(ensureGatewayProfile).mockReturnValueOnce(profileReady.promise)

    setCurrentModel('anthropic/claude-sonnet-4.6')
    setCurrentProvider('anthropic')
    // A real composer pick marks the selection manual; this test drives the
    // atoms directly, so set the source explicitly. Only a manual selection
    // rides along as a per-session override.
    setCurrentModelSource('manual')
    setCurrentReasoningEffort('high')
    setCurrentFastMode(false)

    let createParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        createParams = params

        return { session_id: RUNTIME_SESSION_ID, stored_session_id: null } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(<Harness onReady={next => (handle = next)} requestGateway={requestGateway} />)
    await waitFor(() => expect(handle).not.toBeNull())

    let createPromise!: Promise<null | string>
    act(() => {
      createPromise = handle!.createBackendSessionForSend()
    })
    await waitFor(() => expect(ensureGatewayProfile).toHaveBeenCalled())

    // A background refresh or a second click can mutate the sticky atoms while
    // the profile is waking. This send must still use what was visible at Enter.
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('openai-codex')
    setCurrentReasoningEffort('low')
    setCurrentFastMode(true)
    profileReady.resolve()

    await act(async () => {
      await createPromise
    })

    expect(createParams).toMatchObject({
      fast: false,
      model: 'anthropic/claude-sonnet-4.6',
      provider: 'anthropic',
      reasoning_effort: 'high'
    })
  })

  it('falls back to the entered project cwd when the current cwd is blank', async () => {
    const params = await createWith(() => {
      $projectTree.set([
        {
          id: 'p_app',
          label: 'App',
          path: '/repo/app',
          repos: [{ groups: [], id: '/repo/app', label: 'app', path: '/repo/app', sessionCount: 0 }],
          sessionCount: 0
        }
      ])
      $projectScope.set('p_app')
      $currentCwd.set('')
    })

    expect(params).toMatchObject({ cwd: '/repo/app' })
  })
})

// ── Resume failure recovery (the "stuck loading session window" bug) ──────────
// When session.resume rejects AND the REST transcript fallback ALSO fails, the
// hook must (a) not throw out of the fallback (which stranded the loader), and
// (b) arm $resumeFailedSessionId so use-route-resume can retry. A resume that
// succeeds must NOT leave the flag armed.
function ResumeHarness({
  getRouteToken = () => 'token',
  getRoutedStoredSessionId = () => null,
  navigate = vi.fn() as unknown as NavigateFunction,
  onStateUpdate,
  onViewSync,
  onReady,
  requestGateway,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId = null,
  sessionStateByRuntimeIdRef
}: {
  getRouteToken?: () => string
  getRoutedStoredSessionId?: () => string | null
  navigate?: NavigateFunction
  onStateUpdate?: (sessionId: string, state: ClientSessionState) => void
  onViewSync?: (sessionId: string, state: ClientSessionState) => void
  onReady: (
    resume: (storedSessionId: string, replaceRoute?: boolean, ownerRoute?: SessionProfileRoute) => Promise<unknown>
  ) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  runtimeIdByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
  selectedStoredSessionId?: string | null
  sessionStateByRuntimeIdRef?: MutableRefObject<Map<string, ClientSessionState>>
}) {
  const activeSessionIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)
  const fallbackRuntimeMapRef = useRef(new Map<string, string>())
  const fallbackStateMapRef = useRef(new Map<string, ClientSessionState>())
  const selectedStoredSessionIdRef = useRef<string | null>(selectedStoredSessionId)
  const runtimeMapRef = runtimeIdByStoredSessionIdRef ?? fallbackRuntimeMapRef
  const stateMapRef = sessionStateByRuntimeIdRef ?? fallbackStateMapRef

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken,
    getRoutedStoredSessionId,
    navigate,
    requestGateway,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef: runtimeMapRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: stateMapRef,
    syncSessionStateToView: (sessionId, state) => onViewSync?.(sessionId, state),
    updateSessionState: (sessionId, updater, storedSessionId) => {
      // Full default shape (not a bare {} cast) so seeded/derived fields like
      // turnStartedAt behave as in production state updates.
      const current = stateMapRef.current.get(sessionId) ?? createClientSessionState(storedSessionId ?? null)
      const next = updater(current)

      stateMapRef.current.set(sessionId, next)
      onStateUpdate?.(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

function RouteReaderHarness({
  getRouteToken,
  onReady,
  requestGateway,
  sessionStateByRuntimeIdRef
}: {
  getRouteToken: () => string
  onReady: (resume: (storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
}) {
  const activeSessionIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const selectedStoredSessionIdRef = useRef<string | null>(null)
  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken,
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView: vi.fn(),
    updateSessionState: (sessionId, updater, storedSessionId) => {
      const current =
        sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedSessionId ?? null)
      const next = updater(current)

      sessionStateByRuntimeIdRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

function ResumeTimerHarness({
  onReady,
  requestGateway
}: {
  onReady: (resume: (storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const activeSessionId = useStore($activeSessionId)
  const busyRef = useRef(false)

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId: null,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const actions = useSessionActions({
    activeSessionId,
    activeSessionIdRef: cache.activeSessionIdRef,
    busyRef,
    creatingSessionRef: useRef(false),
    ensureSessionState: cache.ensureSessionState,
    getRouteToken: () => 'timer-contract',
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: cache.resetViewSync,
    runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: cache.sessionStateByRuntimeIdRef,
    holdSessionTranscriptView: cache.holdSessionTranscriptView,
    syncSessionStateToView: cache.syncSessionStateToView,
    getRoutedStoredSessionId: () => null,
    updateSessionState: cache.updateSessionState
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

describe('resumeSession failure recovery', () => {
  beforeEach(() => {
    clearSingleFlightSessionResumeState()
    for (const name of SESSION_OPEN_MARKS) {
      performance.clearMarks(name)
    }
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback: FrameRequestCallback) => {
      callback(0)

      return null as unknown as number
    })
  })

  afterEach(() => {
    clearSingleFlightSessionResumeState()
    setSessionOpenPerfFixtureRunner(null)
    cleanup()
    window.localStorage.clear()
    setActiveSessionId(null)
    setResumeFailedSessionId(null)
    setSelectedStoredSessionId(null)
    setMessages([])
    setSessions([])
    $removedSessionIds.set(new Set())
    $sessionMutationsInFlight.set(new Set())
    clearClarifyRequest()
    vi.restoreAllMocks()
    for (const name of SESSION_OPEN_MARKS) {
      performance.clearMarks(name)
    }
  })

  async function runResume(
    requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>,
    options: {
      runtimeIdByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
      sessionStateByRuntimeIdRef?: MutableRefObject<Map<string, ClientSessionState>>
    } = {}
  ): Promise<void> {
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={r => (resume = r)} requestGateway={requestGateway} {...options} />)
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-1', true)
  }

  it('marks selection, settled resume, and the authoritative history publication', async () => {
    setSessions([storedSession({ id: 'stored-1', message_count: 1 })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 1,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'synthetic persisted answer', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'stored-1',
      session_id: 'stored-1'
    })

    await runResume(async method => {
      if (method === 'session.resume') {
        return {
          info: {},
          messages: [],
          messages_omitted: true,
          resumed: 'stored-1',
          session_id: 'runtime-1',
          session_key: 'stored-1'
        } as never
      }

      return {} as never
    })

    await waitFor(() => expect(performance.getEntriesByName('hermes.session.history.ready')).toHaveLength(1))
    expect(performance.getEntriesByName('hermes.session.select')).toHaveLength(1)
    expect(performance.getEntriesByName('hermes.session.resume.ready')).toHaveLength(1)
    expect(performance.getEntriesByName('hermes.session.rest.commit')).toHaveLength(1)
  })

  it('drives the controlled perf fixture through the mounted production resume orchestration', async () => {
    const authority = {
      connectionId: '',
      displayRevision: 1,
      lineageRootId: 'perf-root',
      profile: 'default',
      resolvedTipId: 'perf-session'
    }
    const outerGateway = vi.fn(async () => ({}) as never)
    let routedSessionId: string | null = null
    let resumeFromRoute:
      | ((storedSessionId: string, replaceRoute?: boolean, ownerRoute?: SessionProfileRoute) => Promise<unknown>)
      | null = null
    const navigateSpy = vi.fn(() => {
      window.setTimeout(() => {
        routedSessionId = 'perf-session'
        void resumeFromRoute?.('perf-session', true)
      }, 25)
    })
    const navigate = navigateSpy as unknown as NavigateFunction
    const fixtureGateway = vi.fn(async (method: string) => {
      expect(method).toBe('session.resume')

      return {
        info: {},
        messages: [],
        messages_omitted: true,
        resumed: 'perf-session',
        session_id: 'perf-runtime',
        session_key: 'perf-session'
      } as never
    })

    vi.mocked(ensureGatewayProfile).mockClear()

    setSessions([
      storedSession({
        _lineage_root_id: authority.lineageRootId,
        display_revision: authority.displayRevision,
        id: authority.resolvedTipId,
        message_count: 1,
        profile: authority.profile
      })
    ])
    saveTranscriptTail(
      'perf-session',
      [{ id: 'cached', parts: [{ type: 'text', text: 'cached answer' }], role: 'assistant' }],
      authority
    )
    render(
      <ResumeHarness
        getRoutedStoredSessionId={() => routedSessionId}
        navigate={navigate}
        onReady={resume => {
          resumeFromRoute = resume
        }}
        requestGateway={outerGateway}
      />
    )
    await waitFor(() => expect(resumeFromRoute).not.toBeNull())

    // ResumeHarness mounts the same hook used by Desktop. Its DEV-only private
    // registration makes this invocation exercise that real orchestration.
    await runSessionOpenPerfFixture({
      delayRuntimeMs: 0,
      fetchLatest: async () => ({
        display_revision: authority.displayRevision,
        lineage_root_id: authority.lineageRootId,
        messages: [{ content: 'persisted answer', role: 'assistant', timestamp: 1 }],
        resolved_tip_id: authority.resolvedTipId,
        session_id: 'perf-session'
      }),
      requestGateway: fixtureGateway,
      storedSessionId: 'perf-session'
    })

    await waitFor(() => expect(performance.getEntriesByName('hermes.session.history.ready')).toHaveLength(1))
    await new Promise(resolve => window.setTimeout(resolve, 50))
    expect(performance.getEntriesByName('hermes.session.select')).toHaveLength(1)
    expect(performance.getEntriesByName('hermes.session.cache.commit').length).toBeGreaterThanOrEqual(1)
    expect(performance.getEntriesByName('hermes.session.rest.commit')).toHaveLength(1)
    expect(performance.getEntriesByName('hermes.session.resume.ready')).toHaveLength(1)
    expect(fixtureGateway).toHaveBeenCalledWith('session.resume', expect.anything())
    expect(fixtureGateway).toHaveBeenCalledTimes(1)
    expect(navigateSpy).toHaveBeenCalledWith(sessionRoute('perf-session'), { replace: true })
    expect(outerGateway).not.toHaveBeenCalled()
    expect(ensureGatewayProfile).not.toHaveBeenCalled()
  })

  it('cleans exact synthetic session state between controlled fixture rounds', async () => {
    const runtimeMapRef: MutableRefObject<Map<string, string>> = { current: new Map() }
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    const originalMessages: ReturnType<typeof $messages.get> = [
      { id: 'baseline-message', parts: [{ type: 'text', text: 'baseline' }], role: 'assistant' }
    ]
    const originalSessions = [storedSession({ id: 'baseline-session', message_count: 1 })]

    setSessions(originalSessions)
    setMessages(originalMessages)
    setActiveSessionId('baseline-runtime')
    setSelectedStoredSessionId('baseline-session')
    let routedSessionId: string | null = 'baseline-session'
    let resumeFromRoute:
      | ((storedSessionId: string, replaceRoute?: boolean, ownerRoute?: SessionProfileRoute) => Promise<unknown>)
      | null = null
    let routedPerfRound = 0
    const navigate = vi.fn(() => {
      routedPerfRound += 1
      routedSessionId = `perf-session-${routedPerfRound}`
      window.setTimeout(() => {
        void resumeFromRoute?.(routedSessionId!, true)
      }, 0)
    }) as unknown as NavigateFunction
    const PerfTranscriptDomMirror = () => {
      const messages = useStore($messages)
      const selectedStoredSessionId = useStore($selectedStoredSessionId)

      return (
        <main data-hermes-perf-session={selectedStoredSessionId ?? undefined}>
          {messages.map(message => (
            <article data-message-id={message.id} key={message.id}>
              {message.parts.map(part => (part.type === 'text' ? part.text : '')).join('')}
            </article>
          ))}
        </main>
      )
    }
    render(
      <>
        <ResumeHarness
          getRoutedStoredSessionId={() => routedSessionId}
          navigate={navigate}
          onReady={resume => {
            resumeFromRoute = resume
          }}
          requestGateway={vi.fn(async () => ({}) as never)}
          runtimeIdByStoredSessionIdRef={runtimeMapRef}
          selectedStoredSessionId="baseline-session"
          sessionStateByRuntimeIdRef={stateMapRef}
        />
        <PerfTranscriptDomMirror />
      </>
    )

    await import('../../chat/perf-probe')

    const drive = window.__PERF_DRIVE__
    expect(drive).toBeDefined()

    for (let round = 0; round < 2; round += 1) {
      await drive!.sessionSwitch({ verifiedCache: true })

      expect($messages.get()).toBe(originalMessages)
      expect($sessions.get()).toBe(originalSessions)
      expect($activeSessionId.get()).toBe('baseline-runtime')
      expect($selectedStoredSessionId.get()).toBe('baseline-session')
      expect([...runtimeMapRef.current.keys()].some(id => id.startsWith('perf-session-'))).toBe(false)
      expect([...stateMapRef.current.keys()].some(id => id.startsWith('perf-runtime-'))).toBe(false)
      expect(
        Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index)).some(key =>
          key?.includes('perf-session-')
        )
      ).toBe(false)
    }
  })

  it('attaches exact persisted-display provenance after a cold revisioned REST hydrate', async () => {
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    const expectedProof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-2'
    }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-2',
        message_count: 2
      })
    ])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [
        { content: 'hello', role: 'user', timestamp: 1 },
        { content: 'hi', role: 'assistant', timestamp: 2 }
      ],
      resolved_tip_id: 'tip-2',
      session_id: 'tip-2'
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 2,
          messages: [],
          messages_omitted: true,
          resumed: 'tip-2',
          session_id: 'runtime-1',
          session_key: 'tip-2'
        } as never
      }

      return {} as never
    })
    await runResume(requestGateway, { sessionStateByRuntimeIdRef: stateMapRef })

    await waitFor(() =>
      expect(stateMapRef.current.get('runtime-1')?.transcriptProvenance).toEqual({
        ...expectedProof,
        coverage: 'latest-page',
        source: 'persisted-display',
        storedSessionId: 'stored-1'
      })
    )
    expect(loadTranscriptTail('stored-1', expectedProof)?.messages[1]).toMatchObject({
      parts: [{ text: 'hi', type: 'text' }],
      role: 'assistant'
    })
  })

  it('does not cache or attach a hydrated proof after the route invalidates while resume is deferred', async () => {
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    const deferredResume = deferred<SessionResumeResponse>()
    const proof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-1'
    }
    let routeToken = 'route-before'

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-1',
        message_count: 2
      })
    ])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [
        { content: 'question', role: 'user', timestamp: 1 },
        { content: 'answer', role: 'assistant', timestamp: 2 }
      ],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1'
    })

    const requestGateway = vi.fn(<T,>(method: string, _params?: Record<string, unknown>): Promise<T> => {
      if (method === 'session.resume') {
        return deferredResume.promise as Promise<T>
      }

      return Promise.resolve({} as T)
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        getRouteToken={() => routeToken}
        onReady={value => (resume = value)}
        requestGateway={requestGateway as <T,>(method: string, params?: Record<string, unknown>) => Promise<T>}
        sessionStateByRuntimeIdRef={stateMapRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const pending = resume!('stored-1', true)
    await waitFor(() => expect($messages.get()).toHaveLength(2))

    routeToken = 'route-after'
    deferredResume.resolve({
      info: {},
      message_count: 2,
      messages: [],
      messages_omitted: true,
      resumed: 'tip-1',
      session_id: 'runtime-1',
      session_key: 'tip-1'
    })
    await pending

    expect(loadTranscriptTail('stored-1', proof)).toBeNull()
    expect(stateMapRef.current.get('runtime-1')?.transcriptProvenance).toBeUndefined()
  })

  it('rejects a changed cold hydration after the accepted runtime is optimistically edited', async () => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    const proof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-1'
    }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-1',
        message_count: 1
      })
    ])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)
    const serialize = vi.spyOn(transcriptTailCache, 'saveTranscriptTail')

    serialize.mockClear()
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 1,
          messages: [{ content: 'runtime baseline', role: 'assistant', timestamp: 0 }],
          resumed: 'tip-1',
          session_id: 'runtime-optimistic-changed',
          session_key: 'tip-1'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway, { sessionStateByRuntimeIdRef: stateMapRef })
    const runtimeState = stateMapRef.current.get('runtime-optimistic-changed')!
    const optimistic = applyRewindOptimistic(
      {
        ...runtimeState,
        messages: [{ id: 'optimistic-edit', parts: [{ text: 'optimistic edit', type: 'text' }], role: 'user' }]
      },
      0
    )

    stateMapRef.current.set('runtime-optimistic-changed', optimistic)
    setMessages(optimistic.messages)
    expect(runtimeState.transcriptAuthorityEpoch ?? 0).toBe(0)
    expect(stateMapRef.current.get('runtime-optimistic-changed')?.transcriptAuthorityEpoch).toBe(1)
    persisted.resolve({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'stale durable answer', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1'
    })
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })

    expect(JSON.stringify($messages.get())).toContain('optimistic edit')
    expect(JSON.stringify($messages.get())).not.toContain('stale durable answer')
    expect(stateMapRef.current.get('runtime-optimistic-changed')?.transcriptProvenance).toBeUndefined()
    expect(loadTranscriptTail('stored-1', proof)).toBeNull()
    expect(serialize).not.toHaveBeenCalled()
  })

  it('does not restore unchanged cache proof after the accepted runtime is optimistically edited', async () => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    const proof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-1'
    }
    const cached = [
      { id: 'cached-before-edit', parts: [{ text: 'cached before edit', type: 'text' as const }], role: 'assistant' as const }
    ]

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-1',
        message_count: 1
      })
    ])
    saveTranscriptTail('stored-1', cached, proof)
    const serialize = vi.spyOn(transcriptTailCache, 'saveTranscriptTail')

    serialize.mockClear()
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 1,
          messages: [],
          messages_omitted: true,
          resumed: 'tip-1',
          session_id: 'runtime-optimistic-unchanged',
          session_key: 'tip-1'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway, { sessionStateByRuntimeIdRef: stateMapRef })
    const runtimeState = stateMapRef.current.get('runtime-optimistic-unchanged')!
    const optimistic = applyRewindOptimistic(
      {
        ...runtimeState,
        messages: [{ id: 'optimistic-edit', parts: [{ text: 'optimistic edit', type: 'text' }], role: 'user' }]
      },
      0
    )

    stateMapRef.current.set('runtime-optimistic-unchanged', optimistic)
    setMessages(optimistic.messages)
    persisted.resolve({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1',
      unchanged: true
    })
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })

    expect(JSON.stringify($messages.get())).toContain('optimistic edit')
    expect(JSON.stringify($messages.get())).not.toContain('cached before edit')
    expect(stateMapRef.current.get('runtime-optimistic-unchanged')?.transcriptProvenance).toBeUndefined()
    expect(serialize).not.toHaveBeenCalled()
  })

  it('uses the latest route-token reader after the harness rerenders', async () => {
    const deferredResume = deferred<SessionResumeResponse>()
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    let currentRouteToken = 'route-before'
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    const onReady = (value: typeof resume) => {
      resume = value
    }

    const requestGatewayMock = vi.fn((method: string, _params?: Record<string, unknown>) =>
      method === 'session.resume' ? deferredResume.promise : Promise.resolve({})
    )

    const requestGateway = <T,>(method: string, params?: Record<string, unknown>): Promise<T> =>
      requestGatewayMock(method, params) as Promise<T>

    setSessions([storedSession({ id: 'stored-1', message_count: 0 })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-1' })

    const mounted = render(
      <RouteReaderHarness
        getRouteToken={() => 'stale-reader'}
        onReady={onReady}
        requestGateway={requestGateway}
        sessionStateByRuntimeIdRef={stateMapRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    mounted.rerender(
      <RouteReaderHarness
        getRouteToken={() => currentRouteToken}
        onReady={onReady}
        requestGateway={requestGateway}
        sessionStateByRuntimeIdRef={stateMapRef}
      />
    )

    const pending = resume!('stored-1', true)

    await waitFor(() => expect(requestGatewayMock).toHaveBeenCalledWith('session.resume', expect.anything()))
    currentRouteToken = 'route-after'

    deferredResume.resolve({
      info: {},
      message_count: 0,
      messages: [],
      resumed: 'stored-1',
      session_id: 'runtime-stale-route',
      session_key: 'stored-1'
    })
    await pending

    expect(stateMapRef.current.has('runtime-stale-route')).toBe(false)
  })

  it('invalidates a pending resume when stored-id rotation removes the old reverse mapping', async () => {
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    const activate = deferred<SessionResumeResponse>()
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-1', 'runtime-before-rotation']])
    }
    const stateBeforeRotation = createClientSessionState('stored-1')

    stateBeforeRotation.messages = [
      { id: 'before', parts: [{ text: 'before rotation', type: 'text' }], role: 'assistant' }
    ]
    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-before-rotation', stateBeforeRotation]])
    }

    const requestGatewayMock = vi.fn((method: string, _params?: Record<string, unknown>) =>
      method === 'session.activate' ? activate.promise : Promise.resolve({})
    )

    const requestGateway = <T,>(method: string, params?: Record<string, unknown>): Promise<T> =>
      requestGatewayMock(method, params) as Promise<T>

    setSessions([storedSession({ id: 'stored-1', message_count: 1 })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [{ content: 'stale pre-rotation transcript', role: 'assistant', timestamp: 1 }],
      session_id: 'stored-1'
    })

    render(
      <ResumeHarness
        onReady={value => (resume = value)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    vi.mocked(getLatestSessionMessages).mockClear()

    const pending = resume!('stored-1', true)

    await waitFor(() => expect(requestGatewayMock).toHaveBeenCalledWith('session.activate', expect.anything()))

    act(() => {
      // Mirror useSessionStateCache.ensureSessionState's real rotation order:
      // remove the obsolete reverse key, install the next id, and invalidate
      // the same runtime state's transcript authority generation.
      runtimeIdByStoredSessionIdRef.current.delete('stored-1')
      runtimeIdByStoredSessionIdRef.current.set('stored-1-next', 'runtime-before-rotation')
      sessionStateByRuntimeIdRef.current.set('runtime-before-rotation', {
        ...stateBeforeRotation,
        storedSessionId: 'stored-1-next',
        transcriptAuthorityEpoch: 1,
        transcriptProvenance: undefined
      })
    })
    expect(runtimeIdByStoredSessionIdRef.current.has('stored-1')).toBe(false)
    expect(runtimeIdByStoredSessionIdRef.current.get('stored-1-next')).toBe('runtime-before-rotation')

    activate.resolve({
      info: {},
      message_count: 1,
      messages: [],
      messages_omitted: true,
      resumed: 'stored-1',
      running: false,
      session_id: 'runtime-before-rotation',
      session_key: 'stored-1'
    })
    await pending

    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect(JSON.stringify($messages.get())).not.toContain('stale pre-rotation transcript')
  })

  it('publishes runtime liveness before a deferred persisted-display read settles', async () => {
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()

    setSessions([storedSession({ id: 'stored-1', message_count: 2 })])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)

    const requestGateway = vi.fn(async (method: string, _params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return {
          info: {},
          inflight: { assistant: 'working', streaming: true, user: 'question' },
          messages: [],
          messages_omitted: true,
          resumed: 'stored-1',
          running: true,
          session_id: 'runtime-live',
          session_key: 'stored-1',
          todo_state: { items: [] }
        } as never
      }

      return {} as never
    })
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={value => (resume = value)}
        requestGateway={requestGateway as <T,>(method: string, params?: Record<string, unknown>) => Promise<T>}
        sessionStateByRuntimeIdRef={stateMapRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const pending = resume!('stored-1', true)

    try {
      await waitFor(() =>
        expect(stateMapRef.current.get('runtime-live')).toMatchObject({
          awaitingResponse: true,
          busy: true,
          turnLive: true
        })
      )
    } finally {
      persisted.resolve({ messages: [], session_id: 'stored-1' })
      await pending
    }
  })

  it('publishes an unproven remote legacy transcript before its deferred resume settles', async () => {
    const deferredResume = deferred<SessionResumeResponse>()
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'remote-legacy',
        id: 'tip-remote',
        message_count: 2,
        profile: 'coder'
      })
    ])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'remote legacy question', role: 'user', timestamp: 1 },
        { content: 'remote legacy answer', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'tip-remote'
    })

    const requestGateway = vi.fn((method: string, _params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return deferredResume.promise
      }

      return Promise.resolve({})
    })
    vi.mocked(requestGatewayForAgent).mockImplementation((_connectionId, _profile, method, params) =>
      requestGateway(method, params) as never
    )

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={value => (resume = value)}
        requestGateway={requestGateway as <T,>(method: string, params?: Record<string, unknown>) => Promise<T>}
        sessionStateByRuntimeIdRef={stateMapRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const pending = resume!('stored-1', true)
    await waitFor(() => expect(JSON.stringify($messages.get())).toContain('remote legacy answer'))
    expect(
      Array.from({ length: window.localStorage.length }, (_value, index) => window.localStorage.key(index)).filter(key =>
        key?.startsWith('hermes.transcript-tail.v3')
      )
    ).toEqual([])

    deferredResume.resolve({
      info: {},
      message_count: 2,
      messages: [],
      messages_omitted: true,
      resumed: 'tip-remote',
      session_id: 'runtime-remote',
      session_key: 'tip-remote'
    })
    await pending

    await waitFor(() => expect(stateMapRef.current.get('runtime-remote')?.transcriptProvenance).toBeUndefined())
  })

  it('keeps a verified cached transcript when a listed session returns a legacy empty page', async () => {
    const proof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-1'
    }
    const cached = [
      {
        id: 'cached-history',
        parts: [{ text: 'cached history remains visible', type: 'text' as const }],
        role: 'assistant' as const
      }
    ]

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-1',
        message_count: 1
      })
    ])
    saveTranscriptTail('stored-1', cached, proof)
    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'tip-1' })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 1,
          messages: [],
          messages_omitted: true,
          resumed: 'tip-1',
          session_id: 'runtime-legacy-empty',
          session_key: 'tip-1'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway)

    expect(JSON.stringify($messages.get())).toContain('cached history remains visible')
    expect($resumeFailedSessionId.get()).toBeNull()
  })

  it('reuses an exact unchanged cache array without converting or serializing history', async () => {
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }

    const proof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-1'
    }
    const cached = [
      {
        id: 'cached-unchanged',
        parts: [{ text: 'same persisted array', type: 'text' as const }],
        role: 'assistant' as const
      }
    ]
    const resumeDeferred = deferred<SessionResumeResponse>()

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-1',
        message_count: 1
      })
    ])
    saveTranscriptTail('stored-1', cached, proof)

    const convert = vi.spyOn(chatMessages, 'toChatMessages')
    const reconcile = vi.spyOn(authoritativeMessageReconciliation, 'reconcileAuthoritativeChatMessages')
    const serialize = vi.spyOn(transcriptTailCache, 'saveTranscriptTail')
    convert.mockClear()
    reconcile.mockClear()
    serialize.mockClear()
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1',
      unchanged: true
    })

    const requestGateway = <T,>(method: string): Promise<T> =>
      (method === 'session.resume' ? resumeDeferred.promise : Promise.resolve({})) as Promise<T>

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={value => (resume = value)}
        requestGateway={requestGateway}
        sessionStateByRuntimeIdRef={stateMapRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    const pending = resume!('stored-1', true)
    await waitFor(() => expect(JSON.stringify($messages.get())).toContain('same persisted array'))
    const painted = $messages.get()

    resumeDeferred.resolve({
      info: {},
      message_count: 1,
      messages: [],
      messages_omitted: true,
      resumed: 'tip-1',
      session_id: 'runtime-unchanged',
      session_key: 'tip-1'
    })
    await pending

    expect(reconcile).not.toHaveBeenCalled()
    expect(convert).not.toHaveBeenCalled()
    expect(serialize).not.toHaveBeenCalled()
    expect($messages.get()).toBe(painted)
    await waitFor(() =>
      expect(stateMapRef.current.get('runtime-unchanged')?.transcriptProvenance).toEqual({
        ...proof,
        coverage: 'latest-page',
        source: 'persisted-display',
        storedSessionId: 'stored-1'
      })
    )
  })

  it('drops an exact unchanged cache paint when resume accepts a different tip', async () => {
    const proof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-a'
    }
    const cached = [
      {
        id: 'cached-tip-a',
        parts: [{ text: 'cached transcript from tip A', type: 'text' as const }],
        role: 'assistant' as const
      }
    ]
    const resumeDeferred = deferred<SessionResumeResponse>()

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-a',
        message_count: 1
      })
    ])
    saveTranscriptTail('stored-1', cached, proof)
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [],
      resolved_tip_id: 'tip-a',
      session_id: 'tip-a',
      unchanged: true
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={value => (resume = value)}
        requestGateway={method =>
          (method === 'session.resume' ? resumeDeferred.promise : Promise.resolve({})) as Promise<never>
        }
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const pending = resume!('stored-1', true)
    await waitFor(() => expect(JSON.stringify($messages.get())).toContain('cached transcript from tip A'))
    resumeDeferred.resolve({
      info: {},
      message_count: 1,
      messages: [],
      messages_omitted: true,
      resumed: 'tip-b',
      session_id: 'runtime-tip-b',
      session_key: 'tip-b'
    })
    await pending

    expect(JSON.stringify($messages.get())).not.toContain('cached transcript from tip A')
  })

  it('uses the persisted-display reconciler when a changed REST page is accepted', async () => {
    setSessions([storedSession({ id: 'stored-1', message_count: 1 })])
    const reconcile = vi.spyOn(authoritativeMessageReconciliation, 'reconcileAuthoritativeChatMessages')
    reconcile.mockClear()
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [{ content: 'changed persisted history', role: 'assistant', timestamp: 1 }],
      session_id: 'stored-1'
    })
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 1,
          messages: [],
          messages_omitted: true,
          resumed: 'stored-1',
          session_id: 'runtime-changed',
          session_key: 'stored-1'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway)

    await waitFor(() => expect(reconcile).toHaveBeenCalledTimes(1))
    expect(JSON.stringify($messages.get())).toContain('changed persisted history')
  })

  it('accepts a revisioned authoritative empty transcript without writing a tail entry', async () => {
    const proof = {
      connectionId: '',
      displayRevision: 8,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-1'
    }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-1',
        message_count: 1
      })
    ])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 8,
      lineage_root_id: 'stored-1',
      messages: [],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1'
    })
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 1,
          messages: [],
          messages_omitted: true,
          resumed: 'tip-1',
          session_id: 'runtime-authoritative-empty',
          session_key: 'tip-1'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway)

    expect($messages.get()).toEqual([])
    expect($activeSessionId.get()).toBe('runtime-authoritative-empty')
    expect(loadTranscriptTail('stored-1', proof)).toBeNull()
    expect($resumeFailedSessionId.get()).toBeNull()
  })

  it('uses the same remote legacy scope for a resume-failure REST fallback', async () => {
    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'remote-legacy',
        id: 'tip-remote',
        message_count: 2,
        profile: 'coder'
      })
    ])
    vi.mocked(requestGatewayForAgent).mockRejectedValue(new Error('remote resume unavailable'))
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [{ content: 'remote fallback history', role: 'assistant', timestamp: 1 }],
      session_id: 'tip-remote'
    })

    await runResume(vi.fn())

    expect(JSON.stringify($messages.get())).toContain('remote fallback history')
    expect($resumeFailedSessionId.get()).toBeNull()
  })

  it('accepts a revisioned fallback response after independent hydration is unavailable', async () => {
    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'conn-1',
        display_revision: 7,
        id: 'tip-1',
        message_count: 2,
        profile: 'coder'
      })
    ])
    vi.mocked(requestGatewayForAgent).mockRejectedValue(new Error('resume unavailable'))
    vi.mocked(getLatestSessionMessages)
      .mockRejectedValueOnce(new Error('hydration unavailable'))
      .mockResolvedValueOnce({
        display_revision: 7,
        lineage_root_id: 'stored-1',
        messages: [{ content: 'revisioned fallback history', role: 'assistant', timestamp: 1 }],
        resolved_tip_id: 'tip-1',
        session_id: 'tip-1'
      })

    await runResume(vi.fn())

    expect(JSON.stringify($messages.get())).toContain('revisioned fallback history')
    expect($resumeFailedSessionId.get()).toBeNull()
  })

  it('caches an accepted revisioned hydration when resume fails without refetching it', async () => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const deferredResume = deferred<SessionResumeResponse>()
    const proof = {
      connectionId: 'conn-fallback',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'coder',
      resolvedTipId: 'tip-1'
    }

    window.localStorage.clear()
    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'conn-fallback',
        display_revision: 7,
        id: 'tip-1',
        message_count: 2,
        profile: 'coder'
      })
    ])
    vi.mocked(getLatestSessionMessages).mockClear()
    vi.mocked(requestGatewayForAgent).mockClear()
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)
    vi.mocked(requestGatewayForAgent).mockReturnValue(deferredResume.promise as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={value => (resume = value)} requestGateway={vi.fn()} />)
    await waitFor(() => expect(resume).not.toBeNull())
    const pending = resume!('stored-1', true)
    await waitFor(() => expect(getLatestSessionMessages).toHaveBeenCalledTimes(1))

    persisted.resolve({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'revisioned fallback history', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1'
    })
    await waitFor(() => expect(JSON.stringify($messages.get())).toContain('revisioned fallback history'))

    deferredResume.reject(new Error('resume unavailable'))
    await pending

    expect(JSON.stringify($messages.get())).toContain('revisioned fallback history')
    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    expect(loadTranscriptTail('stored-1', proof)?.messages[0]).toMatchObject({
      parts: [{ text: 'revisioned fallback history', type: 'text' }],
      role: 'assistant'
    })
    expect($resumeFailedSessionId.get()).toBeNull()
  })

  it('drops a deferred REST result when the same stored id moves to another connection/profile', async () => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const oldProof = {
      connectionId: 'conn-old',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'old-profile',
      resolvedTipId: 'tip-old'
    }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'conn-old',
        display_revision: 7,
        id: 'tip-old',
        message_count: 2,
        profile: 'old-profile'
      })
    ])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)

    const requestGateway = vi.fn(async (method: string, _params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return {
          info: {},
          messages: [{ content: 'runtime fallback', role: 'assistant', timestamp: 1 }],
          resumed: 'tip-old',
          session_id: 'runtime-old',
          session_key: 'tip-old'
        } as never
      }

      return {} as never
    })
    vi.mocked(requestGatewayForAgent).mockImplementation((_connectionId, _profile, method, params) =>
      requestGateway(method, params) as never
    )

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={value => (resume = value)} requestGateway={requestGateway} />)
    await waitFor(() => expect(resume).not.toBeNull())
    const pending = resume!('stored-1', true)
    await waitFor(() => expect(getLatestSessionMessages).toHaveBeenCalled())

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'conn-new',
        display_revision: 8,
        id: 'tip-new',
        message_count: 2,
        profile: 'new-profile'
      })
    ])
    await act(async () => {
      persisted.resolve({
        display_revision: 7,
        lineage_root_id: 'stored-1',
        messages: [{ content: 'stale old connection transcript', role: 'assistant', timestamp: 2 }],
        resolved_tip_id: 'tip-old',
        session_id: 'tip-old'
      })
      await new Promise(resolve => setTimeout(resolve, 50))
    })
    await pending

    expect(loadTranscriptTail('stored-1', oldProof)).toBeNull()
    expect(JSON.stringify($messages.get())).not.toContain('stale old connection transcript')
  })

  it.each([
    ['an older response revision', 7],
    ['an unproven response', undefined]
  ])('drops a deferred REST result when the current same-scope row has a newer display revision than %s', async (_case, responseRevision) => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const deferredResume = deferred<SessionResumeResponse>()
    const oldProof = {
      connectionId: 'conn-1',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'coder',
      resolvedTipId: 'tip-1'
    }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'conn-1',
        display_revision: 7,
        id: 'tip-1',
        message_count: 2,
        profile: 'coder'
      })
    ])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)
    vi.mocked(requestGatewayForAgent).mockReturnValue(deferredResume.promise)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={value => (resume = value)} requestGateway={vi.fn()} />)
    await waitFor(() => expect(resume).not.toBeNull())
    const pending = resume!('stored-1', true)
    await waitFor(() => expect(getLatestSessionMessages).toHaveBeenCalled())

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'conn-1',
        display_revision: 8,
        id: 'tip-1',
        message_count: 2,
        profile: 'coder'
      })
    ])
    persisted.resolve({
      display_revision: responseRevision,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'stale old revision transcript', role: 'assistant', timestamp: 2 }],
      resolved_tip_id: 'tip-1',
      session_id: 'tip-1'
    })
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })
    expect(JSON.stringify($messages.get())).not.toContain('stale old revision transcript')
    deferredResume.resolve({
      info: {},
      message_count: 2,
      messages: [],
      resumed: 'tip-1',
      session_id: 'runtime-1',
      session_key: 'tip-1'
    })
    await pending

    expect(loadTranscriptTail('stored-1', oldProof)).toBeNull()
    expect(JSON.stringify($messages.get())).not.toContain('stale old revision transcript')
  })

  it('does not let the resume-failure REST fallback paint after a same-id scope swap', async () => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'conn-old',
        id: 'tip-old',
        message_count: 2,
        profile: 'old-profile'
      })
    ])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)
    vi.mocked(requestGatewayForAgent).mockRejectedValue(new Error('old connection unavailable'))

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={value => (resume = value)} requestGateway={vi.fn()} />)
    await waitFor(() => expect(resume).not.toBeNull())
    const pending = resume!('stored-1', true)
    await waitFor(() => expect(getLatestSessionMessages).toHaveBeenCalled())

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        connection_id: 'conn-new',
        id: 'tip-new',
        message_count: 2,
        profile: 'new-profile'
      })
    ])
    persisted.resolve({
      messages: [{ content: 'stale fallback transcript', role: 'assistant', timestamp: 2 }],
      session_id: 'tip-old'
    })
    await pending

    expect(JSON.stringify($messages.get())).not.toContain('stale fallback transcript')
  })

  it('clears stale provenance when a cold REST hydrate comes from an old backend', async () => {
    const stale = createClientSessionState('stored-1')
    stale.transcriptProvenance = {
      connectionId: '',
      coverage: 'latest-page',
      displayRevision: 6,
      lineageRootId: 'root-1',
      profile: 'default',
      resolvedTipId: 'tip-old',
      source: 'persisted-display',
      storedSessionId: 'stored-1'
    }
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-1', stale]])
    }

    setSessions([storedSession({ id: 'stored-1', message_count: 2 })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'hello', role: 'user', timestamp: 1 },
        { content: 'legacy reply', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'stored-1'
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 2,
          messages: [],
          messages_omitted: true,
          resumed: 'stored-1',
          session_id: 'runtime-1',
          session_key: 'stored-1'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway, { sessionStateByRuntimeIdRef: stateMapRef })

    await waitFor(() => expect(stateMapRef.current.get('runtime-1')?.transcriptProvenance).toBeUndefined())
  })

  it('rejects REST proof and cache writes when a cold resume resolves a different tip', async () => {
    const stale = createClientSessionState('stored-1')
    stale.transcriptProvenance = {
      connectionId: '',
      coverage: 'latest-page',
      displayRevision: 6,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-stale',
      source: 'persisted-display',
      storedSessionId: 'stored-1'
    }
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-1', stale]])
    }
    const rejectedProof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-a'
    }

    window.localStorage.clear()
    setSessions([storedSession({ _lineage_root_id: 'stored-1', id: 'tip-b', message_count: 2 })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'REST tip A', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'tip-a',
      session_id: 'tip-a'
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 2,
          messages: [{ content: 'resume tip B', role: 'assistant', timestamp: 2 }],
          resumed: 'tip-b',
          session_id: 'runtime-1',
          session_key: 'tip-b'
        } as never
      }

      return {} as never
    })

    try {
      await runResume(requestGateway, { sessionStateByRuntimeIdRef: stateMapRef })

      expect(stateMapRef.current.get('runtime-1')?.messages).toMatchObject([
        { parts: [{ text: 'resume tip B', type: 'text' }], role: 'assistant' }
      ])
      expect(stateMapRef.current.get('runtime-1')?.transcriptProvenance).toBeUndefined()
      expect(loadTranscriptTail('stored-1', rejectedProof)).toBeNull()
    } finally {
      window.localStorage.clear()
    }
  })

  it('rejects a revisioned cold response whose own tip identity differs from the captured row', async () => {
    const mismatchedProof = {
      connectionId: '',
      displayRevision: 8,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-a'
    }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 8,
        id: 'tip-b',
        message_count: 1
      })
    ])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 8,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'response from tip A', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'tip-a',
      session_id: 'tip-a'
    })
    const requestGateway = vi.fn(async () => {
      throw new Error('resume unavailable')
    })

    await runResume(requestGateway)

    expect(JSON.stringify($messages.get())).not.toContain('response from tip A')
    expect(loadTranscriptTail('stored-1', mismatchedProof)).toBeNull()
  })

  it('rejects a legacy cold response whose own tip differs from the captured row', async () => {
    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        id: 'tip-b',
        message_count: 1
      })
    ])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      lineage_root_id: 'stored-1',
      messages: [{ content: 'legacy response from tip A', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'tip-a',
      session_id: 'tip-a'
    })

    const saveTail = vi.spyOn(transcriptTailCache, 'saveTranscriptTail')

    const requestGateway = vi.fn(async () => {
      throw new Error('resume unavailable')
    })

    await runResume(requestGateway)

    expect(JSON.stringify($messages.get())).not.toContain('legacy response from tip A')
    expect(saveTail).not.toHaveBeenCalled()
  })

  it('accepts a legacy cold response that echoes the requested lineage id', async () => {
    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        id: 'tip-b',
        message_count: 1
      })
    ])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [{ content: 'legacy response for requested lineage', role: 'assistant', timestamp: 1 }],
      session_id: 'stored-1'
    })

    const requestGateway = vi.fn(async () => {
      throw new Error('resume unavailable')
    })

    await runResume(requestGateway)

    expect(JSON.stringify($messages.get())).toContain('legacy response for requested lineage')
  })

  it('clears a published legacy tip when deferred resume accepts another tip', async () => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const deferredResume = deferred<SessionResumeResponse>()
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        id: 'tip-a',
        message_count: 1
      })
    ])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)
    const requestGateway = <T,>(method: string): Promise<T> =>
      (method === 'session.resume' ? deferredResume.promise : Promise.resolve({})) as Promise<T>

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={value => (resume = value)}
        requestGateway={requestGateway}
        sessionStateByRuntimeIdRef={stateMapRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    const pending = resume!('stored-1', true)

    persisted.resolve({
      messages: [{ content: 'legacy transcript from tip A', role: 'assistant', timestamp: 1 }],
      session_id: 'tip-a'
    })
    await waitFor(() => expect(JSON.stringify($messages.get())).toContain('legacy transcript from tip A'))
    deferredResume.resolve({
      info: {},
      message_count: 1,
      messages: [{ content: 'runtime transcript from tip B', role: 'assistant', timestamp: 2 }],
      resumed: 'tip-b',
      session_id: 'runtime-tip-b',
      session_key: 'tip-b'
    })
    await pending

    expect(JSON.stringify($messages.get())).toContain('runtime transcript from tip B')
    expect(JSON.stringify($messages.get())).not.toContain('legacy transcript from tip A')
    expect(stateMapRef.current.get('runtime-tip-b')?.transcriptProvenance).toBeUndefined()
  })

  it('does not let a deferred REST transcript replace a runtime-resolved different tip', async () => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    const mismatchedProof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-a'
    }

    setSessions([storedSession({ _lineage_root_id: 'stored-1', display_revision: 7, id: 'tip-b', message_count: 2 })])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)

    const requestGateway = vi.fn(async (method: string, _params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 2,
          messages: [{ content: 'runtime tip B', role: 'assistant', timestamp: 2 }],
          resumed: 'tip-b',
          session_id: 'runtime-b',
          session_key: 'tip-b'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway, { sessionStateByRuntimeIdRef: stateMapRef })
    expect(JSON.stringify($messages.get())).toContain('runtime tip B')

    persisted.resolve({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'deferred REST tip A', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'tip-a',
      session_id: 'tip-a'
    })
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 50))
    })

    expect(JSON.stringify($messages.get())).toContain('runtime tip B')
    expect(JSON.stringify($messages.get())).not.toContain('deferred REST tip A')
    expect(stateMapRef.current.get('runtime-b')?.transcriptProvenance).toBeUndefined()
    expect(loadTranscriptTail('stored-1', mismatchedProof)).toBeNull()
  })

  it('rebinds a runtime-resolved tip, fetches its transcript once, and rejects the stale original tip', async () => {
    const staleTipA = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }
    const runtimeMapRef: MutableRefObject<Map<string, string>> = { current: new Map() }
    const tipAProof = {
      connectionId: '',
      displayRevision: 7,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'tip-a'
    }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-a',
        message_count: 2
      })
    ])
    saveTranscriptTail(
      'stored-1',
      [{ id: 'cached-tip-a', parts: [{ text: 'cached tip A transcript', type: 'text' }], role: 'assistant' }],
      tipAProof
    )
    vi.mocked(getLatestSessionMessages).mockClear()
    vi.mocked(getLatestSessionMessages).mockImplementation(async id => {
      if (id === 'stored-1') {
        return staleTipA.promise
      }

      expect(id).toBe('tip-b')

      return {
        display_revision: 8,
        lineage_root_id: 'stored-1',
        messages: [{ content: 'authoritative tip B transcript', role: 'assistant', timestamp: 2 }],
        pagination: { limit: 120, offset: 0, order: 'latest', returned: 1 },
        resolved_tip_id: 'tip-b',
        session_id: 'tip-b'
      }
    })
    const requestGateway = vi.fn(async method => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 2,
          messages: [],
          messages_omitted: true,
          resumed: 'tip-b',
          session_id: 'runtime-b',
          session_key: 'tip-b'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway, {
      runtimeIdByStoredSessionIdRef: runtimeMapRef,
      sessionStateByRuntimeIdRef: stateMapRef
    })

    expect(vi.mocked(getLatestSessionMessages).mock.calls.filter(([id]) => id === 'tip-b')).toHaveLength(1)
    await waitFor(() => expect(JSON.stringify($messages.get())).toContain('authoritative tip B transcript'))
    expect($activeSessionId.get()).toBe('runtime-b')
    expect(stateMapRef.current.has('runtime-b')).toBe(true)

    staleTipA.resolve({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'stale tip A transcript', role: 'assistant', timestamp: 1 }],
      resolved_tip_id: 'tip-a',
      session_id: 'tip-a'
    })
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 25))
    })

    expect(JSON.stringify($messages.get())).toContain('authoritative tip B transcript')
    expect(JSON.stringify($messages.get())).not.toContain('stale tip A transcript')
    expect(stateMapRef.current.get('runtime-b')?.transcriptProvenance?.resolvedTipId).toBe('tip-b')
    expect(loadTranscriptTail('stored-1', tipAProof)).toBeNull()
  })

  it('activates a live runtime before the rebound tip transcript finishes loading', async () => {
    const reboundTipB = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-a',
        message_count: 2
      })
    ])
    vi.mocked(getLatestSessionMessages).mockClear()
    vi.mocked(getLatestSessionMessages).mockImplementation(id =>
      id === 'tip-b' ? reboundTipB.promise : new Promise(() => undefined)
    )

    let resumeSettled = false
    const resume = runResume(
      async method => {
        if (method === 'session.resume') {
          return {
            info: {},
            message_count: 2,
            messages: [],
            messages_omitted: true,
            resumed: 'tip-b',
            running: true,
            session_id: 'runtime-b',
            session_key: 'tip-b'
          } as never
        }

        return {} as never
      },
      { sessionStateByRuntimeIdRef: stateMapRef }
    ).then(() => {
      resumeSettled = true
    })

    await waitFor(() =>
      expect(vi.mocked(getLatestSessionMessages).mock.calls.filter(([id]) => id === 'tip-b')).toHaveLength(1)
    )
    await act(async () => Promise.resolve())

    expect(resumeSettled).toBe(true)
    expect($activeSessionId.get()).toBe('runtime-b')
    expect(stateMapRef.current.get('runtime-b')).toMatchObject({
      awaitingResponse: true,
      busy: true,
      turnLive: true
    })
    expect($resumeFailedSessionId.get()).not.toBe('stored-1')

    reboundTipB.resolve({
      display_revision: 8,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'late tip B history', role: 'assistant', timestamp: 2 }],
      resolved_tip_id: 'tip-b',
      session_id: 'tip-b'
    })
    await resume
    await waitFor(() => expect(JSON.stringify($messages.get())).toContain('late tip B history'))
    expect(stateMapRef.current.get('runtime-b')?.busy).toBe(true)
  })

  it.each([
    ['full A / short B', { limit: 120, offset: 0, order: 'latest' as const, returned: 1 }, false],
    ['short A / full B', { limit: 1, offset: 0, order: 'latest' as const, returned: 1 }, true]
  ])('keeps pagination bookkeeping bound to accepted tip B for %s', async (_case, tipBPagination, expectedTruncated) => {
    const staleTipA = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()

    clearTranscriptTail('stored-1')
    clearTranscriptTail('tip-a')
    clearTranscriptTail('tip-b')
    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-a',
        message_count: 2
      })
    ])
    vi.mocked(getLatestSessionMessages).mockClear()
    vi.mocked(getLatestSessionMessages).mockImplementation(async id => {
      if (id === 'stored-1') {
        return staleTipA.promise
      }

      return {
        display_revision: 8,
        lineage_root_id: 'stored-1',
        messages: [{ content: 'accepted B', role: 'assistant', timestamp: 2 }],
        pagination: tipBPagination,
        resolved_tip_id: 'tip-b',
        session_id: 'tip-b'
      }
    })

    await runResume(async method =>
      method === 'session.resume'
        ? ({
            info: {},
            message_count: 2,
            messages: [],
            messages_omitted: true,
            resumed: 'tip-b',
            session_id: 'runtime-b',
            session_key: 'tip-b'
          } as never)
        : ({} as never)
    )
    await waitFor(() =>
      expect(transcriptTailState('stored-1')?.possiblyTruncated).toBe(expectedTruncated)
    )

    staleTipA.resolve({
      display_revision: 7,
      lineage_root_id: 'stored-1',
      messages: [{ content: 'late A', role: 'assistant', timestamp: 1 }],
      pagination: expectedTruncated
        ? { limit: 120, offset: 0, order: 'latest', returned: 1 }
        : { limit: 1, offset: 0, order: 'latest', returned: 1 },
      resolved_tip_id: 'tip-a',
      session_id: 'tip-a'
    })
    await act(async () => new Promise(resolve => setTimeout(resolve, 25)))

    expect(transcriptTailState('stored-1')?.possiblyTruncated).toBe(expectedTruncated)
    expect(transcriptTailState('tip-b')?.possiblyTruncated).toBe(expectedTruncated)
    expect(transcriptTailState('tip-a')).toBeUndefined()
  })

  it('keeps an accepted runtime usable when its rebound tip transcript read fails', async () => {
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 7,
        id: 'tip-a',
        message_count: 2
      })
    ])
    vi.mocked(getLatestSessionMessages).mockClear()
    vi.mocked(getLatestSessionMessages).mockImplementation(async id => {
      if (id === 'tip-b') {
        throw new Error('tip B read unavailable')
      }

      return new Promise(() => undefined)
    })

    await runResume(
      async method => {
        if (method === 'session.resume') {
          return {
            info: {},
            message_count: 2,
            messages: [],
            messages_omitted: true,
            resumed: 'tip-b',
            session_id: 'runtime-b',
            session_key: 'tip-b'
          } as never
        }

        return {} as never
      },
      { sessionStateByRuntimeIdRef: stateMapRef }
    )

    expect(vi.mocked(getLatestSessionMessages).mock.calls.filter(([id]) => id === 'tip-b')).toHaveLength(1)
    expect($activeSessionId.get()).toBe('runtime-b')
    expect($resumeFailedSessionId.get()).not.toBe('stored-1')
    expect(stateMapRef.current.has('runtime-b')).toBe(true)
  })

  it('does not resume a tombstoned session after delete', async () => {
    $removedSessionIds.set(new Set(['stored-1']))

    const requestGateway = vi.fn(async () => {
      throw new Error('404: Session not found')
    })

    await runResume(requestGateway)

    expect(requestGateway).not.toHaveBeenCalled()
    expect($resumeFailedSessionId.get()).toBeNull()
    expect($selectedStoredSessionId.get()).toBeNull()
  })

  it.each([
    ['Codex tool-only', ''],
    ['DeepSeek text-plus-tool', 'I found two paths; choose one.']
  ])('restores a pending clarify as a running inline card for a %s transcript', async (_shape, content) => {
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }

    const toolCall = {
      function: { arguments: '{"question":"Which path?","choices":["safe","fast"]}', name: 'clarify' },
      id: 'call-provider'
    }

    setSessions([storedSession({ id: 'stored-1', message_count: 2 })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'help me choose', role: 'user', timestamp: 1 },
        { content, role: 'assistant', timestamp: 2, tool_calls: [toolCall] }
      ],
      session_id: 'stored-1'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 2,
          messages: [],
          messages_omitted: true,
          pending_clarify: {
            choices: ['safe', 'fast'],
            question: 'Which path?',
            request_id: 'req-resumed'
          },
          resumed: 'stored-1',
          running: true,
          session_id: 'runtime-1',
          session_key: 'stored-1'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway, { sessionStateByRuntimeIdRef: stateMapRef })

    await waitFor(() =>
      expect(
        stateMapRef.current
          .get('runtime-1')
          ?.messages.filter(message => message.parts.some(part => part.type === 'tool-call' && part.toolName === 'clarify'))
      ).toHaveLength(1)
    )
    const state = stateMapRef.current.get('runtime-1')
    const clarifyMessages = state?.messages.filter(message =>
      message.parts.some(part => part.type === 'tool-call' && part.toolName === 'clarify')
    ) ?? []

    expect(clarifyMessages[0].pending).toBe(true)
    expect(state?.streamId).toBe(clarifyMessages[0].id)
    expect($clarifyRequests.get()['runtime-1']).toMatchObject({ requestId: 'req-resumed', question: 'Which path?' })
  })

  it('restores a pending batch clarify whose resume snapshot has no top-level question', async () => {
    const stateMapRef: MutableRefObject<Map<string, ClientSessionState>> = { current: new Map() }

    setSessions([storedSession({ id: 'stored-1', message_count: 2 })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'help me choose', role: 'user', timestamp: 1 },
        {
          content: '',
          role: 'assistant',
          timestamp: 2,
          tool_calls: [
            {
              function: {
                arguments: '{"questions":[{"question":"Color?"},{"question":"Size?"}]}',
                name: 'clarify'
              },
              id: 'call-batch-provider'
            }
          ]
        }
      ],
      session_id: 'stored-1'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 2,
          messages: [],
          messages_omitted: true,
          pending_clarify: {
            answers: { q0: 'Blue' },
            questions: [
              { choices: ['Blue', 'Red'], qid: 'q0', question: 'Color?' },
              { choices: ['Small', 'Large'], qid: 'q1', question: 'Size?' }
            ],
            request_id: 'req-batch-resumed'
          },
          resumed: 'stored-1',
          running: true,
          session_id: 'runtime-1',
          session_key: 'stored-1'
        } as never
      }

      return {} as never
    })

    await runResume(requestGateway, { sessionStateByRuntimeIdRef: stateMapRef })

    await waitFor(() => expect($clarifyRequests.get()['runtime-1']).toBeDefined())
    const state = stateMapRef.current.get('runtime-1')
    const request = $clarifyRequests.get()['runtime-1']
    expect(request).toMatchObject({
      lockedAnswers: { q0: 'Blue' },
      question: '',
      requestId: 'req-batch-resumed'
    })
    expect(request.questions).toHaveLength(2)
    expect(
      state?.messages
        .flatMap(message => message.parts)
        .filter(part => part.type === 'tool-call' && part.toolName === 'clarify')
    ).toHaveLength(1)
    expect(state?.messages.filter(message => message.pending)).toHaveLength(1)
    expect(state?.streamId).toBe(state?.messages.find(message => message.pending)?.id)
  })

  it('arms $resumeFailedSessionId when resume RPC and REST fallback both fail', async () => {
    // session.resume rejects (e.g. timeout against a wedged backend)...
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('request timed out: session.resume')
      }

      return {} as never
    })

    // ...and the REST transcript fallback also rejects (backend unreachable).
    vi.mocked(getLatestSessionMessages).mockRejectedValue(new Error('network down'))

    await runResume(requestGateway)

    // The window is no longer silently stranded: the failure latch is armed for
    // the stored session, which use-route-resume consumes to retry.
    expect($resumeFailedSessionId.get()).toBe('stored-1')
  })

  it('keeps verified cached history visible when both REST and resume fail', async () => {
    const proof = {
      connectionId: '',
      displayRevision: 4,
      lineageRootId: 'stored-1',
      profile: 'default',
      resolvedTipId: 'stored-1'
    }
    const cached = [
      {
        id: 'cached-question',
        parts: [{ text: 'cached verified question', type: 'text' as const }],
        role: 'user' as const
      }
    ]

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-1',
        display_revision: 4,
        id: 'stored-1',
        message_count: 1
      })
    ])
    saveTranscriptTail('stored-1', cached, proof)
    vi.mocked(getLatestSessionMessages).mockRejectedValue(new Error('REST unavailable'))
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('request timed out: session.resume')
      }

      return {} as never
    })

    await runResume(requestGateway)

    expect(JSON.stringify($messages.get())).toContain('cached verified question')
    expect($resumeFailedSessionId.get()).toBe('stored-1')
  })

  it('does NOT arm the failure latch when the resume RPC fails but the REST fallback paints history', async () => {
    // session.resume rejects, but the REST transcript fallback succeeds and
    // hydrates a readable transcript — the window is NOT stranded.
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('request timed out: session.resume')
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'hello', role: 'user', timestamp: 1 },
        { content: 'hi there', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'stored-1'
    } as never)
    setSessions([storedSession({ message_count: 2 })])

    await runResume(requestGateway)

    // Arming here would auto-retry a window that already shows history and,
    // on exhaustion, blank that transcript behind the error overlay — a
    // regression vs. plain fallback-success. The latch must stay clear.
    expect($resumeFailedSessionId.get()).toBeNull()
    // The fallback transcript is visible.
    expect($messages.get().length).toBeGreaterThan(0)
  })

  it('paints the REST transcript before a cold session.resume settles and keeps it when resume rejects', async () => {
    // A cold profile build (skills/MCP/memory) can keep session.resume pending
    // past the hydration budget; the already-available REST history must not
    // wait for it (#90130), and a later resume failure must not blank it.
    const runtimeResume = deferred<never>()

    const requestGateway = vi.fn((method: string) =>
      method === 'session.resume' ? runtimeResume.promise : Promise.resolve({} as never)
    ) as <T>(method: string, params?: Record<string, unknown>) => Promise<T>

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'older question', role: 'user', timestamp: 1 },
        { content: 'history visible before runtime', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'stored-1'
    } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={r => (resume = r)} requestGateway={requestGateway} />)
    await waitFor(() => expect(resume).not.toBeNull())

    let settled = false
    const pending = resume!('stored-1', true).finally(() => (settled = true))

    await waitFor(() => expect(JSON.stringify($messages.get())).toContain('history visible before runtime'))
    expect(settled).toBe(false)
    const painted = $messages.get()

    await act(async () => {
      runtimeResume.reject(new Error('request timed out: session.resume'))
      await pending
    })

    expect($messages.get()).toBe(painted)
    expect($resumeFailedSessionId.get()).toBeNull()
  })

  it('preserves an optimistic user message during a same-session reconnect', async () => {
    setMessages([
      {
        id: 'stored-user',
        role: 'user',
        parts: [{ type: 'text', text: 'earlier question' }]
      },
      {
        id: 'stored-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'earlier answer' }]
      },
      {
        id: 'user-optimistic',
        role: 'user',
        parts: [{ type: 'text', text: 'message sent during reconnect' }]
      }
    ])

    const storedMessages = [
      { content: 'earlier question', role: 'user', timestamp: 1 },
      { content: 'earlier answer', role: 'assistant', timestamp: 2 }
    ]

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: storedMessages, session_id: 'stored-1' } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          session_id: 'runtime-1',
          session_key: 'stored-1',
          resumed: 'stored-1',
          message_count: 2,
          messages: storedMessages,
          info: {}
        } as never
      }

      return {} as never
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness onReady={r => (resume = r)} requestGateway={requestGateway} selectedStoredSessionId="stored-1" />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-1', true)

    expect($messages.get().map(message => message.id)).toContain('user-optimistic')
  })

  it('keeps the complete transcript with the live tail after a full renderer restart', async () => {
    const storedMessages = [
      { content: 'older question removed by compression', role: 'user', timestamp: 1 },
      { content: 'older answer removed by compression', role: 'assistant', timestamp: 2 },
      { content: 'recent question', role: 'user', timestamp: 3 },
      { content: 'recent answer', role: 'assistant', timestamp: 4 }
    ]

    const compressedRuntimeMessages = storedMessages.slice(-2)

    setSessions([storedSession({ message_count: storedMessages.length })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: storedMessages, session_id: 'stored-1' } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          session_id: 'runtime-1',
          session_key: 'stored-1',
          resumed: 'stored-1',
          message_count: compressedRuntimeMessages.length,
          messages: compressedRuntimeMessages,
          running: true,
          turn_started_at: 1_700_000_000,
          inflight: {
            user: 'current prompt',
            assistant: 'partial answer',
            streaming: true
          },
          queued: { user: 'newest prompt' },
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, state) => (resumedState = state)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-1', true)

    await waitFor(() => expect(JSON.stringify(resumedState?.messages)).toContain('older question removed by compression'))
    const renderedMessages = JSON.stringify(resumedState?.messages)
    expect(renderedMessages).toContain('older question removed by compression')
    expect(renderedMessages).toContain('current prompt')
    expect(renderedMessages).toContain('partial answer')
    expect(renderedMessages).toContain('newest prompt')
    expect(resumedState?.turnStartedAt).toBe(1_700_000_000_000)
  })

  it('preserves a runtime-cache delta that arrives while cold resume waits for REST', async () => {
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map()
    }

    const compressedRuntimeMessages = [
      { content: 'recent question', role: 'user', timestamp: 3 },
      { content: 'recent answer', role: 'assistant', timestamp: 4 }
    ]

    setSessions([storedSession({ message_count: 4 })])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          session_id: 'runtime-1',
          session_key: 'stored-1',
          resumed: 'stored-1',
          message_count: compressedRuntimeMessages.length,
          messages: compressedRuntimeMessages,
          running: true,
          inflight: {
            user: 'current prompt',
            assistant: 'partial A',
            streaming: true
          },
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, state) => (resumedState = state)}
        requestGateway={requestGateway}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const resumePromise = resume!('stored-1', true)

    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('session.resume', expect.anything()))

    const runtimeState = clientState('stored-1')
    runtimeState.messages = [
      {
        id: 'assistant-stream-live-cold',
        role: 'assistant',
        parts: [{ type: 'text', text: ' + delta B' }],
        pending: true
      }
    ]
    runtimeState.streamId = 'assistant-stream-live-cold'
    sessionStateByRuntimeIdRef.current.set('runtime-1', runtimeState)

    await act(async () => {
      persisted.resolve({
        messages: [
          { content: 'older question removed by compression', role: 'user', timestamp: 1 },
          { content: 'older answer removed by compression', role: 'assistant', timestamp: 2 },
          ...compressedRuntimeMessages
        ],
        session_id: 'stored-1'
      } as never)
      await resumePromise
    })

    await waitFor(() => expect(JSON.stringify(resumedState?.messages)).toContain('older question removed by compression'))
    const renderedText = JSON.stringify(resumedState?.messages)

    const streamingAssistantRows = resumedState?.messages.filter(message => message.id.startsWith('assistant-stream-'))

    expect(renderedText).toContain('older question removed by compression')
    expect(renderedText).toContain('partial A')
    expect(renderedText).toContain('delta B')
    expect(streamingAssistantRows).toHaveLength(1)
    expect(streamingAssistantRows?.[0].id).toBe('assistant-stream-live-cold')
  })

  it('uses the continuation projection when resume rotates an equal-length stored transcript', async () => {
    const parentMessages = [
      { content: 'question before compression', role: 'user', timestamp: 1 },
      { content: 'answer before compression', role: 'assistant', timestamp: 2 }
    ]

    const continuationMessages = [
      { content: 'prompt after compression', role: 'user', timestamp: 3 },
      { content: 'answer after compression', role: 'assistant', timestamp: 4 }
    ]

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: parentMessages,
      session_id: 'stored-1'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          session_id: 'runtime-continuation',
          session_key: 'stored-continuation',
          resumed: 'stored-continuation',
          message_count: continuationMessages.length,
          messages: continuationMessages,
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, state) => (resumedState = state)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-1', true)

    const renderedMessages = JSON.stringify(resumedState?.messages)
    expect(renderedMessages).toContain('prompt after compression')
    expect(renderedMessages).toContain('answer after compression')
    expect(renderedMessages).not.toContain('answer before compression')
  })

  it('does NOT throw out of the fallback when REST also fails (no unhandled rejection)', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('request timed out: session.resume')
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockRejectedValue(new Error('network down'))

    // resumeSession must resolve (swallow the fallback failure), not reject.
    await expect(runResume(requestGateway)).resolves.toBeUndefined()
  })

  it('leaves the failure latch clear when resume succeeds', async () => {
    // Pre-arm to prove a successful resume clears it (entry-clear path).
    setResumeFailedSessionId('stored-1')

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [] } as never)

    await runResume(requestGateway)

    expect($resumeFailedSessionId.get()).toBeNull()
  })

  it('resumes via the gateway default (deferred build) — not lazy, no eager opt-out', async () => {
    // The switch-latency fix lives backend-side: a normal cold resume gets the
    // gateway's default DEFERRED build (transcript returns immediately, agent
    // pre-warms in the background). The client must NOT force the synchronous
    // path (eager_build) and is only `lazy` for subagent watch windows.
    let resumeParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        resumeParams = params

        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [] } as never)

    await runResume(requestGateway)

    expect(resumeParams).not.toHaveProperty('lazy')
    expect(resumeParams).not.toHaveProperty('eager_build')
    expect(resumeParams).toMatchObject({ source: 'desktop', omit_messages: true })
  })

  it('arms the failure latch when resume succeeds with an empty transcript for a non-empty stored session', async () => {
    setSessions([storedSession({ message_count: 4 })])

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-1' } as never)

    await runResume(requestGateway)

    expect($resumeFailedSessionId.get()).toBe('stored-1')
    expect($activeSessionId.get()).toBeNull()
    expect($messages.get()).toEqual([])
  })

  it('does not reuse an empty cached runtime view for a stored session with history', async () => {
    const runtimeIdByStoredSessionIdRef = {
      current: new Map([['stored-1', 'runtime-stale']])
    } satisfies MutableRefObject<Map<string, string>>

    const sessionStateByRuntimeIdRef = {
      current: new Map([
        [
          'runtime-stale',
          {
            awaitingResponse: false,
            branch: '',
            busy: false,
            cwd: '',
            fast: false,
            interimBoundaryPending: false,
            interrupted: false,
            messages: [],
            adoptedRunningTurn: false,
            model: '',
            needsInput: false,
            pendingBranchGroup: null,
            personality: '',
            provider: '',
            reasoningEffort: '',
            sawAssistantPayload: false,
            serviceTier: '',
            storedSessionId: 'stored-1',
            streamId: null,
            turnStartedAt: null,
            turnLive: false,
            usage: null,
            yolo: false
          }
        ]
      ])
    } satisfies MutableRefObject<Map<string, ClientSessionState>>

    setSessions([storedSession({ message_count: 4 })])

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [{ content: 'existing text', role: 'user', timestamp: 1 }],
      session_id: 'stored-1'
    } as never)

    await runResume(requestGateway, {
      runtimeIdByStoredSessionIdRef,
      sessionStateByRuntimeIdRef
    })

    expect(requestGateway).not.toHaveBeenCalledWith('session.usage', { session_id: 'runtime-stale' })
    expect(runtimeIdByStoredSessionIdRef.current.has('stored-1')).toBe(false)
    expect(sessionStateByRuntimeIdRef.current.has('runtime-stale')).toBe(false)
    await waitFor(() => expect($activeSessionId.get()).toBe('runtime-1'))
    await waitFor(() => expect($messages.get().length).toBe(1))
  })
})

describe('session.resume turn timer contract', () => {
  beforeEach(() => {
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback: FrameRequestCallback) => {
      callback(0)

      return null as unknown as number
    })
    setActiveSessionId(null)
    setAwaitingResponse(false)
    setBusy(false)
    setMessages([])
    setSessions([])
    setTurnStartedAt(null)
  })

  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setAwaitingResponse(false)
    setBusy(false)
    setMessages([])
    setSessions([])
    setTurnStartedAt(null)
    vi.restoreAllMocks()
  })

  async function resumeFrom(response: unknown): Promise<void> {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        // Model the JSON-RPC serialization/deserialization boundary. The shared
        // fixture is asserted against the real gateway response in Python.
        return JSON.parse(JSON.stringify(response)) as never
      }

      return {} as never
    })

    vi.mocked(getAllSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-running' } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeTimerHarness onReady={ready => (resume = ready)} requestGateway={requestGateway} />)
    await waitFor(() => expect(resume).not.toBeNull())
    await act(async () => {
      await resume!('stored-running', true)
    })
  }

  it('restores the canonical gateway turn timestamp in milliseconds', async () => {
    await resumeFrom(sessionResumeActiveTurn)

    expect($turnStartedAt.get()).toBe(sessionResumeActiveTurn.turn_started_at * 1000)
  })

  it('clears a stale timer when the gateway response is not running', async () => {
    setTurnStartedAt(1_600_000_000_000)

    await resumeFrom({ ...sessionResumeActiveTurn, running: false })

    expect($turnStartedAt.get()).toBeNull()
  })

  it('clears a stale timer when the running gateway response omits its timestamp', async () => {
    const missingTimestamp: Record<string, unknown> = JSON.parse(JSON.stringify(sessionResumeActiveTurn))
    delete missingTimestamp.turn_started_at
    setTurnStartedAt(1_600_000_000_000)

    await resumeFrom(missingTimestamp)

    expect($turnStartedAt.get()).toBeNull()
  })

  it('clears a stale timer when the running gateway response has a non-numeric timestamp', async () => {
    setTurnStartedAt(1_600_000_000_000)

    await resumeFrom({ ...sessionResumeActiveTurn, turn_started_at: 'not-a-timestamp' })

    expect($turnStartedAt.get()).toBeNull()
  })
})

function BranchHarness({
  activeSessionId = null,
  navigate = vi.fn(),
  onCurrentReady,
  onReady,
  onRefs,
  requestGateway,
  selectedStoredSessionId = null
}: {
  activeSessionId?: string | null
  navigate?: ReturnType<typeof vi.fn>
  onCurrentReady?: (branchCurrentSession: (messageId?: string) => Promise<boolean>) => void
  onReady: (branchStoredSession: (storedSessionId: string, sessionProfile?: string | null) => Promise<boolean>) => void
  onRefs?: (refs: {
    activeSessionIdRef: MutableRefObject<string | null>
    selectedStoredSessionIdRef: MutableRefObject<string | null>
  }) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  selectedStoredSessionId?: string | null
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })
  const activeSessionIdRef = ref<string | null>(activeSessionId)
  const selectedStoredSessionIdRef = ref<string | null>(selectedStoredSessionId)

  onRefs?.({ activeSessionIdRef, selectedStoredSessionIdRef })

  const actions = useSessionActions({
    activeSessionId,
    activeSessionIdRef,
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: navigate as never,
    requestGateway,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions.branchStoredSession)
    onCurrentReady?.(actions.branchCurrentSession)
  }, [actions.branchCurrentSession, actions.branchStoredSession, onCurrentReady, onReady])

  return null
}

describe('branchStoredSession desktop source tagging', () => {
  afterEach(() => {
    cleanup()
    setSessions([])
    $sessionTiles.set([])
    setSelectedStoredSessionId(null)
    vi.restoreAllMocks()
  })

  it('opens the branch as the primary session in the main workspace (#93444)', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.create') {
        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 0,
          messages: [],
          resumed: 'branch-stored',
          session_id: 'branch-runtime',
          session_key: 'branch-stored'
        } as never
      }

      return {} as never
    })

    // Parent is the currently-open (primary) chat.
    setSessions([storedSession({ id: 'stored-parent', message_count: 1 })])
    setSelectedStoredSessionId('stored-parent')
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    const navigate = vi.fn()
    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(
      <BranchHarness
        navigate={navigate}
        onReady={branch => (branchStoredSession = branch)}
        requestGateway={requestGateway}
        selectedStoredSessionId="stored-parent"
      />
    )
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    // The branch becomes the primary session — this is what routes the main
    // workspace area to it, not just a new sidebar row. Selection alone is not
    // enough: leaving the URL on the parent makes chat/index see a permanent
    // routeSessionMismatch and keeps the central loader mounted.
    expect($selectedStoredSessionId.get()).toBe('branch-stored')
    expect(navigate).toHaveBeenCalledWith(sessionRoute('branch-stored'), { replace: true })
    // It must not ALSO exist as a tile: a session is either the main thread or
    // a tile, never both (resumeSession closes any tile with the same id).
    expect($sessionTiles.get().some(tile => tile.storedSessionId === 'branch-stored')).toBe(false)
  })

  it('keeps the current view when branching a different session from the sidebar (does not reintroduce #69750)', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.create') {
        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      if (method === 'session.resume') {
        return {
          info: {},
          message_count: 0,
          messages: [],
          resumed: 'branch-stored',
          session_id: 'branch-runtime',
          session_key: 'branch-stored'
        } as never
      }

      return {} as never
    })

    // The user is looking at "stored-other" and right-clicks a DIFFERENT,
    // unrelated session ("stored-parent") in the sidebar to branch it.
    setSessions([storedSession({ id: 'stored-parent', message_count: 1 })])
    setSelectedStoredSessionId('stored-other')
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    const navigate = vi.fn()
    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(
      <BranchHarness
        navigate={navigate}
        onReady={branch => (branchStoredSession = branch)}
        requestGateway={requestGateway}
        selectedStoredSessionId="stored-other"
      />
    )
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    // Branching a session that is not the one currently open must not steal
    // the user's active view — "stored-other" stays selected.
    expect($selectedStoredSessionId.get()).toBe('stored-other')
    expect(navigate).not.toHaveBeenCalled()
    // The branch instead opens as its own tile.
    expect($sessionTiles.get().some(tile => tile.storedSessionId === 'branch-stored')).toBe(true)
  })

  it('tags desktop branch sessions as desktop sessions', async () => {
    let createParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        createParams = params

        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      return {} as never
    })

    setSessions([storedSession({ id: 'stored-parent', message_count: 1 })])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(<BranchHarness onReady={branch => (branchStoredSession = branch)} requestGateway={requestGateway} />)
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    expect(createParams).toMatchObject({
      parent_session_id: 'stored-parent',
      source: 'desktop'
    })
  })

  // A branch belongs to the backend that OWNS its parent. Routing on profile
  // alone silently sends session.create to whatever socket is active, so a
  // remote-owned parent branched while another connection is active creates
  // the child on the wrong backend — or nowhere — while the sidebar still
  // paints an optimistic row that can never hydrate ("Couldn't load this
  // session"). Same ownership contract removeSession already honours.
  it('routes a connection-tagged parent branch through its owning connection', async () => {
    const ambientRequest = vi.fn(async () => ({}) as never)

    vi.mocked(requestGatewayForAgent).mockImplementation((async (
      _connectionId: null | string,
      _profile: string,
      method: string
    ) => {
      if (method === 'session.create') {
        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      return {} as never
    }) as never)

    setSessions([
      storedSession({ connection_id: 'pandora', id: 'stored-parent', message_count: 1, profile: 'default' })
    ])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(<BranchHarness onReady={branch => (branchStoredSession = branch)} requestGateway={ambientRequest} />)
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    // The create must ride the parent's own (connection, profile) socket...
    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'pandora',
      'default',
      'session.create',
      expect.objectContaining({ parent_session_id: 'stored-parent', source: 'desktop' })
    )
    // ...and never the ambient socket, which may serve a different machine.
    expect(ambientRequest).not.toHaveBeenCalledWith('session.create', expect.anything())
  })

  // The parent transcript read must land on the owning backend too: reading it
  // from the ambient socket returns nothing for a foreign-owned parent, which
  // aborts the branch as "nothing to branch" before any create is attempted.
  it('reads a connection-tagged parent transcript from its owning connection', async () => {
    const ambientRequest = vi.fn(async () => ({}) as never)

    vi.mocked(requestGatewayForAgent).mockImplementation((async (
      _connectionId: null | string,
      _profile: string,
      method: string
    ) => {
      if (method === 'session.create') {
        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      return {} as never
    }) as never)

    setSessions([
      storedSession({ connection_id: 'pandora', id: 'stored-parent', message_count: 1, profile: 'default' })
    ])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(<BranchHarness onReady={branch => (branchStoredSession = branch)} requestGateway={ambientRequest} />)
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    expect(getAllSessionMessages).toHaveBeenCalledWith('stored-parent', {
      connectionId: 'pandora',
      profile: 'default'
    })
  })

  // The create landing on the right backend is only half the job: the sidebar
  // row must be TAGGED with that owner too. An untagged optimistic row inherits
  // the ambient profile, so every later owner lookup (resume/hydrate/prompt)
  // routes to the wrong backend and the chat pane spins forever on a session
  // that backend never had.
  it('tags the optimistic branch row with the parent connection owner', async () => {
    const ambientRequest = vi.fn(async () => ({}) as never)

    vi.mocked(requestGatewayForAgent).mockImplementation((async (
      _connectionId: null | string,
      _profile: string,
      method: string
    ) => {
      if (method === 'session.create') {
        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      return {} as never
    }) as never)

    setSessions([
      storedSession({ connection_id: 'pandora', id: 'stored-parent', message_count: 1, profile: 'default' })
    ])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(<BranchHarness onReady={branch => (branchStoredSession = branch)} requestGateway={ambientRequest} />)
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    const row = $sessions.get().find(session => session.id === 'branch-stored')
    expect(row).toBeDefined()
    expect(row!.connection_id).toBe('pandora')
    expect(row!.profile).toBe('default')
  })

  // An untagged row (single-backend users, the overwhelmingly common case)
  // must keep the ambient path exactly as before — no behaviour change.
  it('keeps an untagged parent branch on the ambient socket', async () => {
    let createParams: Record<string, unknown> | undefined

    const ambientRequest = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        createParams = params

        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      return {} as never
    })

    vi.mocked(requestGatewayForAgent).mockClear()
    setSessions([storedSession({ id: 'stored-parent', message_count: 1 })])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(<BranchHarness onReady={branch => (branchStoredSession = branch)} requestGateway={ambientRequest} />)
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    expect(createParams).toMatchObject({ parent_session_id: 'stored-parent', source: 'desktop' })
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
  })

  it('branches an open live chat via session.branch with a trimmed message count (bug #1/#3 fix)', async () => {
    let branchParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.branch') {
        branchParams = params

        return {
          session_id: 'branch-runtime',
          stored_session_id: 'branch-stored',
          title: 'Branch',
          message_count: 2,
          messages: [],
          info: {}
        } as never
      }

      return {} as never
    })

    setMessages([
      { id: 'q1', role: 'user', parts: [{ type: 'text', text: 'question one' }] },
      { id: 'a1', role: 'assistant', parts: [{ type: 'text', text: 'answer one' }] },
      { id: 'q2', role: 'user', parts: [{ type: 'text', text: 'question two' }] },
      { id: 'a2', role: 'assistant', parts: [{ type: 'text', text: 'answer two' }] }
    ])

    let branchCurrentSession: ((messageId?: string) => Promise<boolean>) | null = null
    render(
      <BranchHarness
        activeSessionId="live-parent"
        onCurrentReady={branch => (branchCurrentSession = branch)}
        onReady={() => undefined}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(branchCurrentSession).not.toBeNull())

    // Branch from the FIRST assistant reply ("a1"), not the last message �
    // this is exactly the scenario that used to drop the question (bug #1):
    // only the clicked message survived instead of everything up to it.
    await expect(branchCurrentSession!('a1')).resolves.toBe(true)

    expect(requestGateway).toHaveBeenCalledWith('session.branch', {
      session_id: 'live-parent',
      count: 2
    })
    expect(branchParams).toEqual({ session_id: 'live-parent', count: 2 })
  })

  it('hydrates the complete persisted display transcript before branching a compacted live chat', async () => {
    let branchParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.branch') {
        branchParams = params

        return {
          session_id: 'branch-runtime',
          stored_session_id: 'branch-stored',
          title: 'Branch',
          message_count: 4,
          messages: [],
          info: {}
        } as never
      }

      return {} as never
    })

    setSessions([storedSession({ id: 'stored-parent', message_count: 4 })])
    setMessages([
      { id: 'summary', role: 'assistant', parts: [{ type: 'text', text: 'compact summary' }] },
      { id: 'tail-user', role: 'user', parts: [{ type: 'text', text: 'second question' }] },
      { id: 'tail-assistant', role: 'assistant', parts: [{ type: 'text', text: 'second answer' }] }
    ])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [
        { content: 'first question', role: 'user', timestamp: 1 },
        { content: 'first answer', role: 'assistant', timestamp: 2 },
        { content: 'second question', role: 'user', timestamp: 3 },
        { content: 'second answer', role: 'assistant', timestamp: 4 }
      ],
      session_id: 'stored-parent'
    } as never)

    let branchCurrentSession: ((messageId?: string) => Promise<boolean>) | null = null
    render(
      <BranchHarness
        activeSessionId="live-parent"
        onCurrentReady={branch => (branchCurrentSession = branch)}
        onReady={() => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionId="stored-parent"
      />
    )
    await waitFor(() => expect(branchCurrentSession).not.toBeNull())

    await expect(branchCurrentSession!()).resolves.toBe(true)

    expect(getAllSessionMessages).toHaveBeenCalledWith('stored-parent', undefined)
    expect(branchParams).toEqual({ session_id: 'live-parent' })
  })

  it('aborts if the active runtime changes while the branch transcript is hydrating', async () => {
    let refs: {
      activeSessionIdRef: MutableRefObject<string | null>
      selectedStoredSessionIdRef: MutableRefObject<string | null>
    } | null = null

    const requestGateway = vi.fn(async () => ({}) as never)

    setMessages([{ id: 'q1', role: 'user', parts: [{ type: 'text', text: 'question' }] }])
    vi.mocked(getAllSessionMessages).mockImplementation(async () => {
      refs!.activeSessionIdRef.current = 'live-other'

      return {
        messages: [{ content: 'question', role: 'user', timestamp: 1 }],
        session_id: 'stored-parent'
      } as never
    })

    let branchCurrentSession: ((messageId?: string) => Promise<boolean>) | null = null
    render(
      <BranchHarness
        activeSessionId="live-parent"
        onCurrentReady={branch => (branchCurrentSession = branch)}
        onReady={() => undefined}
        onRefs={value => (refs = value)}
        requestGateway={requestGateway}
        selectedStoredSessionId="stored-parent"
      />
    )
    await waitFor(() => expect(branchCurrentSession).not.toBeNull())

    await expect(branchCurrentSession!()).resolves.toBe(false)
    expect(requestGateway).not.toHaveBeenCalledWith('session.branch', expect.anything())
  })

  // #67603: right-clicking a session outside the paginated sidebar window is a
  // cache miss. Resolve its owning profile (cache → active → cross-profile) and
  // swap to it before reading the transcript / creating the branch, so the fork
  // is not created on whichever profile happens to be live.
  it('resolves and swaps to the parent profile when the branched session is not cached', async () => {
    setSessions([])
    vi.mocked(getSession).mockResolvedValue(storedSession({ id: 'stored-parent', message_count: 1, profile: 'work' }))
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    let createParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        createParams = params

        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      return {} as never
    })

    let branchStoredSession: ((storedSessionId: string, sessionProfile?: string | null) => Promise<boolean>) | null =
      null

    render(<BranchHarness onReady={branch => (branchStoredSession = branch)} requestGateway={requestGateway} />)
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    expect(ensureGatewayProfile).toHaveBeenCalledWith('work')
    expect(getAllSessionMessages).toHaveBeenCalledWith('stored-parent', 'work')
    // The create itself must carry the owning profile: in app-global remote
    // mode the soft gateway swap alone is not enough — an omitted profile
    // lands the branch on the launch (default) profile's state.db.
    expect(createParams).toMatchObject({ parent_session_id: 'stored-parent', profile: 'work' })

    vi.mocked(getSession).mockReset()
  })

  it('creates the branch on the cached parent session profile', async () => {
    setSessions([storedSession({ id: 'stored-parent', message_count: 1, profile: 'work' })])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    let createParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        createParams = params

        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      return {} as never
    })

    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(<BranchHarness onReady={branch => (branchStoredSession = branch)} requestGateway={requestGateway} />)
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    expect(ensureGatewayProfile).toHaveBeenCalledWith('work')
    expect(createParams).toMatchObject({ profile: 'work' })
  })

  it('omits profile for a profile-less parent so single-profile users are unchanged', async () => {
    setSessions([storedSession({ id: 'stored-parent', message_count: 1 })])
    vi.mocked(getAllSessionMessages).mockResolvedValue({
      messages: [{ content: 'branch me', role: 'user', timestamp: 1 }],
      session_id: 'stored-parent'
    } as never)

    let createParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        createParams = params

        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
      }

      return {} as never
    })

    let branchStoredSession: ((storedSessionId: string) => Promise<boolean>) | null = null
    render(<BranchHarness onReady={branch => (branchStoredSession = branch)} requestGateway={requestGateway} />)
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    expect(createParams).toBeDefined()
    expect(createParams).not.toHaveProperty('profile')
  })
})

// ── Main/tile dedup (the "same session open in main AND its own tab" bug) ─────
// A session is EITHER the main thread OR a tile, never both. openSessionTile
// enforces this from the tile side; resumeSession enforces it from the main
// side by dropping an existing tile when the session loads into main (cold-start
// restore, a pasted/⌘K route, a notification jump), so it can't render twice.
describe('resumeSession drops a redundant tile when the session loads into main', () => {
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setResumeFailedSessionId(null)
    setMessages([])
    setSessions([])
    $sessionTiles.set([])
    vi.restoreAllMocks()
  })

  it('closes the tile so the session is not open in both main and its own tab', async () => {
    // The session is already an open tile (e.g. persisted across a restart)...
    $sessionTiles.set([{ storedSessionId: 'stored-1' }])

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [] } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={r => (resume = r)} requestGateway={requestGateway} />)
    await waitFor(() => expect(resume).not.toBeNull())

    // ...and now it loads into main.
    await resume!('stored-1', true)

    // Its tile is gone — main owns the session, so it renders exactly once.
    expect($sessionTiles.get().some(t => t.storedSessionId === 'stored-1')).toBe(false)
    expect($selectedStoredSessionId.get()).toBe('stored-1')
  })

  it('leaves OTHER sessions tiles untouched', async () => {
    $sessionTiles.set([{ storedSessionId: 'stored-1' }, { storedSessionId: 'stored-2' }])

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [] } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={r => (resume = r)} requestGateway={requestGateway} />)
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-1', true)

    // Only the resumed session's tile closes; the sibling tile stays put.
    expect($sessionTiles.get().map(t => t.storedSessionId)).toEqual(['stored-2'])
  })
})

// ── Warm-cache mapping integrity (the "open chat A, chat B loads" bug) ─────────
// resumeSession's warm fast-path maps storedSessionId -> runtimeId -> cached
// state. A reaped/respawned pooled backend re-mints runtime ids, so a recycled
// id can resolve to a live-but-DIFFERENT session's cache entry. The fast-path
// must verify the cached state still BELONGS to the resumed session before it
// paints, or it shows a totally different thread under the current route.
const clientState = (storedSessionId: string | null): ClientSessionState => createClientSessionState(storedSessionId)

describe('resumeSession warm-cache mapping integrity', () => {
  beforeEach(() => {
    // Earlier describes (branchStoredSession) drive resumes through the
    // profile path on the SAME hoisted mock; drop their recorded calls so the
    // not-called assertions below only see this describe's traffic.
    vi.mocked(requestGatewayForProfile).mockReset()
    vi.mocked(requestGatewayForAgent).mockReset()
  })

  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setResumeFailedSessionId(null)
    setMessages([])
    setSessions([])
    vi.mocked(getSession).mockReset()
    vi.mocked(getLatestSessionMessages)
      .mockReset()
      .mockResolvedValue({ messages: [] } as never)
    vi.mocked(requestGatewayForAgent).mockReset()
    clearClarifyRequest()
    vi.mocked(requestGatewayForProfile).mockReset()
    setConnection(null)
    vi.restoreAllMocks()
  })

  it('pins an untagged row to the active registry connection instead of the same-named local profile', async () => {
    setConnection({ connectionId: 'hermes01', mode: 'remote' } as never)
    setSessions([storedSession({ id: 'remote-stored', profile: 'default' })])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'remote-stored' } as never)
    vi.mocked(requestGatewayForAgent).mockResolvedValue({
      info: {},
      messages: [],
      resumed: 'remote-stored',
      session_id: 'remote-runtime'
    } as never)
    vi.mocked(requestGatewayForProfile).mockResolvedValue({
      info: {},
      messages: [],
      resumed: 'remote-stored',
      session_id: 'wrong-local-runtime'
    } as never)

    const ambientRequest = vi.fn(async () => ({}) as never)
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(<ResumeHarness onReady={ready => (resume = ready)} requestGateway={ambientRequest} />)
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('remote-stored', true)

    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'hermes01',
      'default',
      'session.resume',
      expect.objectContaining({ session_id: 'remote-stored' })
    )
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('pins metadata, transcript, resume, activate, and usage to the captured connection', async () => {
    const ownerRoute: SessionProfileRoute = {
      connectionId: 'source-a',
      mode: 'remote',
      profile: 'default',
      targetProfile: 'backend-default'
    }

    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([
        ['stored-warm', 'runtime-warm'],
        ['stored-legacy', 'runtime-legacy']
      ])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([
        ['runtime-warm', clientState('stored-warm')],
        ['runtime-legacy', clientState('stored-legacy')]
      ])
    }

    // Same-name rows without a source tag are not authoritative for an
    // explicit owner. Metadata must be re-read from the captured connection.
    setSessions([
      storedSession({ id: 'stored-warm', profile: 'default' }),
      storedSession({ id: 'stored-legacy', profile: 'default' })
    ])
    vi.mocked(getSession).mockImplementation(async id => storedSession({ id, profile: 'default' }))
    vi.mocked(getLatestSessionMessages).mockImplementation(async id => ({ messages: [], session_id: id }) as never)
    vi.mocked(requestGatewayForAgent).mockImplementation(async (_connectionId, _profile, method, params) => {
      if (method === 'session.activate') {
        if (params?.session_id === 'runtime-legacy') {
          throw new Error('Method not found')
        }

        return {
          info: {},
          message_count: 0,
          messages: [],
          messages_omitted: true,
          resumed: 'stored-warm',
          running: false,
          session_id: 'runtime-warm',
          session_key: 'stored-warm'
        } as never
      }

      if (method === 'session.usage') {
        return { input: 1, output: 2, total: 3 } as never
      }

      if (method === 'session.resume') {
        return {
          info: {},
          messages: [],
          resumed: params?.session_id,
          session_id: 'runtime-cold'
        } as never
      }

      return {} as never
    })

    const ambientRequest = vi.fn(async () => ({}) as never)

    let resume:
      null | ((storedSessionId: string, replaceRoute?: boolean, ownerRoute?: SessionProfileRoute) => Promise<unknown>) =
      null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        requestGateway={ambientRequest}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    await resume!('stored-warm', true, ownerRoute)
    await resume!('stored-legacy', true, ownerRoute)
    await resume!('stored-cold', true, ownerRoute)
    await resume!('stored-cold', true, {
      connectionId: 'source-b',
      mode: 'remote',
      profile: 'default',
      targetProfile: 'backend-b'
    })

    const restScope = { connectionId: 'source-a', profile: 'backend-default' }
    expect(getSession).toHaveBeenCalledWith('stored-warm', restScope)
    expect(getSession).toHaveBeenCalledWith('stored-cold', restScope)
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-warm', restScope)
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-cold', restScope, {
      deferTailBookkeeping: true
    })
    expect(getSession).toHaveBeenCalledWith('stored-cold', {
      connectionId: 'source-b',
      profile: 'backend-b'
    })
    expect(getLatestSessionMessages).toHaveBeenCalledWith(
      'stored-cold',
      {
        connectionId: 'source-b',
        profile: 'backend-b'
      },
      { deferTailBookkeeping: true }
    )
    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'source-a',
      'default',
      'session.activate',
      expect.objectContaining({ session_id: 'runtime-warm' })
    )
    expect(requestGatewayForAgent).toHaveBeenCalledWith('source-a', 'default', 'session.usage', {
      session_id: 'runtime-legacy'
    })
    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'source-a',
      'default',
      'session.resume',
      expect.objectContaining({ session_id: 'stored-cold' })
    )
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('keeps a registry-tagged cached session on its owning connection without an explicit route', async () => {
    setSessions([
      storedSession({
        connection_id: 'test-amnezia',
        id: 'stored-registry',
        profile: 'default'
      })
    ])
    vi.mocked(getSession).mockImplementation(async id =>
      storedSession({ connection_id: 'test-amnezia', id, profile: 'default' })
    )
    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-registry' } as never)
    vi.mocked(requestGatewayForAgent).mockImplementation(async (_connectionId, _profile, method, params) => {
      if (method === 'session.resume') {
        return {
          info: {},
          messages: [],
          resumed: params?.session_id,
          session_id: 'runtime-registry'
        } as never
      }

      return {} as never
    })

    const ambientRequest = vi.fn(async () => ({}) as never)
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(<ResumeHarness onReady={ready => (resume = ready)} requestGateway={ambientRequest} />)
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-registry', true)

    const restScope = { connectionId: 'test-amnezia', profile: 'default' }
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-registry', restScope, {
      deferTailBookkeeping: true
    })
    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'test-amnezia',
      'default',
      'session.resume',
      expect.objectContaining({ session_id: 'stored-registry' })
    )
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('rejects a cross-wired runtime mapping and falls through to a full resume', async () => {
    // A recycled runtime id ('rt-recycled') is mapped to 'stored-A', but its
    // cached state actually belongs to a DIFFERENT session ('stored-B') — the
    // exact "open chat A, chat B loads" corruption a reaped/respawned pooled
    // backend can leave behind.
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-recycled']])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-recycled', clientState('stored-B')]])
    }

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'rt-A-fresh', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [] } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={r => (resume = r)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    // The fast-path did NOT short-circuit on the cross-wired cache — the full
    // resume RPC ran, for the session that was actually requested.
    const resumeCalls = requestGateway.mock.calls.filter(([method]) => method === 'session.resume')
    expect(resumeCalls.length).toBe(1)
    expect(resumeCalls[0][1]).toMatchObject({
      defer_history: true,
      session_id: 'stored-A'
    })
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-A', undefined, {
      deferTailBookkeeping: true
    })

    // The corrupt mapping was purged so it can't mis-resolve again.
    expect(runtimeIdByStoredSessionIdRef.current.has('stored-A')).toBe(false)
    expect(sessionStateByRuntimeIdRef.current.has('rt-recycled')).toBe(false)
  })

  it('paints the bounded latest transcript before the deferred resume acknowledgement without rebuilding it', async () => {
    const latestPage = Array.from({ length: 500 }, (_, index) => ({
      content: `message-${index}`,
      role: index % 2 === 0 ? ('user' as const) : ('assistant' as const),
      timestamp: index + 1
    }))

    setSessions([storedSession({ id: 'stored-A', message_count: 50_000 })])
    vi.mocked(getLatestSessionMessages).mockReset()
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: latestPage,
      pagination: { limit: 500, offset: 0, order: 'latest', returned: 500 },
      session_id: 'stored-A'
    })

    const deferredResume = deferred<SessionResumeResponse>()

    const requestGatewayMock = vi.fn((method: string, _params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return deferredResume.promise
      }

      return Promise.resolve({})
    })

    const requestGateway = <T,>(method: string, params?: Record<string, unknown>): Promise<T> =>
      requestGatewayMock(method, params) as Promise<T>

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={value => (resume = value)} requestGateway={requestGateway} />)
    await waitFor(() => expect(resume).not.toBeNull())
    const resumePromise = resume!('stored-A', true)

    await waitFor(() => expect(getLatestSessionMessages).toHaveBeenCalledTimes(1))
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-A', undefined, {
      deferTailBookkeeping: true
    })
    await waitFor(() => expect($messages.get()).toHaveLength(500))
    const paintedTranscript = $messages.get()
    expect(requestGatewayMock).toHaveBeenCalledWith(
      'session.resume',
      expect.objectContaining({
        defer_history: true,
        omit_messages: true,
        session_id: 'stored-A'
      })
    )

    deferredResume.resolve({
      session_id: 'rt-A',
      resumed: 'stored-A',
      message_count: 500,
      messages: [],
      info: {}
    })
    await resumePromise
    expect($messages.get()).toBe(paintedTranscript)
  })

  it('honours a warm cache entry whose stored id matches and refreshes its persisted transcript', async () => {
    // Correctly-wired mapping: 'rt-A' <-> 'stored-A'. The fast-path should trust
    // it and never reach session.resume. session.activate refreshes the live
    // projection and, critically, rebinds its event transport after reconnect.
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', clientState('stored-A')]])
    }

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: 0,
          messages: [],
          running: false,
          info: {}
        } as never
      }

      return {} as never
    })

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-A' } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={r => (resume = r)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    // Fast-path served the session from cache: no full resume RPC, mapping intact.
    // The persisted transcript still refreshes in parallel because the runtime
    // projection can differ even when its row count matches.
    const methods = requestGateway.mock.calls.map(([method]) => method)
    expect(methods).toContain('session.activate')
    expect(methods).not.toContain('session.resume')
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-A', undefined)
    expect(requestGateway).toHaveBeenCalledWith(
      'session.activate',
      expect.objectContaining({ omit_messages: true, session_id: 'rt-A' })
    )
    expect(runtimeIdByStoredSessionIdRef.current.get('stored-A')).toBe('rt-A')
  })

  it('rejects a warm persisted response after the current row advances its display revision', async () => {
    const proof = {
      connectionId: '',
      coverage: 'latest-page' as const,
      displayRevision: 7,
      lineageRootId: 'stored-A',
      profile: 'default',
      resolvedTipId: 'stored-A',
      source: 'persisted-display' as const,
      storedSessionId: 'stored-A'
    }
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }
    const state = clientState('stored-A')

    state.messages = [
      { id: 'warm-current', parts: [{ text: 'current warm transcript', type: 'text' }], role: 'assistant' }
    ]
    state.transcriptProvenance = proof

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }
    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-A',
        display_revision: 7,
        id: 'stored-A',
        message_count: 1
      })
    ])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          info: {},
          message_count: 1,
          messages: [],
          messages_omitted: true,
          resumed: 'stored-A',
          running: false,
          session_id: 'rt-A',
          session_key: 'stored-A'
        } as never
      }

      return {} as never
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={value => (resume = value)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    const pending = resume!('stored-A', true)
    await waitFor(() => expect(getLatestSessionMessages).toHaveBeenCalled())

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-A',
        display_revision: 8,
        id: 'stored-A',
        message_count: 1
      })
    ])
    persisted.resolve({
      display_revision: 7,
      lineage_root_id: 'stored-A',
      messages: [{ content: 'stale revision 7 transcript', role: 'assistant', timestamp: 2 }],
      resolved_tip_id: 'stored-A',
      session_id: 'stored-A'
    })
    await pending

    expect(JSON.stringify(sessionStateByRuntimeIdRef.current.get('rt-A')?.messages)).toContain('current warm transcript')
    expect(JSON.stringify(sessionStateByRuntimeIdRef.current.get('rt-A')?.messages)).not.toContain(
      'stale revision 7 transcript'
    )
    expect(loadTranscriptTail('stored-A', proof)).toBeNull()
  })

  it('rejects a revisioned warm response whose own tip identity differs from the captured row', async () => {
    const currentProof = {
      connectionId: '',
      coverage: 'latest-page' as const,
      displayRevision: 8,
      lineageRootId: 'stored-A',
      profile: 'default',
      resolvedTipId: 'tip-b',
      source: 'persisted-display' as const,
      storedSessionId: 'stored-A'
    }
    const mismatchedProof = { ...currentProof, resolvedTipId: 'tip-a' }
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }
    const state = clientState('stored-A')

    state.messages = [
      { id: 'warm-tip-b', parts: [{ text: 'warm transcript from tip B', type: 'text' }], role: 'assistant' }
    ]
    state.transcriptProvenance = currentProof
    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    setSessions([
      storedSession({
        _lineage_root_id: 'stored-A',
        display_revision: 8,
        id: 'tip-b',
        message_count: 1
      })
    ])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      display_revision: 8,
      lineage_root_id: 'stored-A',
      messages: [{ content: 'wrong warm response from tip A', role: 'assistant', timestamp: 2 }],
      resolved_tip_id: 'tip-a',
      session_id: 'stored-A'
    })
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          info: {},
          message_count: 1,
          messages: [],
          messages_omitted: true,
          resumed: 'stored-A',
          running: false,
          session_id: 'rt-A',
          session_key: 'stored-A'
        } as never
      }

      return {} as never
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={value => (resume = value)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    expect(JSON.stringify(sessionStateByRuntimeIdRef.current.get('rt-A')?.messages)).toContain(
      'warm transcript from tip B'
    )
    expect(JSON.stringify(sessionStateByRuntimeIdRef.current.get('rt-A')?.messages)).not.toContain(
      'wrong warm response from tip A'
    )
    expect(loadTranscriptTail('stored-A', mismatchedProof)).toBeNull()
  })

  it('re-arms a pending clarify in place on the warm session.activate path', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      { id: 'cached-user', role: 'user', parts: [{ type: 'text', text: 'help me choose' }] },
      {
        id: 'cached-assistant',
        role: 'assistant',
        parts: [
          { type: 'text', text: 'I found two paths.' },
          {
            type: 'tool-call',
            toolCallId: 'call-provider',
            toolName: 'clarify',
            args: { choices: ['safe', 'fast'], question: 'Which path?' },
            argsText: '{"question":"Which path?","choices":["safe","fast"]}'
          }
        ]
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'help me choose', role: 'user', timestamp: 1 },
        {
          content: 'I found two paths.',
          role: 'assistant',
          timestamp: 2,
          tool_calls: [
            {
              function: {
                arguments: '{"question":"Which path?","choices":["safe","fast"]}',
                name: 'clarify'
              },
              id: 'call-provider'
            }
          ]
        }
      ],
      session_id: 'stored-A'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          info: {},
          message_count: 2,
          messages: [],
          messages_omitted: true,
          pending_clarify: {
            choices: ['safe', 'fast'],
            question: 'Which path?',
            request_id: 'req-warm'
          },
          resumed: 'stored-A',
          running: true,
          session_id: 'rt-A',
          session_key: 'stored-A'
        } as never
      }

      return {} as never
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    const resumedState = sessionStateByRuntimeIdRef.current.get('rt-A')

    const clarifyMessages =
      resumedState?.messages.filter(message =>
        message.parts.some(part => part.type === 'tool-call' && part.toolName === 'clarify')
      ) ?? []

    expect(requestGateway.mock.calls.map(([method]) => method)).toContain('session.activate')
    expect(clarifyMessages).toHaveLength(1)
    expect(clarifyMessages[0].pending).toBe(true)
    expect(
      clarifyMessages[0].parts.find(part => part.type === 'tool-call' && part.toolName === 'clarify')
    ).toMatchObject({ toolCallId: 'call-provider' })
    expect(resumedState?.streamId).toBe(clarifyMessages[0].id)
    expect($clarifyRequests.get()['rt-A']).toMatchObject({ requestId: 'req-warm' })
  })

  it.each([
    ['with a stale request-store entry', true],
    ['after the request store was already cleared', false]
  ])('de-arms stale clarify state %s when session.activate reports no pending request', async (_case, keepStore) => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.busy = true
    state.needsInput = true
    state.streamId = 'cached-assistant'
    state.messages = [
      { id: 'cached-user', role: 'user', parts: [{ type: 'text', text: 'help me choose' }] },
      {
        id: 'cached-assistant',
        role: 'assistant',
        pending: true,
        parts: [
          {
            type: 'tool-call',
            toolCallId: 'call-provider',
            toolName: 'clarify',
            args: { choices: ['safe', 'fast'], question: 'Which path?' },
            argsText: '{"question":"Which path?","choices":["safe","fast"]}'
          }
        ]
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    if (keepStore) {
      setClarifyRequest({
        choices: ['safe', 'fast'],
        multiSelect: false,
        question: 'Which path?',
        requestId: 'req-stale',
        sessionId: 'rt-A'
      })
    }

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'help me choose', role: 'user', timestamp: 1 },
        {
          content: '',
          role: 'assistant',
          timestamp: 2,
          tool_calls: [
            {
              function: {
                arguments: '{"question":"Which path?","choices":["safe","fast"]}',
                name: 'clarify'
              },
              id: 'call-provider'
            }
          ]
        }
      ],
      session_id: 'stored-A'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          info: {},
          message_count: 2,
          messages: [],
          messages_omitted: true,
          resumed: 'stored-A',
          running: false,
          session_id: 'rt-A',
          session_key: 'stored-A'
        } as never
      }

      return {} as never
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    const resumedState = sessionStateByRuntimeIdRef.current.get('rt-A')

    const clarifyPart = resumedState?.messages
      .flatMap(message => message.parts)
      .find(part => part.type === 'tool-call' && part.toolName === 'clarify')

    expect($clarifyRequests.get()['rt-A']).toBeUndefined()
    expect(resumedState).toMatchObject({ needsInput: false, streamId: null })
    expect(resumedState?.messages.find(message => message.id === resumedState.streamId)).toBeUndefined()
    expect(clarifyPart).toHaveProperty('result')
  })

  it('does not let an older activate response clear a newer clarify request', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', clientState('stored-A')]])
    }

    const activated = deferred<SessionResumeResponse>()

    const requestGateway = vi.fn((method: string) =>
      method === 'session.activate' ? activated.promise : Promise.resolve({})
    )

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-A' } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        requestGateway={requestGateway as never}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    const resumePromise = resume!('stored-A', true)
    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('session.activate', expect.anything()))

    setClarifyRequest({
      choices: ['new'],
      multiSelect: false,
      question: 'New question?',
      receivedAt: Date.now() / 1000 + 60,
      requestId: 'req-newer',
      sessionId: 'rt-A'
    })
    activated.resolve({
      info: {},
      message_count: 0,
      messages: [],
      messages_omitted: true,
      resumed: 'stored-A',
      running: false,
      session_id: 'rt-A',
      session_key: 'stored-A'
    })
    await resumePromise

    expect($clarifyRequests.get()['rt-A']).toMatchObject({ requestId: 'req-newer' })
  })

  it('reads the terminal transcript after warm reconnect transport reattachment', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      {
        id: 'cached-user',
        role: 'user',
        parts: [{ type: 'text', text: 'long running prompt' }]
      },
      {
        id: 'cached-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'partial before disconnect' }],
        pending: true
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    let transportAttached = false
    let attachedWhenTranscriptRead: boolean | null = null

    vi.mocked(getLatestSessionMessages).mockReset()
    vi.mocked(getLatestSessionMessages).mockImplementation(async () => {
      attachedWhenTranscriptRead = transportAttached

      return {
        messages: [
          { content: 'long running prompt', role: 'user', timestamp: 1 },
          {
            content: transportAttached ? 'complete answer persisted during disconnect' : 'partial before disconnect',
            role: 'assistant',
            timestamp: 2
          }
        ],
        session_id: 'stored-A'
      } as never
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        // The backend rebinds the session transport before returning this
        // terminal snapshot. A transcript read issued earlier can miss the
        // final persisted row and no live event will arrive to repair it.
        transportAttached = true

        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: 2,
          messages: [],
          messages_omitted: true,
          running: false,
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, next) => (resumedState = next)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    expect(attachedWhenTranscriptRead).toBe(true)
    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(resumedState?.messages)).toContain('complete answer persisted during disconnect')
    expect(JSON.stringify(resumedState?.messages)).not.toContain('partial before disconnect')
  })

  it('keeps a terminal live state when a running reconnect finishes during transcript hydration', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.busy = true
    state.awaitingResponse = true
    state.turnLive = true
    state.turnStartedAt = 1_700_000_123_000
    state.messages = [
      {
        id: 'cached-user',
        role: 'user',
        parts: [{ type: 'text', text: 'long running prompt' }]
      },
      {
        id: 'assistant-stream-rt-A',
        role: 'assistant',
        parts: [{ type: 'text', text: 'partial before terminal event' }],
        pending: true
      }
    ]
    state.streamId = 'assistant-stream-rt-A'

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()

    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: 2,
          messages: [],
          messages_omitted: true,
          running: true,
          turn_started_at: 1_700_000_123,
          inflight: {
            user: 'long running prompt',
            assistant: 'partial before terminal event',
            streaming: true
          },
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, next) => (resumedState = next)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const resumePromise = resume!('stored-A', true)

    await waitFor(() => expect(getLatestSessionMessages).toHaveBeenCalledTimes(1))
    expect(sessionStateByRuntimeIdRef.current.get('rt-A')).toMatchObject({
      awaitingResponse: true,
      busy: true,
      turnLive: true
    })

    // The rebound transport delivers the terminal state while REST is still
    // pending. Settle the same stream row the real terminal event owns; the
    // later durable hydration may reconcile messages but cannot revive it.
    const liveTerminalState = sessionStateByRuntimeIdRef.current.get('rt-A')!
    sessionStateByRuntimeIdRef.current.set('rt-A', {
      ...liveTerminalState,
      adoptedRunningTurn: false,
      awaitingResponse: false,
      busy: false,
      messages: liveTerminalState.messages.map(message =>
        message.id === 'assistant-stream-rt-A'
          ? {
              ...message,
              parts: [{ type: 'text' as const, text: 'complete durable answer' }],
              pending: false
            }
          : message
      ),
      streamId: null,
      turnLive: false,
      turnStartedAt: null
    })

    await act(async () => {
      persisted.resolve({
        messages: [
          { content: 'long running prompt', role: 'user', timestamp: 1 },
          { content: 'complete durable answer', role: 'assistant', timestamp: 2 }
        ],
        session_id: 'stored-A'
      } as never)
      await resumePromise
    })

    expect(JSON.stringify(resumedState?.messages)).toContain('complete durable answer')
    expect(JSON.stringify(resumedState?.messages)).not.toContain('partial before terminal event')
    expect(resumedState).toMatchObject({
      adoptedRunningTurn: false,
      awaitingResponse: false,
      busy: false,
      turnLive: false,
      turnStartedAt: null
    })
  })

  it('preserves cached image attachments through an idle persisted transcript refresh', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      {
        id: 'cached-user',
        role: 'user',
        parts: [{ type: 'text', text: 'describe this image' }],
        attachmentRefs: ['@image:/tmp/photo.png']
      },
      {
        id: 'cached-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'It is a photo.' }]
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    const persistedMessages = [
      { content: 'describe this image', role: 'user', timestamp: 1 },
      { content: 'It is a photo.', role: 'assistant', timestamp: 2 }
    ]

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: persistedMessages,
      session_id: 'stored-A'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: persistedMessages.length,
          messages: persistedMessages,
          running: false,
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, next) => (resumedState = next)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    expect(requestGateway.mock.calls.map(([method]) => method)).toContain('session.activate')
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-A', undefined)
    expect(resumedState?.messages[0]?.attachmentRefs).toEqual(['@image:/tmp/photo.png'])
  })

  it('restores the warm reconnect turn clock from session.activate', async () => {
    const turnStartedAtSeconds = 1_700_000_123

    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const cachedState = clientState('stored-A')
    cachedState.busy = true
    cachedState.turnStartedAt = null

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', cachedState]])
    }

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: 0,
          messages: [],
          running: true,
          turn_started_at: turnStartedAtSeconds,
          inflight: {
            user: 'current prompt',
            assistant: 'partial answer',
            streaming: true
          },
          info: {}
        } as never
      }

      return {} as never
    })

    vi.mocked(getAllSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-A' } as never)

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, state) => (resumedState = state)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    expect(resumedState).toMatchObject({
      awaitingResponse: true,
      busy: true,
      turnStartedAt: turnStartedAtSeconds * 1000
    })
    expect(JSON.stringify(resumedState?.messages)).toContain('partial answer')
  })

  it('repairs an idle warm cache from a divergent equal-length persisted transcript', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      {
        id: 'cached-user',
        role: 'user',
        parts: [{ type: 'text', text: 'stale runtime prompt' }]
      },
      {
        id: 'cached-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'stale runtime answer' }]
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    const staleRuntimeMessages = [
      { content: 'stale runtime prompt', role: 'user', timestamp: 1 },
      { content: 'stale runtime answer', role: 'assistant', timestamp: 2 }
    ]

    const persistedMessages = [
      { content: 'prompt saved after compression', role: 'user', timestamp: 3 },
      { content: 'answer saved after compression', role: 'assistant', timestamp: 4 }
    ]

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: persistedMessages,
      session_id: 'stored-A'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: staleRuntimeMessages.length,
          messages: staleRuntimeMessages,
          running: false,
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, next) => (resumedState = next)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    const renderedMessages = JSON.stringify(resumedState?.messages)
    expect(renderedMessages).toContain('prompt saved after compression')
    expect(renderedMessages).toContain('answer saved after compression')
    expect(renderedMessages).not.toContain('stale runtime answer')
  })

  it('keeps the activated transcript when a persisted transcript refresh returns empty rows', async () => {
    // Regression: after a wake/reconnect, session.activate can legitimately
    // rebind a session with a non-empty transcript while the concurrent REST
    // refresh (getLatestSessionMessages) races a just-respawned backend and
    // resolves with zero rows. That empty page must not be trusted over the
    // transcript activate already restored.
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      {
        id: 'cached-user',
        role: 'user',
        parts: [{ type: 'text', text: 'still here after wake' }]
      },
      {
        id: 'cached-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'still here after wake too' }]
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    const activatedMessages = [
      { content: 'still here after wake', role: 'user', timestamp: 1 },
      { content: 'still here after wake too', role: 'assistant', timestamp: 2 }
    ]

    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-A' } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: activatedMessages.length,
          messages: activatedMessages,
          running: false,
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, next) => (resumedState = next)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    const renderedMessages = JSON.stringify(resumedState?.messages)
    expect(renderedMessages).toContain('still here after wake')
    expect(renderedMessages).toContain('still here after wake too')
  })

  it('keeps the complete persisted transcript when activating a compressed running session', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      {
        id: 'runtime-user',
        role: 'user',
        parts: [{ type: 'text', text: 'recent prompt' }]
      },
      {
        id: 'runtime-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'recent answer' }]
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    const persistedMessages = [
      { content: 'older prompt that compression removed', role: 'user', timestamp: 1 },
      { content: 'older answer that compression removed', role: 'assistant', timestamp: 2 },
      { content: 'recent prompt', role: 'user', timestamp: 3 },
      { content: 'recent answer', role: 'assistant', timestamp: 4 }
    ]

    const compressedRuntimeMessages = persistedMessages.slice(2)

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: persistedMessages,
      session_id: 'stored-A'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: compressedRuntimeMessages.length,
          messages: compressedRuntimeMessages,
          running: true,
          inflight: {
            user: 'current prompt',
            assistant: 'partial answer',
            streaming: true
          },
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, next) => (resumedState = next)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    const renderedMessages = resumedState?.messages ?? []
    const renderedText = JSON.stringify(renderedMessages)

    expect(renderedText).toContain('older prompt that compression removed')
    expect(renderedText).toContain('older answer that compression removed')
    expect(renderedText).toContain('recent prompt')
    expect(renderedText).toContain('recent answer')
    expect(renderedText).toContain('partial answer')
    expect(renderedMessages.filter(message => JSON.stringify(message).includes('current prompt'))).toHaveLength(1)
  })

  it('preserves live cache updates that arrive while the persisted transcript is loading', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      {
        id: 'runtime-user',
        role: 'user',
        parts: [{ type: 'text', text: 'recent prompt' }],
        timestamp: 3
      },
      {
        id: 'runtime-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'recent answer' }],
        timestamp: 4
      },
      {
        id: 'user-inflight-rt-A',
        role: 'user',
        parts: [{ type: 'text', text: 'current prompt' }]
      },
      {
        id: 'assistant-stream-live-123',
        role: 'assistant',
        parts: [{ type: 'text', text: 'partial A' }],
        pending: true
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    const persisted = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()

    const compressedRuntimeMessages = [
      { content: 'recent prompt', role: 'user', timestamp: 3 },
      { content: 'recent answer', role: 'assistant', timestamp: 4 }
    ]

    vi.mocked(getLatestSessionMessages).mockReturnValue(persisted.promise)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: compressedRuntimeMessages.length,
          messages: compressedRuntimeMessages,
          running: true,
          inflight: {
            user: 'current prompt',
            assistant: 'partial A',
            streaming: true
          },
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, next) => (resumedState = next)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const resumePromise = resume!('stored-A', true)

    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('session.activate', expect.anything()))

    const liveState = sessionStateByRuntimeIdRef.current.get('rt-A')!

    const liveMessages = liveState.messages.map(message =>
      message.id === 'assistant-stream-live-123'
        ? { ...message, parts: [{ type: 'text' as const, text: 'partial A + delta B' }] }
        : message
    )

    sessionStateByRuntimeIdRef.current.set('rt-A', {
      ...liveState,
      messages: [
        ...liveMessages,
        {
          id: 'user-racing',
          role: 'user',
          parts: [{ type: 'text', text: 'racing prompt' }]
        }
      ]
    })

    await act(async () => {
      persisted.resolve({
        messages: [
          { content: 'older prompt', role: 'user', timestamp: 1 },
          { content: 'older answer', role: 'assistant', timestamp: 2 },
          ...compressedRuntimeMessages
        ],
        session_id: 'stored-A'
      } as never)
      await resumePromise
    })

    const renderedText = JSON.stringify(resumedState?.messages)

    expect(renderedText).toContain('older prompt')
    expect(renderedText).toContain('partial A + delta B')
    expect(renderedText).toContain('racing prompt')

    const streamingAssistantRows = resumedState?.messages.filter(message => message.id.startsWith('assistant-stream-'))

    expect(streamingAssistantRows).toHaveLength(1)
    expect(streamingAssistantRows?.[0].id).toBe('assistant-stream-live-123')
  })

  it('does not duplicate an in-flight user prompt already present in the persisted suffix', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      {
        id: 'runtime-user',
        role: 'user',
        parts: [{ type: 'text', text: 'earlier prompt' }]
      },
      {
        id: 'runtime-assistant',
        role: 'assistant',
        parts: [{ type: 'text', text: 'earlier answer' }]
      },
      {
        id: 'user-optimistic',
        role: 'user',
        parts: [{ type: 'text', text: 'current prompt' }]
      },
      {
        id: 'assistant-stream-rt-A',
        role: 'assistant',
        parts: [{ type: 'text', text: 'partial answer' }],
        pending: true
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    const compressedRuntimeMessages = [
      { content: 'earlier prompt', role: 'user', timestamp: 1 },
      { content: 'earlier answer', role: 'assistant', timestamp: 2 }
    ]

    const persistedMessages = [
      { content: 'older prompt removed by compression', role: 'user', timestamp: -1 },
      { content: 'older answer removed by compression', role: 'assistant', timestamp: 0 },
      ...compressedRuntimeMessages,
      { content: 'current prompt', role: 'user', timestamp: 3 }
    ]

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: persistedMessages,
      session_id: 'stored-A'
    } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          session_id: 'rt-A',
          session_key: 'stored-A',
          resumed: 'stored-A',
          message_count: compressedRuntimeMessages.length,
          messages: compressedRuntimeMessages,
          running: true,
          inflight: {
            user: 'current prompt',
            assistant: 'partial answer',
            streaming: true
          },
          info: {}
        } as never
      }

      return {} as never
    })

    let resumedState: ClientSessionState | undefined
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null

    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onStateUpdate={(_sessionId, next) => (resumedState = next)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    const currentPromptRows = (resumedState?.messages ?? []).filter(message =>
      JSON.stringify(message).includes('current prompt')
    )

    expect(currentPromptRows).toHaveLength(1)
    expect(JSON.stringify(resumedState?.messages)).toContain('partial answer')
  })

  it('keeps a warm runtime and optimistic turn on a transient activation timeout', async () => {
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const state = clientState('stored-A')
    state.messages = [
      {
        id: 'user-optimistic',
        role: 'user',
        parts: [{ type: 'text', text: 'do not lose me' }]
      }
    ]

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', state]])
    }

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        throw new Error('request timed out: session.activate')
      }

      return {} as never
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={r => (resume = r)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    expect(requestGateway.mock.calls.map(([method]) => method)).not.toContain('session.resume')
    expect(runtimeIdByStoredSessionIdRef.current.get('stored-A')).toBe('rt-A')
    expect(sessionStateByRuntimeIdRef.current.get('rt-A')?.messages[0]?.id).toBe('user-optimistic')
  })

  it('never publishes a tail-only warm cache before the full persisted history', async () => {
    const cachedState = clientState('stored-1')
    cachedState.messages = [
      {
        id: 'user-latest',
        role: 'user',
        parts: [{ type: 'text', text: 'latest question after long-context completion' }]
      },
      {
        id: 'assistant-latest',
        role: 'assistant',
        parts: [{ type: 'text', text: 'latest answer after long-context completion' }]
      }
    ]

    const runtimeIdByStoredSessionIdRef = {
      current: new Map([['stored-1', 'runtime-warm']])
    } satisfies MutableRefObject<Map<string, string>>

    const sessionStateByRuntimeIdRef = {
      current: new Map([['runtime-warm', cachedState]])
    } satisfies MutableRefObject<Map<string, ClientSessionState>>

    const persistedAuthority = deferred<{
      messages: Array<{ content: string; role: 'assistant' | 'user'; timestamp: number }>
      session_id: string
    }>()

    const publications: Array<{ older: boolean; latest: boolean }> = []

    setSessions([storedSession({ message_count: 4 })])
    vi.mocked(getLatestSessionMessages).mockReturnValue(persistedAuthority.promise as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.activate') {
        return {
          info: {},
          message_count: 2,
          messages: [],
          messages_omitted: true,
          resumed: 'stored-1',
          running: false,
          session_id: 'runtime-warm',
          session_key: 'stored-1'
        } as never
      }

      return {} as never
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={ready => (resume = ready)}
        onViewSync={(_sessionId, state) => {
          const snapshot = JSON.stringify(state.messages)
          publications.push({
            latest: snapshot.includes('latest question after long-context completion'),
            older: snapshot.includes('earlier question before long-context completion')
          })
        }}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const resumePromise = resume!('stored-1', true)
    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'session.activate',
        expect.objectContaining({ omit_messages: true, session_id: 'runtime-warm' })
      )
    )

    expect(
      publications.filter(snapshot => snapshot.latest && !snapshot.older),
      `Tail-only warm-cache publication escaped before persisted authority: ${JSON.stringify(publications)}`
    ).toEqual([])

    persistedAuthority.resolve({
      messages: [
        { content: 'earlier question before long-context completion', role: 'user', timestamp: 1 },
        { content: 'earlier answer before long-context completion', role: 'assistant', timestamp: 2 },
        { content: 'latest question after long-context completion', role: 'user', timestamp: 3 },
        { content: 'latest answer after long-context completion', role: 'assistant', timestamp: 4 }
      ],
      session_id: 'stored-1'
    })
    await resumePromise

    expect(publications.some(snapshot => snapshot.latest && snapshot.older)).toBe(true)
    expect(
      publications.filter(snapshot => snapshot.latest && !snapshot.older),
      `Tail-only warm-cache publication escaped: ${JSON.stringify(publications)}`
    ).toEqual([])
  })
})

describe('createBackendSessionForSend workspace target', () => {
  afterEach(() => {
    cleanup()
    $newChatProfile.set(null)
    $activeGatewayProfile.set('default')
    $projectScope.set(ALL_PROJECTS)
    setCurrentCwd('')
    setNewChatWorkspaceTarget(undefined)
    vi.restoreAllMocks()
  })

  it('omits cwd for an explicit no-workspace draft even when global cwd changes before send', async () => {
    const params = await createWith(
      () => {
        $activeGatewayProfile.set('default')
      },
      handle => {
        handle.startFreshSessionDraft({ workspaceTarget: null })
        $currentCwd.set('/project-open-in-file-browser')
      }
    )

    expect(params).not.toHaveProperty('cwd')
    expect($newChatWorkspaceTarget.get()).toBeUndefined()
  })

  it('uses the clicked workspace target instead of a later global cwd value', async () => {
    const params = await createWith(
      () => {
        $activeGatewayProfile.set('default')
      },
      handle => {
        handle.startFreshSessionDraft({ workspaceTarget: '/clicked-workspace' })
        $currentCwd.set('/project-open-in-file-browser')
      }
    )

    expect(params).toMatchObject({ cwd: '/clicked-workspace' })
  })

  it('does not inherit a stale cwd when Home is the active project scope', async () => {
    const params = await createWith(
      () => {
        $projectScope.set(NO_PROJECT_ID)
      },
      () => {
        // Simulate the stale live path left by the previously selected project
        // before the new draft is submitted.
        $currentCwd.set('/previous-project')
      }
    )

    expect(params).not.toHaveProperty('cwd')
  })
})

describe('openNewSessionTile workspace target', () => {
  afterEach(() => {
    cleanup()
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    vi.restoreAllMocks()
  })

  it('omits cwd for a Home tile even when project scope resolves to a repo', async () => {
    $projectScope.set('p_voice')
    $projectTree.set([
      {
        id: 'p_voice',
        label: 'Voice Assistant',
        path: '/Users/oschmidt/Checkouts/voice-assistant',
        repos: [],
        sessionCount: 0
      } as never
    ])

    let createParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.create') {
        createParams = params

        return {
          info: { cwd: '', model: 'test-model', tools: {}, skills: {} },
          session_id: RUNTIME_SESSION_ID,
          stored_session_id: 'stored-home-tile'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(<Harness onReady={value => (handle = value)} requestGateway={requestGateway} />)
    await waitFor(() => expect(handle).not.toBeNull())

    await act(async () => {
      await handle!.openNewSessionTile('center', { cwd: null, listed: false })
    })

    expect(createParams).not.toHaveProperty('cwd')
  })
})
describe('selectSidebarItem', () => {
  it('fronts the workspace pane when navigating to a sidebar route (issue #72602)', async () => {
    const navigate = vi.fn()
    const requestGateway = vi.fn(async () => ({}) as never)
    let handle: HarnessHandle | null = null

    render(<Harness navigate={navigate} onReady={value => (handle = value)} requestGateway={requestGateway} />)
    await waitFor(() => expect(handle).not.toBeNull())

    act(() => {
      handle!.selectSidebarItem({ icon: (() => null) as never, id: 'skills', label: 'Capabilities', route: '/skills' })
    })

    expect(navigate).toHaveBeenCalledWith('/skills', undefined)
    expect(noteActiveTreeGroup).toHaveBeenCalledWith(null)
    expect(revealTreePane).toHaveBeenCalledWith('workspace')
  })
})

const mockDeleteSession = vi.mocked(deleteSession)
const mockGetSession = vi.mocked(getSession)
const mockSetSessionArchived = vi.mocked(setSessionArchived)
const profiles = (...names: string[]) => names.map(name => ({ name }) as never)

describe('removeSession / archiveSession profile routing (#78836)', () => {
  beforeEach(() => {
    setSessions([])
    setMessagingSessions([])
    setCronSessions([])
    $profiles.set(profiles('default', 'winefox'))
    $activeGatewayProfile.set('default')
    $pinnedSessionIds.set([])
    $removedSessionIds.set(new Set())
    $sessionMutationsInFlight.set(new Set())
    $sessionSeenCounts.set({})
    $unreadFinishedMarkers.set({})
    mockDeleteSession.mockReset()
    mockGetSession.mockReset()
    mockSetSessionArchived.mockReset()
  })

  afterEach(() => {
    cleanup()
    setSessions([])
    setMessagingSessions([])
    setCronSessions([])
    $profiles.set([])
    $activeGatewayProfile.set('default')
    $pinnedSessionIds.set([])
    $removedSessionIds.set(new Set())
    $sessionMutationsInFlight.set(new Set())
    $sessionSeenCounts.set({})
    $unreadFinishedMarkers.set({})
  })

  async function readyActions() {
    let handle: HarnessHandle | null = null
    render(<Harness onReady={value => (handle = value)} requestGateway={vi.fn(async () => ({}) as never)} />)
    await waitFor(() => expect(handle).not.toBeNull())

    return handle!
  }

  it('DELETEs a stamped messaging session against its owning profile', async () => {
    mockDeleteSession.mockResolvedValue({ ok: true })
    setMessagingSessions([
      storedSession({ id: 'tg-winefox-1', profile: 'winefox', source: 'telegram', title: 'TG chat' })
    ])

    const handle = await readyActions()
    await act(async () => {
      await handle.removeSession('tg-winefox-1')
    })

    expect(mockDeleteSession).toHaveBeenCalledWith('tg-winefox-1', 'winefox')
    expect($messagingSessions.get()).toEqual([])
    expect($sessions.get()).toEqual([])
  })

  it('resolves a profile-less messaging DELETE before drop, without leaking into recents', async () => {
    $sessionSeenCounts.set({
      winefox: { 'tg-1': 4 },
      default: { 'desk-keep': 2 }
    })
    $unreadFinishedMarkers.set({
      winefox: ['tg-1'],
      default: ['desk-keep']
    })
    setMessagingSessions([storedSession({ id: 'tg-1', source: 'telegram', title: 'QQ/TG' })])
    mockGetSession.mockImplementation(async (id: string, scope?: ProfileScope) => {
      expect($messagingSessions.get().some(session => session.id === 'tg-1')).toBe(true)
      const profile = scope && typeof scope === 'object' ? scope.profile : scope

      if (!profile) {
        throw new Error('404: Session not found')
      }

      if (profile === 'winefox') {
        return storedSession({ id, profile: 'winefox', source: 'telegram' })
      }

      throw new Error('404: Session not found')
    })
    mockDeleteSession.mockResolvedValue({ ok: true })

    const handle = await readyActions()
    await act(async () => {
      await handle.removeSession('tg-1')
    })

    expect(mockGetSession).toHaveBeenCalled()
    expect(mockDeleteSession).toHaveBeenCalledWith('tg-1', 'winefox')
    expect($messagingSessions.get()).toEqual([])
    expect($sessions.get()).toEqual([])
    expect($sessionSeenCounts.get().winefox?.['tg-1']).toBeUndefined()
    expect($sessionSeenCounts.get().default?.['desk-keep']).toBe(2)
    expect($unreadFinishedMarkers.get().winefox ?? []).not.toContain('tg-1')
    expect($unreadFinishedMarkers.get().default).toEqual(['desk-keep'])
  })

  it('restores a failed DELETE to the messaging slice, not recents', async () => {
    const row = storedSession({ id: 'tg-roll', profile: 'winefox', source: 'telegram' })
    setMessagingSessions([row])
    $pinnedSessionIds.set(['tg-roll'])
    mockDeleteSession.mockRejectedValue(new Error('backend down'))

    const handle = await readyActions()
    await act(async () => {
      await handle.removeSession('tg-roll')
    })

    expect($messagingSessions.get().map(session => session.id)).toEqual(['tg-roll'])
    expect($sessions.get()).toEqual([])
    expect($pinnedSessionIds.get()).toEqual(['tg-roll'])
    expect($removedSessionIds.get().has('tg-roll')).toBe(false)
    expect($sessionMutationsInFlight.get().has('tg-roll')).toBe(false)
  })

  it('archives a messaging row against its owning profile', async () => {
    mockSetSessionArchived.mockResolvedValue({ ok: true })
    setMessagingSessions([storedSession({ id: 'tg-arch', profile: 'winefox', source: 'telegram' })])

    const handle = await readyActions()
    await act(async () => {
      await handle.archiveSession('tg-arch')
    })

    expect(mockSetSessionArchived).toHaveBeenCalledWith('tg-arch', true, 'winefox')
    expect($messagingSessions.get()).toEqual([])
  })

  it('restores a failed archive to the messaging slice', async () => {
    setMessagingSessions([storedSession({ id: 'tg-arch-fail', profile: 'winefox', source: 'telegram' })])
    mockSetSessionArchived.mockRejectedValue(new Error('archive failed'))

    const handle = await readyActions()
    await act(async () => {
      await handle.archiveSession('tg-arch-fail')
    })

    expect($messagingSessions.get().map(session => session.id)).toEqual(['tg-arch-fail'])
    expect($sessions.get()).toEqual([])
  })

  it('still DELETEs a desktop-native session with its listed profile', async () => {
    mockDeleteSession.mockResolvedValue({ ok: true })
    setSessions([storedSession({ id: 'desk-1', profile: 'default', source: 'desktop' })])

    const handle = await readyActions()
    await act(async () => {
      await handle.removeSession('desk-1')
    })

    expect(mockDeleteSession).toHaveBeenCalledWith('desk-1', 'default')
    expect($sessions.get()).toEqual([])
  })

  it('does not restore a failed cron DELETE into recents', async () => {
    setCronSessions([storedSession({ id: 'cron-1', profile: 'winefox', source: 'cron' })])
    mockDeleteSession.mockRejectedValue(new Error('cron delete failed'))

    const handle = await readyActions()
    await act(async () => {
      await handle.removeSession('cron-1')
    })

    expect($cronSessions.get().map(session => session.id)).toEqual(['cron-1'])
    expect($sessions.get()).toEqual([])
    expect($messagingSessions.get()).toEqual([])
  })

  it('restores a dual-listed messaging row to messaging, not recents', async () => {
    const row = storedSession({ id: 'tg-dual', profile: 'winefox', source: 'telegram' })
    setMessagingSessions([row])
    setSessions([row])
    mockDeleteSession.mockRejectedValue(new Error('backend down'))

    const handle = await readyActions()
    await act(async () => {
      await handle.removeSession('tg-dual')
    })

    expect($messagingSessions.get().map(session => session.id)).toEqual(['tg-dual'])
    expect($sessions.get()).toEqual([])
  })

  it('fails closed when a listed profile-less messaging DELETE cannot resolve an owner', async () => {
    const row = storedSession({ id: 'tg-unresolved', source: 'telegram', title: 'QQ/TG' })
    setMessagingSessions([row])
    $pinnedSessionIds.set(['tg-unresolved'])
    $sessionSeenCounts.set({
      winefox: { 'tg-unresolved': 3 },
      default: { 'desk-keep': 1 }
    })
    $unreadFinishedMarkers.set({
      winefox: ['tg-unresolved'],
      default: ['desk-keep']
    })
    mockGetSession.mockRejectedValue(new Error('404: Session not found'))

    const handle = await readyActions()
    await act(async () => {
      await handle.removeSession('tg-unresolved')
    })

    expect(mockDeleteSession).not.toHaveBeenCalled()
    expect($messagingSessions.get().map(session => session.id)).toEqual(['tg-unresolved'])
    expect($sessions.get()).toEqual([])
    expect($pinnedSessionIds.get()).toEqual(['tg-unresolved'])
    expect($sessionSeenCounts.get().winefox?.['tg-unresolved']).toBe(3)
    expect($unreadFinishedMarkers.get().winefox).toEqual(['tg-unresolved'])
    expect($removedSessionIds.get().has('tg-unresolved')).toBe(false)
    expect($sessionMutationsInFlight.get().has('tg-unresolved')).toBe(false)
  })

  it('fails closed when a listed profile-less messaging archive cannot resolve an owner', async () => {
    setMessagingSessions([storedSession({ id: 'tg-arch-unresolved', source: 'telegram' })])
    mockGetSession.mockRejectedValue(new Error('404: Session not found'))

    const handle = await readyActions()
    await act(async () => {
      await handle.archiveSession('tg-arch-unresolved')
    })

    expect(mockSetSessionArchived).not.toHaveBeenCalled()
    expect($messagingSessions.get().map(session => session.id)).toEqual(['tg-arch-unresolved'])
    expect($sessions.get()).toEqual([])
  })
})

// A fresh chat created through $newChatRoute must keep the route as its EXACT
// owner after session.create. The create RPC already rode
// requestGatewayForAgent(capturedRoute); but the optimistic row was stamped
// from $activeGatewayProfile (still `default` in All-profiles / Bot routing)
// and no owner hint was recorded, so the first turn ran on omar while every
// later session-scoped RPC resolved the row as `default` → "session not
// found" on the default backend, and the orphaned omar runtime was eventually
// ws-orphan-reaped.
describe('routed fresh chat keeps its exact owner across turns', () => {
  const route: SessionProfileRoute = { connectionId: 'local', mode: 'local', profile: 'omar' }
  const STORED = 'stored-omar-fresh'

  afterEach(() => {
    cleanup()
    $newChatProfile.set(null)
    $newChatRoute.set(null)
    $activeGatewayProfile.set('default')
    $sessionTiles.set([])
    setSessions([])
    setCurrentCwd('')
    setNewChatWorkspaceTarget(undefined)
    vi.restoreAllMocks()
  })

  // The SAME sync ladder contrib/wiring's requestGateway runs for every
  // session-scoped RPC (prompt.submit, session.resume, attach, interrupt,
  // redirect, recovery): tile route → exact unique hint → row owner.
  const ownerFor = (storedSessionId: string) =>
    resolveSessionRpcOwner({
      routingSessionId: storedSessionId,
      sessionOwnerHint: id => getSessionOwnerHint(id),
      sessionRowOwner: (id: string) => knownSessionOwner($sessions.get(), id),
      tileOwnerRoute: sessionTileOwnerRoute
    })

  async function createRoutedFreshChat() {
    // Ambient dispatcher = the DEFAULT backend. It never heard of the session:
    // any session-scoped RPC landing here is exactly the bug.
    const ambientRequest = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (typeof params?.session_id === 'string') {
        throw new Error(`Session not found: ${params.session_id} (ambient/default backend, ${method})`)
      }

      return {} as never
    })

    // The owning backend, local::omar. Anything else 4001s.
    vi.mocked(requestGatewayForAgent).mockImplementation(async (connectionId, profile, method) => {
      if (connectionId === 'local' && profile === 'omar') {
        if (method === 'session.create') {
          return { session_id: RUNTIME_SESSION_ID, stored_session_id: STORED } as never
        }

        return { ok: true } as never
      }

      throw new Error(`Session not found (${connectionId}::${profile}, ${method})`)
    })

    // Ambient profile = default; the draft is routed at local::omar.
    $activeGatewayProfile.set('default')
    $newChatProfile.set(route.profile)
    $newChatRoute.set({ ...route })

    let handle: HarnessHandle | null = null
    render(<Harness onReady={value => (handle = value)} requestGateway={ambientRequest} />)
    await waitFor(() => expect(handle).not.toBeNull())

    let runtimeId: null | string = null

    await act(async () => {
      runtimeId = await handle!.createBackendSessionForSend('hello omar')
    })

    expect(runtimeId).toBe(RUNTIME_SESSION_ID)

    return { ambientRequest, runtimeId: runtimeId as unknown as string }
  }

  it('unit: an explicitly routed fresh create never resolves its follow-up owner to the ambient default', async () => {
    await createRoutedFreshChat()

    const followupOwner = ownerFor(STORED)
    const foregroundScope = registryBackendScopeKey(route.connectionId, route.profile)

    const followupOwnerKey =
      followupOwner && typeof followupOwner === 'object'
        ? `${followupOwner.connectionId}::${followupOwner.profile}`
        : followupOwner

    // The exact failing shape: the foreground socket is omar's, the follow-up
    // routes to default.
    expect({ followupOwner: followupOwnerKey, foregroundScope }).not.toEqual({
      followupOwner: 'default',
      foregroundScope: 'conn:local::omar'
    })
    expect({ followupOwner: followupOwnerKey, foregroundScope }).toEqual({
      followupOwner: 'local::omar',
      foregroundScope: 'conn:local::omar'
    })

    // Both owner records carry the route, and the ambient profile never moved.
    expect(getSessionOwnerHint(STORED)).toEqual(route)
    expect($sessions.get().find(session => sessionMatchesStoredId(session, STORED))).toMatchObject({
      connection_id: 'local',
      is_default_profile: false,
      profile: 'omar'
    })
    expect($activeGatewayProfile.get()).toBe('default')
  })

  it('integration: both turns hit local::omar with default ambient — no session-not-found, no orphaned runtime', async () => {
    const { ambientRequest, runtimeId } = await createRoutedFreshChat()

    const submitTurn = (text: string) =>
      requestForSessionProfile(ownerFor(STORED), ambientRequest, 'prompt.submit', { session_id: runtimeId, text })

    // First turn on omar.
    await expect(submitTurn('first turn')).resolves.toEqual({ ok: true })

    // Keep default as the ambient profile (All-profiles / Bot routing never
    // moved it) and submit the second turn.
    $activeGatewayProfile.set('default')
    await expect(submitTurn('second turn')).resolves.toEqual({ ok: true })

    const submits = vi.mocked(requestGatewayForAgent).mock.calls.filter(call => call[2] === 'prompt.submit')

    expect(submits.map(call => [call[0], call[1], (call[3] as { text: string }).text])).toEqual([
      ['local', 'omar', 'first turn'],
      ['local', 'omar', 'second turn']
    ])
    // No session-not-found: the default backend never saw a session-scoped RPC.
    expect(ambientRequest).not.toHaveBeenCalledWith('prompt.submit', expect.anything())
    expect(ambientRequest.mock.calls.filter(call => typeof call[1]?.session_id === 'string')).toEqual([])
    // No ws_orphan_reap: the client never closed or abandoned the runtime it
    // minted on omar — no session.close on any route, the binding stands.
    expect(vi.mocked(requestGatewayForAgent).mock.calls.filter(call => call[2] === 'session.close')).toEqual([])
    expect(ambientRequest).not.toHaveBeenCalledWith('session.close', expect.anything())
    expect(getSessionOwnerHint(STORED)).toEqual(route)
  })
})
