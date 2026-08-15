import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { noteActiveTreeGroup, revealTreePane } from '@/components/pane-shell/tree/store'
import {
  getAllSessionMessages,
  getLatestSessionMessages,
  getSession,
  type SessionInfo,
  type SessionResumeResponse
} from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearSessionDraft, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { $activeGatewayProfile, $newChatProfile, ensureGatewayProfile } from '@/store/profile'
import { $projectScope, $projectTree, ALL_PROJECTS } from '@/store/projects'
import {
  $activeSessionId,
  $activeSessionStoredIdRotation,
  $currentCwd,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $messages,
  $newChatWorkspaceTarget,
  $resumeFailedSessionId,
  $selectedStoredSessionId,
  $turnStartedAt,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setAwaitingResponse,
  setBusy,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setMessages,
  setNewChatWorkspaceTarget,
  setResumeFailedSessionId,
  setSelectedStoredSessionId,
  setSessions,
  setTurnStartedAt
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import sessionResumeActiveTurn from '../../../../../../tests/fixtures/session-resume-active-turn.json'
import { sessionRoute } from '../../routes'
import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'
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
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  noteActiveTreeGroup: vi.fn(),
  revealTreePane: vi.fn()
}))

const RUNTIME_SESSION_ID = 'rt-new-001'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

type HarnessHandle = Pick<
  ReturnType<typeof useSessionActions>,
  'createBackendSessionForSend' | 'selectSidebarItem' | 'startFreshSessionDraft'
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
  navigate = vi.fn(),
  onReady,
  requestGateway
}: {
  navigate?: ReturnType<typeof vi.fn>
  onReady: (handle: HarnessHandle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: navigate as never,
    requestGateway,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

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
    $activeGatewayProfile.set('default')
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    $currentCwd.set('')
    $currentFastMode.set(false)
    $currentModel.set('')
    $currentProvider.set('')
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

  it('passes the current workspace cwd into session.create', async () => {
    const params = await createWith(() => {
      $currentCwd.set('/remote/worktree')
    })

    expect(params).toMatchObject({ cwd: '/remote/worktree' })
  })

  it('freezes the visible selector state before profile readiness and sends fast: false explicitly', async () => {
    const profileReady = deferred<void>()
    vi.mocked(ensureGatewayProfile).mockReturnValueOnce(profileReady.promise)

    setCurrentModel('anthropic/claude-sonnet-4.6')
    setCurrentProvider('anthropic')
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
  onStateUpdate,
  onReady,
  requestGateway,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId = null,
  sessionStateByRuntimeIdRef
}: {
  onStateUpdate?: (sessionId: string, state: ClientSessionState) => void
  onReady: (resume: (storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  runtimeIdByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
  selectedStoredSessionId?: string | null
  sessionStateByRuntimeIdRef?: MutableRefObject<Map<string, ClientSessionState>>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })
  const runtimeMapRef = runtimeIdByStoredSessionIdRef ?? ref(new Map<string, string>())
  const stateMapRef = sessionStateByRuntimeIdRef ?? ref(new Map<string, ClientSessionState>())

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef: runtimeMapRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef: ref<string | null>(selectedStoredSessionId),
    sessionStateByRuntimeIdRef: stateMapRef,
    syncSessionStateToView: vi.fn(),
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
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setResumeFailedSessionId(null)
    setMessages([])
    setSessions([])
    vi.restoreAllMocks()
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

    await runResume(requestGateway)

    // Arming here would auto-retry a window that already shows history and,
    // on exhaustion, blank that transcript behind the error overlay — a
    // regression vs. plain fallback-success. The latch must stay clear.
    expect($resumeFailedSessionId.get()).toBeNull()
    // The fallback transcript is visible.
    expect($messages.get().length).toBeGreaterThan(0)
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
    expect($activeSessionId.get()).toBe('runtime-1')
    expect($messages.get().length).toBe(1)
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

  it('opens the branch as a new tab and leaves the parent chat selected', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.create') {
        return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as never
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
      />
    )
    await waitFor(() => expect(branchStoredSession).not.toBeNull())

    await expect(branchStoredSession!('stored-parent')).resolves.toBe(true)

    // The branch opened as its own tab...
    expect($sessionTiles.get().some(tile => tile.storedSessionId === 'branch-stored')).toBe(true)
    // ...without stealing the primary selection or navigating away from the parent.
    expect($selectedStoredSessionId.get()).toBe('stored-parent')
    expect(navigate).not.toHaveBeenCalledWith(sessionRoute('branch-stored'))
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
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setResumeFailedSessionId(null)
    setMessages([])
    setSessions([])
    vi.restoreAllMocks()
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
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-A', undefined)

    // The corrupt mapping was purged so it can't mis-resolve again.
    expect(runtimeIdByStoredSessionIdRef.current.has('stored-A')).toBe(false)
    expect(sessionStateByRuntimeIdRef.current.has('rt-recycled')).toBe(false)
  })

  it('paints the bounded latest transcript after the deferred resume acknowledgement', async () => {
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
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-A', undefined)
    expect($messages.get()).toHaveLength(0)
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
    expect($messages.get()).toHaveLength(500)
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
})

describe('createBackendSessionForSend workspace target', () => {
  afterEach(() => {
    cleanup()
    $newChatProfile.set(null)
    $activeGatewayProfile.set('default')
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
