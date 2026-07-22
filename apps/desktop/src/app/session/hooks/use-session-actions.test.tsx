import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages, type SessionInfo, setSessionArchived } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearSessionDraft, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { $activeGatewayProfile, $newChatProfile, ensureGatewayProfile } from '@/store/profile'
import { $projectScope, $projectTree, $removedSessionIds, ALL_PROJECTS } from '@/store/projects'
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
  $messagingPlatformTotals,
  $messagingSessions,
  $newChatWorkspaceTarget,
  $resumeFailedSessionId,
  $selectedStoredSessionId,
  $sessionProfileTotals,
  $sessions,
  $sessionsTotal,
  canApplySessionListResponse,
  getSessionArchiveGeneration,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setCronSessions,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setMessages,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setNewChatWorkspaceTarget,
  setResumeFailedSessionId,
  setSelectedStoredSessionId,
  setSessionProfileTotals,
  setSessions,
  setSessionsTotal
} from '@/store/session'
import { $sessionTiles, discardSessionTileForProfile, openSessionTile } from '@/store/session-states'

import { sessionRoute } from '../../routes'
import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

const RUNTIME_SESSION_ID = 'rt-new-001'

function deferred<T>() {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>((done, fail) => {
    reject = fail
    resolve = done
  })

  return { promise, reject, resolve }
}

type HarnessHandle = Pick<
  ReturnType<typeof useSessionActions>,
  'createBackendSessionForSend' | 'startFreshSessionDraft'
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

type ArchiveHandle = Pick<ReturnType<typeof useSessionActions>, 'archiveSession'>

function ArchiveHarness({
  activeSessionId = 'runtime-1',
  getRouteToken = () => 'token',
  navigate = vi.fn(),
  onReady,
  runtimeIdByStoredSessionIdRef,
  runtimeProfileByStoredSessionIdRef,
  selectedStoredSessionIdRef: selectedStoredSessionIdRefProp,
  sessionStateByRuntimeIdRef,
  selectedStoredSessionId
}: {
  activeSessionId?: null | string
  getRouteToken?: () => string
  navigate?: ReturnType<typeof vi.fn>
  onReady: (handle: ArchiveHandle) => void
  runtimeIdByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
  runtimeProfileByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
  selectedStoredSessionIdRef?: MutableRefObject<string | null>
  sessionStateByRuntimeIdRef?: MutableRefObject<Map<string, ClientSessionState>>
  selectedStoredSessionId: string
}) {
  const activeSessionIdRef: MutableRefObject<string | null> = { current: activeSessionId }

  const selectedStoredSessionIdRef: MutableRefObject<string | null> = selectedStoredSessionIdRefProp ?? {
    current: selectedStoredSessionId
  }

  const busyRef: MutableRefObject<boolean> = { current: false }
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId,
    activeSessionIdRef,
    busyRef,
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken,
    getRoutedStoredSessionId: () => selectedStoredSessionId,
    navigate: navigate as never,
    requestGateway: async () => ({}) as never,
    resetViewSync: vi.fn(),
    runtimeIdByStoredSessionIdRef:
      runtimeIdByStoredSessionIdRef ?? ref(new Map([[selectedStoredSessionId, activeSessionId ?? 'runtime-1']])),
    runtimeProfileByStoredSessionIdRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: sessionStateByRuntimeIdRef ?? ref(new Map()),
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
    runtimeIdByStoredSessionIdRef: runtimeIdByStoredSessionIdRef ?? ref(new Map<string, string>()),
    selectedStoredSessionId,
    selectedStoredSessionIdRef: ref<string | null>(selectedStoredSessionId),
    sessionStateByRuntimeIdRef: sessionStateByRuntimeIdRef ?? ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: (sessionId, updater) => {
      const next = updater({} as ClientSessionState)
      onStateUpdate?.(sessionId, next)

      return next
    }
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
    vi.mocked(getSessionMessages).mockRejectedValue(new Error('network down'))

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

    vi.mocked(getSessionMessages).mockResolvedValue({
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

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: storedMessages, session_id: 'stored-1' } as never)

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

  it('restores the in-flight turn and queued user prompt after a full renderer restart', async () => {
    const storedMessages = [
      { content: 'earlier question', role: 'user', timestamp: 1 },
      { content: 'earlier answer', role: 'assistant', timestamp: 2 }
    ]

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: storedMessages, session_id: 'stored-1' } as never)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return {
          session_id: 'runtime-1',
          session_key: 'stored-1',
          resumed: 'stored-1',
          message_count: storedMessages.length,
          messages: storedMessages,
          running: true,
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
    expect(renderedMessages).toContain('current prompt')
    expect(renderedMessages).toContain('partial answer')
    expect(renderedMessages).toContain('newest prompt')
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

    vi.mocked(getSessionMessages).mockResolvedValue({
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

    vi.mocked(getSessionMessages).mockRejectedValue(new Error('network down'))

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

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)

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

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)

    await runResume(requestGateway)

    expect(resumeParams).not.toHaveProperty('lazy')
    expect(resumeParams).not.toHaveProperty('eager_build')
    expect(resumeParams).toMatchObject({ source: 'desktop' })
  })

  it('arms the failure latch when resume succeeds with an empty transcript for a non-empty stored session', async () => {
    setSessions([storedSession({ message_count: 4 })])

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-1' } as never)

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

    vi.mocked(getSessionMessages).mockResolvedValue({
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

function BranchHarness({
  onReady,
  requestGateway
}: {
  onReady: (branchStoredSession: (storedSessionId: string, sessionProfile?: string | null) => Promise<boolean>) => void
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
    navigate: vi.fn() as never,
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
    onReady(actions.branchStoredSession)
  }, [actions.branchStoredSession, onReady])

  return null
}

describe('branchStoredSession desktop source tagging', () => {
  afterEach(() => {
    cleanup()
    setSessions([])
    vi.restoreAllMocks()
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
    vi.mocked(getSessionMessages).mockResolvedValue({
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

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)

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
    expect(resumeCalls[0][1]).toMatchObject({ session_id: 'stored-A' })

    // The corrupt mapping was purged so it can't mis-resolve again.
    expect(runtimeIdByStoredSessionIdRef.current.has('stored-A')).toBe(false)
    expect(sessionStateByRuntimeIdRef.current.has('rt-recycled')).toBe(false)
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

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-A' } as never)

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
    expect(getSessionMessages).toHaveBeenCalledWith('stored-A', undefined)
    expect(runtimeIdByStoredSessionIdRef.current.get('stored-A')).toBe('rt-A')
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

    vi.mocked(getSessionMessages).mockResolvedValue({
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

describe('archiveSession refresh races', () => {
  beforeEach(() => {
    $activeGatewayProfile.set('default')
  })

  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setCronSessions([])
    setMessages([])
    setMessagingPlatformTotals({})
    setMessagingSessions([])
    setSelectedStoredSessionId(null)
    setSessionProfileTotals({})
    setSessions([])
    setSessionsTotal(0)
    vi.restoreAllMocks()
  })

  it('blocks stale list publication while archiving the selected compression lineage', async () => {
    const continuation = storedSession({ id: 'tip-2', _lineage_root_id: 'root-1', message_count: 2 })
    const archive = deferred<{ ok: boolean }>()

    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['root-1', 'runtime-1']])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-1', createClientSessionState('root-1')]])
    }

    setSessions([continuation])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    setSelectedStoredSessionId('root-1')
    vi.mocked(setSessionArchived).mockReturnValue(archive.promise)

    let handle: ArchiveHandle | null = null
    render(
      <ArchiveHarness
        onReady={value => (handle = value)}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId="root-1"
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const pending = handle!.archiveSession('tip-2', 'default', 'desktop')
    await waitFor(() => expect($sessions.get()).toEqual([]))

    const generation = getSessionArchiveGeneration()
    expect(canApplySessionListResponse(generation)).toBe(false)
    expect($selectedStoredSessionId.get()).toBe('root-1')

    archive.resolve({ ok: true })
    await pending

    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
    expect($sessions.get()).toEqual([])
    expect($sessionsTotal.get()).toBe(0)
    expect($sessionProfileTotals.get()).toEqual({ default: 0 })
    expect($selectedStoredSessionId.get()).toBeNull()
    expect(runtimeIdByStoredSessionIdRef.current.has('root-1')).toBe(false)
    expect(sessionStateByRuntimeIdRef.current.has('runtime-1')).toBe(false)
  })

  it('does not clear a newer selection when a selected archive settles', async () => {
    const session = storedSession({ id: 'selected-archive', message_count: 2, profile: 'default', source: 'desktop' })
    const archive = deferred<{ ok: boolean }>()

    setSessions([session])
    setSessionsTotal(1)
    setSelectedStoredSessionId('selected-archive')
    vi.mocked(setSessionArchived).mockReturnValue(archive.promise)

    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'selected-archive' }
    let handle: ArchiveHandle | null = null
    render(
      <ArchiveHarness
        onReady={value => (handle = value)}
        selectedStoredSessionId="selected-archive"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const pending = handle!.archiveSession('selected-archive', 'default', 'desktop')

    setSelectedStoredSessionId('new-selection')
    selectedStoredSessionIdRef.current = 'new-selection'
    archive.resolve({ ok: true })
    await pending

    expect($selectedStoredSessionId.get()).toBe('new-selection')
  })

  it('does not clear newer route intent while selection synchronization lags', async () => {
    const session = storedSession({ id: 'selected-archive', message_count: 2, profile: 'default', source: 'desktop' })
    const archive = deferred<{ ok: boolean }>()
    const routeToken = { current: 'route-a' }

    setSessions([session])
    setSessionsTotal(1)
    setSelectedStoredSessionId('selected-archive')
    vi.mocked(setSessionArchived).mockReturnValue(archive.promise)

    let handle: ArchiveHandle | null = null
    render(
      <ArchiveHarness
        getRouteToken={() => routeToken.current}
        onReady={value => (handle = value)}
        selectedStoredSessionId="selected-archive"
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const pending = handle!.archiveSession('selected-archive', 'default', 'desktop')

    routeToken.current = 'route-b'
    archive.resolve({ ok: true })
    await pending

    expect($selectedStoredSessionId.get()).toBe('selected-archive')
  })

  it('restores the selected compression lineage and totals when archive fails', async () => {
    const continuation = storedSession({ id: 'tip-2', _lineage_root_id: 'root-1', message_count: 2 })

    const previousMessages = [
      {
        id: 'message-1',
        role: 'user' as const,
        parts: [{ type: 'text' as const, text: 'Keep me' }]
      }
    ]

    setSessions([continuation])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    setSelectedStoredSessionId('root-1')
    setMessages(previousMessages)
    vi.mocked(setSessionArchived).mockRejectedValue(new Error('boom'))

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="root-1" />)
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('tip-2', 'default', 'desktop')

    expect($selectedStoredSessionId.get()).toBe('root-1')
    expect($messages.get()).toBe(previousMessages)
    expect($sessions.get()).toEqual([continuation])
    expect($sessionsTotal.get()).toBe(1)
    expect($sessionProfileTotals.get()).toEqual({ default: 1 })
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })

  it('does not restore a failed archive into a newly active profile', async () => {
    const session = storedSession({ id: 'work-session', message_count: 2, profile: 'work', source: 'desktop' })
    const archive = deferred<{ ok: boolean }>()

    $activeGatewayProfile.set('work')
    setSessions([session])
    setSessionsTotal(1)
    setSessionProfileTotals({ work: 1 })
    setSelectedStoredSessionId('other-session')
    vi.mocked(setSessionArchived).mockReturnValue(archive.promise)

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="other-session" />)
    await waitFor(() => expect(handle).not.toBeNull())

    const pending = handle!.archiveSession('work-session', 'work', 'desktop')

    $activeGatewayProfile.set('default')
    setSessions([])
    setSessionsTotal(0)
    setSessionProfileTotals({ default: 0 })
    archive.reject(new Error('archive failed'))
    await pending

    expect($sessions.get()).toEqual([])
    expect($sessionsTotal.get()).toBe(0)
    expect($sessionProfileTotals.get()).toEqual({ default: 0 })
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })

  it('evicts the owning profile tile and runtime after switching profiles', async () => {
    const session = storedSession({ id: 'profile-tile', message_count: 2, profile: 'work', source: 'desktop' })
    const archive = deferred<{ ok: boolean }>()

    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['profile-tile', 'runtime-work']])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-work', createClientSessionState('profile-tile')]])
    }

    $activeGatewayProfile.set('work')
    discardSessionTileForProfile('profile-tile', 'work')
    setSelectedStoredSessionId('other-session')
    openSessionTile('profile-tile')
    setSessions([session])
    setSessionsTotal(1)
    setSessionProfileTotals({ work: 1 })
    vi.mocked(setSessionArchived).mockReturnValue(archive.promise)

    let handle: ArchiveHandle | null = null
    render(
      <ArchiveHarness
        onReady={value => (handle = value)}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId="other-session"
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const pending = handle!.archiveSession('profile-tile', 'work', 'desktop')

    $activeGatewayProfile.set('default')
    archive.resolve({ ok: true })
    await pending

    $activeGatewayProfile.set('work')
    expect($sessionTiles.get().some(tile => tile.storedSessionId === 'profile-tile')).toBe(false)
    expect(runtimeIdByStoredSessionIdRef.current.has('profile-tile')).toBe(false)
    expect(sessionStateByRuntimeIdRef.current.has('runtime-work')).toBe(false)
  })

  it('evicts an inactive profile runtime without touching the active profile', async () => {
    const session = storedSession({ id: 'inactive-runtime', message_count: 2, profile: 'work', source: 'desktop' })

    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['inactive-runtime', 'runtime-work']])
    }

    const runtimeProfileByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['inactive-runtime', 'work']])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-work', createClientSessionState('inactive-runtime')]])
    }

    $activeGatewayProfile.set('default')
    setSessions([session])
    setSessionsTotal(1)
    setSessionProfileTotals({ work: 1 })
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    let handle: ArchiveHandle | null = null
    render(
      <ArchiveHarness
        onReady={value => (handle = value)}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        runtimeProfileByStoredSessionIdRef={runtimeProfileByStoredSessionIdRef}
        selectedStoredSessionId="other-session"
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('inactive-runtime', 'work', 'desktop')

    expect($activeGatewayProfile.get()).toBe('default')
    expect(runtimeIdByStoredSessionIdRef.current.has('inactive-runtime')).toBe(false)
    expect(runtimeProfileByStoredSessionIdRef.current.has('inactive-runtime')).toBe(false)
    expect(sessionStateByRuntimeIdRef.current.has('runtime-work')).toBe(false)
  })

  it('preserves a runtime binding reassigned to another profile while archive is pending', async () => {
    const session = storedSession({ id: 'shared-runtime', message_count: 2, profile: 'work', source: 'desktop' })
    const archive = deferred<{ ok: boolean }>()

    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['shared-runtime', 'runtime-shared']])
    }

    const runtimeProfileByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['shared-runtime', 'work']])
    }

    const activeState = createClientSessionState('shared-runtime')

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-shared', activeState]])
    }

    $activeGatewayProfile.set('default')
    setSessions([session])
    setSessionsTotal(1)
    vi.mocked(setSessionArchived).mockReturnValue(archive.promise)

    let handle: ArchiveHandle | null = null
    render(
      <ArchiveHarness
        onReady={value => (handle = value)}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        runtimeProfileByStoredSessionIdRef={runtimeProfileByStoredSessionIdRef}
        selectedStoredSessionId="other-session"
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const pending = handle!.archiveSession('shared-runtime', 'work', 'desktop')

    runtimeProfileByStoredSessionIdRef.current.set('shared-runtime', 'default')
    archive.resolve({ ok: true })
    await pending

    expect(runtimeIdByStoredSessionIdRef.current.get('shared-runtime')).toBe('runtime-shared')
    expect(runtimeProfileByStoredSessionIdRef.current.get('shared-runtime')).toBe('default')
    expect(sessionStateByRuntimeIdRef.current.get('runtime-shared')).toBe(activeState)
  })

  it('restores totals when a row subscriber throws during backend rollback', async () => {
    const session = storedSession({
      id: 'rollback-subscriber',
      message_count: 2,
      profile: 'default',
      source: 'desktop'
    })

    const archive = deferred<{ ok: boolean }>()

    setSessions([session])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    vi.mocked(setSessionArchived).mockReturnValue(archive.promise)

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="other-session" />)
    await waitFor(() => expect(handle).not.toBeNull())

    const pending = handle!.archiveSession('rollback-subscriber', 'default', 'desktop')
    await waitFor(() => expect($sessions.get()).toEqual([]))

    const unlisten = $sessions.listen(() => {
      throw new Error('rollback subscriber boom')
    })

    try {
      archive.reject(new Error('archive failed'))
      await pending
    } finally {
      unlisten()
    }

    expect($sessions.get()).toEqual([session])
    expect($sessionsTotal.get()).toBe(1)
    expect($sessionProfileTotals.get()).toEqual({ default: 1 })
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })

  it('releases the archive barrier after a synchronous mutation failure', async () => {
    const session = storedSession({ id: 'sync-failure', message_count: 2, profile: 'default', source: 'desktop' })

    setSessions([session])
    setSessionsTotal(1)
    vi.mocked(setSessionArchived).mockImplementation(() => {
      throw new Error('sync boom')
    })

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="other-session" />)
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('sync-failure', 'default', 'desktop')

    expect($sessions.get()).toEqual([session])
    expect($sessionsTotal.get()).toBe(1)
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })

  it('releases the archive barrier when optimistic rollback subscribers throw', async () => {
    const session = storedSession({ id: 'subscriber-failure', message_count: 2, profile: 'default', source: 'desktop' })

    setSessions([session])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    const removedBefore = new Set($removedSessionIds.get())

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="other-session" />)
    await waitFor(() => expect(handle).not.toBeNull())

    const unlisten = $sessions.listen(() => {
      throw new Error('subscriber boom')
    })

    try {
      await handle!.archiveSession('subscriber-failure', 'default', 'desktop')
    } finally {
      unlisten()
    }

    expect($sessions.get()).toEqual([session])
    expect($sessionsTotal.get()).toBe(1)
    expect($sessionProfileTotals.get()).toEqual({ default: 1 })
    expect($removedSessionIds.get()).toEqual(removedBefore)
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })

  it('restores messaging totals when a row subscriber throws during backend rollback', async () => {
    const session = storedSession({ id: 'telegram-rollback', message_count: 2, profile: 'work', source: 'telegram' })
    const archive = deferred<{ ok: boolean }>()

    $activeGatewayProfile.set('work')
    setMessagingSessions([session])
    setMessagingPlatformTotals({ telegram: 1 })
    vi.mocked(setSessionArchived).mockReturnValue(archive.promise)

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="other-session" />)
    await waitFor(() => expect(handle).not.toBeNull())

    const pending = handle!.archiveSession('telegram-rollback', 'work', 'telegram')
    await waitFor(() => expect($messagingSessions.get()).toEqual([]))

    const unlisten = $messagingSessions.listen(() => {
      throw new Error('messaging rollback subscriber boom')
    })

    try {
      archive.reject(new Error('archive failed'))
      await pending
    } finally {
      unlisten()
    }

    expect($messagingSessions.get()).toEqual([session])
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 1 })
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })

  it('removes an archived messaging session from its platform bucket', async () => {
    const telegram = storedSession({ id: 'telegram-1', message_count: 2, profile: 'work', source: 'telegram' })

    setMessagingSessions([telegram])
    setMessagingPlatformTotals({ telegram: 1 })
    setSelectedStoredSessionId('telegram-1')
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="telegram-1" />)
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('telegram-1', 'work', 'telegram')

    expect(setSessionArchived).toHaveBeenCalledWith('telegram-1', true, 'work')
    expect($messagingSessions.get()).toEqual([])
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 0 })
  })

  it('preserves an unknown messaging total while archiving', async () => {
    const telegram = storedSession({ id: 'telegram-unknown', message_count: 2, profile: 'work', source: 'telegram' })

    setMessagingSessions([telegram])
    setMessagingPlatformTotals({})
    setSelectedStoredSessionId('telegram-unknown')
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="telegram-unknown" />)
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('telegram-unknown', 'work', 'telegram')

    expect($messagingSessions.get()).toEqual([])
    expect($messagingPlatformTotals.get()).toEqual({})
  })

  it('removes only the matching profile when a messaging bucket contains duplicate ids', async () => {
    const defaultRow = storedSession({
      id: 'same-message-id',
      message_count: 2,
      profile: 'default',
      source: 'telegram'
    })

    const workRow = storedSession({ id: 'same-message-id', message_count: 2, profile: 'work', source: 'telegram' })

    setMessagingSessions([defaultRow, workRow])
    setMessagingPlatformTotals({ telegram: 2 })
    setSelectedStoredSessionId('same-message-id')
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="same-message-id" />)
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('same-message-id', 'work', 'telegram')

    expect(setSessionArchived).toHaveBeenCalledWith('same-message-id', true, 'work')
    expect($messagingSessions.get()).toEqual([defaultRow])
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 1 })
  })

  it('removes an archived cron session from the cron bucket', async () => {
    const cron = storedSession({ id: 'cron-1', message_count: 2, profile: 'work', source: 'cron' })

    setCronSessions([cron])
    setSelectedStoredSessionId('cron-1')
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="cron-1" />)
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('cron-1', 'work', 'cron')

    expect(setSessionArchived).toHaveBeenCalledWith('cron-1', true, 'work')
    expect($cronSessions.get()).toEqual([])
  })

  it('does not touch a loaded sibling when a profile-bound tile row is paged out', async () => {
    const defaultRow = storedSession({ id: 'paged-out-id', message_count: 2, profile: 'default', source: 'desktop' })

    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['paged-out-id', 'runtime-work']])
    }

    $activeGatewayProfile.set('work')
    setSessions([defaultRow])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    setSelectedStoredSessionId('other-session')
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    let handle: ArchiveHandle | null = null
    render(
      <ArchiveHarness
        onReady={value => (handle = value)}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId="other-session"
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('paged-out-id', 'work')

    expect(setSessionArchived).toHaveBeenCalledWith('paged-out-id', true, 'work')
    expect($sessions.get()).toEqual([defaultRow])
    expect($sessionsTotal.get()).toBe(1)
    expect($sessionProfileTotals.get()).toEqual({ default: 1 })
    expect(runtimeIdByStoredSessionIdRef.current.has('paged-out-id')).toBe(false)
  })

  it('uses profile and source to disambiguate identical sidebar ids', async () => {
    const local = storedSession({ id: 'same-id', message_count: 2, profile: 'default', source: 'desktop' })
    const telegram = storedSession({ id: 'same-id', message_count: 2, profile: 'work', source: 'telegram' })

    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['same-id', 'runtime-default']])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-default', createClientSessionState('same-id')]])
    }

    setSessions([local])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    setMessagingSessions([telegram])
    setMessagingPlatformTotals({ telegram: 1 })
    setSelectedStoredSessionId('same-id')
    vi.mocked(setSessionArchived).mockResolvedValue({ ok: true })

    let handle: ArchiveHandle | null = null
    render(
      <ArchiveHarness
        onReady={value => (handle = value)}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId="same-id"
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.archiveSession('same-id', 'work', 'telegram')

    expect(setSessionArchived).toHaveBeenCalledWith('same-id', true, 'work')
    expect($sessions.get()).toEqual([local])
    expect($sessionsTotal.get()).toBe(1)
    expect($sessionProfileTotals.get()).toEqual({ default: 1 })
    expect($messagingSessions.get()).toEqual([])
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 0 })
    expect($selectedStoredSessionId.get()).toBe('same-id')
    expect(runtimeIdByStoredSessionIdRef.current.get('same-id')).toBe('runtime-default')
    expect(sessionStateByRuntimeIdRef.current.has('runtime-default')).toBe(true)
  })

  it('does not let an older failed archive roll back a newer success', async () => {
    const session = storedSession({ id: 'same-target', message_count: 2, profile: 'default', source: 'desktop' })
    const first = deferred<{ ok: boolean }>()
    const second = deferred<{ ok: boolean }>()

    setSessions([session])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    setSelectedStoredSessionId('other-session')
    vi.mocked(setSessionArchived).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="other-session" />)
    await waitFor(() => expect(handle).not.toBeNull())

    const older = handle!.archiveSession('same-target', 'default', 'desktop')
    const newer = handle!.archiveSession('same-target', 'default', 'desktop')

    second.resolve({ ok: true })
    await newer
    first.reject(new Error('older request failed'))
    await older

    expect($sessions.get()).toEqual([])
    expect($sessionsTotal.get()).toBe(0)
    expect($sessionProfileTotals.get()).toEqual({ default: 0 })
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })

  it('does not let a newer failed archive roll back an older success', async () => {
    const session = storedSession({ id: 'same-target', message_count: 2, profile: 'default', source: 'desktop' })
    const first = deferred<{ ok: boolean }>()
    const second = deferred<{ ok: boolean }>()

    setSessions([session])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    setSelectedStoredSessionId('other-session')
    vi.mocked(setSessionArchived).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="other-session" />)
    await waitFor(() => expect(handle).not.toBeNull())

    const older = handle!.archiveSession('same-target', 'default', 'desktop')
    const newer = handle!.archiveSession('same-target', 'default', 'desktop')

    first.resolve({ ok: true })
    await older
    second.reject(new Error('newer request failed'))
    await newer

    expect($sessions.get()).toEqual([])
    expect($sessionsTotal.get()).toBe(0)
    expect($sessionProfileTotals.get()).toEqual({ default: 0 })
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })

  it('shares concurrent archive ownership across lineage and source aliases', async () => {
    const session = storedSession({
      id: 'lineage-tip',
      _lineage_root_id: 'lineage-root',
      message_count: 2,
      profile: 'default',
      source: 'desktop'
    })

    const first = deferred<{ ok: boolean }>()
    const second = deferred<{ ok: boolean }>()

    setSessions([session])
    setSessionsTotal(1)
    setSessionProfileTotals({ default: 1 })
    setSelectedStoredSessionId('other-session')
    vi.mocked(setSessionArchived).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)

    let handle: ArchiveHandle | null = null
    render(<ArchiveHarness onReady={value => (handle = value)} selectedStoredSessionId="other-session" />)
    await waitFor(() => expect(handle).not.toBeNull())

    const older = handle!.archiveSession('lineage-tip', 'default', 'desktop')
    const newer = handle!.archiveSession('lineage-root', 'default')

    second.resolve({ ok: true })
    await newer
    first.reject(new Error('older request failed'))
    await older

    expect($sessions.get()).toEqual([])
    expect($sessionsTotal.get()).toBe(0)
    expect($sessionProfileTotals.get()).toEqual({ default: 0 })
    expect(canApplySessionListResponse(getSessionArchiveGeneration())).toBe(true)
  })
})
