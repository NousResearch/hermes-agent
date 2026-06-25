import { cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import { $currentCwd, $messages, $resumeFailedSessionId, setMessages, setResumeFailedSessionId } from '@/store/session'

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

const RUNTIME_SESSION_ID = 'rt-new-001'

function Harness({
  onReady,
  requestGateway
}: {
  onReady: (create: (preview?: string | null) => Promise<string | null>) => void
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
    navigate: vi.fn() as never,
    requestGateway,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions.createBackendSessionForSend)
  }, [actions.createBackendSessionForSend, onReady])

  return null
}

async function createWith(profileSetup: () => void): Promise<Record<string, unknown> | undefined> {
  let createParams: Record<string, unknown> | undefined

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.create') {
      createParams = params

      return { session_id: RUNTIME_SESSION_ID, stored_session_id: null } as never
    }

    return {} as never
  })

  $currentCwd.set('')
  profileSetup()

  let create: ((preview?: string | null) => Promise<string | null>) | null = null
  render(<Harness onReady={c => (create = c)} requestGateway={requestGateway} />)
  await waitFor(() => expect(create).not.toBeNull())
  await create!()

  return createParams
}

describe('createBackendSessionForSend profile routing', () => {
  afterEach(() => {
    cleanup()
    $newChatProfile.set(null)
    $activeGatewayProfile.set('default')
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
})

// ── Resume failure recovery (the "stuck loading session window" bug) ──────────
// When session.resume rejects AND the REST transcript fallback ALSO fails, the
// hook must (a) not throw out of the fallback (which stranded the loader), and
// (b) arm $resumeFailedSessionId so use-route-resume can retry. A resume that
// succeeds must NOT leave the flag armed.
function ResumeHarness({
  onReady,
  requestGateway
}: {
  onReady: (resume: (storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) => void
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
    navigate: vi.fn() as never,
    requestGateway,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: (_sessionId, updater) => updater({} as ClientSessionState)
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

describe('resumeSession failure recovery', () => {
  afterEach(() => {
    cleanup()
    setResumeFailedSessionId(null)
    setMessages([])
    vi.restoreAllMocks()
  })

  async function runResume(
    requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  ): Promise<void> {
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={r => (resume = r)} requestGateway={requestGateway} />)
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
  })
})

interface CacheHarnessProps {
  onReady: (resume: (storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
}

// Harness that lets the test own the two cache maps so it can pre-seed a
// cross-wired runtime-id mapping and observe whether the warm fast-path trusts
// it. Mirrors the production wiring from use-session-state-cache.
function CacheHarness({
  onReady,
  requestGateway,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionIdRef,
  sessionStateByRuntimeIdRef
}: CacheHarnessProps) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    navigate: vi.fn() as never,
    requestGateway,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    syncSessionStateToView: vi.fn(),
    updateSessionState: (_sessionId, updater) => updater({} as ClientSessionState)
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

const clientState = (storedSessionId: string | null): ClientSessionState => createClientSessionState(storedSessionId)

describe('resumeSession warm-cache mapping integrity', () => {
  afterEach(() => {
    cleanup()
    setResumeFailedSessionId(null)
    setMessages([])
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

    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'rt-A-fresh', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <CacheHarness
        onReady={r => (resume = r)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
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

  it('honours a warm cache entry whose stored id matches (no needless refetch)', async () => {
    // Correctly-wired mapping: 'rt-A' <-> 'stored-A'. The fast-path should trust
    // it and never reach session.resume (only the lightweight usage probe).
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-A', 'rt-A']])
    }

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['rt-A', clientState('stored-A')]])
    }

    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.usage') {
        return { input: 0, output: 0, total: 0 } as never
      }

      return {} as never
    })

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <CacheHarness
        onReady={r => (resume = r)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-A', true)

    // Fast-path served the session from cache: no full resume RPC, mapping intact.
    const methods = requestGateway.mock.calls.map(([method]) => method)
    expect(methods).not.toContain('session.resume')
    expect(runtimeIdByStoredSessionIdRef.current.get('stored-A')).toBe('rt-A')
  })
})
