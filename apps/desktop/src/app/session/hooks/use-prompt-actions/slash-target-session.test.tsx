import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'
import { $yoloActive, setSessions, setYoloActive } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import type { SubmitTextOptions } from './utils'

import { usePromptActions } from '.'

vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

// The scenario every assertion below is built on: the user streamed in chat A,
// then clicked chat B in the sidebar (or swapped profile / the gateway
// reconnected). The durable route names B, but the runtime ref still points at
// A's runtime session and the cache cannot confirm a runtime for B — exactly
// `resolveTargetSessionId`'s "routed needs resume" rung.
const RUNTIME_A = 'rt-chat-a'
const RUNTIME_B = 'rt-chat-b'
const STORED_B = 'stored-chat-b'

interface GatewayCall {
  method: string
  params?: Record<string, unknown>
}

function storedSessionRow(): SessionInfo {
  return {
    ended_at: null,
    id: STORED_B,
    input_tokens: 0,
    is_active: true,
    last_active: 0,
    message_count: 3,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: 'Chat B',
    tool_call_count: 0
  }
}

interface HarnessHandle {
  submitText: (text: string, options?: SubmitTextOptions) => Promise<boolean>
}

interface HarnessProps {
  activeSessionId?: null | string
  createBackendSessionForSend?: (preview?: null | string) => Promise<null | string>
  onReady: (handle: HarnessHandle) => void
  onUpdateState?: (sessionId: string) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>
  routedStoredSessionId?: null | string
  runtimeIdForStoredSession?: null | string
  selectedStoredSessionId?: null | string
}

function Harness({
  activeSessionId = RUNTIME_A,
  createBackendSessionForSend,
  onReady,
  onUpdateState,
  requestGateway,
  routedStoredSessionId = STORED_B,
  runtimeIdForStoredSession = null,
  selectedStoredSessionId = STORED_B
}: HarnessProps) {
  const activeSessionIdRef: MutableRefObject<null | string> = { current: activeSessionId }
  const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: selectedStoredSessionId }
  const busyRef: MutableRefObject<boolean> = { current: false }

  const actions = usePromptActions({
    activeSessionId,
    activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef,
    createBackendSessionForSend: createBackendSessionForSend ?? (async () => RUNTIME_A),
    getRoutedStoredSessionId: () => routedStoredSessionId,
    getRouteToken: () => 'token',
    getRuntimeIdForStoredSession: () => runtimeIdForStoredSession,
    handleSkinCommand: () => 'Skin set to midnight.',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: () => undefined,
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState: (sessionId, updater) => {
      onUpdateState?.(sessionId)

      return updater({ messages: [], busy: false, awaitingResponse: false, interrupted: false } as never)
    }
  })

  useEffect(() => {
    onReady({
      submitText: (...args: Parameters<typeof actions.submitText>) =>
        act(async () => actions.submitText(...args)) as Promise<boolean>
    })
  }, [actions.submitText, onReady])

  return null
}

async function actRender(ui: React.ReactElement) {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(ui)
  })

  return result!
}

describe('slash commands that act on an existing session resolve it through the shared ladder', () => {
  beforeEach(() => {
    // Seed the sidebar row so the resolver's profile lookup resolves from cache
    // instead of probing the backend.
    setSessions(() => [storedSessionRow()])
    setYoloActive(false)
    clearNotifications()
  })

  afterEach(() => {
    cleanup()
    setSessions(() => [])
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('writes /yolo to the routed chat, not the stale runtime ref', async () => {
    const calls: GatewayCall[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        return { session_id: RUNTIME_B } as never
      }

      return { value: '1' } as never
    })

    let handle: HarnessHandle | null = null
    await actRender(<Harness onReady={h => (handle = h)} requestGateway={requestGateway} />)

    await handle!.submitText('/yolo')

    // The approval bypass is a per-session safety flag. Landing it on the chat
    // the user left leaves that conversation auto-approving tool calls while
    // the visible one keeps prompting.
    expect(calls).toContainEqual({
      method: 'config.set',
      params: { key: 'yolo', session_id: RUNTIME_B, value: '1' }
    })
    expect(calls.some(call => call.params?.session_id === RUNTIME_A)).toBe(false)
    expect(calls.some(call => call.method === 'session.resume')).toBe(true)
  })

  it('hands off the routed chat, not the stale runtime ref', async () => {
    vi.useFakeTimers()
    const calls: GatewayCall[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        return { session_id: RUNTIME_B } as never
      }

      if (method === 'handoff.state') {
        return { state: 'completed' } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(<Harness onReady={h => (handle = h)} requestGateway={requestGateway} />)

    const pending = handle!.submitText('/handoff telegram')
    await vi.advanceTimersByTimeAsync(2_000)
    await pending

    // A handoff leaves the app — the wrong target ships another conversation's
    // transcript to Telegram/Discord.
    expect(calls).toContainEqual({
      method: 'handoff.request',
      params: { platform: 'telegram', session_id: RUNTIME_B }
    })
    expect(calls.some(call => call.method === 'handoff.request' && call.params?.session_id === RUNTIME_A)).toBe(false)
  })

  it('prints /skin confirmation into the routed chat, not the stale runtime ref', async () => {
    const updated: string[] = []

    const requestGateway = vi.fn(
      async (method: string) => (method === 'session.resume' ? { session_id: RUNTIME_B } : {}) as never
    )

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} onUpdateState={id => updated.push(id)} requestGateway={requestGateway} />
    )

    await handle!.submitText('/skin midnight')

    expect(updated).toContain(RUNTIME_B)
    expect(updated).not.toContain(RUNTIME_A)
  })

  it('still arms /yolo locally on a new-chat draft instead of minting a session', async () => {
    const createBackendSessionForSend = vi.fn(async () => 'rt-should-not-exist')
    const calls: GatewayCall[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        createBackendSessionForSend={createBackendSessionForSend}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
        routedStoredSessionId={null}
        selectedStoredSessionId={null}
      />
    )

    await handle!.submitText('/yolo')

    // No durable conversation is in play, so `/yolo` keeps its deliberate
    // local-arm behaviour; the session-create path applies it on the first
    // message. Spinning up a backend session just to hold the flag would leave
    // an empty chat in the sidebar.
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(calls).toEqual([])
    expect($yoloActive.get()).toBe(true)
    expect($notifications.get().length).toBe(1)
  })
})
