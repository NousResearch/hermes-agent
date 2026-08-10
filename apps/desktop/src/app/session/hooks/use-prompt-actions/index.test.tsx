import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSession } from '@/hermes'
import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $composerAttachments, $composerDraft, type ComposerAttachment, setComposerDraft } from '@/store/composer'
import { $queuedPromptsBySession, getQueuedPrompts } from '@/store/composer-queue'
import { $hudMode } from '@/store/hud'
import { $notifications, clearNotifications } from '@/store/notifications'
import {
  $busy,
  $connection,
  $currentCwd,
  $currentUsage,
  $messages,
  $sessions,
  $terminalBackend,
  $turnStartedAt,
  setCurrentUsage,
  setMessages,
  setSessions
} from '@/store/session'
import { dropSessionState, publishSessionState } from '@/store/session-states'
import { $wakeWord, resetWakeWordState } from '@/store/wake-word'
import type { SessionInfo } from '@/types/hermes'

import type { SubmitTextOptions } from './utils'

import { uploadComposerAttachment, usePromptActions } from '.'

vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

// The active id the desktop holds is the *runtime* session id from
// session.create — deliberately distinct from the stored DB id here, because
// that mismatch is the bug: the REST renameSession endpoint resolves against
// the stored sessions table and 404s on a runtime id. session.title accepts
// the runtime id directly.
const RUNTIME_SESSION_ID = 'rt-abc123'

function sessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: RUNTIME_SESSION_ID,
    input_tokens: 0,
    is_active: true,
    last_active: 0,
    message_count: 3,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: 'Old title',
    tool_call_count: 0,
    ...overrides
  }
}

// Wrap render() in act() so the Harness's useEffect (onReady callback +
// internal state from usePromptActions) flushes synchronously instead of
// spilling async state updates outside act().
async function actRender(ui: React.ReactElement) {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(ui)
  })

  return result!
}

interface HarnessHandle {
  activeSessionIdRef: MutableRefObject<string | null>
  cancelRun: () => Promise<void>
  editMessage: (edited: Parameters<ReturnType<typeof usePromptActions>['editMessage']>[0]) => Promise<void>
  reloadFromMessage: (parentId: null | string) => Promise<void>
  restoreToMessage: (messageId: string, target?: { text?: string; userOrdinal?: number | null }) => Promise<void>
  redirectPrompt: (text: string) => Promise<boolean>
  /** @deprecated Use `redirectPrompt`. */
  steerPrompt: (text: string) => Promise<boolean>
  submitTextRaw: (text: string, options?: SubmitTextOptions) => Promise<boolean>
  submitText: (text: string, options?: SubmitTextOptions) => Promise<boolean>
}

function Harness({
  activeSessionIdRef: activeSessionIdRefProp,
  busyRef,
  getRoutedStoredSessionId,
  getRuntimeIdForStoredSession,
  getRouteToken,
  onUpdateState,
  onReady,
  onSeedState,
  openMemoryGraph,
  refreshSessions,
  requestGateway,
  resumeStoredSession,
  seedMessages,
  selectedStoredSessionIdRef: selectedStoredSessionIdRefProp,
  storedSessionId,
  activeSessionId,
  createBackendSessionForSend
}: {
  activeSessionIdRef?: MutableRefObject<string | null>
  busyRef?: MutableRefObject<boolean>
  getRoutedStoredSessionId?: () => null | string
  getRuntimeIdForStoredSession?: (storedSessionId: string) => null | string
  getRouteToken?: () => string
  onUpdateState?: (
    sessionId: string,
    storedSessionId: null | string | undefined,
    state: Record<string, unknown>
  ) => void
  onReady: (handle: HarnessHandle) => void
  onSeedState?: (state: Record<string, unknown>) => void
  openMemoryGraph?: () => void
  refreshSessions: () => Promise<void>
  requestGateway: <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>
  resumeStoredSession?: (storedSessionId: string) => Promise<void> | void
  seedMessages?: unknown[]
  selectedStoredSessionIdRef?: MutableRefObject<string | null>
  storedSessionId?: null | string
  activeSessionId?: null | string
  createBackendSessionForSend?: (preview?: null | string) => Promise<null | string>
}) {
  const localActiveSessionIdRef = useRef<string | null>(
    activeSessionId === undefined ? RUNTIME_SESSION_ID : activeSessionId
  )

  const activeSessionIdRef = activeSessionIdRefProp ?? localActiveSessionIdRef

  const selectedStoredSessionIdRef: MutableRefObject<string | null> = selectedStoredSessionIdRefProp ?? {
    current: storedSessionId === undefined ? RUNTIME_SESSION_ID : storedSessionId
  }

  const localBusyRef = busyRef ?? { current: false }

  const stateRef = useRef({
    messages: seedMessages ?? [],
    busy: false,
    awaitingResponse: false,
    interrupted: true
  } as never)

  const actions = usePromptActions({
    activeSessionId: activeSessionId === undefined ? RUNTIME_SESSION_ID : activeSessionId,
    activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef: localBusyRef,
    createBackendSessionForSend: createBackendSessionForSend ?? (async () => RUNTIME_SESSION_ID),
    getRoutedStoredSessionId: getRoutedStoredSessionId ?? (() => null),
    getRuntimeIdForStoredSession: getRuntimeIdForStoredSession ?? (() => null),
    getRouteToken: getRouteToken ?? (() => 'token'),
    handleSkinCommand: () => '',
    openMemoryGraph: openMemoryGraph ?? (() => undefined),
    refreshSessions,
    requestGateway,
    resumeStoredSession: resumeStoredSession ?? (() => undefined),
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState: (sessionId, updater, storedSessionId) => {
      // Seed with interrupted:true so we can prove a fresh submit clears it.
      const next = updater(stateRef.current) as unknown as Record<string, unknown>
      stateRef.current = next as never
      onSeedState?.(next)
      onUpdateState?.(sessionId, storedSessionId, next)

      return next as never
    }
  })

  useEffect(() => {
    onReady({
      activeSessionIdRef,
      cancelRun: (...args: Parameters<typeof actions.cancelRun>) =>
        act(async () => actions.cancelRun(...args)) as Promise<void>,
      editMessage: (...args: Parameters<typeof actions.editMessage>) =>
        act(async () => actions.editMessage(...args)) as Promise<void>,
      reloadFromMessage: (...args: Parameters<typeof actions.reloadFromMessage>) =>
        act(async () => actions.reloadFromMessage(...args)) as Promise<void>,
      restoreToMessage: (...args: Parameters<typeof actions.restoreToMessage>) =>
        act(async () => actions.restoreToMessage(...args)) as Promise<void>,
      redirectPrompt: (...args: Parameters<typeof actions.redirectPrompt>) =>
        act(async () => actions.redirectPrompt(...args)) as Promise<boolean>,
      steerPrompt: (...args: Parameters<typeof actions.steerPrompt>) =>
        act(async () => actions.steerPrompt(...args)) as Promise<boolean>,
      submitTextRaw: actions.submitText,
      submitText: (...args: Parameters<typeof actions.submitText>) =>
        act(async () => actions.submitText(...args)) as Promise<boolean>
    })
  }, [
    actions.cancelRun,
    actions.editMessage,
    actions.reloadFromMessage,
    actions.restoreToMessage,
    actions.redirectPrompt,
    actions.steerPrompt,
    actions.submitText,
    activeSessionIdRef,
    onReady
  ])

  return null
}

describe('usePromptActions /title', () => {
  beforeEach(() => {
    setSessions(() => [sessionInfo()])
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renames via the session.title RPC (with the runtime id), updates the sidebar store, and refreshes', async () => {
    const refreshSessions = vi.fn(async () => undefined)

    const requestGateway = vi.fn(
      async (method: string) => (method === 'session.title' ? { pending: false, title: 'New title' } : {}) as never
    )

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={refreshSessions} requestGateway={requestGateway} />
    )

    await handle!.submitText('/title New title')

    // Routes through session.title with the runtime session id — NOT the slash
    // worker (slash.exec) and NOT the REST endpoint. This is the path that
    // resolves the runtime id and persists reliably across platforms.
    expect(requestGateway).toHaveBeenCalledWith('session.title', {
      session_id: RUNTIME_SESSION_ID,
      title: 'New title'
    })
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
    expect(refreshSessions).toHaveBeenCalledTimes(1)
    expect($sessions.get()[0]?.title).toBe('New title')
  })

  it('reports the queued state when the session row is not persisted yet', async () => {
    const refreshSessions = vi.fn(async () => undefined)

    const requestGateway = vi.fn(
      async (method: string) => (method === 'session.title' ? { pending: true, title: 'Fresh chat' } : {}) as never
    )

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={refreshSessions} requestGateway={requestGateway} />
    )

    await handle!.submitText('/title Fresh chat')

    expect(requestGateway).toHaveBeenCalledWith('session.title', {
      session_id: RUNTIME_SESSION_ID,
      title: 'Fresh chat'
    })
    // Even when queued, the sidebar reflects the chosen title optimistically.
    expect(refreshSessions).toHaveBeenCalledTimes(1)
    expect($sessions.get()[0]?.title).toBe('Fresh chat')
  })

  it('falls through to the slash worker for a bare /title (show current title)', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    const requestGateway = vi.fn(async () => ({ output: 'Title: Old title' }) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={refreshSessions} requestGateway={requestGateway} />
    )

    await handle!.submitText('/title')

    expect(requestGateway).not.toHaveBeenCalledWith('session.title', expect.anything())
    expect(requestGateway).toHaveBeenCalledWith('slash.exec', expect.objectContaining({ command: 'title' }))
  })

  it('surfaces a rename error without touching the sidebar store', async () => {
    const refreshSessions = vi.fn(async () => undefined)

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.title') {
        throw new Error('Title too long')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={refreshSessions} requestGateway={requestGateway} />
    )

    await handle!.submitText('/title way too long title')

    expect(requestGateway).toHaveBeenCalledWith(
      'session.title',
      expect.objectContaining({ title: 'way too long title' })
    )
    expect(refreshSessions).not.toHaveBeenCalled()
    expect($sessions.get()[0]?.title).toBe('Old title')
  })
})

// Helper: extract rendered text parts from captured updateSessionState seeds.
function renderedSeedTexts(seeds: Record<string, unknown>[]): string[] {
  return seeds.flatMap(state => {
    const messages = Array.isArray(state.messages)
      ? (state.messages as Array<{ parts?: Array<{ text?: string }> }>)
      : []

    return messages.flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))
  })
}

// The HUD floats over the app the user is really working in, so the gateway
// turns this flag into a per-turn hint: read the window underneath and work in
// it, rather than reaching for Hermes's own browser and panes.
describe('usePromptActions HUD surface', () => {
  afterEach(() => {
    cleanup()
    $hudMode.set(false)
    vi.restoreAllMocks()
  })

  async function submitFrom(window: 'app' | 'hud') {
    $hudMode.set(window === 'hud')

    const submitted: (Record<string, unknown> | undefined)[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submitted.push(params)
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await handle!.submitText("what's under you rn?")

    return submitted[0]
  }

  it('tags a message typed into the HUD', async () => {
    expect(await submitFrom('hud')).toMatchObject({ surface: 'hud' })
  })

  it('says nothing about the surface from the app window', async () => {
    expect(await submitFrom('app')).not.toHaveProperty('surface')
  })
})

describe('usePromptActions slash session targeting', () => {
  const STORED_SESSION_ID = 'stored-db-xyz789'
  const RECOVERED_SESSION_ID = 'rt-recovered-456'

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('runs /goal status against the ROUTED stored session instead of minting a new one', async () => {
    // Teknium's report: start a goal in the desktop app, then `/goal status`
    // says there is no goal. `/goal` state lives per-session in SessionDB
    // (`goal:<session_id>`), and slash.ts used to resolve its target with a
    // bare `hint || activeRef || createSession()`. With the runtime binding
    // momentarily absent (profile swap / reconnect / orphan-reap / timeout) it
    // minted a NEW session, so the status query asked a session that never had
    // a goal. submit.ts already resumes the routed chat here; both pipelines
    // must resolve identically.
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    let boundRuntimeId: null | string = null

    const createBackendSessionForSend = vi.fn(async () => 'rt-brand-new-WRONG')

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        boundRuntimeId = RECOVERED_SESSION_ID
        selectedStoredSessionIdRef.current = STORED_SESSION_ID
        activeSessionIdRef.current = RECOVERED_SESSION_ID

        return { session_id: RECOVERED_SESSION_ID } as never
      }

      if (method === 'slash.exec') {
        return { output: '⊙ Goal (active, 1/20 turns): build a rocket' } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => boundRuntimeId}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )

    await handle!.submitText('/goal status')

    // Never fork the conversation to answer a question about it.
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(calls.map(c => c.method)).toEqual(['session.resume', 'slash.exec'])
    expect(calls[0]?.params).toMatchObject({ session_id: STORED_SESSION_ID })
    // The command lands on the recovered runtime that owns the goal.
    expect(calls[1]?.params).toEqual({ command: 'goal status', session_id: RECOVERED_SESSION_ID })
  })

  it('does not fork the chat when the routed session cannot be rebound', async () => {
    const calls: string[] = []
    const createBackendSessionForSend = vi.fn(async () => 'rt-brand-new-WRONG')

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      if (method === 'session.resume') {
        throw new Error('4007 session not found')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={{ current: null }}
        createBackendSessionForSend={createBackendSessionForSend}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => null}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={{ current: null }}
        storedSessionId={null}
      />
    )

    await handle!.submitText('/goal status')

    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(calls).not.toContain('slash.exec')
  })
})

describe('usePromptActions /wake', () => {
  beforeEach(() => {
    setSessions(() => [sessionInfo()])
    resetWakeWordState()
  })

  afterEach(() => {
    cleanup()
    resetWakeWordState()
    vi.restoreAllMocks()
  })

  it('starts the GUI-owned listener through wake.start and never spawns the slash worker', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, _params?: Record<string, unknown>, _timeoutMs?: number) => {
      if (method === 'wake.start') {
        return {
          owner_surface: 'gui',
          phrase: 'hey hermes',
          provider: 'openwakeword',
          started: true
        } as never
      }

      if (method === 'wake.status') {
        return {
          available: true,
          configured_surface: 'gui',
          enabled: true,
          input_device: {
            hostapi: 'Windows WASAPI',
            name: 'Microphone Array',
            selector: 'Microphone Array'
          },
          listening: true,
          owner_surface: 'gui',
          phrase: 'hey hermes',
          provider: 'openwakeword'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => seeds.push(state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/wake on')

    expect(requestGateway).toHaveBeenCalledWith(
      'wake.start',
      { client_capture: true, persist: true, surface: 'gui' },
      180_000
    )
    expect(requestGateway).toHaveBeenCalledWith('wake.status', { client_capture: true, surface: 'gui' })
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
    expect(requestGateway).not.toHaveBeenCalledWith('command.dispatch', expect.anything())
    expect($wakeWord.get()).toMatchObject({ available: true, enabled: true, listening: true })
    expect(renderedSeedTexts(seeds).join('\n')).toContain('Input: Microphone Array (Windows WASAPI)')
  })

  it('uses gateway truth for a bare toggle and stops through wake.stop', async () => {
    let statusCalls = 0

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'wake.status') {
        statusCalls += 1

        return {
          available: true,
          enabled: statusCalls === 1,
          listening: statusCalls === 1,
          owner_surface: statusCalls === 1 ? 'gui' : null,
          phrase: 'hey hermes',
          provider: 'openwakeword'
        } as never
      }

      if (method === 'wake.stop') {
        return { disabled_persisted: true, stopped: true } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await handle!.submitText('/wake')

    expect(requestGateway.mock.calls.map(([method]) => method)).toEqual(['wake.status', 'wake.stop', 'wake.status'])
    expect(requestGateway).toHaveBeenCalledWith('wake.stop', { persist: true })
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
    expect(requestGateway).not.toHaveBeenCalledWith('command.dispatch', expect.anything())
    expect($wakeWord.get()).toMatchObject({ enabled: false, listening: false })
  })
})

describe('usePromptActions /compress', () => {
  beforeEach(() => {
    setSessions(() => [sessionInfo()])
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
    setCurrentUsage({ calls: 0, input: 0, output: 0, total: 0 })
    setMessages([])
    vi.restoreAllMocks()
  })

  it('routes through session.compress (not slash.exec) with a 120s timeout and renders the summary', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, _params?: Record<string, unknown>, _timeoutMs?: number) => {
      if (method === 'session.compress') {
        return {
          removed: 8,
          summary: {
            headline: 'Compressed: 234 → 226 messages',
            noop: false,
            token_line: 'Approx request size: ~285,727 → ~198,104 tokens'
          }
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/compress')

    // The dedicated RPC is the TUI's path and has no slash-worker pipe
    // timeout — and the call carries an LLM-sized client timeout instead of
    // the 30s WS default, so a large session's compression can finish.
    expect(requestGateway).toHaveBeenCalledWith(
      'session.compress',
      expect.objectContaining({ session_id: RUNTIME_SESSION_ID }),
      120_000
    )
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
    expect(requestGateway).not.toHaveBeenCalledWith('command.dispatch', expect.anything())
  })

  it('replaces the transcript from the response messages', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, _params?: Record<string, unknown>, _timeoutMs?: number) => {
      if (method === 'session.compress') {
        return {
          removed: 2,
          summary: { headline: 'Compressed: 4 → 2 messages' },
          messages: [
            { role: 'user', content: 'summarized context' },
            { role: 'assistant', content: 'sure, here is the summary' }
          ]
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/compress')

    // The transcript was replaced with the post-compress history.
    const finalMessages = seeds[seeds.length - 1]?.messages as Array<{ parts?: Array<{ text?: string }> }>

    const renderedText = (finalMessages ?? [])
      .flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))
      .join('\n')

    expect(renderedText).toContain('summarized context')
    expect(renderedText).toContain('sure, here is the summary')
  })

  it('uses the compute-host response transcript and success output', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.compress') {
        return {
          host_ack: { output: 'Compressed 4 → 2 messages' },
          messages: [
            { role: 'user', content: 'compute-host summary' },
            { role: 'assistant', content: 'compute-host answer' }
          ]
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/compress')

    const computeHostTexts = renderedSeedTexts(seeds)
    expect(computeHostTexts).toEqual(expect.arrayContaining(['compute-host summary', 'compute-host answer']))
    expect(computeHostTexts.some(text => text.includes('Compressed 4 → 2 messages'))).toBe(true)
    expect($notifications.get()).toEqual(
      expect.arrayContaining([expect.objectContaining({ message: 'Compressed 4 → 2 messages' })])
    )
  })

  it('renders an aborted compression as an error, not a success', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.compress') {
        return {
          status: 'aborted',
          summary: {
            aborted: true,
            headline: 'Compression aborted: 6 messages preserved',
            note: 'No compression provider is configured.'
          }
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await handle!.submitText('/compress')

    expect($notifications.get()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'error', message: expect.stringContaining('Compression aborted') })
      ])
    )
    expect($notifications.get()).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: 'success', message: expect.stringContaining('Compression aborted') })
      ])
    )
  })

  it('passes a focus topic through as focus_topic', async () => {
    const requestGateway = vi.fn(
      async (_method: string, _params?: Record<string, unknown>, _timeoutMs?: number) => ({ removed: 0 }) as never
    )

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await handle!.submitText('/compress the auth refactor')

    expect(requestGateway).toHaveBeenCalledWith(
      'session.compress',
      expect.objectContaining({ focus_topic: 'the auth refactor' }),
      120_000
    )
  })

  it('surfaces the RPC error verbatim (e.g. the busy guard) instead of a routing error', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.compress') {
        throw new Error('session busy — /interrupt the current turn before /compress')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/compress')

    const texts = renderedSeedTexts(seeds)
    expect(texts.some(text => text.includes('session busy'))).toBe(true)
    expect(texts.some(text => text.includes('not a quick/plugin/skill command'))).toBe(false)
  })

  it('falls back to the slash worker when an older gateway lacks session.compress', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.compress') {
        throw new Error('method not found: session.compress')
      }

      if (method === 'slash.exec') {
        return { output: 'compressed by legacy gateway' } as never
      }

      throw new Error(`unexpected method: ${method}`)
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/compress')

    expect(requestGateway).toHaveBeenCalledWith('slash.exec', expect.objectContaining({ command: 'compress' }))
    expect(renderedSeedTexts(seeds).some(text => text.includes('compressed by legacy gateway'))).toBe(true)
  })

  it('does not clobber the foreground transcript when compression resolves after a session switch', async () => {
    const RUNTIME_SESSION_B = 'rt-session-b'
    const storedSessionIdRef: MutableRefObject<string | null> = { current: 'stored-a' }
    const updates: Array<{ sessionId: string; storedSessionId: null | string | undefined }> = []

    let resolveCompress: (value: unknown) => void = () => undefined

    const compressResult = new Promise(resolve => {
      resolveCompress = resolve
    })

    const activeSessionIdRef: MutableRefObject<string | null> = { current: RUNTIME_SESSION_ID }

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.compress') {
        return (await compressResult) as never
      }

      throw new Error(`unexpected method: ${method}`)
    })

    setMessages([{ id: 'foreground-b', parts: [textPart('session B transcript')], role: 'user', timestamp: 0 }])
    setCurrentUsage({ calls: 7, input: 70, output: 30, total: 100 })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        onUpdateState={(sessionId, storedSessionId) => updates.push({ sessionId, storedSessionId })}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={storedSessionIdRef}
      />
    )

    // Start the request in a synchronous act. The async `submitText` harness
    // helper cannot be used here: this test intentionally keeps its promise
    // pending while the user switches sessions, which would leave React's
    // async act scope open and overlap the wait below.
    let submitted: Promise<boolean>
    act(() => {
      submitted = handle!.submitTextRaw('/compress')
    })
    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('session.compress', expect.anything(), 120_000))

    // Switch to session B before compression resolves.
    activeSessionIdRef.current = RUNTIME_SESSION_B
    resolveCompress({
      info: { usage: { context_used: 4_000, total: 12_000 } },
      messages: [{ content: 'compressed session A transcript', role: 'system' }],
      removed: 5
    })
    await act(async () => {
      await submitted
    })

    // The foreground transcript + usage are unchanged — the late result only
    // updated session A's cache, not the active session B's.
    expect($messages.get()).toEqual([
      { id: 'foreground-b', parts: [textPart('session B transcript')], role: 'user', timestamp: 0 }
    ])
    expect($currentUsage.get()).toEqual(expect.objectContaining({ calls: 7, input: 70, output: 30, total: 100 }))
    expect(updates).toContainEqual({ sessionId: RUNTIME_SESSION_ID, storedSessionId: 'stored-a' })
  })

  it('keeps a late compression error bound to its invocation-time stored session', async () => {
    const RUNTIME_SESSION_B = 'rt-session-b'
    const storedSessionIdRef: MutableRefObject<string | null> = { current: 'stored-a' }
    const activeSessionIdRef: MutableRefObject<string | null> = { current: RUNTIME_SESSION_ID }
    const updates: Array<{ sessionId: string; storedSessionId: null | string | undefined }> = []
    let rejectCompress: (reason?: unknown) => void = () => undefined

    const compressResult = new Promise((_, reject) => {
      rejectCompress = reject
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.compress') {
        return (await compressResult) as never
      }

      throw new Error(`unexpected method: ${method}`)
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        onUpdateState={(sessionId, storedSessionId) => updates.push({ sessionId, storedSessionId })}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={storedSessionIdRef}
      />
    )

    // Keep the RPC pending while the selected stored session changes without
    // leaving an async React act scope open (see the foreground-race test).
    let submitted: Promise<boolean>
    act(() => {
      submitted = handle!.submitTextRaw('/compress')
    })
    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('session.compress', expect.anything(), 120_000))
    activeSessionIdRef.current = RUNTIME_SESSION_B
    storedSessionIdRef.current = 'stored-b'
    rejectCompress(new Error('compression failed'))
    await act(async () => {
      await submitted
    })

    expect(updates).toContainEqual({ sessionId: RUNTIME_SESSION_ID, storedSessionId: 'stored-a' })
  })
  it('shows a compression progress toast outside the transcript', async () => {
    let resolveCompress: (value: unknown) => void = () => undefined

    const compressResult = new Promise(resolve => {
      resolveCompress = resolve
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.compress') {
        return (await compressResult) as never
      }

      throw new Error(`unexpected method: ${method}`)
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const submitted = handle!.submitTextRaw('/compress')
    await waitFor(() => expect($notifications.get().some(item => item.message === 'compressing context...')).toBe(true))
    resolveCompress({ messages: [{ content: 'compressed transcript', role: 'system' }] })
    await submitted
  })
})

describe('usePromptActions exec fallback error reporting', () => {
  beforeEach(() => {
    setSessions(() => [sessionInfo()])
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('surfaces the slash.exec failure when command.dispatch only adds "not a quick/plugin/skill command"', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'slash.exec') {
        throw new Error('slash worker timed out')
      }

      if (method === 'command.dispatch') {
        throw new Error('not a quick/plugin/skill command: debug')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    // /debug still goes through exec (no dedicated RPC), so it exercises the
    // slash.exec → command.dispatch fallback + error unmasking path.
    await handle!.submitText('/debug')

    // The dispatch fallback knowing nothing about /debug is routing noise;
    // the worker timeout is what actually went wrong (#44456).
    const texts = renderedSeedTexts(seeds)
    expect(texts.some(text => text.includes('slash worker timed out'))).toBe(true)
    expect(texts.some(text => text.includes('not a quick/plugin/skill command'))).toBe(false)
  })

  it('falls back to slash.exec when an older gateway lacks a dedicated RPC', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.status') {
        throw new Error('method not found: session.status')
      }

      if (method === 'slash.exec') {
        return { output: 'session status from slash worker' } as never
      }

      throw new Error(`unexpected method: ${method}`)
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/status')

    expect(requestGateway).toHaveBeenCalledWith('session.status', expect.anything(), undefined)
    expect(requestGateway).toHaveBeenCalledWith('slash.exec', expect.objectContaining({ command: 'status' }))
    expect(renderedSeedTexts(seeds).some(text => text.includes('session status from slash worker'))).toBe(true)
  })

  it('still reports a real command.dispatch failure for skill/quick commands', async () => {
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'slash.exec') {
        throw new Error('skill command: use command.dispatch for /my-skill')
      }

      if (method === 'command.dispatch') {
        throw new Error('quick command failed with exit code 1')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/my-skill')

    const texts = renderedSeedTexts(seeds)
    expect(texts.some(text => text.includes('quick command failed with exit code 1'))).toBe(true)
  })
})

describe('usePromptActions slash.exec dispatch payloads', () => {
  afterEach(() => {
    cleanup()
    $busy.set(false)
    vi.restoreAllMocks()
  })

  it('executes /approvals against the focused profile session and persists its mode', async () => {
    const focusedProfile = 'work'
    const focusedSessionId = 'work-runtime-session'
    const persistedModes = new Map<string, string>()
    const sessionProfiles = new Map([[focusedSessionId, focusedProfile]])

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'slash.exec') {
        const sessionId = String(params?.session_id ?? '')
        const profile = sessionProfiles.get(sessionId)
        const command = String(params?.command ?? '')

        if (profile && command === 'approvals off') {
          persistedModes.set(profile, 'off')
        }

        return { output: 'Approval mode: off (persistent profile setting).' } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null

    render(
      <Harness
        activeSessionId={focusedSessionId}
        activeSessionIdRef={{ current: focusedSessionId }}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={focusedSessionId}
      />
    )

    await handle!.submitText('/approvals off')

    expect(requestGateway).toHaveBeenCalledWith('slash.exec', {
      command: 'approvals off',
      session_id: focusedSessionId
    })
    expect(persistedModes.get(focusedProfile)).toBe('off')
    expect(persistedModes.has('default')).toBe(false)
  })

  it('submits /goal send directives returned directly by slash.exec instead of rendering no output', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    const states: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'slash.exec') {
        return {
          type: 'send',
          notice: '⊙ Goal set. Starting now.',
          message: 'write the implementation plan'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => states.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/goal write the implementation plan')

    expect(calls.map(c => c.method)).toEqual(['slash.exec', 'prompt.submit'])
    expect(calls[0]?.params).toEqual({
      command: 'goal write the implementation plan',
      session_id: RUNTIME_SESSION_ID
    })
    expect(calls[1]?.params).toEqual({
      session_id: RUNTIME_SESSION_ID,
      text: 'write the implementation plan'
    })

    const renderedText = states
      .flatMap(state => {
        const messages = Array.isArray(state.messages)
          ? (state.messages as Array<{ parts?: Array<{ text?: string }> }>)
          : []

        return messages.flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))
      })
      .join('\n')

    expect(renderedText).toContain('⊙ Goal set. Starting now.')
    expect(renderedText).not.toContain('/goal: no output')
  })

  it('queues the /goal kickoff instead of dropping it when the session is busy (#63352)', async () => {
    // The backend sets the goal the moment slash.exec runs — dropping the
    // returned kickoff message because busyRef was true left a goal the agent
    // never heard about. The busy path must park the kickoff on the composer
    // queue so the settle drain sends it.
    $queuedPromptsBySession.set({})

    const calls: { method: string; params?: Record<string, unknown> }[] = []
    const states: Record<string, unknown>[] = []
    const busyRef = { current: true }

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'slash.exec') {
        return {
          type: 'send',
          notice: '⊙ Goal set (20-turn budget): ship the release notes',
          message: 'ship the release notes'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        busyRef={busyRef}
        onReady={h => (handle = h)}
        onSeedState={s => states.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/goal ship the release notes')

    // The kickoff must NOT submit mid-turn — and must NOT vanish either.
    expect(calls.map(c => c.method)).toEqual(['slash.exec'])

    const queued = getQueuedPrompts(RUNTIME_SESSION_ID)
    expect(queued.map(entry => entry.text)).toEqual(['ship the release notes'])

    const renderedText = states
      .flatMap(state => {
        const messages = Array.isArray(state.messages)
          ? (state.messages as Array<{ parts?: Array<{ text?: string }> }>)
          : []

        return messages.flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))
      })
      .join('\n')

    // The notice still renders, and the busy line reports a queue, not a demand
    // to /interrupt.
    expect(renderedText).toContain('⊙ Goal set (20-turn budget): ship the release notes')
    expect(renderedText).toContain('queued')

    $queuedPromptsBySession.set({})
  })

  it('gates the busy queue on the TARGET session, not the foreground busy flag', async () => {
    // `busyRef` is the FOREGROUND view's busy flag; a slash command runs against
    // the session `resolveTargetSessionId` picked, which is frequently not the
    // foreground one (tile, route rebind, freshly created session). A stale
    // foreground `true` — e.g. left behind by a warm resume of a *different*,
    // still-running session — parked the kickoff of an idle session's command
    // on the queue and told the user "session busy" about a session that was
    // doing nothing.
    $queuedPromptsBySession.set({})
    publishSessionState(RUNTIME_SESSION_ID, createClientSessionState(RUNTIME_SESSION_ID))

    const calls: string[] = []
    const busyRef = { current: true }

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      return (method === 'slash.exec' ? { type: 'send', message: 'audit the session states' } : {}) as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        busyRef={busyRef}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/audit-only audit the session states')

    // The target session is idle, so the command sends now — nothing queues.
    expect(calls).toContain('prompt.submit')
    expect(getQueuedPrompts(RUNTIME_SESSION_ID)).toEqual([])

    dropSessionState(RUNTIME_SESSION_ID)
    $queuedPromptsBySession.set({})
  })

  it('still queues when the TARGET session is busy and the foreground flag is not', async () => {
    // The converse leak: a background/tile command must not submit mid-turn
    // just because the foreground view happens to be idle.
    $queuedPromptsBySession.set({})
    publishSessionState(RUNTIME_SESSION_ID, {
      ...createClientSessionState(RUNTIME_SESSION_ID),
      busy: true
    })

    const calls: string[] = []
    const busyRef = { current: false }

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      return (method === 'slash.exec' ? { type: 'send', message: 'audit the session states' } : {}) as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        busyRef={busyRef}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/audit-only audit the session states')

    expect(calls).toEqual(['slash.exec'])
    expect(getQueuedPrompts(RUNTIME_SESSION_ID).map(entry => entry.text)).toEqual(['audit the session states'])

    dropSessionState(RUNTIME_SESSION_ID)
    $queuedPromptsBySession.set({})
  })

  it('binds slash output and the busy queue to the TARGET session, not the foreground selection', async () => {
    // A tile (⌘T tab, split pane) routes its slash commands through this hook
    // with an explicit runtime id while the foreground selection names a
    // different conversation. Binding the output writer to the foreground
    // selection re-keyed the tile's cache entry onto the primary's stored
    // session and parked its queued payload on the primary's queue.
    const tileRuntimeId = 'tile-runtime'
    const tileStoredId = 'tile-stored'

    $queuedPromptsBySession.set({})
    publishSessionState(tileRuntimeId, {
      ...createClientSessionState(tileStoredId),
      busy: true
    })

    const boundStoredIds: (null | string | undefined)[] = []

    const requestGateway = vi.fn(
      async (method: string) => (method === 'slash.exec' ? { type: 'send', message: 'run it in the tab' } : {}) as never
    )

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onUpdateState={(_sessionId, storedSessionId) => boundStoredIds.push(storedSessionId)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId="primary-stored"
      />
    )

    await handle!.submitText('/audit-only run it in the tab', { sessionId: tileRuntimeId })

    // Every transcript write lands on the tile's own stored session.
    expect(boundStoredIds).not.toHaveLength(0)
    expect(new Set(boundStoredIds)).toEqual(new Set([tileStoredId]))
    // …and the kickoff queues against the tile, never the foreground chat.
    expect(getQueuedPrompts(tileStoredId).map(entry => entry.text)).toEqual(['run it in the tab'])
    expect(getQueuedPrompts('primary-stored')).toEqual([])

    dropSessionState(tileRuntimeId)
    $queuedPromptsBySession.set({})
  })

  it("sends a skill's kickoff into the TAB that invoked it, not the foreground chat", async () => {
    // `/work` in a fresh ⌘T tab: slash.exec returns a skill dispatch whose
    // `message` is the kickoff prompt. The dispatcher resolved the tab as its
    // target, then submitted the kickoff with no target at all, so submit
    // re-resolved from activeSessionIdRef and fired it as a user message into
    // whatever conversation was on screen.
    const tabRuntimeId = 'tab-runtime'
    const tabStoredId = 'tab-stored'

    $queuedPromptsBySession.set({})
    publishSessionState(tabRuntimeId, createClientSessionState(tabStoredId))

    const submitted: (Record<string, unknown> | undefined)[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submitted.push(params)
      }

      return (
        method === 'slash.exec'
          ? { type: 'skill', name: 'work', message: 'Load the work skill, then: fix the tab bug' }
          : {}
      ) as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId="foreground-runtime"
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId="foreground-stored"
      />
    )

    await handle!.submitText('/work fix the tab bug', { sessionId: tabRuntimeId })

    expect(submitted).toEqual([
      expect.objectContaining({
        session_id: tabRuntimeId,
        text: 'Load the work skill, then: fix the tab bug'
      })
    ])

    dropSessionState(tabRuntimeId)
    $queuedPromptsBySession.set({})
  })

  it('renders a skill turn as its invocation — the expanded body never reaches a bubble', async () => {
    // A `/skill` dispatch's `message` is the whole skill body (model-facing
    // scaffolding). The agent must receive it verbatim; every UI surface —
    // the user bubble and any system line — must show only `/work fix it`.
    const skillBody =
      '[IMPORTANT: The user has invoked the "work" skill, indicating they want you to follow its instructions.\n' +
      'The full skill content is loaded below.]\n\nSPIN UP A WORKTREE, never the primary checkout.\n\n' +
      'The user has provided the following instruction alongside the skill invocation: fix it'

    const states: Record<string, unknown>[] = []
    const submitted: (Record<string, unknown> | undefined)[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submitted.push(params)
      }

      return (
        method === 'slash.exec' ? { type: 'skill', name: 'work', message: skillBody, display: '/work fix it' } : {}
      ) as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => states.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/work fix it')

    // The agent still gets the full skill.
    expect(submitted).toEqual([expect.objectContaining({ text: skillBody })])

    const rendered = states.flatMap(state => {
      const messages = Array.isArray(state.messages)
        ? (state.messages as Array<{ parts?: Array<{ text?: string }> }>)
        : []

      return messages.flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))
    })

    expect(rendered).toContain('/work fix it')
    expect(rendered.join('\n')).not.toContain('SPIN UP A WORKTREE')
    expect(rendered.join('\n')).not.toContain('IMPORTANT: The user has invoked')
  })

  it('slash status header carries the command token, not the full invocation', async () => {
    // `/goal <long prose>` used to echo the entire invocation in the mono
    // header AND the goal text again in the backend notice right under it.
    const states: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'slash.exec') {
        return {
          type: 'send',
          notice: '⊙ Goal set: build the whole thing',
          message: 'build the whole thing end to end with tests'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => states.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/goal build the whole thing end to end with tests')

    const systemTexts = states
      .flatMap(state => {
        const messages = Array.isArray(state.messages)
          ? (state.messages as Array<{ role?: string; parts?: Array<{ text?: string }> }>)
          : []

        return messages
          .filter(message => message.role === 'system')
          .flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))
      })
      .join('\n')

    expect(systemTexts).toContain('slash:/goal\n')
    expect(systemTexts).not.toContain('slash:/goal build the whole thing')
  })

  it('dispatches a slash command with a multiline arg instead of "empty slash command" (#41323, #55510)', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    const states: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'slash.exec') {
        return { type: 'send', message: 'Write a Python script\nthat prints Hello World' } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => states.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/goal Write a Python script\nthat prints Hello World')

    // The newline lives in the arg — the command still reaches the gateway
    // whole, exactly as the CLI and Telegram handle it.
    expect(calls.map(c => c.method)).toEqual(['slash.exec', 'prompt.submit'])
    expect(calls[0]?.params).toEqual({
      command: 'goal Write a Python script\nthat prints Hello World',
      session_id: RUNTIME_SESSION_ID
    })

    const renderedText = states
      .flatMap(state => {
        const messages = Array.isArray(state.messages)
          ? (state.messages as Array<{ parts?: Array<{ text?: string }> }>)
          : []

        return messages.flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))
      })
      .join('\n')

    expect(renderedText).not.toContain('empty slash command')
  })

  it('restores a degenerate slash payload to the composer instead of losing it', async () => {
    setComposerDraft('')

    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    // `/ text` parses to an empty command name on every surface (CLI parity).
    // The composer draft was already cleared on submit and slash input never
    // enters the Up-arrow history ring, so the payload must be handed back.
    await handle!.submitText('/ pasted context that must not vanish')

    expect($composerDraft.get()).toBe('/ pasted context that must not vanish')
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
  })
})

describe('usePromptActions desktop slash pickers', () => {
  beforeEach(() => {
    setSessions(() => [sessionInfo({ id: '20260610_120000_abcdef', title: 'Loaded session' })])
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('resumes an exact session id even when it is not in the loaded sidebar cache', async () => {
    const resumeStoredSession = vi.fn(async () => undefined)
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
      />
    )

    await handle!.submitText('/resume 20260610_130000_123abc')

    expect(resumeStoredSession).toHaveBeenCalledWith('20260610_130000_123abc')
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
  })

  it('opens the memory graph overlay for /journey and its aliases instead of hitting the backend', async () => {
    const openMemoryGraph = vi.fn()
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        openMemoryGraph={openMemoryGraph}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('/journey')
    await handle!.submitText('/memory-graph')
    await handle!.submitText('/learning')

    expect(openMemoryGraph).toHaveBeenCalledTimes(3)
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
    expect(requestGateway).not.toHaveBeenCalledWith('command.dispatch', expect.anything())
  })

  it('marks a timed-out handoff as failed so the next attempt can retry', async () => {
    vi.useFakeTimers()
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'handoff.state') {
        return { state: 'pending' } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const result = handle!.submitText('/handoff telegram')
    await vi.advanceTimersByTimeAsync(61_000)
    await result

    expect(calls.some(call => call.method === 'handoff.request')).toBe(true)
    expect(calls).toContainEqual({
      method: 'handoff.fail',
      params: {
        error: expect.stringContaining('Timed out'),
        session_id: RUNTIME_SESSION_ID
      }
    })
  })
})

describe('usePromptActions submit / queue drain semantics', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('clears a leftover interrupted flag on a fresh submit (so the new turn streams)', async () => {
    const seeds: Record<string, unknown>[] = []
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.submitText('hello after a stop')

    // The optimistic seed must reset interrupted:false even though the prior
    // session state had interrupted:true — otherwise the message stream drops
    // every delta of this brand-new turn.
    expect(seeds.length).toBeGreaterThan(0)
    expect(seeds.every(s => s.interrupted === false)).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        session_id: RUNTIME_SESSION_ID,
        text: 'hello after a stop'
      },
      1_800_000
    )
  })

  it('flags prompt.submit with interrupted:true after a voice-playback barge', async () => {
    const { markVoicePlaybackInterrupted } = await import('@/lib/voice-playback')
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    markVoicePlaybackInterrupted()
    await handle!.submitText('stop! rude interruption')

    // The latch is one-shot: the flag rides this submit, the next is clean.
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        session_id: RUNTIME_SESSION_ID,
        text: 'stop! rude interruption',
        interrupted: true
      },
      1_800_000
    )

    await handle!.submitText('follow-up without a barge')
    expect(requestGateway).toHaveBeenLastCalledWith(
      'prompt.submit',
      {
        session_id: RUNTIME_SESSION_ID,
        text: 'follow-up without a barge'
      },
      1_800_000
    )
  })

  it('a fromQueue drain sends even when busyRef is still true on the settle edge', async () => {
    // busyRef lags $busy by one effect tick on the busy→false settle edge, so a
    // drained queue send would otherwise hit the busy guard and silently no-op.
    const busyRef = { current: true }
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        busyRef={busyRef}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    const accepted = await handle!.submitText('queued message', { fromQueue: true })

    expect(accepted).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        queued: true,
        session_id: RUNTIME_SESSION_ID,
        text: 'queued message'
      },
      1_800_000
    )
  })

  it('a fromQueue drain sends to its queued session even after the active session changes', async () => {
    $busy.set(false)

    const updates: { sessionId: string; state: Record<string, unknown>; storedSessionId: null | string | undefined }[] =
      []

    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    render(
      <Harness
        getRuntimeIdForStoredSession={storedId => (storedId === 'stored-session-a' ? 'rt-session-a' : null)}
        onReady={h => (handle = h)}
        onUpdateState={(sessionId, storedSessionId, state) => updates.push({ sessionId, state, storedSessionId })}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    const accepted = await handle!.submitText('queued for background session', {
      fromQueue: true,
      sessionId: 'rt-session-a',
      storedSessionId: 'stored-session-a'
    })

    expect(accepted).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        queued: true,
        session_id: 'rt-session-a',
        text: 'queued for background session'
      },
      1_800_000
    )
    expect(requestGateway).not.toHaveBeenCalledWith('session.resume', expect.anything())
    expect(
      updates.some(update => update.sessionId === 'rt-session-a' && update.storedSessionId === 'stored-session-a')
    ).toBe(true)
    // Offscreen queue drains must not flip the foreground composer into Thinking.
    expect($busy.get()).toBe(false)
  })

  it('a fromQueue drain carrying a stale runtime id re-homes via session.resume instead of landing in the foreground session', async () => {
    // The session-switch window this guards: the composer's queue key has
    // already flipped to session B (route-driven) while the foreground runtime
    // id prop still reads session A (resume-driven, one settle behind). Without
    // the central-binding check, prompt.submit fires with session_id=A and B's
    // queued prompt — plus its whole answer turn — lands inside A. With no
    // binding recorded for B yet, the stale id must be dropped and the drain
    // re-homed through the stored-session resume path.
    const updates: { sessionId: string; state: Record<string, unknown>; storedSessionId: null | string | undefined }[] =
      []

    const requestGateway = vi.fn(
      async (method: string, _params?: Record<string, unknown>) =>
        (method === 'session.resume' ? { session_id: 'rt-session-b' } : {}) as never
    )

    let handle: HarnessHandle | null = null
    render(
      <Harness
        onReady={h => (handle = h)}
        onUpdateState={(sessionId, storedSessionId, state) => updates.push({ sessionId, state, storedSessionId })}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    const accepted = await handle!.submitText('queued for B mid-switch', {
      fromQueue: true,
      sessionId: 'rt-session-a',
      storedSessionId: 'stored-session-b'
    })

    expect(accepted).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith('session.resume', {
      session_id: 'stored-session-b',
      source: 'desktop',
      omit_messages: true
    })
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        queued: true,
        session_id: 'rt-session-b',
        text: 'queued for B mid-switch'
      },
      1_800_000
    )
    // The invariant: the stale foreground runtime never receives the prompt.
    expect(
      requestGateway.mock.calls.every(
        ([method, params]) =>
          method !== 'prompt.submit' || (params as { session_id?: string }).session_id !== 'rt-session-a'
      )
    ).toBe(true)
    expect(
      updates.some(update => update.sessionId === 'rt-session-b' && update.storedSessionId === 'stored-session-b')
    ).toBe(true)
  })

  it('a fromQueue drain rebinds to the centrally recorded runtime when its explicit id is stale', async () => {
    // Same window, but B's runtime binding is already known centrally — the
    // drain should adopt the authoritative binding directly (no resume
    // round-trip) rather than trusting the leftover foreground id.
    const requestGateway = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({}) as never)

    let handle: HarnessHandle | null = null
    render(
      <Harness
        getRuntimeIdForStoredSession={storedId => (storedId === 'stored-session-b' ? 'rt-session-b-live' : null)}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    const accepted = await handle!.submitText('queued for B, B already re-bound', {
      fromQueue: true,
      sessionId: 'rt-session-a',
      storedSessionId: 'stored-session-b'
    })

    expect(accepted).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        queued: true,
        session_id: 'rt-session-b-live',
        text: 'queued for B, B already re-bound'
      },
      1_800_000
    )
    expect(requestGateway).not.toHaveBeenCalledWith('session.resume', expect.anything())
    expect(
      requestGateway.mock.calls.every(
        ([method, params]) =>
          method !== 'prompt.submit' || (params as { session_id?: string }).session_id !== 'rt-session-a'
      )
    ).toBe(true)
  })

  it('a NON-queue explicit target keeps its runtime id even with no central binding recorded', async () => {
    // The scoping invariant for the check above. A slash skill dispatch into a
    // fresh ⌘T tab passes the same shape a stale drain does — sessionId and
    // storedSessionId differ, and the tab has no central binding yet — but its
    // two ids were resolved in the same tick, so the explicit target IS
    // authoritative. Validating this caller against the (empty) binding would
    // null the target and silently drop the kickoff into nowhere.
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    render(
      <Harness
        getRuntimeIdForStoredSession={() => null}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    const accepted = await handle!.submitText('kickoff for the tab', {
      sessionId: 'rt-tab',
      storedSessionId: 'stored-tab'
    })

    expect(accepted).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        session_id: 'rt-tab',
        text: 'kickoff for the tab'
      },
      1_800_000
    )
  })

  it('a fromQueue drain with null runtime id does NOT land in the foreground session (cross-session leak guard)', async () => {
    // The cross-session leak: a background drain fires with sessionId=null
    // (the stored session's runtime was reaped by the gateway). Without the
    // guard, `null ?? activeSessionIdRef.current` falls back to whichever
    // runtime id the foreground happens to hold — landing the queued prompt
    // in the chat the user is currently viewing, NOT the session that owns
    // the queue entry. The drain must instead go through session.resume to
    // rebind the correct runtime before submitting.
    const requestGateway = vi.fn(
      async (method: string, _params?: Record<string, unknown>, _timeoutMs?: number) =>
        (method === 'session.resume' ? { session_id: 'rt-session-a-rebound' } : {}) as never
    )

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={'rt-foreground'}
        getRuntimeIdForStoredSession={() => null}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={'rt-foreground'}
      />
    )

    // Background drain: sessionId=null (binding reaped), storedSessionId
    // points to a DIFFERENT session than the foreground.
    const accepted = await handle!.submitText('queued for background session', {
      fromQueue: true,
      sessionId: null,
      storedSessionId: 'stored-session-a'
    })

    expect(accepted).toBe(true)
    // Must resume the correct stored session to get the right runtime id.
    expect(requestGateway).toHaveBeenCalledWith('session.resume', {
      session_id: 'stored-session-a',
      source: 'desktop',
      omit_messages: true
    })
    // The prompt must land in the resumed session, NOT the foreground.
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        queued: true,
        session_id: 'rt-session-a-rebound',
        text: 'queued for background session'
      },
      1_800_000
    )
    // The invariant: the foreground runtime never receives the prompt.
    expect(
      requestGateway.mock.calls.every(
        ([method, params]) => method !== 'prompt.submit' || params?.session_id !== 'rt-foreground'
      )
    ).toBe(true)
  })

  it('a rejected fromQueue drain returns false (entry stays queued) and a later retry sends it', async () => {
    // A stale-session 404 must not strand the queued entry: submitPrompt returns
    // false on failure so the composer keeps it, and the edge-independent
    // auto-drain re-attempts once the session is idle again. storedSessionId is
    // null so the session.resume recovery path is skipped and the error surfaces.
    let attempt = 0

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'prompt.submit') {
        attempt += 1

        if (attempt === 1) {
          throw new Error('404: {"detail":"Session not found"}')
        }
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={null}
      />
    )

    const first = await handle!.submitText('please send me', { fromQueue: true })
    expect(first).toBe(false)

    const second = await handle!.submitText('please send me', { fromQueue: true })
    expect(second).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        queued: true,
        session_id: RUNTIME_SESSION_ID,
        text: 'please send me'
      },
      1_800_000
    )
  })

  it('rides out a transient "session busy" so the user never sees it (retries, no error bubble)', async () => {
    // A submit racing the settle edge can hit a transient 4009 before the turn
    // has fully wound down. It must be invisible: retried in place until the
    // gateway accepts, never a red "session busy" bubble.
    let attempt = 0
    const seeds: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'prompt.submit') {
        attempt += 1

        if (attempt === 1) {
          throw new Error('4009: session busy')
        }
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => seeds.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    expect(await handle!.submitText('sent while settling')).toBe(true)
    expect(attempt).toBe(2) // rode past the busy on the second try
    // No assistant-error message was appended for the transient busy.
    expect(seeds.some(s => Array.isArray(s.messages) && (s.messages as { error?: string }[]).some(m => m.error))).toBe(
      false
    )
  })

  it('a normal (non-queue) submit still respects the busyRef guard', async () => {
    const busyRef = { current: true }
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        busyRef={busyRef}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    const accepted = await handle!.submitText('should be blocked')

    expect(accepted).toBe(false)
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything())
  })
})

describe('usePromptActions redirectPrompt', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('redirects the live turn with trimmed correction text', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'redirected' }) as never)

    let handle: HarnessHandle | null = null
    const capturedStates: Record<string, unknown>[] = []
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => capturedStates.push(state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    const accepted = await handle!.redirectPrompt('  nudge the run  ')

    expect(accepted).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith('session.redirect', {
      session_id: RUNTIME_SESSION_ID,
      text: 'nudge the run'
    })
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything())
    expect((capturedStates.at(-1)?.messages as unknown[]).at(-1)).toMatchObject({
      role: 'user',
      parts: [{ type: 'text', text: 'nudge the run' }]
    })
  })

  it('reports rejection so the caller queues when the turn already ended', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'rejected' }) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.redirectPrompt('too late')).toBe(false)
  })

  it('reports rejection without throwing when the redirect RPC errors', async () => {
    const requestGateway = vi.fn(async () => {
      throw new Error('agent does not support redirect')
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.redirectPrompt('boom')).toBe(false)
  })

  it('skips the RPC entirely for empty text', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'redirected' }) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.redirectPrompt('   ')).toBe(false)
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('accepts a queued redirect during the agent-build window and records the correction', async () => {
    // running=True but the agent is still building: the gateway queues the
    // correction instead of rejecting, so the composer must NOT re-queue it.
    const requestGateway = vi.fn(async () => ({ status: 'queued' }) as never)

    let handle: HarnessHandle | null = null
    const capturedStates: Record<string, unknown>[] = []
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => capturedStates.push(state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    expect(await handle!.redirectPrompt('build-window nudge')).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith('session.redirect', {
      session_id: RUNTIME_SESSION_ID,
      text: 'build-window nudge'
    })
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything())
    expect((capturedStates.at(-1)?.messages as unknown[]).at(-1)).toMatchObject({
      role: 'user',
      parts: [{ type: 'text', text: 'build-window nudge' }]
    })
  })

  it('resumes the stored session and retries once when session.redirect reports "session not found"', async () => {
    const STORED_SESSION_ID = 'stored-db-xyz789'
    const RECOVERED_SESSION_ID = 'rt-recovered-456'
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let redirectAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.redirect') {
        redirectAttempts += 1

        if (redirectAttempts === 1) {
          throw new Error('session not found')
        }

        return { status: 'redirected' } as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    expect(await handle!.redirectPrompt('reconnect nudge')).toBe(true)
    expect(calls.map(c => c.method)).toEqual(['session.redirect', 'session.resume', 'session.redirect'])
    expect(calls[0]?.params).toEqual({ session_id: RUNTIME_SESSION_ID, text: 'reconnect nudge' })
    expect(calls[1]?.params).toEqual({ session_id: STORED_SESSION_ID, source: 'desktop', omit_messages: true })
    expect(calls[2]?.params).toEqual({ session_id: RECOVERED_SESSION_ID, text: 'reconnect nudge' })
    expect(handle!.activeSessionIdRef.current).toBe(RECOVERED_SESSION_ID)
  })
})

describe('usePromptActions restoreToMessage', () => {
  beforeEach(() => {
    $busy.set(false)
    $messages.set([
      { id: 'u1', role: 'user', parts: [textPart('first prompt')] },
      { id: 'a1', role: 'assistant', parts: [textPart('first answer')] },
      { id: 'u2', role: 'user', parts: [textPart('second prompt')] },
      { id: 'a2', role: 'assistant', parts: [textPart('second answer')] }
    ])
  })

  afterEach(() => {
    cleanup()
    $busy.set(false)
    $messages.set([])
    vi.restoreAllMocks()
  })

  it('rewinds to the target user turn and resubmits its text', async () => {
    const requestGateway = vi.fn(async () => ({}) as never)
    let lastState: Record<string, unknown> = {}

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => (lastState = state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={$messages.get()}
      />
    )

    await handle!.restoreToMessage('u1')

    // Ordinal 0 = "truncate before the first visible user message": the gateway
    // drops that turn and everything after, then runs the same text again.
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        session_id: RUNTIME_SESSION_ID,
        text: 'first prompt',
        confirm_truncate: true,
        truncate_before_user_ordinal: 0,
        truncate_before_message_id: 'u1',
        confirm_empty_truncate: true
      },
      1_800_000
    )
    expect((lastState.messages as { id: string }[]).map(m => m.id)).toEqual(['u1'])
    expect(lastState.busy).toBe(true)
  })

  it('rethrows gateway failures and clears the busy flags for the dialog to surface', async () => {
    const requestGateway = vi.fn(async () => {
      throw new Error('gateway exploded')
    })

    let lastState: Record<string, unknown> = {}
    let handle: HarnessHandle | null = null

    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => (lastState = state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await expect(handle!.restoreToMessage('u2')).rejects.toThrow('gateway exploded')
    expect(lastState.busy).toBe(false)
  })

  it('interrupts the live turn and retries past "session busy" when reverting mid-stream', async () => {
    $busy.set(true)

    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'prompt.submit') {
        submitAttempts += 1

        // The cooperative interrupt hasn't wound the turn down yet on the first
        // try; the second attempt lands once the gateway reports idle.
        if (submitAttempts === 1) {
          throw new Error('session busy')
        }
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={$messages.get()}
      />
    )

    await handle!.restoreToMessage('u1')

    expect(requestGateway).toHaveBeenCalledWith('session.interrupt', { session_id: RUNTIME_SESSION_ID })
    expect(submitAttempts).toBe(2)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        session_id: RUNTIME_SESSION_ID,
        text: 'first prompt',
        confirm_truncate: true,
        truncate_before_user_ordinal: 0,
        truncate_before_message_id: 'u1',
        confirm_empty_truncate: true
      },
      1_800_000
    )
  })

  it('rejects non-user targets and unknown ids without touching the gateway', async () => {
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await expect(handle!.restoreToMessage('a1')).rejects.toThrow('Could not find the message to restore.')
    await expect(handle!.restoreToMessage('missing')).rejects.toThrow('Could not find the message to restore.')

    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('uses the clicked runtime user ordinal when the rendered message id is stale', async () => {
    const requestGateway = vi.fn(async () => ({}) as never)

    let lastState: Record<string, unknown> = {}
    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => (lastState = state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={$messages.get()}
      />
    )

    await handle!.restoreToMessage('runtime-user-id-not-in-store', {
      text: 'first prompt',
      userOrdinal: 0
    })

    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        session_id: RUNTIME_SESSION_ID,
        text: 'first prompt',
        confirm_truncate: true,
        truncate_before_user_ordinal: 0,
        truncate_before_message_id: 'u1',
        confirm_empty_truncate: true
      },
      1_800_000
    )
    expect((lastState.messages as { id: string }[]).map(m => m.id)).toEqual(['u1'])
  })
})

describe('usePromptActions file attachment sync', () => {
  afterEach(() => {
    cleanup()
    $connection.set(null)
    $currentCwd.set('')
    vi.restoreAllMocks()
  })

  function fileAttachment(): ComposerAttachment {
    return {
      id: 'file:report.txt',
      kind: 'file',
      label: 'report.txt',
      path: '/Users/alice/Downloads/report.txt',
      refText: '@file:`/Users/alice/Downloads/report.txt`'
    }
  }

  it('uploads file bytes via file.attach on a remote gateway and submits the rewritten ref', async () => {
    // Remote gateway can't read the client-disk path, so the desktop must upload
    // the bytes and submit the workspace-relative ref the gateway hands back —
    // not the original /Users/... path (which would dead-end as "outside the
    // allowed workspace").
    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => 'data:text/plain;base64,aGVsbG8=') }
    })

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'file.attach') {
        return {
          attached: true,
          path: '/remote/work/.hermes/desktop-attachments/report.txt',
          ref_text: '@file:.hermes/desktop-attachments/report.txt',
          uploaded: true
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('convert this to epub', { attachments: [fileAttachment()] })

    expect(ok).toBe(true)
    expect(calls.map(c => c.method)).toEqual(['file.attach', 'prompt.submit'])
    expect(calls[0]?.params).toMatchObject({
      session_id: RUNTIME_SESSION_ID,
      path: '/Users/alice/Downloads/report.txt',
      name: 'report.txt',
      data_url: 'data:text/plain;base64,aGVsbG8='
    })
    expect(calls[1]?.params).toEqual({
      session_id: RUNTIME_SESSION_ID,
      text: '@file:.hermes/desktop-attachments/report.txt\n\nconvert this to epub'
    })
  })

  it('uploads Windows file bytes when local mode fronts a POSIX WSL/Docker backend', async () => {
    $connection.set({ mode: 'local' } as never)
    $currentCwd.set('/root')
    const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,aGVsbG8=')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const attachment: ComposerAttachment = {
      ...fileAttachment(),
      path: 'C:\\Users\\alice\\Downloads\\report.txt',
      refText: '@file:`C:\\Users\\alice\\Downloads\\report.txt`'
    }

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'file.attach') {
        return {
          attached: true,
          path: '/root/.hermes/desktop-attachments/report.txt',
          ref_text: '@file:.hermes/desktop-attachments/report.txt',
          uploaded: true
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.submitText('summarize', { attachments: [attachment] })).toBe(true)
    expect(readFileDataUrl).toHaveBeenCalledWith('C:\\Users\\alice\\Downloads\\report.txt')
    expect(calls[0]).toEqual({
      method: 'file.attach',
      params: {
        data_url: 'data:text/plain;base64,aGVsbG8=',
        name: 'report.txt',
        path: 'C:\\Users\\alice\\Downloads\\report.txt',
        session_id: RUNTIME_SESSION_ID
      }
    })
    expect(calls[1]).toEqual({
      method: 'prompt.submit',
      params: { session_id: RUNTIME_SESSION_ID, text: '@file:.hermes/desktop-attachments/report.txt\n\nsummarize' }
    })
  })

  it('uses image.attach_bytes for a Windows image when the local backend cwd is POSIX', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:image/jpeg;base64,aGVsbG8=')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'image.attach_bytes') {
        return { attached: true, path: '/root/tmp/photo.jpg' } as never
      }

      return {} as never
    })

    const uploaded = await uploadComposerAttachment(
      {
        id: 'image:photo.jpg',
        kind: 'image',
        label: 'photo.jpg',
        path: 'C:\\Users\\alice\\Pictures\\photo.jpg'
      },
      {
        backendCwd: '/root',
        remote: false,
        requestGateway,
        sessionId: RUNTIME_SESSION_ID
      }
    )

    expect(readFileDataUrl).toHaveBeenCalledWith('C:\\Users\\alice\\Pictures\\photo.jpg')
    expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', {
      content_base64: 'aGVsbG8=',
      filename: 'photo.jpg',
      session_id: RUNTIME_SESSION_ID
    })
    expect(requestGateway).not.toHaveBeenCalledWith('image.attach', expect.anything())
    expect(uploaded.path).toBe('/root/tmp/photo.jpg')
  })

  it('uploads file bytes when the terminal backend is a container (docker)', async () => {
    // Container backends have their own filesystem: the host drop path would
    // dangle inside the sandbox, so the bytes must cross via file.attach's
    // data_url pipeline and be staged into a bind-mounted cache dir (#76577).
    $connection.set({ mode: 'local' } as never)
    $terminalBackend.set('docker')
    const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,aGVsbG8=')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'file.attach') {
        return {
          attached: true,
          path: '/root/.hermes/attachments/report.txt',
          ref_text: '@file:/root/.hermes/attachments/report.txt',
          uploaded: true
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.submitText('summarize', { attachments: [fileAttachment()] })).toBe(true)
    expect(readFileDataUrl).toHaveBeenCalledWith('/Users/alice/Downloads/report.txt')
    expect(calls[0]).toEqual({
      method: 'file.attach',
      params: {
        data_url: 'data:text/plain;base64,aGVsbG8=',
        name: 'report.txt',
        path: '/Users/alice/Downloads/report.txt',
        session_id: RUNTIME_SESSION_ID
      }
    })

    // Don't leak the container-backend state into later tests.
    $terminalBackend.set('')
  })

  it('passes a path-less @file: ref straight through (no path = nothing to upload)', async () => {
    // Submit-layer contract: only attachments that carry a `path` are upload
    // candidates. A path-less ref (an @-mention/context ref or pasted text)
    // has no bytes to send, so syncAttachments leaves it untouched and the ref
    // reaches the gateway as-is — correct for workspace-relative refs.
    //
    // The MahmoudR drag-drop bug (a Finder PDF that became a local-path text
    // ref in remote mode) is fixed upstream at the DROP layer: OS drops now
    // carry a path and route through the upload pipeline instead of becoming a
    // path-less inline ref. See partitionDroppedFiles in use-composer-actions.
    $connection.set({ mode: 'remote' } as never)
    const readFileDataUrl = vi.fn(async () => 'data:application/pdf;base64,JVBERi0=')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const pathlessRef: ComposerAttachment = {
      id: 'file:devis',
      kind: 'file',
      label: 'DEVIS_signed.pdf',
      // NOTE: no `path` field — only the pre-baked local @file: ref.
      refText: '@file:`/Users/mahmoud/Downloads/DEVIS_signed.pdf`'
    }

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('read this file', { attachments: [pathlessRef] })

    expect(ok).toBe(true)
    // No path → no file.attach, no byte read: the ref passes through unchanged.
    expect(calls.map(c => c.method)).toEqual(['prompt.submit'])
    expect(readFileDataUrl).not.toHaveBeenCalled()
    expect(calls[0]?.params?.text).toContain('@file:`/Users/mahmoud/Downloads/DEVIS_signed.pdf`')
  })

  it('passes a Windows path directly for a native Windows local backend', async () => {
    $connection.set({ mode: 'local' } as never)
    $currentCwd.set('C:\\Users\\alice\\project')
    const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,c2hvdWxkLW5vdC1iZS1yZWFk')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const attachment: ComposerAttachment = {
      ...fileAttachment(),
      path: 'C:\\Users\\alice\\Downloads\\report.txt',
      refText: '@file:`C:\\Users\\alice\\Downloads\\report.txt`'
    }

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'file.attach') {
        return { attached: true, ref_text: '@file:data/report.txt', uploaded: false } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('summarize', { attachments: [attachment] })

    expect(ok).toBe(true)
    expect(calls[0]?.method).toBe('file.attach')
    expect(readFileDataUrl).not.toHaveBeenCalled()
    // Native Windows local mode shares the same path namespace.
    expect(calls[0]?.params).not.toHaveProperty('data_url')
    expect(calls[1]).toEqual({
      method: 'prompt.submit',
      params: { session_id: RUNTIME_SESSION_ID, text: '@file:data/report.txt\n\nsummarize' }
    })
  })
})

describe('usePromptActions eager-upload races', () => {
  beforeEach(() => {
    setSessions(() => [sessionInfo()])
    $composerAttachments.set([])
  })

  afterEach(() => {
    cleanup()
    $composerAttachments.set([])
    $connection.set(null)
    vi.restoreAllMocks()
  })

  it('joins an in-flight eager upload at submit instead of staging the file twice', async () => {
    // Drop-then-immediately-Enter: the drop kicks off an eager file.attach; if
    // submit doesn't join it, both calls stage the file and leave a duplicate
    // under .hermes/desktop-attachments/. Submit must await the in-flight upload
    // and reuse its gateway-side ref.
    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => 'data:application/pdf;base64,JVBERi0=') }
    })

    let releaseAttach: () => void = () => {}
    const methods: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      methods.push(method)

      if (method === 'file.attach') {
        // Block until released so submit runs while the upload is in flight.
        await new Promise<void>(resolve => {
          releaseAttach = resolve
        })

        return { attached: true, ref_text: '@file:.hermes/desktop-attachments/doc.pdf', uploaded: true } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    // Drop a file → the eager effect fires file.attach and blocks on it.
    $composerAttachments.set([{ id: 'file:doc.pdf', kind: 'file', label: 'doc.pdf', path: '/Users/me/doc.pdf' }])
    await waitFor(() => expect(methods.filter(m => m === 'file.attach').length).toBe(1))

    // Submit reads the store, sees the upload in flight, and joins it.
    const submitting = handle!.submitText('here you go')
    releaseAttach()

    expect(await submitting).toBe(true)
    // Exactly one file.attach (submit reused the eager result), then the send.
    expect(methods.filter(m => m === 'file.attach').length).toBe(1)
    expect(methods).toContain('prompt.submit')
  })
})

describe('usePromptActions sleep/wake session recovery', () => {
  const STORED_SESSION_ID = 'stored-db-xyz789'
  const RECOVERED_SESSION_ID = 'rt-recovered-456'

  afterEach(() => {
    cleanup()
    $turnStartedAt.set(null)
    vi.restoreAllMocks()
  })

  it('resumes the stored session and retries once when prompt.submit reports "session not found"', async () => {
    // After sleep/wake the gateway's in-memory session table is cleared, so the
    // first prompt.submit with the stale runtime id fails. The hook resumes the
    // durable stored id (which survives gateway restarts), gets a fresh live id,
    // and retries the send transparently.
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    const ok = await handle!.submitText('message after wake')

    expect(ok).toBe(true)
    // First submit (stale id) → session.resume (stored id) → retry submit (fresh id).
    expect(calls.map(c => c.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[1]?.params).toEqual({ session_id: STORED_SESSION_ID, source: 'desktop', omit_messages: true })
    expect(calls[2]?.params).toEqual({ session_id: RECOVERED_SESSION_ID, text: 'message after wake' })
  })

  // #67603 (second symptom): a recovery resume must re-register on the session's
  // OWNING profile. Resuming on whichever profile is live forks the conversation
  // into the wrong profile's DB — the session then appears under both profiles.
  it('carries the owning profile from the cache into the recovery resume', async () => {
    setSessions(() => [sessionInfo({ id: STORED_SESSION_ID, profile: 'work' })])

    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('message after wake')).toBe(true)
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true,
      profile: 'work'
    })

    setSessions(() => [])
  })

  // The session lives on another profile and is outside the paginated sidebar
  // cache: resolve it by id across profiles rather than resuming profile-blind.
  it('resolves the owning profile across profiles when the session is not cached', async () => {
    // module-factory vi.fn is not reset by restoreAllMocks — reset explicitly in
    // the finally below so this resolved value never leaks into sibling tests.
    setSessions(() => [])
    vi.mocked(getSession).mockResolvedValue(sessionInfo({ id: STORED_SESSION_ID, profile: 'work' }))

    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('message after wake')).toBe(true)
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true,
      profile: 'work'
    })

    vi.mocked(getSession).mockReset()
    setSessions(() => [])
  })

  it('background queue resume uses the queued stored id and leaves foreground runtime selected', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        // The central binding is stale in lockstep with the caller here: the
        // sleep/wake reaper only clears the GATEWAY's in-memory session, so
        // client-side state still swears by the old runtime id. That is what
        // routes this case to the reactive 404→resume→retry path instead of
        // the proactive binding check (covered by the cross-session drain
        // tests above).
        getRuntimeIdForStoredSession={storedId => (storedId === STORED_SESSION_ID ? 'rt-background-stale' : null)}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId="stored-foreground"
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const ok = await handle!.submitText('queued background message after wake', {
      fromQueue: true,
      sessionId: 'rt-background-stale',
      storedSessionId: STORED_SESSION_ID
    })

    expect(ok).toBe(true)
    expect(calls.map(c => c.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[0]?.params).toEqual({
      queued: true,
      session_id: 'rt-background-stale',
      text: 'queued background message after wake'
    })
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true
    })
    expect(calls[2]?.params).toEqual({
      queued: true,
      session_id: RECOVERED_SESSION_ID,
      text: 'queued background message after wake'
    })
    expect(handle!.activeSessionIdRef.current).toBe(RUNTIME_SESSION_ID)
  })

  it('resumes the stored session and retries once when session.interrupt reports "session not found"', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let interruptAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.interrupt') {
        interruptAttempts += 1

        if (interruptAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.cancelRun()

    expect(calls.map(c => c.method)).toEqual(['session.interrupt', 'session.resume', 'session.interrupt'])
    expect(calls[0]?.params).toEqual({ session_id: RUNTIME_SESSION_ID })
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true
    })
    expect(calls[2]?.params).toEqual({ session_id: RECOVERED_SESSION_ID })
  })

  it('clears the active and cached turn clocks when stopping a turn', async () => {
    const states: Record<string, unknown>[] = []
    const requestGateway = vi.fn(async () => ({}) as never)
    $turnStartedAt.set(1_700_000_000_000)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => states.push(state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.cancelRun()

    expect($turnStartedAt.get()).toBeNull()
    expect(states.at(-1)).toMatchObject({
      awaitingResponse: false,
      busy: false,
      interrupted: true,
      turnStartedAt: null
    })
  })

  it('surfaces the original error (no resume) when the failure is not "session not found"', async () => {
    const calls: string[] = []
    const states: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      if (method === 'prompt.submit') {
        throw new Error('gateway exploded')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => states.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    // submitText swallows the error into an inline bubble and returns false.
    expect(await handle!.submitText('message')).toBe(false)
    // No resume attempt for a non-recoverable error.
    expect(calls).not.toContain('session.resume')
  })

  it('surfaces "session not found" (no resume) when there is no stored session id', async () => {
    const calls: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      if (method === 'prompt.submit') {
        throw new Error('session not found')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={null}
      />
    )

    // With a null stored ref, the `&& selectedStoredSessionIdRef.current` guard
    // short-circuits — no resume is attempted and the error surfaces normally.
    expect(await handle!.submitText('message')).toBe(false)
    expect(calls).not.toContain('session.resume')
  })

  it('recovers via session.resume when prompt.submit TIMES OUT and a stored session is selected (#55578)', async () => {
    // A starved gateway loop rejects with "request timed out: prompt.submit".
    // With a stored session selected, that must recover exactly like
    // "session not found" — resume + retry — not surface an error that leaves
    // activeSessionId null and lets the next send mint a new session.
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('request timed out: prompt.submit')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    const ok = await handle!.submitText('message during starved loop')

    expect(ok).toBe(true)
    expect(calls.map(c => c.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true
    })
    expect(calls[2]?.params).toEqual({
      session_id: RECOVERED_SESSION_ID,
      text: 'message during starved loop'
    })
  })

  it('resumes the SELECTED stored session instead of minting a new one when activeSessionId is null (#55578 split)', async () => {
    // The exact split path from #55578 symptom (b): the runtime binding is
    // gone (orphan-reaped / cleared by a timeout) but a stored session is
    // still selected in the sidebar. A follow-up submit must continue that
    // conversation via session.resume — createBackendSessionForSend would
    // silently fork the user's chat in two.
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    const createBackendSessionForSend = vi.fn(async () => 'brand-new-session-WRONG')

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        createBackendSessionForSend={createBackendSessionForSend}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    const ok = await handle!.submitText('follow-up in the selected chat')

    expect(ok).toBe(true)
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(calls.map(c => c.method)).toEqual(['session.resume', 'prompt.submit'])
    expect(calls[0]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true
    })
    expect(calls[1]?.params).toMatchObject({ session_id: RECOVERED_SESSION_ID })
  })

  it('never replaces a selected stored session when its direct runtime resume fails', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    const busyRef: MutableRefObject<boolean> = { current: false }
    const createBackendSessionForSend = vi.fn(async () => 'brand-new-session-WRONG')

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('4007 session not found on the active profile')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        busyRef={busyRef}
        createBackendSessionForSend={createBackendSessionForSend}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('keep me in the selected conversation')).toBe(false)
    expect(busyRef.current).toBe(false)
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything(), expect.anything())
  })

  it('resumes the ROUTED stored session instead of minting a new one when profile switching cleared both session refs', async () => {
    // A profile swap/reconnect can temporarily clear both volatile ids while
    // the durable route still points at the conversation the user is viewing.
    // Enter during that window must resume the routed chat, never create a
    // contextless session (or create it against the transient wrong profile).
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'rt-wrong-profile' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    let boundRuntimeId: string | null = null
    const createBackendSessionForSend = vi.fn(async () => 'brand-new-session-WRONG')
    const requestGateway = vi.fn(async () => ({}) as never)

    const resumeStoredSession = vi.fn(async (storedSessionId: string) => {
      expect(storedSessionId).toBe(STORED_SESSION_ID)
      selectedStoredSessionIdRef.current = STORED_SESSION_ID
      activeSessionIdRef.current = RECOVERED_SESSION_ID
      boundRuntimeId = RECOVERED_SESSION_ID
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId="rt-wrong-profile"
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => boundRuntimeId}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )

    expect(await handle!.submitText('follow-up while the profile route is rebinding')).toBe(true)
    expect(resumeStoredSession).toHaveBeenCalledWith(STORED_SESSION_ID)
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      { session_id: RECOVERED_SESSION_ID, text: 'follow-up while the profile route is rebinding' },
      1_800_000
    )
  })

  it('lets the durable route replace a stale selected session and runtime before submit', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'rt-wrong-profile' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'stored-wrong-profile' }
    let boundRuntimeId: string | null = null
    const requestGateway = vi.fn(async () => ({}) as never)

    const resumeStoredSession = vi.fn(async () => {
      selectedStoredSessionIdRef.current = STORED_SESSION_ID
      activeSessionIdRef.current = RECOVERED_SESSION_ID
      boundRuntimeId = RECOVERED_SESSION_ID
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId="rt-wrong-profile"
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => boundRuntimeId}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('stay in the routed profile session')).toBe(true)
    expect(resumeStoredSession).toHaveBeenCalledWith(STORED_SESSION_ID)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      { session_id: RECOVERED_SESSION_ID, text: 'stay in the routed profile session' },
      1_800_000
    )
  })

  it('submits directly when the routed stored session already owns the live runtime', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: RECOVERED_SESSION_ID }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED_SESSION_ID }
    const requestGateway = vi.fn(async () => ({}) as never)
    const resumeStoredSession = vi.fn()

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={RECOVERED_SESSION_ID}
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => RECOVERED_SESSION_ID}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('normal follow-up')).toBe(true)
    expect(resumeStoredSession).not.toHaveBeenCalled()
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      { session_id: RECOVERED_SESSION_ID, text: 'normal follow-up' },
      1_800_000
    )
  })

  it('never falls through to session.create or a stale runtime when routed-session recovery fails', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'rt-wrong-profile' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED_SESSION_ID }
    const busyRef: MutableRefObject<boolean> = { current: false }
    let recoverySucceeds = false
    let boundRuntimeId: string | null = null

    const createBackendSessionForSend = vi.fn(async () => 'brand-new-session-WRONG')
    const requestGateway = vi.fn(async () => ({}) as never)

    const resumeStoredSession = vi.fn(async () => {
      if (!recoverySucceeds) {
        return
      }

      activeSessionIdRef.current = RECOVERED_SESSION_ID
      boundRuntimeId = RECOVERED_SESSION_ID
    })

    $messages.set([])

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId="rt-wrong-profile"
        activeSessionIdRef={activeSessionIdRef}
        busyRef={busyRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => boundRuntimeId}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('do not fork me')).toBe(false)
    expect(busyRef.current).toBe(false)
    expect($messages.get()).toEqual([])
    expect(resumeStoredSession).toHaveBeenCalledWith(STORED_SESSION_ID)
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything(), expect.anything())

    // Prove the failed attempt released the per-session submit lock. The next
    // send can recover and submit instead of being silently rejected forever.
    recoverySucceeds = true
    expect(await handle!.submitText('retry after recovery')).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      { session_id: RECOVERED_SESSION_ID, text: 'retry after recovery' },
      1_800_000
    )
  })

  it('still creates a new session for a genuine new-chat draft (no stored session selected)', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }

    // Mirror the real createBackendSessionForSend: a successful create
    // re-homes the active runtime ref to the session it minted BEFORE
    // returning. An inert stub here is what let the new-chat drift-abort
    // regression ship green.
    const createBackendSessionForSend = vi.fn(async () => {
      activeSessionIdRef.current = RUNTIME_SESSION_ID

      return RUNTIME_SESSION_ID
    })

    const calls: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={null}
      />
    )

    const ok = await handle!.submitText('first message of a new chat')

    expect(ok).toBe(true)
    expect(createBackendSessionForSend).toHaveBeenCalledTimes(1)
    expect(calls).not.toContain('session.resume')
  })
})

describe('usePromptActions submit session-context isolation (#54527)', () => {
  const STORED_SESSION_A = 'stored-project-a'
  const STORED_SESSION_B = 'stored-project-b'
  const RUNTIME_SESSION_B = 'rt-session-b-wrong'

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    setSessions(() => [])
  })

  it('aborts submit when the user switches sessions during session.resume (no misroute)', async () => {
    // Exact #54527 failure: user submits in Session A while its runtime binding
    // is gone; before resume returns they switch to Session B. Without a pinned
    // context the resumed runtime id belongs to B and A's text lands in the
    // wrong chat — permanently lost from A.
    let releaseResume: () => void = () => {}
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED_SESSION_A }
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        await new Promise<void>(resolve => {
          releaseResume = resolve
        })

        // Simulate the user switching to Session B while resume is in flight.
        selectedStoredSessionIdRef.current = STORED_SESSION_B
        activeSessionIdRef.current = RUNTIME_SESSION_B

        return { session_id: RUNTIME_SESSION_B } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_SESSION_A}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const submitting = handle!.submitText('carefully composed prompt for project A')
    await waitFor(() => expect(calls.some(c => c.method === 'session.resume')).toBe(true))
    releaseResume()

    expect(await submitting).toBe(false)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(false)
    expect(calls.find(c => c.method === 'session.resume')?.params).toEqual({
      session_id: STORED_SESSION_A,
      source: 'desktop',
      omit_messages: true
    })
  })

  it('does not false-positive-abort when the session has rotated via compression (lineage root vs tip)', async () => {
    // The composer keys drafts/attachments on the DURABLE lineage root
    // (resolveComposerSessionKey / sessionPinId — survives auto-compression
    // tip rotation), but selectedStoredSessionIdRef tracks the CURRENT TIP.
    // For any session that has compressed at least once, root !== tip — if
    // composerScope is compared against the raw tip, every legitimate submit
    // into that session would look like drift.
    const ROOT_ID = 'stored-root-original'
    const TIP_ID = 'stored-tip-after-compression'

    setSessions(() => [sessionInfo({ id: TIP_ID, _lineage_root_id: ROOT_ID })])

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={TIP_ID}
      />
    )

    // The composer's scope is the lineage root (what resolveComposerSessionKey
    // actually returns for this session) — a legitimate, non-drifted submit.
    const ok = await handle!.submitText('message into the rotated session', { composerScope: ROOT_ID })

    expect(ok).toBe(true)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(true)
  })

  it('aborts submit when the composer scope disagrees with the resolved target (#59305)', async () => {
    // The composer (ChatBar) and the session-side refs live in separate React
    // subtrees; each can be internally consistent yet still disagree with each
    // other at the instant of send if the two updated on different commits.
    // composerScope carries the composer's own snapshot of "what session was
    // loaded" into submit.ts, which must refuse to send when it disagrees with
    // the session the submit is actually about to target.
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_A}
      />
    )

    const ok = await handle!.submitText('typed while B was on screen', { composerScope: STORED_SESSION_B })

    expect(ok).toBe(false)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(false)
  })

  it('submits normally when the composer scope agrees with the resolved target', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_A}
      />
    )

    const ok = await handle!.submitText('typed while A was on screen', { composerScope: STORED_SESSION_A })

    expect(ok).toBe(true)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(true)
  })

  it('aborts recovery submit when the user switches sessions during timeout resume', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    let releaseResume: () => void = () => {}

    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED_SESSION_A }

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('request timed out: prompt.submit')
        }
      }

      if (method === 'session.resume') {
        await new Promise<void>(resolve => {
          releaseResume = resolve
        })
        selectedStoredSessionIdRef.current = STORED_SESSION_B

        return { session_id: RUNTIME_SESSION_B } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_SESSION_A}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const submitting = handle!.submitText('message that must not land in session B')
    await waitFor(() => expect(calls.some(c => c.method === 'session.resume')).toBe(true))
    releaseResume()

    expect(await submitting).toBe(false)
    expect(submitAttempts).toBe(1)
    expect(calls.filter(c => c.method === 'prompt.submit')).toHaveLength(1)
    expect(calls.find(c => c.method === 'session.resume')?.params).toMatchObject({
      session_id: STORED_SESSION_A
    })
  })

  it('submits the first prompt of a new chat — the create pipeline re-homing selection/route is not user drift', async () => {
    // Regression for the #54527 guard breaking every NEW chat: on a fresh draft
    // (no stored session, no runtime session) createBackendSessionForSend
    // legitimately sets selectedStoredSessionIdRef + navigates to the new
    // session's route. Comparing against the pre-create (null) baseline made
    // the guard read that self-inflicted move as a user switch and abort, so
    // prompt.submit never fired: the message vanished, no DB row was ever
    // persisted, and the desktop stranded on a route whose REST reads 404
    // ("Session not found").
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let routeToken = '/'

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    // Mirror the real createBackendSessionForSend: on success it re-homes the
    // refs AND the route to the session it just created.
    const createBackendSessionForSend = vi.fn(async () => {
      activeSessionIdRef.current = 'rt-new-chat'
      selectedStoredSessionIdRef.current = 'stored-new-chat'
      routeToken = '/stored-new-chat'

      return 'rt-new-chat'
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRouteToken={() => routeToken}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    expect(await handle!.submitText('first message of a brand-new chat')).toBe(true)
    expect(createBackendSessionForSend).toHaveBeenCalledTimes(1)
    expect(calls.find(c => c.method === 'prompt.submit')?.params).toMatchObject({
      session_id: 'rt-new-chat'
    })
  })

  it('aborts when the user switches sessions during the tail of a successful create', async () => {
    // createBackendSessionForSend awaits once more (armed-YOLO apply) AFTER
    // committing the refs and returning a real id, so a switch in that window
    // escapes its internal null-return drift check. The active ref is the
    // tell: every switch path retargets it synchronously, so it no longer
    // equals the id create returned. The submit must abort, not adopt the
    // switched-to context as its re-pinned baseline.
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let routeToken = '/'

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    const createBackendSessionForSend = vi.fn(async () => {
      // The user switched to Session B during the post-commit await: the
      // switch path re-homed all three context markers before create returned.
      activeSessionIdRef.current = RUNTIME_SESSION_B
      selectedStoredSessionIdRef.current = STORED_SESSION_B
      routeToken = `/${STORED_SESSION_B}`

      return 'rt-new-chat'
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRouteToken={() => routeToken}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    expect(await handle!.submitText('message that must not land in session B')).toBe(false)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(false)
  })
})

describe('usePromptActions new-chat first-send delivery (#63078)', () => {
  const NEW_RUNTIME_ID = 'rt-first-send'
  const NEW_STORED_ID = 'stored-first-send'

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $connection.set(null)
    $composerAttachments.set([])
  })

  it('delivers the first message of a new chat through the intentional route transition (#62562)', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    let routeToken = '/::'

    // Mirror the real session creator: session.create selects the persisted
    // row and replaces the new-chat route with the created session's URL
    // BEFORE returning. The submit pipeline must adopt that intentional
    // transition as its new pinned target — not mistake it for the user
    // switching conversations and stop before prompt.submit.
    const createBackendSessionForSend = vi.fn(async (preview?: null | string) => {
      expect(preview).toBe('first message of a new chat')
      activeSessionIdRef.current = NEW_RUNTIME_ID
      selectedStoredSessionIdRef.current = NEW_STORED_ID
      routeToken = `/${NEW_STORED_ID}::`

      return NEW_RUNTIME_ID
    })

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRouteToken={() => routeToken}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    expect(await handle!.submitText('first message of a new chat')).toBe(true)
    expect(createBackendSessionForSend).toHaveBeenCalledTimes(1)
    // The FULL RPC transcript: exactly one prompt.submit, addressed to the
    // created runtime session, carrying the user's text — no session.resume
    // detour and, critically, no silent drop before the submit.
    expect(calls).toEqual([
      {
        method: 'prompt.submit',
        params: { session_id: NEW_RUNTIME_ID, text: 'first message of a new chat' }
      }
    ])
  })

  it('delivers the first prompt when React Router commits the created-session route late (#62990)', async () => {
    // The creator requests the navigation, but React Router can still expose
    // the OLD new-chat route to the submit continuation for a beat, committing
    // the created session's URL only before the next await settles. Neither
    // route snapshot (stale '/', then the late-committed session route) is a
    // user switch — both are the pipeline's own transition and the first
    // prompt must still reach the gateway.
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    let sessionCreated = false
    let routeReadsAfterCreate = 0

    const createBackendSessionForSend = vi.fn(async () => {
      activeSessionIdRef.current = NEW_RUNTIME_ID
      selectedStoredSessionIdRef.current = NEW_STORED_ID
      sessionCreated = true

      return NEW_RUNTIME_ID
    })

    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRouteToken={() => {
          if (!sessionCreated) {
            return '/::'
          }

          routeReadsAfterCreate += 1

          // React Router can still expose / to the outer submit continuation,
          // then commit the created-session route before the next await settles.
          return routeReadsAfterCreate === 1 ? '/::' : `/${NEW_STORED_ID}::`
        }}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    expect(await handle!.submitText('hello')).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        session_id: NEW_RUNTIME_ID,
        text: 'hello'
      },
      1_800_000
    )
  })

  it('aborts a new-session submit when sidebar navigation changes the route before its selected ref (#62562)', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    let routeToken = '/::'

    let releaseAttach: () => void = () => {}

    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => 'data:text/plain;base64,aGVsbG8=') }
    })

    const createBackendSessionForSend = vi.fn(async () => {
      activeSessionIdRef.current = NEW_RUNTIME_ID
      selectedStoredSessionIdRef.current = NEW_STORED_ID
      routeToken = `/${NEW_STORED_ID}::`

      return NEW_RUNTIME_ID
    })

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'file.attach') {
        await new Promise<void>(resolve => {
          releaseAttach = resolve
        })

        return {
          attached: true,
          path: '/remote/work/report.txt',
          ref_text: '@file:report.txt',
          uploaded: true
        } as never
      }

      return {} as never
    })

    const attachment: ComposerAttachment = {
      id: 'file:report.txt',
      kind: 'file',
      label: 'report.txt',
      path: '/Users/alice/report.txt',
      refText: '@file:`/Users/alice/report.txt`'
    }

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRouteToken={() => routeToken}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const submitting = handle!.submitText('first message', { attachments: [attachment] })
    await waitFor(() => expect(calls.some(call => call.method === 'file.attach')).toBe(true))

    // selectSidebarItem calls navigate() first. The routed effect has not yet
    // entered resumeSession(), so the selected-session ref still points at the
    // just-created session when attachment sync settles — the route move to a
    // DIFFERENT chat must abort on its own.
    routeToken = '/sidebar-target::'
    releaseAttach()

    expect(await submitting).toBe(false)
    expect(selectedStoredSessionIdRef.current).toBe(NEW_STORED_ID)
    expect(calls.some(call => call.method === 'prompt.submit')).toBe(false)
  })

  it('still aborts when the user genuinely switches chats after create, during attachment sync (#62805)', async () => {
    // The post-create re-baseline (adopting the created chat as the pinned
    // target) must not mask a REAL switch later in the pipeline: the user
    // clicks a different session after createBackendSessionForSend lands but
    // before attachment sync settles — selection AND route both move to the
    // other chat, and the drift check at the post-attachments boundary must
    // abort rather than deliver the text into whichever chat won the race.
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    let routeToken = '/::'

    let releaseFileAttach: () => void = () => {}

    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => 'data:application/pdf;base64,JVBERi0=') }
    })

    const createBackendSessionForSend = vi.fn(async () => {
      activeSessionIdRef.current = NEW_RUNTIME_ID
      selectedStoredSessionIdRef.current = NEW_STORED_ID
      routeToken = `/${NEW_STORED_ID}::`

      return NEW_RUNTIME_ID
    })

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'file.attach') {
        // Block here so the user can switch sessions mid-sync.
        await new Promise<void>(resolve => {
          releaseFileAttach = resolve
        })

        return {
          attached: true,
          ref_text: '@file:.hermes/desktop-attachments/test.pdf',
          uploaded: true
        } as never
      }

      return {} as never
    })

    const attachment: ComposerAttachment = {
      id: 'file:test',
      kind: 'file',
      label: 'test.pdf',
      path: '/abs/test.pdf',
      refText: '@file:`/abs/test.pdf`'
    }

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRouteToken={() => routeToken}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const submitting = handle!.submitText('message before the switch', { attachments: [attachment] })
    await waitFor(() => expect(calls.some(c => c.method === 'file.attach')).toBe(true))

    // Simulate a user switching to a different session after the new session
    // was created and the sync phase started.
    selectedStoredSessionIdRef.current = 'stored-other-session'
    routeToken = '/stored-other-session::'
    releaseFileAttach()

    expect(await submitting).toBe(false)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(false)
  })
})

describe('usePromptActions busy-gateway churn tolerance (#64327)', () => {
  const STORED_ID = 'stored-busy-gw'
  const RESUMED_RUNTIME_ID = 'rt-busy-gw'

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('does not abort a send when programmatic gateway churn fires mid-submit (selection null-reset + search/hash route churn)', async () => {
    // The busy-gateway superset of #63078: with background streaming sessions,
    // per-minute cron sessions, or a messaging surface active, the selected
    // stored id gets null-reset by gateway/profile reconnects and overlays
    // park state in location.search/hash. None of that is the user changing
    // chats — a send from a second chat must ride through it and reach
    // prompt.submit instead of silently aborting.
    let releaseResume: () => void = () => {}
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED_ID }
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let routeToken = `/${STORED_ID}::`

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        await new Promise<void>(resolve => {
          releaseResume = resolve
        })

        return { session_id: RESUMED_RUNTIME_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        getRouteToken={() => routeToken}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_ID}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const submitting = handle!.submitText('send from a second chat on a busy gateway')
    await waitFor(() => expect(calls.some(c => c.method === 'session.resume')).toBe(true))

    // Programmatic churn while resume is in flight — NOT user switches:
    // a gateway/profile reconnect null-resets the selection...
    selectedStoredSessionIdRef.current = null
    // ...a background event retargets the active runtime ref (#47709 class)...
    activeSessionIdRef.current = 'rt-some-background-session'
    // ...and an overlay parks state in search/hash (pathname unchanged).
    routeToken = `/${STORED_ID}:?panel=preview:#reply`
    releaseResume()

    expect(await submitting).toBe(true)
    expect(calls.find(c => c.method === 'prompt.submit')?.params).toMatchObject({
      session_id: RESUMED_RUNTIME_ID,
      text: 'send from a second chat on a busy gateway'
    })
  })

  it('still aborts when the user genuinely moves to a different chat mid-submit', async () => {
    // The churn tolerance must not weaken the real guard: selection AND route
    // moving to another actual chat is a user switch and must abort.
    let releaseResume: () => void = () => {}
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED_ID }
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    let routeToken = `/${STORED_ID}::`

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        await new Promise<void>(resolve => {
          releaseResume = resolve
        })

        return { session_id: RESUMED_RUNTIME_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        getRouteToken={() => routeToken}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_ID}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const submitting = handle!.submitText('must not land in the other chat')
    await waitFor(() => expect(calls.some(c => c.method === 'session.resume')).toBe(true))

    // A genuine switch: the user clicks another chat, which retargets
    // selection and route synchronously.
    selectedStoredSessionIdRef.current = 'stored-other-chat'
    routeToken = '/stored-other-chat::'
    releaseResume()

    expect(await submitting).toBe(false)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(false)
  })
})

describe('usePromptActions eager attachment upload (drop-time)', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $connection.set(null)
    $composerAttachments.set([])
  })

  it('uploads a dropped file the moment it lands (active session) and rewrites the chip with the gateway ref', async () => {
    // A Finder drop adds a chip with a local path but no attachedSessionId. With
    // a session already open, the hook should stage it right away — so the send
    // is instant and the card can show a spinner while bytes upload — instead of
    // waiting for submit.
    $connection.set({ mode: 'remote' } as never)
    const readFileDataUrl = vi.fn(async () => 'data:application/pdf;base64,JVBERi0=')
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { readFileDataUrl } })

    const calls: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      if (method === 'file.attach') {
        return {
          attached: true,
          ref_text: '@file:.hermes/desktop-attachments/DEVIS_signed.pdf',
          uploaded: true
        } as never
      }

      return {} as never
    })

    $composerAttachments.set([
      { id: 'file:devis', kind: 'file', label: 'DEVIS_signed.pdf', path: '/Users/mahmoud/Downloads/DEVIS_signed.pdf' }
    ])

    await actRender(
      <Harness onReady={() => undefined} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await waitFor(() => expect(calls).toContain('file.attach'))
    await waitFor(() => expect($composerAttachments.get()[0]?.attachedSessionId).toBe(RUNTIME_SESSION_ID))

    const chip = $composerAttachments.get()[0]!
    expect(chip.refText).toBe('@file:.hermes/desktop-attachments/DEVIS_signed.pdf')
    expect(chip.uploadState).toBeUndefined()
    expect(readFileDataUrl).toHaveBeenCalledWith('/Users/mahmoud/Downloads/DEVIS_signed.pdf')
  })

  it('flags the chip uploadState=error when the eager upload fails, keeping the path so submit can retry', async () => {
    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => 'data:application/pdf;base64,JVBERi0=') }
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'file.attach') {
        throw new Error('[Errno 13] Permission denied')
      }

      return {} as never
    })

    $composerAttachments.set([{ id: 'file:x', kind: 'file', label: 'x.pdf', path: '/abs/x.pdf' }])

    await actRender(
      <Harness onReady={() => undefined} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await waitFor(() => expect($composerAttachments.get()[0]?.uploadState).toBe('error'))
    expect($composerAttachments.get()[0]?.attachedSessionId).toBeUndefined()
    expect($composerAttachments.get()[0]?.path).toBe('/abs/x.pdf')
  })

  it('does not eagerly re-upload a chip already attached to this session', async () => {
    $connection.set({ mode: 'remote' } as never)
    const requestGateway = vi.fn(async () => ({}) as never)

    $composerAttachments.set([
      {
        id: 'file:done',
        kind: 'file',
        label: 'done.pdf',
        path: '/abs/done.pdf',
        refText: '@file:data/done.pdf',
        attachedSessionId: RUNTIME_SESSION_ID
      }
    ])

    await actRender(
      <Harness onReady={() => undefined} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    await Promise.resolve()
    expect(requestGateway).not.toHaveBeenCalledWith('file.attach', expect.anything())
  })
})

describe('uploadComposerAttachment remote read failures', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('turns the raw 16MB IPC cap error into a friendly remote-gateway message', async () => {
    // electron/hardening.ts rejects the readFileDataUrl IPC with this exact
    // shape when a file exceeds the configured data-URL read cap.
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        readFileDataUrl: vi.fn(async () => {
          throw new Error('File preview failed: file is too large (20971520 bytes; limit 16777216 bytes).')
        })
      }
    })

    const requestGateway = vi.fn(async () => ({}) as never)

    await expect(
      uploadComposerAttachment(
        { id: 'file:big', kind: 'file', label: 'huge.csv', path: '/abs/huge.csv' },
        { remote: true, requestGateway, sessionId: RUNTIME_SESSION_ID }
      )
    ).rejects.toThrow('huge.csv is too large to upload to the remote gateway (max 16 MB).')

    // The cap is hit before any gateway round-trip.
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('passes non-cap read errors through unchanged', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        readFileDataUrl: vi.fn(async () => {
          throw new Error('ENOENT: no such file')
        })
      }
    })

    await expect(
      uploadComposerAttachment(
        { id: 'file:gone', kind: 'file', label: 'gone.csv', path: '/abs/gone.csv' },
        { remote: true, requestGateway: vi.fn(async () => ({}) as never), sessionId: RUNTIME_SESSION_ID }
      )
    ).rejects.toThrow('ENOENT: no such file')
  })
})

// The actions bag is a STABLE ref that wiring.tsx mutates in place
// (Object.assign), and the pane surfaces are memoized on that stable ref — so a
// surface does NOT re-render when the active session changes and its props keep
// holding whichever `usePromptActions` closure was current when it last
// rendered. `activeSessionIdRef` is therefore the authority (mirrored during
// render in use-session-state-cache, and pinned imperatively mid-flight by
// submit.ts / use-session-actions without touching the source prop), while the
// closure-captured `activeSessionId` prop is stale by design.
//
// Every action below used `activeSessionId || activeSessionIdRef.current`, which
// prefers the STALE value whenever it is non-null and only consults the fresh
// ref once the prop is null. That routed history-mutating writes and live-turn
// corrections into the previously-focused chat: content the user typed in chat B
// reached chat A's agent, and rewinds truncated the wrong session's transcript.
// `cancelRun` in the same file already reads the ref exclusively and documents
// exactly this hazard.
describe('usePromptActions stale-closure session routing', () => {
  const RUNTIME_SESSION_B = 'rt-session-b-current'

  beforeEach(() => {
    // Earlier suites in this file leave `$busy` true (it is a module-level
    // store, shared across tests). reloadFromMessage bails on a busy session,
    // so without this reset the regeneration case never reaches its routing
    // decision and would pass vacuously.
    $busy.set(false)
  })

  afterEach(() => {
    cleanup()
    setMessages([])
    $busy.set(false)
    vi.restoreAllMocks()
  })

  type GatewayCall = [string, Record<string, unknown>?]
  type GatewayRequestFn = <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>
  type GatewayMock = GatewayRequestFn & { mock: { calls: unknown[][] } }

  function gatewayCalls(requestGateway: GatewayMock): GatewayCall[] {
    return requestGateway.mock.calls as unknown as GatewayCall[]
  }

  // Renders with `activeSessionId` (the prop) pinned to session A, then moves
  // the ref to session B — the exact split a memoized surface holds after the
  // user switches chats. Every action must target B.
  //
  // The rewind/reload planners read the GLOBAL `$messages` store (the view
  // transcript), not the harness's per-session state, so the fixture has to
  // seed both or the action bails on a null plan and the test would pass
  // vacuously without ever reaching the routing decision.
  async function renderWithStaleClosure(requestGateway: GatewayMock, seedMessages?: unknown[]) {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: RUNTIME_SESSION_ID }
    const updated: string[] = []

    if (seedMessages) {
      setMessages(seedMessages as never)
    }

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={RUNTIME_SESSION_ID}
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        onUpdateState={sessionId => updated.push(sessionId)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={seedMessages}
        selectedStoredSessionIdRef={{ current: null }}
      />
    )

    // The user switches to session B. The ref follows; the prop captured in the
    // memoized surface's closure still says session A.
    activeSessionIdRef.current = RUNTIME_SESSION_B

    return { handle: handle!, updated }
  }

  it('redirects the live turn into the CURRENT session, not the stale closure session', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'redirected' }) as never) as unknown as GatewayMock
    const { handle } = await renderWithStaleClosure(requestGateway)

    await handle.redirectPrompt('actually use Postgres')

    // A redirect reaches the model mid-turn. Sent to the stale session, the
    // correction lands in a conversation the user is no longer looking at —
    // this is the observed "session suddenly working on another chat's task".
    expect(requestGateway).toHaveBeenCalledWith('session.redirect', {
      session_id: RUNTIME_SESSION_B,
      text: 'actually use Postgres'
    })
    expect(requestGateway).not.toHaveBeenCalledWith(
      'session.redirect',
      expect.objectContaining({ session_id: RUNTIME_SESSION_ID })
    )
  })

  it('regenerates against the CURRENT session, not the stale closure session', async () => {
    const requestGateway = vi.fn(async () => ({}) as never) as unknown as GatewayMock

    const { handle, updated } = await renderWithStaleClosure(requestGateway, [
      { id: 'u1', parts: [textPart('original prompt')], role: 'user', timestamp: 0 },
      { id: 'a1', parts: [textPart('reply')], role: 'assistant', timestamp: 1 }
    ])

    await handle.reloadFromMessage('u1')

    // prompt.submit with a truncate ordinal DELETES history after that point.
    // Aimed at the stale session it destroys the wrong transcript.
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      expect.objectContaining({ session_id: RUNTIME_SESSION_B }),
      expect.anything()
    )
    expect(requestGateway).not.toHaveBeenCalledWith(
      'prompt.submit',
      expect.objectContaining({ session_id: RUNTIME_SESSION_ID }),
      expect.anything()
    )
    expect(updated).toContain(RUNTIME_SESSION_B)
    expect(updated).not.toContain(RUNTIME_SESSION_ID)
  })

  it('restores a checkpoint in the CURRENT session, not the stale closure session', async () => {
    const requestGateway = vi.fn(async () => ({}) as never) as unknown as GatewayMock

    const { handle, updated } = await renderWithStaleClosure(requestGateway, [
      { id: 'u1', parts: [textPart('first prompt')], role: 'user', timestamp: 0 },
      { id: 'a1', parts: [textPart('first reply')], role: 'assistant', timestamp: 1 },
      { id: 'u2', parts: [textPart('second prompt')], role: 'user', timestamp: 2 }
    ])

    await handle.restoreToMessage('u2')

    // A rewind is destructive; the optimistic truncation must also be applied
    // to the session that actually receives the rewind.
    expect(updated).toContain(RUNTIME_SESSION_B)
    expect(updated).not.toContain(RUNTIME_SESSION_ID)

    for (const [, params] of gatewayCalls(requestGateway)) {
      if (params && 'session_id' in params) {
        expect(params.session_id).toBe(RUNTIME_SESSION_B)
      }
    }
  })

  it('edits a message in the CURRENT session, not the stale closure session', async () => {
    const requestGateway = vi.fn(async () => ({}) as never) as unknown as GatewayMock

    const { handle, updated } = await renderWithStaleClosure(requestGateway, [
      { id: 'u1', parts: [textPart('original prompt')], role: 'user', timestamp: 0 },
      { id: 'a1', parts: [textPart('reply')], role: 'assistant', timestamp: 1 }
    ])

    await handle.editMessage({
      content: [{ text: 'edited prompt', type: 'text' }],
      parentId: null,
      role: 'user',
      sourceId: 'u1'
    } as never)

    expect(updated).toContain(RUNTIME_SESSION_B)
    expect(updated).not.toContain(RUNTIME_SESSION_ID)

    for (const [, params] of gatewayCalls(requestGateway)) {
      if (params && 'session_id' in params) {
        expect(params.session_id).toBe(RUNTIME_SESSION_B)
      }
    }
  })
})
