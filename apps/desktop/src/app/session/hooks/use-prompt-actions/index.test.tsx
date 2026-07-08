import { cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { textPart } from '@/lib/chat-messages'
import { $composerAttachments, $composerDraft, type ComposerAttachment, setComposerDraft } from '@/store/composer'
import { $busy, $connection, $messages, $sessions, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { uploadComposerAttachment, usePromptActions } from '.'

vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
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

interface HarnessHandle {
  cancelRun: () => Promise<void>
  restoreToMessage: (messageId: string, target?: { text?: string; userOrdinal?: number | null }) => Promise<void>
  steerPrompt: (text: string) => Promise<boolean>
  submitText: (text: string, options?: { attachments?: ComposerAttachment[]; fromQueue?: boolean }) => Promise<boolean>
}

function Harness({
  busyRef,
  onReady,
  onSeedState,
  openMemoryGraph,
  refreshSessions,
  requestGateway,
  resumeStoredSession,
  seedMessages,
  storedSessionId,
  activeSessionId,
  createBackendSessionForSend
}: {
  busyRef?: MutableRefObject<boolean>
  onReady: (handle: HarnessHandle) => void
  onSeedState?: (state: Record<string, unknown>) => void
  openMemoryGraph?: () => void
  refreshSessions: () => Promise<void>
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  resumeStoredSession?: (storedSessionId: string) => Promise<void> | void
  seedMessages?: unknown[]
  storedSessionId?: null | string
  activeSessionId?: null | string
  createBackendSessionForSend?: () => Promise<null | string>
}) {
  const activeSessionIdRef: MutableRefObject<string | null> = {
    current: activeSessionId === undefined ? RUNTIME_SESSION_ID : activeSessionId
  }

  const selectedStoredSessionIdRef: MutableRefObject<string | null> = {
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
    handleSkinCommand: () => '',
    openMemoryGraph: openMemoryGraph ?? (() => undefined),
    refreshSessions,
    requestGateway,
    resumeStoredSession: resumeStoredSession ?? (() => undefined),
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState: (_sessionId, updater) => {
      // Seed with interrupted:true so we can prove a fresh submit clears it.
      const next = updater(stateRef.current) as unknown as Record<string, unknown>
      stateRef.current = next as never
      onSeedState?.(next)

      return next as never
    }
  })

  useEffect(() => {
    onReady({
      cancelRun: actions.cancelRun,
      restoreToMessage: actions.restoreToMessage,
      steerPrompt: actions.steerPrompt,
      submitText: actions.submitText
    })
  }, [actions.cancelRun, actions.restoreToMessage, actions.steerPrompt, actions.submitText, onReady])

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
    render(<Harness onReady={h => (handle = h)} refreshSessions={refreshSessions} requestGateway={requestGateway} />)

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
    render(<Harness onReady={h => (handle = h)} refreshSessions={refreshSessions} requestGateway={requestGateway} />)

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
    render(<Harness onReady={h => (handle = h)} refreshSessions={refreshSessions} requestGateway={requestGateway} />)

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
    render(<Harness onReady={h => (handle = h)} refreshSessions={refreshSessions} requestGateway={requestGateway} />)

    await handle!.submitText('/title way too long title')

    expect(requestGateway).toHaveBeenCalledWith(
      'session.title',
      expect.objectContaining({ title: 'way too long title' })
    )
    expect(refreshSessions).not.toHaveBeenCalled()
    expect($sessions.get()[0]?.title).toBe('Old title')
  })
})

describe('usePromptActions slash.exec dispatch payloads', () => {
  afterEach(() => {
    cleanup()
    $busy.set(false)
    vi.restoreAllMocks()
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
    render(
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
    render(
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
    render(
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
    render(
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
    render(
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
    render(
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
    render(
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

  it('a fromQueue drain sends even when busyRef is still true on the settle edge', async () => {
    // busyRef lags $busy by one effect tick on the busy→false settle edge, so a
    // drained queue send would otherwise hit the busy guard and silently no-op.
    const busyRef = { current: true }
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    render(
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
        session_id: RUNTIME_SESSION_ID,
        text: 'queued message'
      },
      1_800_000
    )
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
    render(
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
    render(
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
    render(
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

describe('usePromptActions steerPrompt', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('injects the trimmed text via session.steer and reports acceptance on a queued status', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'queued' }) as never)

    let handle: HarnessHandle | null = null
    render(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const accepted = await handle!.steerPrompt('  nudge the run  ')

    expect(accepted).toBe(true)
    // Steer never starts a turn — it rides the live run via session.steer only.
    expect(requestGateway).toHaveBeenCalledWith('session.steer', {
      session_id: RUNTIME_SESSION_ID,
      text: 'nudge the run'
    })
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything())
  })

  it('reports rejection (so the caller queues) when the gateway has no live tool window', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'rejected' }) as never)

    let handle: HarnessHandle | null = null
    render(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.steerPrompt('too late')).toBe(false)
  })

  it('reports rejection (never throws) when the steer RPC errors', async () => {
    const requestGateway = vi.fn(async () => {
      throw new Error('agent does not support steer')
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.steerPrompt('boom')).toBe(false)
  })

  it('skips the RPC entirely for empty text', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'queued' }) as never)

    let handle: HarnessHandle | null = null
    render(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    expect(await handle!.steerPrompt('   ')).toBe(false)
    expect(requestGateway).not.toHaveBeenCalled()
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
    render(
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
        truncate_before_user_ordinal: 0
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

    render(
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
    render(
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
        truncate_before_user_ordinal: 0
      },
      1_800_000
    )
  })

  it('rejects non-user targets and unknown ids without touching the gateway', async () => {
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    render(
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
    render(
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
        truncate_before_user_ordinal: 0
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
    render(
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
    render(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('read this file', { attachments: [pathlessRef] })

    expect(ok).toBe(true)
    // No path → no file.attach, no byte read: the ref passes through unchanged.
    expect(calls.map(c => c.method)).toEqual(['prompt.submit'])
    expect(readFileDataUrl).not.toHaveBeenCalled()
    expect(calls[0]?.params?.text).toContain('@file:`/Users/mahmoud/Downloads/DEVIS_signed.pdf`')
  })

  it('passes the path directly via file.attach in local mode (no byte upload)', async () => {
    $connection.set({ mode: 'local' } as never)

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'file.attach') {
        return { attached: true, ref_text: '@file:data/report.txt', uploaded: false } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('summarize', { attachments: [fileAttachment()] })

    expect(ok).toBe(true)
    expect(calls[0]?.method).toBe('file.attach')
    // Local mode sends no data_url — the gateway shares this disk.
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
    render(
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
    render(
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
    expect(calls[1]?.params).toEqual({ session_id: STORED_SESSION_ID, source: 'desktop' })
    expect(calls[2]?.params).toEqual({ session_id: RECOVERED_SESSION_ID, text: 'message after wake' })
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
    render(
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
    expect(calls[1]?.params).toEqual({ session_id: STORED_SESSION_ID, source: 'desktop' })
    expect(calls[2]?.params).toEqual({ session_id: RECOVERED_SESSION_ID })
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
    render(
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
    render(
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
    render(
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
    expect(calls[1]?.params).toEqual({ session_id: STORED_SESSION_ID })
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
    render(
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
    expect(calls[0]?.params).toEqual({ session_id: STORED_SESSION_ID })
    expect(calls[1]?.params).toMatchObject({ session_id: RECOVERED_SESSION_ID })
  })

  it('still creates a new session for a genuine new-chat draft (no stored session selected)', async () => {
    const createBackendSessionForSend = vi.fn(async () => RUNTIME_SESSION_ID)
    const calls: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionId={null}
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


describe('usePromptActions submit pasted screenshot attachments', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $connection.set(null)
    $composerAttachments.set([])
  })

  it('uploads a pathless pasted screenshot data URL before prompt.submit', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'image.attach_bytes') {
        return {
          attached: true,
          path: '/tmp/hermes/images/upload_20260708_120000_1.png'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(<Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />)

    const attachment: ComposerAttachment = {
      id: 'image:pasted',
      kind: 'image',
      label: 'pasted-screenshot.png',
      previewUrl: 'data:image/png;base64,aGVsbG8='
    }

    const ok = await handle!.submitText('describe this', { attachments: [attachment] })

    expect(ok).toBe(true)
    expect(calls.map(c => c.method)).toEqual(['image.attach_bytes', 'prompt.submit'])
    expect(calls[0]?.params).toEqual({
      session_id: RUNTIME_SESSION_ID,
      content_base64: 'aGVsbG8=',
      filename: 'pasted-screenshot.png'
    })
    expect(calls[1]?.params).toEqual({ session_id: RUNTIME_SESSION_ID, text: 'describe this' })
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

    render(
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

    render(
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

    render(
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


  it('uploads an in-memory pasted screenshot data URL without reading a disk path', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'image.attach_bytes') {
        return { attached: true, path: '/tmp/hermes/images/upload_1.png' } as never
      }

      return {} as never
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        readFileDataUrl: vi.fn(async () => {
          throw new Error('should not read disk for pathless pasted screenshots')
        })
      }
    })

    const uploaded = await uploadComposerAttachment(
      {
        id: 'image:pasted',
        kind: 'image',
        label: 'pasted-screenshot.png',
        previewUrl: 'data:image/png;base64,aGVsbG8='
      },
      { remote: false, requestGateway, sessionId: RUNTIME_SESSION_ID }
    )

    expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', {
      session_id: RUNTIME_SESSION_ID,
      content_base64: 'aGVsbG8=',
      filename: 'pasted-screenshot.png'
    })
    expect(window.hermesDesktop.readFileDataUrl).not.toHaveBeenCalled()
    expect(uploaded.path).toBe('/tmp/hermes/images/upload_1.png')
    expect(uploaded.attachedSessionId).toBe(RUNTIME_SESSION_ID)
  })

  it('turns the raw 16MB IPC cap error into a friendly remote-gateway message', async () => {
    // electron/hardening.cjs rejects the readFileDataUrl IPC with this exact
    // shape when a file exceeds DATA_URL_READ_MAX_BYTES.
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
