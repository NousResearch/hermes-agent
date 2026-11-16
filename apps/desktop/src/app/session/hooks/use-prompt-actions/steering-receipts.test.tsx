import { cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications, clearNotifications } from '@/store/notifications'
import { dropSessionState, publishSessionState, setLiveTurnBackend } from '@/store/session-states'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'

// Where an accepted correction lands, as the receipt names it (#141973).

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

describe('usePromptActions correction receipts', () => {
  afterEach(() => {
    cleanup()
    clearNotifications()
    vi.restoreAllMocks()
  })

  it('keeps a steer the compression guard queued out of the live turn and says it runs next (#141973)', async () => {
    // Legacy session.steer answers `queued` for both destinations; `disposition: 'queue'` names
    // the next-turn queue, so the stream must keep running and the user is told where it went.
    const requestGateway = vi.fn(
      async () => ({ status: 'queued', text: 'mid-compaction nudge', disposition: 'queue', reason: 'compression' }) as never
    )

    $notifications.set([])
    let handle: HarnessHandle | null = null
    const capturedStates: Record<string, unknown>[] = []
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => capturedStates.push(state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={[
          { id: 'user-1', role: 'user', parts: [{ type: 'text', text: 'long task' }] },
          { id: 'assistant-stream-1', role: 'assistant', parts: [{ type: 'text', text: 'working' }], pending: true }
        ]}
        seedStreamId="assistant-stream-1"
      />
    )

    expect(await handle!.redirectPrompt('mid-compaction nudge', 'steer')).toBe(true)

    const state = capturedStates.at(-1) as { messages: { id: string; pending?: boolean }[]; streamId: string | null }
    expect(state.streamId).toBe('assistant-stream-1')
    expect(state.messages[1]).toMatchObject({ id: 'assistant-stream-1', pending: true })
    expect($notifications.get()).toMatchObject([{ id: 'composer-steer-queued', kind: 'info' }])
  })

  it('seals a redirect the gateway steered into the live turn between model requests', async () => {
    const requestGateway = vi.fn(async () => ({ status: 'queued', text: 'join now', disposition: 'steer' }) as never)

    $notifications.set([])
    let handle: HarnessHandle | null = null
    const capturedStates: Record<string, unknown>[] = []
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => capturedStates.push(state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={[
          { id: 'user-1', role: 'user', parts: [{ type: 'text', text: 'long task' }] },
          { id: 'assistant-stream-1', role: 'assistant', parts: [{ type: 'text', text: 'working' }], pending: true }
        ]}
        seedStreamId="assistant-stream-1"
      />
    )

    expect(await handle!.redirectPrompt('join now')).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith('session.redirect', { session_id: RUNTIME_SESSION_ID, text: 'join now' })
    // It joined this turn: the stream is sealed above the correction, and nothing says "queued".
    expect(capturedStates.at(-1)?.streamId).toBeNull()
    expect($notifications.get()).toEqual([])
  })

  it('steers when the backend confirms a turn the stale slice calls settled (#141973)', async () => {
    // A long-lived or reattached pane can hold a settled slice while its turn still runs: the
    // owning backend is asked, and only its "running" lets the correction through.
    publishSessionState(RUNTIME_SESSION_ID, createClientSessionState(RUNTIME_SESSION_ID))

    const release = setLiveTurnBackend({
      request: vi.fn(async () => ({ sessions: [{ id: RUNTIME_SESSION_ID, status: 'working' }] })) as never
    })

    try {
      const requestGateway = vi.fn(async () => ({ status: 'redirected' }) as never)

      let handle: HarnessHandle | null = null
      await actRender(
        <Harness
          busyRef={{ current: true }}
          onReady={h => (handle = h)}
          refreshSessions={async () => undefined}
          requestGateway={requestGateway}
        />
      )

      expect(await handle!.redirectPrompt('still running')).toBe(true)
      expect(requestGateway).toHaveBeenCalledWith('session.redirect', { session_id: RUNTIME_SESSION_ID, text: 'still running' })
    } finally {
      release()
      dropSessionState(RUNTIME_SESSION_ID)
    }
  })
})
