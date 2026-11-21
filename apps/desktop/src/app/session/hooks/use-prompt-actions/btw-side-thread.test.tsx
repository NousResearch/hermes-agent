import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $queuedPromptsBySession, getQueuedPrompts } from '@/store/composer-queue'
import { dropSessionState, publishSessionState } from '@/store/session-states'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

// /btw side exploration threads: the gateway parses the flags (agent/side_thread.py); the
// desktop renders the reply and submits a promotion as an ordinary turn, or queues it behind a
// running one exactly like a /goal kickoff.

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

const SLASH_METRIC = 'shared_metrics.slash_command'

function renderedTexts(states: Record<string, unknown>[]): string {
  return states
    .flatMap(state => {
      const messages = Array.isArray(state.messages)
        ? (state.messages as Array<{ parts?: Array<{ text?: string }> }>)
        : []

      return messages.flatMap(message => (message.parts ?? []).map(part => part.text ?? ''))
    })
    .join('\n')
}

async function mount(reply: Record<string, string>, busy = false) {
  const calls: { method: string; params?: Record<string, unknown> }[] = []
  const states: Record<string, unknown>[] = []

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === SLASH_METRIC) {
      return {} as never
    }

    calls.push({ method, params })

    return (method === 'prompt.btw' ? reply : {}) as never
  })

  let handle: HarnessHandle | null = null
  await actRender(
    <Harness
      busyRef={{ current: busy }}
      onReady={h => (handle = h)}
      onSeedState={s => states.push(s)}
      refreshSessions={async () => undefined}
      requestGateway={requestGateway}
    />
  )

  return { calls, handle: handle!, states }
}

describe('usePromptActions /btw side threads', () => {
  beforeEach(() => {
    clearSingleFlightSessionResumeState()
    vi.mocked(getLatestSessionMessages).mockImplementation(async () => ({ messages: [], session_id: 'session' }))
    $queuedPromptsBySession.set({})
  })

  afterEach(() => {
    cleanup()
    dropSessionState(RUNTIME_SESSION_ID)
    $queuedPromptsBySession.set({})
    vi.restoreAllMocks()
  })

  it('shows the start notice the gateway returns for --explore', async () => {
    const { calls, handle, states } = await mount({ task_id: 'btw_ab12cd', text: '🔭 Side exploration btw_ab12cd' })

    await handle.submitText('/btw --explore find the retry logic')

    expect(calls[0]).toEqual({
      method: 'prompt.btw',
      params: { session_id: RUNTIME_SESSION_ID, text: '--explore find the retry logic' }
    })
    expect(renderedTexts(states)).toContain('🔭 Side exploration btw_ab12cd')
    expect(renderedTexts(states)).not.toContain('answering from a conversation snapshot')
  })

  it('submits a promotion as the next turn when the session is idle', async () => {
    const { calls, handle } = await mount({ text: '⤴ Bringing btw_ab12cd in', prompt: 'PROMOTED REPORT' })

    await handle.submitText('/btw --promote btw_ab12cd')

    const submit = calls.find(call => call.method === 'prompt.submit')
    expect(submit?.params).toMatchObject({ session_id: RUNTIME_SESSION_ID, text: 'PROMOTED REPORT' })
  })

  it('queues a promotion behind a running turn instead of submitting it', async () => {
    publishSessionState(RUNTIME_SESSION_ID, { ...createClientSessionState(RUNTIME_SESSION_ID), busy: true })
    const { calls, handle, states } = await mount({ text: '⤴ Bringing it in', prompt: 'PROMOTED REPORT' }, true)

    await handle.submitText('/btw --promote')

    expect(calls.map(call => call.method)).toEqual(['prompt.btw'])
    expect(getQueuedPrompts(RUNTIME_SESSION_ID).map(entry => entry.text)).toEqual(['PROMOTED REPORT'])
    expect(renderedTexts(states)).toContain('queued')
  })
})
