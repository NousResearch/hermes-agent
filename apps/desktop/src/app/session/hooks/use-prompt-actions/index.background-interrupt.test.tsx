import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { textPart } from '@/lib/chat-messages'
import { $backgroundStatusBySession, reconcileBackgroundProcesses } from '@/store/composer-status'
import { $gateway } from '@/store/gateway'
import { clearNotifications } from '@/store/notifications'
import { $busy, setMessages } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

beforeEach(() => {
  clearSingleFlightSessionResumeState()
  vi.mocked(getLatestSessionMessages).mockReset()
  vi.mocked(getLatestSessionMessages).mockImplementation(async () => ({ messages: [], session_id: 'session' }))
})

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

// Plain interrupts must not kill background work; rewinds must. cancelRun is
// Stop and "send now" on a queued message while busy — the timeline survives,
// and the gateway's session.interrupt leaves terminal(background=true) running.
// reload / restore / edit truncate history, so processes from the discarded
// turns get reaped.
describe('usePromptActions background processes: interrupt vs rewind', () => {
  const seed = [
    { id: 'u1', parts: [textPart('first')], role: 'user' as const, timestamp: 0 },
    { id: 'a1', parts: [textPart('reply')], role: 'assistant' as const, timestamp: 1 }
  ]

  let gatewayCalls: Array<{ method: string; params?: Record<string, unknown> }>
  // resetSessionBackground remembers dismissed ids per session at module
  // level, so each test seeds a fresh process id.
  let procSeq = 0
  let procId = ''

  beforeEach(() => {
    procId = `proc_live_${++procSeq}`
    $busy.set(false)
    $backgroundStatusBySession.set({})
    setMessages(seed as never)
    gatewayCalls = []
    $gateway.set({
      request: async (method: string, params?: Record<string, unknown>) => {
        gatewayCalls.push({ method, params })

        return {} as never
      }
    } as never)
    reconcileBackgroundProcesses(RUNTIME_SESSION_ID, [
      { command: 'npm run dev', session_id: procId, status: 'running' }
    ] as never)
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
    setMessages([])
    $busy.set(false)
    $backgroundStatusBySession.set({})
    $gateway.set(null)
    vi.restoreAllMocks()
  })

  const killed = () => gatewayCalls.some(call => call.method === 'process.kill' && call.params?.process_id === procId)

  async function mount(requestGateway: ReturnType<typeof vi.fn>) {
    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway as never}
        seedMessages={seed}
      />
    )

    return handle!
  }

  it('cancelRun interrupts the turn but leaves running background processes and subagents alone', async () => {
    const requestGateway = vi.fn(async () => ({}) as never)
    const handle = await mount(requestGateway)
    upsertSubagent(RUNTIME_SESSION_ID, { goal: 'review', status: 'running', subagent_id: 'bg-review', task_index: 0 })
    upsertSubagent(RUNTIME_SESSION_ID, { goal: 'old', status: 'completed', subagent_id: 'done-1', task_index: 1 })

    await handle.cancelRun()

    expect(($subagentsBySession.get()[RUNTIME_SESSION_ID] ?? []).map(row => [row.id, row.status])).toEqual([
      ['bg-review', 'running']
    ])

    expect(requestGateway).toHaveBeenCalledWith('session.interrupt', { session_id: RUNTIME_SESSION_ID })
    expect(requestGateway).not.toHaveBeenCalledWith('process.kill', expect.anything())
    expect(killed()).toBe(false)

    const rows = $backgroundStatusBySession.get()[RUNTIME_SESSION_ID] ?? []

    expect(rows.map(row => [row.id, row.state])).toEqual([[procId, 'running']])
  })

  it('reloadFromMessage (regenerate) kills processes from the truncated turns', async () => {
    const requestGateway = vi.fn(async () => ({}) as never)
    const handle = await mount(requestGateway)

    await handle.reloadFromMessage('u1')

    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      expect.objectContaining({ confirm_truncate: true, truncate_before_message_id: 'u1' }),
      expect.anything()
    )
    expect(killed()).toBe(true)
    expect($backgroundStatusBySession.get()[RUNTIME_SESSION_ID] ?? []).toEqual([])
  })

  it('restoreToMessage still kills processes from the discarded turns', async () => {
    const requestGateway = vi.fn(async () => ({}) as never)
    const handle = await mount(requestGateway)

    await handle.restoreToMessage('u1')

    expect(killed()).toBe(true)
    expect($backgroundStatusBySession.get()[RUNTIME_SESSION_ID] ?? []).toEqual([])
  })
})
