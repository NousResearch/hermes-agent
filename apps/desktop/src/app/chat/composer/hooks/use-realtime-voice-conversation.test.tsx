/**
 * Behavior tests for the realtime (xAI S2S supervisor) voice conversation:
 * consult dispatch → Hermes turn → spoken result, steering, and stop words.
 * The shared client is faked; no sockets or audio.
 */

import type { RealtimeTokenGrant } from '@hermes/shared'
import type * as HermesShared from '@hermes/shared'
import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { routeVoiceSupervisorGatewayEvent, routeVoiceSupervisorServerRequest } from '@/store/voice-supervisor-events'

import { useRealtimeVoiceConversation } from './use-realtime-voice-conversation'

interface FakeCall {
  name: string
  callId: string
  args: Record<string, unknown>
}

interface FakeCallbacks {
  onFunctionCall: (call: FakeCall) => void | Promise<void>
  onStatus?: (status: string, detail?: string) => void
  onUserTranscript?: (text: string) => void
}

const { FakeRealtimeClient, fakeClients, notifyError } = vi.hoisted(() => {
  const notifyError = vi.fn()

  class FakeRealtimeClient {
    callbacks: FakeCallbacks | null = null
    closed = false
    lastResponseHadAudio = false
    readonly outputs: [string, string][] = []
    readonly acks: number[] = []
    readonly verbatim: [string, boolean | undefined][] = []
    muted = false

    constructor() {
      fakeClients.push(this)
    }

    async connect(_grant: unknown, callbacks: FakeCallbacks) {
      this.callbacks = callbacks
    }

    close() {
      this.closed = true
    }

    sendFunctionOutput(callId: string, output: string) {
      this.outputs.push([callId, output])
    }

    speakAcknowledgment() {
      this.acks.push(Date.now())
    }

    speakVerbatim(text: string, interruptible?: boolean) {
      this.verbatim.push([text, interruptible])
    }

    setMuted(muted: boolean) {
      this.muted = muted
    }
  }

  const fakeClients: InstanceType<typeof FakeRealtimeClient>[] = []

  return { FakeRealtimeClient, fakeClients, notifyError }
})

type FakeClient = InstanceType<typeof FakeRealtimeClient>

vi.mock('@/store/notifications', () => ({
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

vi.mock('@hermes/shared', async importOriginal => {
  const actual = await importOriginal<typeof HermesShared>()

  return {
    ...actual,
    RealtimeVoiceClient: FakeRealtimeClient
  }
})

interface HarnessProps {
  busy: boolean
  pending: { id: string; pending: boolean; text: string; userText?: string | null } | null
}

interface HarnessOverrides {
  initialPending?: HarnessProps['pending']
  requestToken?: () => Promise<RealtimeTokenGrant | null>
}

function makeHarness(overrides: HarnessOverrides = {}) {
  let currentProps: HarnessProps = { busy: false, pending: overrides.initialPending ?? null }
  let pendingSource = currentProps.pending
  const submitTask = vi.fn(async (_text: string) => true)
  const onInterrupt = vi.fn(async () => undefined)
  const onStopWord = vi.fn()
  const onFatalError = vi.fn()

  const consumePendingResponse = vi.fn(() => {
    pendingSource = null
  })

  const requestToken =
    overrides.requestToken ??
    vi.fn(async () => ({
      token: 'eph',
      url: 'wss://api.x.ai/v1/realtime?model=m',
      session_update: {}
    }))

  const view = renderHook(
    (props: HarnessProps) =>
      useRealtimeVoiceConversation({
        busy: props.busy,
        enabled: true,
        sessionId: 'voice-session',
        requestToken,
        submitTask,
        onInterrupt,
        onFatalError,
        onStopWord,
        isStopWord: text => text.trim().toLowerCase() === 'stop',
        pendingResponse: () => pendingSource,
        consumePendingResponse,
        failureLabel: 'voice failed'
      }),
    { initialProps: currentProps }
  )

  return {
    view,
    submitTask,
    onInterrupt,
    onFatalError,
    onStopWord,
    consumePendingResponse,
    requestToken,
    setProps(next: Partial<HarnessProps>) {
      currentProps = { ...currentProps, ...next }

      if ('pending' in next) {
        pendingSource = next.pending ?? null
      }

      view.rerender(currentProps)
    },
    async client(): Promise<FakeClient> {
      await waitFor(() => expect(fakeClients.length).toBeGreaterThan(0))

      return fakeClients[fakeClients.length - 1]
    },
    async fire(client: FakeClient, call: FakeCall) {
      await act(async () => {
        await client.callbacks?.onFunctionCall(call)
      })
    }
  }
}

beforeEach(() => {
  fakeClients.length = 0
  notifyError.mockClear()
})

describe('consult lifecycle', () => {
  it('baselines old chat history before attributing a new consult result', async () => {
    const h = makeHarness({
      initialPending: {
        id: 'old-assistant',
        pending: false,
        text: 'An answer from before voice mode.',
        userText: 'old typed prompt'
      }
    })

    const client = await h.client()

    expect(h.consumePendingResponse).toHaveBeenCalledOnce()
    await h.fire(client, {
      name: 'consult_hermes',
      callId: 'c1',
      args: { task: 'which repository are you in?' }
    })
    h.setProps({
      pending: {
        id: 'new-assistant',
        pending: false,
        text: 'You are in hermes-agent.',
        userText: 'which repository are you in?'
      }
    })

    await waitFor(() => expect(client.outputs).toContainEqual(['c1', 'You are in hermes-agent.']))
  })

  it('runs a consult as a turn and speaks the settled result', async () => {
    const h = makeHarness()
    const client = await h.client()

    await h.fire(client, {
      name: 'consult_hermes',
      callId: 'c1',
      args: { task: 'check disk usage' }
    })
    expect(h.submitTask).toHaveBeenCalledWith('check disk usage')
    // Model called silently → instant acknowledgment.
    expect(client.acks).toHaveLength(1)

    // Turn settles: busy false + finalized reply text.
    h.setProps({ busy: false, pending: { id: 'm1', pending: false, text: 'Disk is 42% full.' } })
    await waitFor(() => expect(client.outputs).toContainEqual(['c1', 'Disk is 42% full.']))
    expect(h.consumePendingResponse).toHaveBeenCalled()
  })

  it('reports a dropped submit to the voice model instead of tracking a phantom consult', async () => {
    const h = makeHarness()
    const client = await h.client()
    h.submitTask.mockResolvedValueOnce(false) // busy guard / read-only / cancelled

    await h.fire(client, { name: 'consult_hermes', callId: 'c1', args: { task: 'check disk usage' } })

    const reply = client.outputs.find(([id]) => id === 'c1')
    expect(reply?.[1]).toMatch(/could not start/i)
    expect(client.acks).toHaveLength(0)
    // Nothing is tracked, so the next consult is accepted (no "still working").
    await h.fire(client, { name: 'consult_hermes', callId: 'c2', args: { task: 'second' } })
    expect(h.submitTask).toHaveBeenCalledTimes(2)
    expect(client.outputs.find(([id]) => id === 'c2')).toBeUndefined()
  })

  it('skips the acknowledgment when the model already spoke', async () => {
    const h = makeHarness()
    const client = await h.client()
    client.lastResponseHadAudio = true
    await h.fire(client, {
      name: 'consult_hermes',
      callId: 'c1',
      args: { task: 't' }
    })
    expect(client.acks).toHaveLength(0)
  })

  it('rejects a second consult while one is in flight', async () => {
    const h = makeHarness()
    const client = await h.client()
    await h.fire(client, { name: 'consult_hermes', callId: 'c1', args: { task: 'first' } })
    await h.fire(client, { name: 'consult_hermes', callId: 'c2', args: { task: 'second' } })
    expect(h.submitTask).toHaveBeenCalledTimes(1)
    const busyReply = client.outputs.find(([id]) => id === 'c2')
    expect(busyReply?.[1]).toMatch(/still working/)
  })

  it('does not send the result while the turn is still pending', async () => {
    const h = makeHarness()
    const client = await h.client()
    await h.fire(client, { name: 'consult_hermes', callId: 'c1', args: { task: 't' } })
    h.setProps({ busy: true, pending: { id: 'm1', pending: true, text: 'partial…' } })
    await new Promise(resolve => setTimeout(resolve, 700))
    expect(client.outputs).toHaveLength(0)
  })

  it("never speaks another submission's reply as the consult result", async () => {
    const h = makeHarness()
    const client = await h.client()
    await h.fire(client, {
      name: 'consult_hermes',
      callId: 'c1',
      args: { task: 'check disk usage' }
    })

    // A typed message's turn settles first — consumed, not attributed.
    h.setProps({
      busy: false,
      pending: { id: 'm1', pending: false, text: 'Hi there!', userText: 'hello' }
    })
    await waitFor(() => expect(h.consumePendingResponse).toHaveBeenCalled())
    expect(client.outputs).toHaveLength(0)

    // The consult turn's own reply (matching trigger) completes the consult.
    h.setProps({
      pending: { id: 'm2', pending: false, text: 'Disk is 42% full.', userText: 'check disk usage' }
    })
    await waitFor(() => expect(client.outputs).toContainEqual(['c1', 'Disk is 42% full.']))
  })

  it('fails out a dead consult so a new one can start', async () => {
    const h = makeHarness()
    const client = await h.client()
    const t0 = Date.now()
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(t0)

    try {
      await h.fire(client, { name: 'consult_hermes', callId: 'c1', args: { task: 'first' } })

      // Turn died: idle, no reply, well past the stale window.
      nowSpy.mockReturnValue(t0 + 31_000)
      await h.fire(client, { name: 'consult_hermes', callId: 'c2', args: { task: 'second' } })

      expect(client.outputs).toContainEqual(['c1', 'That task failed without producing a result.'])
      expect(h.submitTask).toHaveBeenLastCalledWith('second')
    } finally {
      nowSpy.mockRestore()
    }
  })
})

describe('steering', () => {
  it('retargets the consult, interrupts the busy turn, and confirms', async () => {
    const h = makeHarness()
    const client = await h.client()
    await h.fire(client, { name: 'consult_hermes', callId: 'c1', args: { task: 'original' } })
    h.setProps({ busy: true })
    await h.fire(client, {
      name: 'steer_hermes',
      callId: 's1',
      args: { instruction: 'also check logs' }
    })
    expect(h.submitTask).toHaveBeenLastCalledWith('also check logs')
    expect(h.onInterrupt).toHaveBeenCalled()
    expect(client.outputs).toContainEqual(['s1', 'Steering applied — Hermes is adjusting course.'])

    // The steered continuation still answers the ORIGINAL consult call id.
    h.setProps({ busy: false, pending: { id: 'm2', pending: false, text: 'done' } })
    await waitFor(() => expect(client.outputs).toContainEqual(['c1', 'done']))
  })

  it('reports nothing-to-steer without a consult', async () => {
    const h = makeHarness()
    const client = await h.client()
    await h.fire(client, {
      name: 'steer_hermes',
      callId: 's1',
      args: { instruction: 'go faster' }
    })
    expect(h.submitTask).not.toHaveBeenCalled()
    expect(client.outputs[0][1]).toMatch(/No Hermes task is running/)
  })
})

describe('session control', () => {
  it('fires the stop word from the user transcript sidecar', async () => {
    const h = makeHarness()
    const client = await h.client()
    act(() => {
      client.callbacks?.onUserTranscript?.('stop')
    })
    expect(h.onStopWord).toHaveBeenCalled()
    act(() => {
      client.callbacks?.onUserTranscript?.('stop the deploy and check logs')
    })
    expect(h.onStopWord).toHaveBeenCalledTimes(1)
  })

  it('never surfaces transcript text — the sidecar ASR is stop-word only', async () => {
    const h = makeHarness()
    const client = await h.client()
    act(() => {
      client.callbacks?.onUserTranscript?.('what repo is this')
    })
    expect(h.onStopWord).not.toHaveBeenCalled()
    expect('transcript' in h.view.result.current).toBe(false)
  })

  it('speaks throttled tool progress and blocking notices for its session', async () => {
    const h = makeHarness()
    const client = await h.client()
    await h.fire(client, {
      name: 'consult_hermes',
      callId: 'c1',
      args: { task: 'inspect the project' }
    })

    act(() => {
      routeVoiceSupervisorGatewayEvent({
        payload: { name: 'delegate_task' },
        session_id: 'voice-session',
        type: 'tool.start'
      })
      routeVoiceSupervisorServerRequest({
        method: 'approval',
        params: { session_id: 'voice-session' }
      })
    })

    expect(client.verbatim).toEqual([
      ['Running delegate task.', true],
      ['Hermes needs your approval in the app.', true]
    ])
  })

  it('mints a fresh token per dial and closes the client on unmount', async () => {
    const h = makeHarness()
    const client = await h.client()
    expect(h.requestToken).toHaveBeenCalledTimes(1)
    h.view.unmount()
    expect(client.closed).toBe(true)
  })

  it('stands down quietly when realtime is unavailable (null grant)', async () => {
    const h = makeHarness({ requestToken: vi.fn(async () => null) })
    await waitFor(() => expect(h.requestToken).toHaveBeenCalled())
    await waitFor(() => expect(h.view.result.current.status).toBe('idle'))

    // Not fatal and not an error toast — the surface swaps to the classic
    // loop on its own; killing the conversation would defeat the fallback.
    expect(fakeClients).toHaveLength(0)
    expect(h.onFatalError).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('hands a dead realtime transport to the surface fallback', async () => {
    const h = makeHarness()
    const client = await h.client()

    act(() => {
      client.callbacks?.onStatus?.('closed')
    })

    expect(h.onFatalError).toHaveBeenCalledOnce()
    expect(h.view.result.current.status).toBe('idle')
  })
})
