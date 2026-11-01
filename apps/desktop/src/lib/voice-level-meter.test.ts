import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { startVoiceLevelMeter } from './voice-level-meter'

type SampleEvent = MessageEvent<{ level: number; audioTime: number; nextReportTime: number }>

class FakeWorkletNode {
  static instances: FakeWorkletNode[] = []
  onprocessorerror: (() => void) | null = null
  connected = false
  port = {
    onmessage: null as ((event: SampleEvent) => void) | null,
    closed: false,
    close: () => {
      this.port.closed = true
    }
  }

  constructor() {
    FakeWorkletNode.instances.push(this)
  }

  connect() {
    this.connected = true
  }
  disconnect() {
    this.connected = false
  }
}

function setup() {
  let resolve!: () => void
  let reject!: (error: Error) => void
  const loading = new Promise<void>((done, fail) => {
    resolve = done
    reject = fail
  })
  const context = Object.assign(new EventTarget(), {
    audioWorklet: { addModule: () => loading },
    destination: {},
    currentTime: 12.5,
    state: 'running'
  }) as unknown as AudioContext
  const connections = new Set<FakeWorkletNode>()
  const source = {
    connect: (node: FakeWorkletNode) => connections.add(node),
    disconnect: (node: FakeWorkletNode) => connections.delete(node)
  } as unknown as MediaStreamAudioSourceNode

  return { context, source, connections, resolve, reject }
}

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

beforeEach(() => {
  FakeWorkletNode.instances = []
  vi.stubGlobal('AudioWorkletNode', FakeWorkletNode)
})
afterEach(() => vi.unstubAllGlobals())

describe('voice level meter ownership', () => {
  it('does not reconnect capture after cancellation during module loading', async () => {
    const capture = setup()
    const onLevel = vi.fn()
    const onError = vi.fn()
    const stop = startVoiceLevelMeter(capture.context, capture.source, onLevel, onError)
    stop()
    capture.resolve()
    await flush()

    expect(capture.connections.size).toBe(0)
    expect(FakeWorkletNode.instances).toHaveLength(0)
    expect(onLevel).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()
  })

  it('uses captured audio time and ignores queued samples after stopping', async () => {
    const capture = setup()
    const onLevel = vi.fn()
    const stop = startVoiceLevelMeter(capture.context, capture.source, onLevel, vi.fn())
    capture.resolve()
    await flush()
    const node = FakeWorkletNode.instances[0]
    const deliver = node.port.onmessage!
    deliver({ data: { level: 0.2, audioTime: 12.5, nextReportTime: 12.5 + 1 / 60 } } as SampleEvent)
    stop()
    stop()
    deliver({ data: { level: 0.5, audioTime: 13, nextReportTime: 13 + 1 / 60 } } as SampleEvent)

    expect(onLevel.mock.calls).toEqual([[0.2, 12_500, true]])
    expect(capture.connections.size).toBe(0)
    expect(node.connected).toBe(false)
    expect(node.port.closed).toBe(true)
  })

  it('delivers the entire backlog but marks only the current reporting interval as safe to end capture', async () => {
    const capture = setup()
    const onLevel = vi.fn()
    const stop = startVoiceLevelMeter(capture.context, capture.source, onLevel, vi.fn())
    capture.resolve()
    await flush()
    const deliver = FakeWorkletNode.instances[0].port.onmessage!

    // The audio clock is already at 12.5s before any of these messages arrive.
    deliver({ data: { level: 0, audioTime: 12, nextReportTime: 12 + 1 / 60 } } as SampleEvent)
    deliver({ data: { level: 0.2, audioTime: 12.2, nextReportTime: 12.2 + 1 / 60 } } as SampleEvent)
    deliver({ data: { level: 0.3, audioTime: 12.5, nextReportTime: 12.5 + 1 / 60 } } as SampleEvent)

    expect(onLevel.mock.calls).toEqual([
      [0, 12_000, false],
      [0.2, 12_200, false],
      [0.3, 12_500, true]
    ])
    Object.defineProperty(capture.context, 'currentTime', { value: 12.5 + 128 / 48_000, configurable: true })
    deliver({ data: { level: 0, audioTime: 12.5, nextReportTime: 12.5 + 1 / 60 } } as SampleEvent)
    expect(onLevel).toHaveBeenLastCalledWith(0, 12_500, true)
    // At the next scheduled report, the preceding sample is historical even
    // if its successor's message has not been dispatched yet.
    Object.defineProperty(capture.context, 'currentTime', { value: 12.5 + 1 / 60 })
    deliver({ data: { level: 0, audioTime: 12.5, nextReportTime: 12.5 + 1 / 60 } } as SampleEvent)
    expect(onLevel).toHaveBeenLastCalledWith(0, 12_500, false)
    stop()
  })

  it('releases capture and reports a crashed processor only once', async () => {
    const capture = setup()
    const onError = vi.fn()
    const stop = startVoiceLevelMeter(capture.context, capture.source, vi.fn(), onError)
    capture.resolve()
    await flush()
    const node = FakeWorkletNode.instances[0]
    const fail = node.onprocessorerror!
    fail()
    fail()
    stop()

    expect(onError).toHaveBeenCalledOnce()
    expect(capture.connections.size).toBe(0)
    expect(node.connected).toBe(false)
    expect(node.port.closed).toBe(true)
  })

  it.each([false, true])('reports module failure only while its capture is owned (cancelled: %s)', async cancelled => {
    const capture = setup()
    const onError = vi.fn()
    const stop = startVoiceLevelMeter(capture.context, capture.source, vi.fn(), onError)

    if (cancelled) {
      stop()
    }

    capture.reject(new Error('worklet asset unavailable'))
    await flush()
    expect(onError).toHaveBeenCalledTimes(cancelled ? 0 : 1)
    expect(capture.connections.size).toBe(0)
    expect(FakeWorkletNode.instances).toHaveLength(0)
  })
})

it('fails a pending meter when its audio context closes unexpectedly', async () => {
  const capture = setup()
  const onError = vi.fn()
  startVoiceLevelMeter(capture.context, capture.source, vi.fn(), onError)
  capture.resolve()
  await flush()
  Object.defineProperty(capture.context, 'state', { value: 'closed' })
  capture.context.dispatchEvent(new Event('statechange'))
  capture.context.dispatchEvent(new Event('error'))

  expect(onError).toHaveBeenCalledOnce()
  expect(capture.connections.size).toBe(0)
  expect(FakeWorkletNode.instances[0].port.closed).toBe(true)
})
