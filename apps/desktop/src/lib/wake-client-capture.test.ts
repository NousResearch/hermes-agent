import { afterEach, describe, expect, it, vi } from 'vitest'

import { startClientWakeCapture } from './wake-client-capture'

function fakeNode() {
  return {
    connect: vi.fn(),
    disconnect: vi.fn()
  }
}

function fakeStream(label: string) {
  const track = new EventTarget() as EventTarget & {
    label: string
    stop: ReturnType<typeof vi.fn>
  }

  track.label = label
  track.stop = vi.fn()

  return {
    getTracks: vi.fn(() => [track]),
    track
  }
}

function installAudioContext(options: { failReplacementConnect?: Error } = {}) {
  const sources: Array<ReturnType<typeof fakeNode> & { stream: unknown }> = []

  const processor = {
    ...fakeNode(),
    onaudioprocess: null as ((event: AudioProcessingEvent) => void) | null
  }

  const mute = {
    ...fakeNode(),
    gain: { value: 1 }
  }

  const context = {
    close: vi.fn().mockResolvedValue(undefined),
    createGain: vi.fn(() => mute),
    createMediaStreamSource: vi.fn((stream: unknown) => {
      const source = { ...fakeNode(), stream }

      if (sources.length > 0 && options.failReplacementConnect) {
        source.connect.mockImplementation(() => {
          throw options.failReplacementConnect
        })
      }

      sources.push(source)

      return source
    }),
    createScriptProcessor: vi.fn(() => processor),
    destination: {},
    resume: vi.fn().mockResolvedValue(undefined),
    sampleRate: 48_000,
    state: 'running'
  }

  function FakeAudioContext() {
    return context
  }

  vi.stubGlobal('AudioContext', FakeAudioContext)

  return { context, mute, processor, sources }
}

async function flushAsyncWork() {
  await Promise.resolve()
  await Promise.resolve()
}

function deferred<T>() {
  let resolve: (value: T) => void = () => undefined
  let reject: (reason?: unknown) => void = () => undefined

  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, reject, resolve }
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('startClientWakeCapture microphone handoff', () => {
  it('swaps to the new default input when media devices change', async () => {
    const first = fakeStream('Built-in Microphone')
    const second = fakeStream('AirPods Max')

    const mediaDevices = new EventTarget() as EventTarget & {
      getUserMedia: ReturnType<typeof vi.fn>
    }

    mediaDevices.getUserMedia = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second)
    vi.stubGlobal('navigator', { mediaDevices })
    const { context, processor, sources } = installAudioContext()

    const capture = await startClientWakeCapture({ request: vi.fn().mockResolvedValue({}) })

    mediaDevices.dispatchEvent(new Event('devicechange'))
    await flushAsyncWork()

    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(2)
    expect(context.createMediaStreamSource).toHaveBeenNthCalledWith(2, second)
    expect(sources[1]?.connect).toHaveBeenCalledWith(processor)
    expect(sources[0]?.disconnect).toHaveBeenCalledOnce()
    expect(first.track.stop).toHaveBeenCalledOnce()
    expect(second.track.stop).not.toHaveBeenCalled()

    capture.stop()
  })

  it('serializes rapid device changes so an older stream cannot win the race', async () => {
    const first = fakeStream('Built-in Microphone')
    const second = fakeStream('AirPods Max — connecting')
    const third = fakeStream('AirPods Max')
    const firstHandoff = deferred<typeof second>()
    const queuedHandoff = deferred<typeof third>()

    const mediaDevices = new EventTarget() as EventTarget & {
      getUserMedia: ReturnType<typeof vi.fn>
    }

    mediaDevices.getUserMedia = vi
      .fn()
      .mockResolvedValueOnce(first)
      .mockImplementationOnce(() => firstHandoff.promise)
      .mockImplementationOnce(() => queuedHandoff.promise)
    vi.stubGlobal('navigator', { mediaDevices })
    installAudioContext()

    const capture = await startClientWakeCapture({ request: vi.fn().mockResolvedValue({}) })

    mediaDevices.dispatchEvent(new Event('devicechange'))
    mediaDevices.dispatchEvent(new Event('devicechange'))
    await flushAsyncWork()

    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(2)

    firstHandoff.resolve(second)
    await flushAsyncWork()
    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(3)

    queuedHandoff.resolve(third)
    await flushAsyncWork()

    expect(first.track.stop).toHaveBeenCalledOnce()
    expect(second.track.stop).toHaveBeenCalledOnce()
    expect(third.track.stop).not.toHaveBeenCalled()

    capture.stop()
  })

  it('reacquires the default input when the active track ends', async () => {
    const first = fakeStream('AirPods Max')
    const second = fakeStream('Built-in Microphone')

    const mediaDevices = new EventTarget() as EventTarget & {
      getUserMedia: ReturnType<typeof vi.fn>
    }

    mediaDevices.getUserMedia = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second)
    vi.stubGlobal('navigator', { mediaDevices })
    const { context } = installAudioContext()

    const capture = await startClientWakeCapture({ request: vi.fn().mockResolvedValue({}) })

    first.track.dispatchEvent(new Event('ended'))
    await flushAsyncWork()

    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(2)
    expect(context.createMediaStreamSource).toHaveBeenNthCalledWith(2, second)
    expect(first.track.stop).toHaveBeenCalledOnce()

    capture.stop()
  })

  it('ignores a failed handoff after wake capture has stopped', async () => {
    const first = fakeStream('AirPods Max')
    const handoff = deferred<ReturnType<typeof fakeStream>>()

    const mediaDevices = new EventTarget() as EventTarget & {
      getUserMedia: ReturnType<typeof vi.fn>
    }

    mediaDevices.getUserMedia = vi
      .fn()
      .mockResolvedValueOnce(first)
      .mockImplementationOnce(() => handoff.promise)
    vi.stubGlobal('navigator', { mediaDevices })
    installAudioContext()
    const onError = vi.fn()

    const capture = await startClientWakeCapture({ onError, request: vi.fn().mockResolvedValue({}) })

    mediaDevices.dispatchEvent(new Event('devicechange'))
    capture.stop()
    handoff.reject(new Error('device vanished'))
    await flushAsyncWork()

    expect(onError).not.toHaveBeenCalled()

    mediaDevices.dispatchEvent(new Event('devicechange'))
    await flushAsyncWork()
    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(2)
  })

  it('releases a replacement stream that resolves after capture has stopped', async () => {
    const first = fakeStream('AirPods Max')
    const replacement = fakeStream('Built-in Microphone')
    const handoff = deferred<typeof replacement>()

    const mediaDevices = new EventTarget() as EventTarget & {
      getUserMedia: ReturnType<typeof vi.fn>
    }

    mediaDevices.getUserMedia = vi
      .fn()
      .mockResolvedValueOnce(first)
      .mockImplementationOnce(() => handoff.promise)
    vi.stubGlobal('navigator', { mediaDevices })
    const { context } = installAudioContext()
    const onError = vi.fn()

    const capture = await startClientWakeCapture({ onError, request: vi.fn().mockResolvedValue({}) })

    mediaDevices.dispatchEvent(new Event('devicechange'))
    capture.stop()
    handoff.resolve(replacement)
    await flushAsyncWork()

    expect(replacement.track.stop).toHaveBeenCalledOnce()
    expect(context.createMediaStreamSource).toHaveBeenCalledTimes(1)
    expect(onError).not.toHaveBeenCalled()
  })

  it('moves track-ended recovery to each replacement and removes all listeners on stop', async () => {
    const first = fakeStream('Built-in Microphone')
    const second = fakeStream('AirPods Max')
    const third = fakeStream('Studio Display Microphone')

    const mediaDevices = new EventTarget() as EventTarget & {
      getUserMedia: ReturnType<typeof vi.fn>
    }

    mediaDevices.getUserMedia = vi
      .fn()
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(second)
      .mockResolvedValueOnce(third)
    vi.stubGlobal('navigator', { mediaDevices })
    installAudioContext()

    const capture = await startClientWakeCapture({ request: vi.fn().mockResolvedValue({}) })

    mediaDevices.dispatchEvent(new Event('devicechange'))
    await flushAsyncWork()

    first.track.dispatchEvent(new Event('ended'))
    await flushAsyncWork()
    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(2)

    second.track.dispatchEvent(new Event('ended'))
    await flushAsyncWork()
    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(3)

    capture.stop()
    second.track.dispatchEvent(new Event('ended'))
    third.track.dispatchEvent(new Event('ended'))
    mediaDevices.dispatchEvent(new Event('devicechange'))
    await flushAsyncWork()

    expect(mediaDevices.getUserMedia).toHaveBeenCalledTimes(3)
  })

  it('keeps the current stream after a transient acquisition failure and recovers later', async () => {
    const first = fakeStream('Built-in Microphone')
    const second = fakeStream('AirPods Max')
    const failure = new Error('default input is changing')

    const mediaDevices = new EventTarget() as EventTarget & {
      getUserMedia: ReturnType<typeof vi.fn>
    }

    mediaDevices.getUserMedia = vi
      .fn()
      .mockResolvedValueOnce(first)
      .mockRejectedValueOnce(failure)
      .mockResolvedValueOnce(second)
    vi.stubGlobal('navigator', { mediaDevices })
    const { processor, sources } = installAudioContext()
    const onError = vi.fn()

    const capture = await startClientWakeCapture({ onError, request: vi.fn().mockResolvedValue({}) })

    mediaDevices.dispatchEvent(new Event('devicechange'))
    await flushAsyncWork()

    expect(onError).toHaveBeenCalledOnce()
    expect(onError).toHaveBeenCalledWith(failure)
    expect(sources[0]?.disconnect).not.toHaveBeenCalled()
    expect(first.track.stop).not.toHaveBeenCalled()

    mediaDevices.dispatchEvent(new Event('devicechange'))
    await flushAsyncWork()

    expect(sources[1]?.connect).toHaveBeenCalledWith(processor)
    expect(sources[0]?.disconnect).toHaveBeenCalledOnce()
    expect(first.track.stop).toHaveBeenCalledOnce()
    expect(second.track.stop).not.toHaveBeenCalled()

    capture.stop()
  })

  it('releases a replacement stream when its audio source cannot connect', async () => {
    const first = fakeStream('Built-in Microphone')
    const replacement = fakeStream('AirPods Max')

    const mediaDevices = new EventTarget() as EventTarget & {
      getUserMedia: ReturnType<typeof vi.fn>
    }

    mediaDevices.getUserMedia = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(replacement)
    vi.stubGlobal('navigator', { mediaDevices })
    const failure = new Error('audio graph rejected replacement')
    installAudioContext({ failReplacementConnect: failure })
    const onError = vi.fn()

    const capture = await startClientWakeCapture({ onError, request: vi.fn().mockResolvedValue({}) })

    mediaDevices.dispatchEvent(new Event('devicechange'))
    await flushAsyncWork()

    expect(onError).toHaveBeenCalledWith(failure)
    expect(replacement.track.stop).toHaveBeenCalledOnce()
    expect(first.track.stop).not.toHaveBeenCalled()

    capture.stop()
  })
})
