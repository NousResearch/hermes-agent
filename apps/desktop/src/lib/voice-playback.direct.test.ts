import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setVoicePlaybackState } from '@/store/voice-playback'

import type * as VoiceClientDirect from './voice-client-direct'

const mocks = vi.hoisted(() => ({
  directTtsConfig: vi.fn(),
  synthesizeSpeechClientDirect: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  speakText: vi.fn()
}))

vi.mock('@/lib/voice-client-direct', async importOriginal => {
  const actual = await importOriginal<typeof VoiceClientDirect>()

  return {
    ...actual,
    directTtsConfig: mocks.directTtsConfig,
    synthesizeSpeechClientDirect: mocks.synthesizeSpeechClientDirect
  }
})

import { startSpeechStream, stopVoicePlayback } from './voice-playback'

interface Deferred<T> {
  promise: Promise<T>
  reject: (reason?: unknown) => void
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined
  let reject: (reason?: unknown) => void = () => undefined

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, reject, resolve }
}

class MockAudio {
  static instances: MockAudio[] = []

  readonly listeners = new Map<string, () => void>()
  readonly load = vi.fn()
  readonly pause = vi.fn()
  readonly play = vi.fn(async () => undefined)
  src: string

  constructor(src: string) {
    this.src = src
    MockAudio.instances.push(this)
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    this.listeners.set(type, () => {
      if (typeof listener === 'function') {
        listener(new Event(type))
      } else {
        listener.handleEvent(new Event(type))
      }
    })
  }

  emit(type: string) {
    this.listeners.get(type)?.()
  }
}

const tts = {
  mode: 'direct' as const,
  wire: 'openai-speech' as const,
  provider: 'openai',
  base_url: 'http://127.0.0.1:7851/v1',
  api_key: 'local-test-key',
  model: 'local-tts',
  voice: 'test-voice',
  speed: null
}

const firstSentence = '这是第一段用于验证流水线预取行为的完整测试句子，长度已经超过二十四个字符。'
const secondSentence = '这是第二段用于验证严格顺序播放的完整测试句子，长度同样超过二十四个字符。'
const thirdSentence = '这是第三段用于验证中断后丢弃结果的完整测试句子，长度也超过二十四个字符。'

async function flushPromises() {
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
}

describe('client-direct speech prefetch', () => {
  let createObjectURL: ReturnType<typeof vi.fn>
  let revokeObjectURL: ReturnType<typeof vi.fn>
  let syntheses: Array<Deferred<ArrayBuffer>>

  beforeEach(() => {
    MockAudio.instances = []
    syntheses = []
    createObjectURL = vi.fn(() => `blob:voice-${createObjectURL.mock.calls.length}`)
    revokeObjectURL = vi.fn()
    vi.stubGlobal('Audio', MockAudio)
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })

    mocks.directTtsConfig.mockReset()
    mocks.synthesizeSpeechClientDirect.mockReset()
    mocks.directTtsConfig.mockResolvedValue(tts)
    mocks.synthesizeSpeechClientDirect.mockImplementation(() => {
      const next = deferred<ArrayBuffer>()

      syntheses.push(next)

      return next.promise
    })
    setVoicePlaybackState({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
  })

  afterEach(() => {
    stopVoicePlayback()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  async function openSession(text: string) {
    const session = await startSpeechStream({ messageId: 'reply-1', source: 'read-aloud' })
    expect(session).not.toBeNull()
    session!.append(text)
    session!.finish()
    await flushPromises()

    return session!
  }

  it('prefetches one sentence in FIFO order and aborts in-flight work on stop', async () => {
    const session = await openSession(`${firstSentence} ${secondSentence} ${thirdSentence}`)
    expect(mocks.synthesizeSpeechClientDirect).toHaveBeenCalledTimes(1)

    syntheses[0].resolve(new Uint8Array([1]).buffer)
    await flushPromises()

    expect(MockAudio.instances).toHaveLength(1)
    expect(MockAudio.instances[0].play).toHaveBeenCalledTimes(1)
    expect(mocks.synthesizeSpeechClientDirect).toHaveBeenCalledTimes(2)

    syntheses[1].resolve(new Uint8Array([2]).buffer)
    await flushPromises()
    expect(MockAudio.instances).toHaveLength(1)

    MockAudio.instances[0].emit('ended')
    await flushPromises()
    expect(MockAudio.instances).toHaveLength(2)
    expect(MockAudio.instances[1].play).toHaveBeenCalledTimes(1)
    expect(mocks.synthesizeSpeechClientDirect).toHaveBeenCalledTimes(3)

    const signal = mocks.synthesizeSpeechClientDirect.mock.calls[2]?.[2]?.signal as AbortSignal
    expect(signal.aborted).toBe(false)

    stopVoicePlayback()
    await expect(session.done).resolves.toBe('done')
    expect(signal.aborted).toBe(true)
    expect(MockAudio.instances[1].pause).toHaveBeenCalled()

    syntheses[2].resolve(new Uint8Array([3]).buffer)
    await flushPromises()
    expect(MockAudio.instances).toHaveLength(2)
  })

  it('lets the current sentence finish when prefetching the next sentence fails', async () => {
    const session = await startSpeechStream({ messageId: 'reply-1', source: 'read-aloud' })
    expect(session).not.toBeNull()
    session!.append(`${firstSentence} ${secondSentence} trailing`)
    await flushPromises()

    syntheses[0].resolve(new Uint8Array([1]).buffer)
    await flushPromises()

    syntheses[1].reject(new Error('provider unavailable'))
    await flushPromises()
    expect(MockAudio.instances[0].pause).not.toHaveBeenCalled()
    expect(mocks.synthesizeSpeechClientDirect).toHaveBeenCalledTimes(2)

    session!.append(thirdSentence)
    await flushPromises()
    expect(mocks.synthesizeSpeechClientDirect).toHaveBeenCalledTimes(2)

    MockAudio.instances[0].emit('ended')
    await expect(session!.done).resolves.toBe('done')
  })
})
