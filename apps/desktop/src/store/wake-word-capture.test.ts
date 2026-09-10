import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $wakeWord, applyWakeStartResult, resetWakeWordState, stopClientCapture } from './wake-word'

const { gatewayRequest } = vi.hoisted(() => ({ gatewayRequest: vi.fn().mockResolvedValue({ stopped: true }) }))

vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return { $gateway: atom({ request: gatewayRequest }) }
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, resolve, reject }
}

function microphone() {
  const stop = vi.fn()

  return { stop, stream: { getTracks: () => [{ stop }] } as unknown as MediaStream }
}

const getUserMedia = vi.fn<() => Promise<MediaStream>>()
const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')

beforeEach(() => {
  resetWakeWordState()
  vi.clearAllMocks()
  getUserMedia.mockReset()
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } })
  vi.stubGlobal(
    'AudioContext',
    class {
      sampleRate = 16_000
      state = 'running'
      destination = {}
      createMediaStreamSource = () => ({ connect: vi.fn(), disconnect: vi.fn() })
      createScriptProcessor = () => ({ connect: vi.fn(), disconnect: vi.fn(), onaudioprocess: null })
      createGain = () => ({ connect: vi.fn(), disconnect: vi.fn(), gain: { value: 1 } })
      close = async () => undefined
    }
  )
})

afterEach(() => {
  resetWakeWordState()
  vi.unstubAllGlobals()

  if (originalMediaDevices) {
    Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices)
  } else {
    Reflect.deleteProperty(navigator, 'mediaDevices')
  }
})

describe('pending wake microphone ownership', () => {
  it.each([false, true])('releases a late microphone after voice takeover (wake restarted: %s)', async restart => {
    const opening = deferred<MediaStream>()
    const abandoned = microphone()
    const replacement = microphone()

    getUserMedia.mockReturnValueOnce(opening.promise).mockResolvedValueOnce(replacement.stream)
    applyWakeStartResult({ started: true, capture: 'client', phrase: 'hey hermes' })
    expect(getUserMedia).toHaveBeenCalledOnce()
    stopClientCapture()

    if (restart) {
      applyWakeStartResult({ started: true, capture: 'client', phrase: 'hey hermes' })
      await Promise.resolve()
      await Promise.resolve()
    }

    opening.resolve(abandoned.stream)
    await opening.promise
    await Promise.resolve()

    expect(abandoned.stop).toHaveBeenCalledOnce()
    expect(replacement.stop).not.toHaveBeenCalled()
    stopClientCapture()
    expect(replacement.stop).toHaveBeenCalledTimes(restart ? 1 : 0)
    expect(abandoned.stop).toHaveBeenCalledOnce()
  })

  it('ignores an obsolete microphone failure after a new listener acquired the same gateway', async () => {
    const opening = deferred<MediaStream>()
    const replacement = microphone()

    getUserMedia.mockReturnValueOnce(opening.promise).mockResolvedValueOnce(replacement.stream)
    applyWakeStartResult({ started: true, capture: 'client', phrase: 'hey hermes' })
    applyWakeStartResult({ started: true, capture: 'client', phrase: 'hey hermes' })
    await Promise.resolve()
    await Promise.resolve()
    const state = $wakeWord.get()

    opening.reject(new Error('Old microphone request denied'))
    await opening.promise.catch(() => undefined)
    await Promise.resolve()

    expect($wakeWord.get()).toBe(state)
    expect(gatewayRequest).not.toHaveBeenCalled()
    expect(replacement.stop).not.toHaveBeenCalled()
    stopClientCapture()
    expect(replacement.stop).toHaveBeenCalledOnce()
  })
})
