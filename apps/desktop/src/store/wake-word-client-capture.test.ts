// Issue #119089: a dead client PCM chain must land the ear in an honest off
// state with the reason visible — not a "listening" toggle that can never fire.
import { beforeEach, describe, expect, it, vi } from 'vitest'

const capture = vi.hoisted(() => ({
  options: null as {
    frameLength?: number
    onError?: (error: Error) => void
  } | null,
  stop: vi.fn()
}))

vi.mock('@/lib/wake-client-capture', () => ({
  startClientWakeCapture: vi.fn(async (options: typeof capture.options) => {
    capture.options = options

    return { active: true, stop: capture.stop }
  })
}))

import { $wakeWord, applyWakeStartResult, resetWakeWordState } from './wake-word'

const flush = () => new Promise<void>(resolve => setTimeout(resolve, 0))

beforeEach(() => {
  capture.options = null
  capture.stop.mockClear()
  resetWakeWordState()
})

describe('client capture failure surfacing (issue #119089)', () => {
  it('lands the ear off with the reason when the PCM chain dies after arming', async () => {
    applyWakeStartResult({ capture: 'client', frame_length: 1280, phrase: 'hey hermes', started: true })
    await flush()

    expect($wakeWord.get()).toMatchObject({ listening: true, notice: '' })
    expect(capture.options?.onError).toBeTypeOf('function')

    capture.options?.onError?.(new Error('client wake capture hears only silence'))

    expect(capture.stop).toHaveBeenCalled()
    expect($wakeWord.get()).toMatchObject({
      listening: false,
      notice: 'client wake capture hears only silence'
    })
  })
})
