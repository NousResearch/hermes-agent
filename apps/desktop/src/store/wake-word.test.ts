import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $wakeWord,
  applyWakeStartResult,
  applyWakeStatus,
  applyWakeStopResult,
  armWakeWord,
  releaseWakeWord,
  resetWakeWordState,
  resumeWakeAfterVoice,
  toggleWakeWord,
  WAKE_OWNED_RETRY_MS,
  wakeListenerRole,
  type WakeRequester
} from './wake-word'

const requester = (impl: (method: string, params?: Record<string, unknown>) => unknown) =>
  vi.fn(async (method: string, params: Record<string, unknown> = {}) =>
    impl(method, params)
  ) as unknown as WakeRequester

beforeEach(() => {
  resetWakeWordState()
})

describe('applyWakeStatus', () => {
  it('syncs availability, listening and phrase from wake.status', () => {
    applyWakeStatus({
      available: true,
      hint: '',
      listening: true,
      owned_by_caller: true,
      owner_surface: 'gui',
      phrase: 'hey hermes',
      provider: 'openwakeword'
    })

    expect($wakeWord.get()).toMatchObject({
      available: true,
      listening: true,
      notice: '',
      phrase: 'hey hermes'
    })
  })

  it('tracks unavailability and carries the hint for the tooltip', () => {
    applyWakeStatus({ available: false, hint: 'pip install openwakeword', listening: false, phrase: 'hey hermes' })

    const state = $wakeWord.get()
    expect(state.available).toBe(false)
    expect(state.listening).toBe(false)
    expect(state.notice).toBe('pip install openwakeword')
  })

  it('keeps the dead-mic hint visible while listening (audio_silent)', () => {
    applyWakeStatus({
      audio_silent: true,
      available: true,
      hint: 'Microphone delivers only silence — grant mic access',
      listening: true,
      phrase: 'hey hermes'
    })

    const state = $wakeWord.get()
    expect(state.listening).toBe(true)
    expect(state.notice).toBe('Microphone delivers only silence — grant mic access')
  })
})

describe('toggleWakeWord', () => {
  it('starts via wake.start with surface gui when off, and flips to listening', async () => {
    applyWakeStatus({ available: true, listening: false, phrase: 'hey hermes' })

    const request = requester(method => {
      expect(method).toBe('wake.start')

      return { owner_surface: 'gui', phrase: 'hey hermes', provider: 'porcupine', started: true }
    })

    await toggleWakeWord(request)

    expect(request).toHaveBeenCalledWith('wake.start', { client_capture: true, persist: true, surface: 'gui' })
    expect($wakeWord.get()).toMatchObject({ listening: true, notice: '', pending: false })
  })

  it('stops via wake.stop when listening', async () => {
    applyWakeStatus({ available: true, listening: true, phrase: 'hey hermes' })

    const request = requester(method => {
      expect(method).toBe('wake.stop')

      return { reason: null, stopped: true }
    })

    await toggleWakeWord(request)

    expect(request).toHaveBeenCalledWith('wake.stop', { persist: true })
    expect($wakeWord.get()).toMatchObject({ listening: false, notice: '', pending: false })
  })

  it('does NOT flip state on {started:false, reason} and surfaces the reason', async () => {
    applyWakeStatus({ available: true, listening: false, phrase: 'hey hermes' })

    await toggleWakeWord(requester(() => ({ owner_surface: 'tui', reason: 'owned', started: false })))

    const state = $wakeWord.get()
    expect(state.listening).toBe(false)
    expect(state.notice).toBe('another surface owns the listener')
    expect(state.available).toBe(true)
  })

  it('marks the feature unavailable when start refuses with reason unavailable', async () => {
    applyWakeStatus({ available: true, listening: false, phrase: 'hey hermes' })

    await toggleWakeWord(requester(() => ({ hint: 'Set PORCUPINE_ACCESS_KEY', reason: 'unavailable', started: false })))

    const state = $wakeWord.get()
    expect(state.available).toBe(false)
    expect(state.listening).toBe(false)
    expect(state.notice).toBe('Set PORCUPINE_ACCESS_KEY')
  })

  it('stays off and keeps the error as the notice when the RPC throws', async () => {
    applyWakeStatus({ available: true, listening: false, phrase: 'hey hermes' })

    await toggleWakeWord(
      requester(() => {
        throw new Error('Hermes gateway unavailable')
      })
    )

    expect($wakeWord.get()).toMatchObject({
      listening: false,
      notice: 'Hermes gateway unavailable',
      pending: false
    })
  })

  it('ignores clicks while a toggle is already in flight', async () => {
    applyWakeStatus({ available: true, listening: false, phrase: 'hey hermes' })

    let resolveStart: (value: unknown) => void = () => undefined

    const request = vi.fn(
      async () =>
        new Promise(resolve => {
          resolveStart = resolve
        })
    ) as unknown as WakeRequester

    const first = toggleWakeWord(request)
    await toggleWakeWord(request)

    expect(request).toHaveBeenCalledTimes(1)

    resolveStart({ phrase: 'hey hermes', started: true })
    await first

    expect($wakeWord.get().listening).toBe(true)
  })
})

describe('armWakeWord (gateway-ready auto-arm)', () => {
  it('queries wake.status then arms and syncs the store', async () => {
    const calls: string[] = []

    const request = requester(method => {
      calls.push(method)

      if (method === 'wake.status') {
        return { available: true, listening: false, phrase: 'hey hermes', provider: 'porcupine' }
      }

      return { phrase: 'hey hermes', started: true }
    })

    await armWakeWord(request)

    expect(calls).toEqual(['wake.status', 'wake.start'])
    expect($wakeWord.get()).toMatchObject({ available: true, listening: true, phrase: 'hey hermes' })
  })

  it('does not attempt to arm when the wake word is unavailable', async () => {
    const calls: string[] = []

    const request = requester(method => {
      calls.push(method)

      return { available: false, hint: 'no mic', listening: false, phrase: 'hey hermes' }
    })

    await armWakeWord(request)

    expect(calls).toEqual(['wake.status'])
    expect($wakeWord.get()).toMatchObject({ available: false, listening: false, notice: 'no mic' })
  })

  it('skips arming when this surface already listens (status sync only)', async () => {
    const calls: string[] = []

    const request = requester(method => {
      calls.push(method)

      return { available: true, listening: true, owned_by_caller: true, phrase: 'hey hermes' }
    })

    await armWakeWord(request)

    expect(calls).toEqual(['wake.status'])
    expect($wakeWord.get()).toMatchObject({ available: true, listening: true })
  })

  it('keeps the default hidden state when the backend lacks wake.* methods', async () => {
    await armWakeWord(
      requester(() => {
        throw new Error('Unknown method: wake.status')
      })
    )

    expect($wakeWord.get()).toMatchObject({ available: false, listening: false })
  })

  it('keeps the toggle off when auto-arm is refused (e.g. TUI owns the mic)', async () => {
    const request = requester(method =>
      method === 'wake.status'
        ? { available: true, listening: false, owner_surface: 'tui', phrase: 'hey hermes' }
        : { owner_surface: 'tui', reason: 'owned', started: false }
    )

    await armWakeWord(request)

    const state = $wakeWord.get()
    expect(state.available).toBe(true)
    expect(state.listening).toBe(false)
    expect(state.notice).toBe('another surface owns the listener')
  })
})

describe('applyWakeStopResult', () => {
  it('lands on off even when the backend says not_owner', () => {
    applyWakeStatus({ available: true, listening: true, phrase: 'hey hermes' })

    applyWakeStopResult({ reason: 'not_owner', stopped: false })

    const state = $wakeWord.get()
    expect(state.listening).toBe(false)
    expect(state.notice).toBe('another surface owns the listener')
  })
})

describe('applyWakeStartResult', () => {
  it('adopts the backend phrase when the listener starts', () => {
    applyWakeStartResult({ phrase: 'computer', provider: 'porcupine', started: true })

    expect($wakeWord.get()).toMatchObject({ available: true, listening: true, phrase: 'computer' })
  })
})

describe('resumeWakeAfterVoice (post-voice reconcile)', () => {
  it('re-arms when config says enabled but the listener is down', async () => {
    const calls: string[] = []

    const request = requester(method => {
      calls.push(method)

      if (method === 'wake.resume') {
        return { reason: 'not_owner', resumed: false }
      }

      if (method === 'wake.status') {
        return { available: true, enabled: true, listening: false, phrase: 'hey hermes' }
      }

      return { phrase: 'hey hermes', started: true }
    })

    await resumeWakeAfterVoice(request)

    expect(calls).toEqual(['wake.resume', 'wake.status', 'wake.start'])
    expect($wakeWord.get()).toMatchObject({ listening: true })
  })

  it('re-arm start never passes persist (passive path must not write config)', async () => {
    const startParams: Array<Record<string, unknown> | undefined> = []

    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'wake.resume') {
        return { resumed: false }
      }

      if (method === 'wake.status') {
        return { available: true, enabled: true, listening: false }
      }

      startParams.push(params)

      return { started: true }
    }) as unknown as WakeRequester

    await resumeWakeAfterVoice(request)

    expect(startParams).toEqual([{ client_capture: true, surface: 'gui' }])
  })

  it('stops after the resume alone brings the listener back', async () => {
    const calls: string[] = []

    const request = requester(method => {
      calls.push(method)

      if (method === 'wake.resume') {
        return { resumed: true }
      }

      return { available: true, enabled: true, listening: true, owned_by_caller: true }
    })

    await resumeWakeAfterVoice(request)

    expect(calls).toEqual(['wake.resume', 'wake.status'])
    expect($wakeWord.get()).toMatchObject({ listening: true })
  })

  it('leaves the listener off when config says disabled', async () => {
    const calls: string[] = []

    const request = requester(method => {
      calls.push(method)

      if (method === 'wake.resume') {
        return { resumed: false }
      }

      return { available: true, enabled: false, listening: false }
    })

    await resumeWakeAfterVoice(request)

    expect(calls).toEqual(['wake.resume', 'wake.status'])
    expect($wakeWord.get().listening).toBe(false)
  })

  it('yields when another surface owns the mic lease', async () => {
    const calls: string[] = []

    const request = requester(method => {
      calls.push(method)

      if (method === 'wake.resume') {
        return { resumed: false }
      }

      if (method === 'wake.status') {
        return { available: true, enabled: true, listening: false, owner_surface: 'tui' }
      }

      return { owner_surface: 'tui', reason: 'owned', started: false }
    })

    await resumeWakeAfterVoice(request)

    expect(calls).toEqual(['wake.resume', 'wake.status', 'wake.start'])
    expect($wakeWord.get().listening).toBe(false)
  })

  it('is a no-op against older backends without wake.* methods', async () => {
    const request = requester(() => {
      throw new Error('Unknown method: wake.resume')
    })

    await resumeWakeAfterVoice(request)

    expect($wakeWord.get()).toMatchObject({ available: false, listening: false })
  })
})

describe('HUD handoff: one window holds the wake listener', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('the app window releases while a HUD is up; the HUD and a lone app window claim', () => {
    expect(wakeListenerRole(false, true)).toBe('release')
    expect(wakeListenerRole(false, false)).toBe('claim')
    expect(wakeListenerRole(true, true)).toBe('claim')
  })

  it('release stops without persist, so wake_word.enabled stays on', async () => {
    applyWakeStartResult({ phrase: 'hey hermes', started: true })

    const request = requester(() => ({ stopped: true }))

    await releaseWakeWord(request)

    expect(request).toHaveBeenCalledWith('wake.stop', {})
    expect($wakeWord.get()).toMatchObject({ enabled: true, listening: false, notice: '' })
  })

  it('a claiming window retries while the previous owner still holds the lease', async () => {
    vi.useFakeTimers()
    let starts = 0

    const request = requester(method => {
      if (method === 'wake.status') {
        return { available: true, listening: false, phrase: 'hey hermes' }
      }

      starts += 1

      // The app window's release lands after the HUD's first two attempts.
      return starts < 3
        ? { owner_surface: 'gui', reason: 'owned', started: false }
        : { phrase: 'hey hermes', started: true }
    })

    const armed = armWakeWord(request, { retryOwned: 5 })
    await vi.advanceTimersByTimeAsync(2 * WAKE_OWNED_RETRY_MS)
    await armed

    expect(starts).toBe(3)
    expect($wakeWord.get()).toMatchObject({ listening: true, notice: '' })
  })

  it('gives up after retryOwned and keeps the refusal as the notice', async () => {
    vi.useFakeTimers()

    const request = requester(method =>
      method === 'wake.status'
        ? { available: true, listening: false, phrase: 'hey hermes' }
        : { owner_surface: 'tui', reason: 'owned', started: false }
    )

    const armed = armWakeWord(request, { retryOwned: 2 })
    await vi.advanceTimersByTimeAsync(2 * WAKE_OWNED_RETRY_MS)
    await armed

    expect(request).toHaveBeenCalledTimes(4) // status + first start + 2 retries
    expect($wakeWord.get()).toMatchObject({ listening: false, notice: 'another surface owns the listener' })
  })

  it('does not retry other refusals', async () => {
    const request = requester(method =>
      method === 'wake.status'
        ? { available: true, listening: false, phrase: 'hey hermes' }
        : { reason: 'disabled', started: false }
    )

    await armWakeWord(request, { retryOwned: 5 })

    expect(request).toHaveBeenCalledTimes(2)
  })
})
