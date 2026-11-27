import { VOICE_ENGINE_AREA } from '@hermes/plugin-sdk'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import type { LiveVoiceEngine, LiveVoiceEngineIO } from '@/contrib/voice-engines'
import { en } from '@/i18n/en'
import { $notifications, clearNotifications } from '@/store/notifications'

import { useExtensionVoiceConversation } from './use-extension-voice-conversation'

// The engine is the only seam: a fake that records the io core hands it and a
// stop() the tests can count. Everything else (registry, plugin context,
// notifications) is real.
interface FakeEngine extends LiveVoiceEngine {
  ios: LiveVoiceEngineIO[]
  stops: number
}

const unloads: Array<() => void> = []

function loadFakeEngine(overrides: Partial<LiveVoiceEngine> = {}): FakeEngine {
  const engine: FakeEngine = {
    ios: [],
    label: 'Fake Voice',
    start: async io => {
      engine.ios.push(io)

      return {
        setMuted: vi.fn(),
        stop: async () => {
          engine.stops += 1
        }
      }
    },
    status: async () => ({ available: true }),
    stops: 0,
    ...overrides
  }

  const disposers: Array<() => void> = []
  const ctx = createPluginContext('fake', dispose => disposers.push(dispose))
  ctx.register({ area: VOICE_ENGINE_AREA, data: engine, id: 'live' })
  unloads.push(() => disposers.splice(0).forEach(dispose => dispose()))

  return engine
}

interface Reply {
  id: string
  pending: boolean
  text: string
}

function mount(overrides: Partial<Parameters<typeof useExtensionVoiceConversation>[0]> = {}) {
  const state = { busy: false, reply: null as null | Reply }
  const onSubmit = vi.fn(async () => undefined)
  const onUnavailable = vi.fn()
  const onFatalError = vi.fn()
  const onStopWord = vi.fn()

  const hook = renderHook(
    ({ enabled }: { enabled: boolean }) =>
      useExtensionVoiceConversation({
        busy: state.busy,
        consumePendingResponse: () => {
          state.reply = null
        },
        enabled,
        engineId: 'fake:live',
        onFatalError,
        onStopWord,
        onSubmit,
        onUnavailable,
        pendingResponse: () => state.reply,
        seedHistory: () => [{ role: 'user', text: 'earlier question' }],
        sessionId: 'session-1',
        ...overrides
      }),
    { initialProps: { enabled: false } }
  )

  return { hook, onFatalError, onStopWord, onSubmit, onUnavailable, state }
}

async function enable(hook: ReturnType<typeof mount>['hook']) {
  await act(async () => {
    hook.rerender({ enabled: true })
  })
  await act(async () => undefined)
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  clearNotifications()
})

afterEach(() => {
  cleanup()
  unloads.splice(0).forEach(unload => unload())
  vi.useRealTimers()
})

describe('useExtensionVoiceConversation', () => {
  it('starts the engine with the chat id and history, and stops it when disabled', async () => {
    const engine = loadFakeEngine()
    const { hook } = mount()

    await enable(hook)

    expect(engine.ios).toHaveLength(1)
    expect(engine.ios[0]).toMatchObject({
      history: [{ role: 'user', text: 'earlier question' }],
      sessionId: 'session-1'
    })
    expect(hook.result.current.status).toBe('listening')

    await act(async () => {
      hook.rerender({ enabled: false })
    })

    expect(engine.stops).toBe(1)
    expect(hook.result.current.status).toBe('idle')
  })

  it('delegate submits a voice-live turn and resolves with the settled reply', async () => {
    const engine = loadFakeEngine()
    const { hook, onSubmit, state } = mount()
    await enable(hook)

    let answer: null | string = null

    await act(async () => {
      void engine.ios[0].delegate('  what is on my calendar  ', 'User: what is on my calendar').then(text => {
        answer = text
      })
    })

    expect(onSubmit).toHaveBeenCalledWith('what is on my calendar', 'User: what is on my calendar')
    expect(hook.result.current.status).toBe('thinking')

    // The turn streams, then settles.
    state.busy = true
    hook.rerender({ enabled: true })
    state.reply = { id: 'a1', pending: true, text: 'You have' }
    await act(async () => vi.advanceTimersByTimeAsync(400))
    expect(answer).toBeNull()

    state.busy = false
    hook.rerender({ enabled: true })
    state.reply = { id: 'a1', pending: false, text: '**You have** two meetings.' }
    await act(async () => vi.advanceTimersByTimeAsync(400))

    expect(answer).toBe('You have two meetings.')
    expect(hook.result.current.status).toBe('listening')
  })

  it('a newer request supersedes the one in flight', async () => {
    const engine = loadFakeEngine()
    const { hook, onSubmit } = mount()
    await enable(hook)

    let first: null | string = null
    let second: null | string = null

    await act(async () => {
      void engine.ios[0].delegate('first', '').then(text => (first = text))
      void engine.ios[0].delegate('second', '').then(text => (second = text))
    })

    expect(onSubmit).toHaveBeenCalledTimes(2)
    expect(first).toBe('')
    expect(second).toBeNull()

    // Ending the call answers the abandoned request instead of leaving it hanging.
    await act(async () => {
      await hook.result.current.end()
    })
    expect(second).toBe('')
  })

  it('rejects the delegation when the turn could not be submitted', async () => {
    const engine = loadFakeEngine()
    const { hook } = mount({ onSubmit: vi.fn(async () => Promise.reject(new Error('gateway offline'))) })
    await enable(hook)

    await expect(engine.ios[0].delegate('do the thing', '')).rejects.toThrow('gateway offline')
    expect($notifications.get().some(n => n.title === en.notifications.voice.liveDelegationFailed)).toBe(true)
  })

  it('a spoken stop phrase ends the call instead of becoming a turn', async () => {
    const engine = loadFakeEngine()
    const { hook, onStopWord, onSubmit } = mount()
    await enable(hook)

    await act(async () => {
      await engine.ios[0].delegate('stop', '')
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(onStopWord).toHaveBeenCalled()
    expect(engine.stops).toBe(1)
  })

  it('refuses a second call while one is active (one call at a time)', async () => {
    const engine = loadFakeEngine()
    const first = mount()
    const second = mount()

    await enable(first.hook)
    await enable(second.hook)

    expect(engine.ios).toHaveLength(1)
    expect(second.onFatalError).toHaveBeenCalled()

    // Ending the first frees the slot.
    await act(async () => {
      await first.hook.result.current.end()
    })
    await act(async () => {
      await second.hook.result.current.start()
    })
    expect(engine.ios).toHaveLength(2)
  })

  it('falls back when the engine reports unavailable at start, with its reason', async () => {
    const engine = loadFakeEngine({ status: async () => ({ available: false, reason: 'no xAI sign-in' }) })
    const { hook, onUnavailable } = mount()

    await enable(hook)

    expect(engine.ios).toHaveLength(0)
    expect(onUnavailable).toHaveBeenCalled()
    expect($notifications.get().map(n => n.message)).toContain(
      en.notifications.voice.extensionUnavailable('Fake Voice', 'no xAI sign-in')
    )
  })

  it('falls back when no registered engine matches the selected mode (plugin gone)', async () => {
    const { hook, onUnavailable } = mount()

    await enable(hook)

    expect(onUnavailable).toHaveBeenCalled()
    expect($notifications.get().map(n => n.message)).toContain(en.notifications.voice.extensionMissing('fake:live'))
  })

  it('an engine-side close tears the call down and tells the user', async () => {
    const engine = loadFakeEngine()
    const { hook, onFatalError } = mount()
    await enable(hook)

    await act(async () => {
      engine.ios[0].onClosed('connection_lost')
    })

    expect(onFatalError).toHaveBeenCalled()
    expect(engine.stops).toBe(1)
    expect($notifications.get().map(n => n.message)).toContain(en.notifications.voice.liveEndedConnectionLost)
  })

  it('a goodbye the engine reports as close_requested ends the call without a warning', async () => {
    const engine = loadFakeEngine()
    const { hook, onFatalError, onStopWord } = mount()
    await enable(hook)

    await act(async () => {
      engine.ios[0].onClosed('close_requested')
    })

    expect(onStopWord).toHaveBeenCalled()
    expect(onFatalError).not.toHaveBeenCalled()
    expect(engine.stops).toBe(1)
    expect($notifications.get()).toEqual([])
  })

  it('stops a session that finished starting after the call was ended', async () => {
    let finishStart: () => void = () => undefined
    const stop = vi.fn(async () => undefined)

    loadFakeEngine({
      start: () =>
        new Promise(resolve => {
          finishStart = () => resolve({ stop })
        })
    })
    const { hook } = mount()
    await enable(hook)

    await act(async () => {
      hook.rerender({ enabled: false })
    })
    await act(async () => finishStart())
    await act(async () => undefined)

    expect(stop).toHaveBeenCalledTimes(1)
  })

  it('unmount stops the engine', async () => {
    const engine = loadFakeEngine()
    const { hook } = mount()
    await enable(hook)

    await act(async () => hook.unmount())

    expect(engine.stops).toBe(1)
  })
})
