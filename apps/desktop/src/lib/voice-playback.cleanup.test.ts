import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { startSpeechStream, stopVoicePlayback } from './voice-playback'
import { directTtsConfig, synthesizeSpeechClientDirect } from './voice-client-direct'

vi.mock('./voice-client-direct', () => ({
  directTtsConfig: vi.fn(),
  synthesizeSpeechClientDirect: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  speakText: vi.fn()
}))

// A client-direct speech session parks one AudioContext for the whole reply.
// Chrome warns past ~6 live contexts and then refuses to start new ones, so a
// leaked context per reply degrades into silent playback with only console
// warnings. The relay path has always closed its context on settle; the direct
// path did not. These pin that, plus the dead-{ stop } closures that used to
// accumulate for the length of a long reply.
describe('client-direct speech session resource cleanup', () => {
  let contexts: FakeAudioContext[]
  let sources: FakeSource[]

  class FakeSource {
    listeners: Record<string, Array<() => void>> = {}
    stop = vi.fn()
    buffer: unknown = null
    // Every source is chained source → highpass → gain → destination, so
    // connect() must exist or the pump throws mid-schedule.
    connect() {
      return this
    }
    disconnect() {
      return this
    }
    addEventListener(name: string, fn: () => void) {
      ;(this.listeners[name] ??= []).push(fn)
    }
    start() {
      // Real sources fire `ended` when the scheduled buffer finishes playing.
      // Without this the pump awaits forever and the test hangs on a promise
      // the real implementation would already have resolved.
      setTimeout(() => this.fireEnded(), 0)
    }
    fireEnded() {
      for (const fn of [...(this.listeners.ended ?? [])]) fn()
    }
  }

  class FakeAudioContext {
    state = 'running'
    // The scheduler reads currentTime and connects the gain chain to
    // destination. Missing either throws inside the pump before the source is
    // ever scheduled, leaving the session promise unsettled.
    currentTime = 0
    destination = {}
    close = vi.fn(async () => {
      this.state = 'closed'
    })
    resume = vi.fn(async () => {
      this.state = 'running'
    })
    createBufferSource() {
      const src = new FakeSource()
      sources.push(src)
      return src
    }
    // The DC-offset click removal chains every source through a high-pass
    // filter and a gain envelope. Omitting these threw inside the pump, which
    // left the schedule promise unsettled and the session hanging forever.
    createBiquadFilter() {
      return {
        type: '',
        frequency: { value: 0 },
        Q: { value: 0 },
        connect: () => undefined,
        disconnect: () => undefined
      }
    }
    createGain() {
      return {
        gain: { value: 1, setValueAtTime() {}, linearRampToValueAtTime() {} },
        connect: () => undefined,
        disconnect: () => undefined
      }
    }
    createBuffer = () => ({ getChannelData: () => new Float32Array(1), length: 1 })
    // decodeAudioData is invoked with (bytes, resolve, reject) callbacks and
    // wrapped in an outer promise. Returning a value instead of calling the
    // callbacks leaves that outer promise unsettled, so the pump never
    // proceeds to schedule a source and the session hangs on `done`.
    decodeAudioData = (
      _bytes: ArrayBuffer,
      onSuccess: (buffer: unknown) => void
    ): void => {
      onSuccess({
        getChannelData: () => new Float32Array(1),
        length: 1,
        sampleRate: 24000,
        duration: 0.01
      })
    }
  }

  beforeEach(() => {
    contexts = []
    sources = []
    vi.mocked(directTtsConfig).mockResolvedValue({
      provider: 'deepgram',
      model: 'aura-2-hermes-en',
      voice: 'atmos'
    } as never)
    vi.mocked(synthesizeSpeechClientDirect).mockResolvedValue(new ArrayBuffer(8))
    // ensureCtx() reads window.AudioContext, so stub it there rather than on
    // the bare global — in jsdom those are not the same object.
    const ctor = function (this: FakeAudioContext) {
      const ctx = new FakeAudioContext()
      contexts.push(ctx)
      return ctx
    } as unknown as typeof AudioContext
    vi.stubGlobal('AudioContext', ctor)
    window.AudioContext = ctor
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  // cutSentences() only releases the buffered tail on flush, and the session
  // flushes at finish(). Feeding text and waiting is not enough on its own.
  async function speakAndSettle(text: string) {
    const session = await startSpeechStream({ connectionId: null, profile: null } as never)
    expect(session).not.toBeNull()
    session!.append(text)
    session!.finish()
    await session!.done
    return session!
  }

  it('closes the AudioContext when the session settles', async () => {
    await speakAndSettle('The transmission has begun without incident tonight.')

    expect(vi.mocked(synthesizeSpeechClientDirect)).toHaveBeenCalled()
    expect(contexts.length).toBeGreaterThan(0)

    // The leak: without close(), every voice reply parks a live context for
    // the life of the page.
    await vi.waitFor(() => {
      expect(contexts[0]!.close).toHaveBeenCalledTimes(1)
    })
  })

  it('releases the context when a reply is cut short by a barge-in', async () => {
    const session = await startSpeechStream({ connectionId: null, profile: null } as never)
    expect(session).not.toBeNull()

    session!.append('A long first clause that will start playing. And a second one follows it.')
    await vi.waitFor(() => expect(contexts.length).toBeGreaterThan(0))

    // stopVoicePlayback() settles the session mid-reply — exactly the moment a
    // naive implementation leaks a context it has already created.
    stopVoicePlayback()
    await session!.done

    await vi.waitFor(() => {
      expect(contexts[0]!.close).toHaveBeenCalled()
    })
  })

  it('settle does not re-stop sources that already played out', async () => {
    await speakAndSettle('First clause here. Second clause here. Third clause here.')

    // Every source is stopped once at SCHEDULE time (the explicit
    // source.stop(startAt + duration)), so that is the baseline. If an ended
    // source were still in the tracking list, settle's cut loop would stop it
    // a second time. One call each means the pruning worked.
    expect(sources.length).toBeGreaterThan(1)
    for (const src of sources) {
      expect(src.stop.mock.calls).toHaveLength(1)
    }
  })
})
