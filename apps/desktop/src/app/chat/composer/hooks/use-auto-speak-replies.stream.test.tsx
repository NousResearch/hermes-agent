import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { renderMessageStream } from '@/app/session/hooks/use-message-stream/test-harness'
import { STREAM_DELTA_FLUSH_MS } from '@/app/session/hooks/use-message-stream/utils'
import type { ClientSessionState } from '@/app/types'
import { assistantTextPart, type ChatMessage, textPart, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearSpokenRepliesForTests, markAssistantIdSpoken, pendingSpeechReply } from '@/lib/spoken-reply'
import { stopVoicePlayback } from '@/lib/voice-playback'
import { $hapticsMuted } from '@/store/haptics'
import { $voicePlayback } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { createEventDeduper } from '../../../../../electron/event-dedupe'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/lib/voice-client-direct', () => ({ directTtsConfig: vi.fn(async () => null) }))

class TestAudioContext {
  static instances: TestAudioContext[] = []
  currentTime = 0
  destination = {}
  state = 'running'
  close = vi.fn(async () => undefined)
  resume = vi.fn(async () => undefined)
  starts: number[] = []

  constructor() {
    TestAudioContext.instances.push(this)
  }

  createBuffer(_channels: number, length: number, rate: number) {
    return { duration: length / rate, getChannelData: () => new Float32Array(length) }
  }

  createBufferSource() {
    return { buffer: null, connect: vi.fn(), start: (at: number) => this.starts.push(at) }
  }
}

class TestSocket {
  static readonly OPEN = 1
  static readonly CONNECTING = 0
  static instances: TestSocket[] = []
  readyState = TestSocket.OPEN
  binaryType = ''
  sent: string[] = []
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  close = vi.fn(() => {
    this.readyState = 3
  })

  constructor() {
    TestSocket.instances.push(this)
  }

  send(data: string) {
    this.sent.push(data)
  }

  audio(samples = 24_000) {
    this.onmessage?.({ data: JSON.stringify({ type: 'start', sample_rate: 24_000 }) })
    this.onmessage?.({ data: new Int16Array(samples).buffer })
  }
}

const SID = 'voice-stream-completion'

const reply = (id: string, text: string, pending = false): ChatMessage => ({
  id,
  role: 'assistant',
  pending,
  parts: [assistantTextPart(text)]
})

function mountSpeech(initial: ChatMessage[] = []) {
  const $messages = atom<ChatMessage[]>(initial)
  $autoSpeakReplies.set(true)
  renderHook(
    () =>
      useAutoSpeakReplies({
        conversationActive: false,
        failureLabel: 'Speech failed',
        sessionId: SID,
        markSpoken: () => {
          const message = $messages.get().findLast(m => m.role === 'assistant')

          if (message) {
            markAssistantIdSpoken(SID, $messages.get(), message.id)
          }
        },
        pendingReply: () => pendingSpeechReply(SID, $messages.get())
      }),
    {
      wrapper: ({ children }) => (
        <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
      )
    }
  )

  return $messages
}

const tick = async (ms = 0) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

describe('auto-speech with real playback and completion events', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $hapticsMuted.set(true)
    TestSocket.instances = []
    TestAudioContext.instances = []
    vi.stubGlobal('WebSocket', TestSocket)
    vi.stubGlobal('AudioContext', TestAudioContext)
    const duplicateCue = createEventDeduper()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        claimAmbientCue: vi.fn(async (key: string) => !duplicateCue(key)),
        getConnection: vi.fn(async () => ({ authMode: 'token', wsUrl: 'ws://127.0.0.1/api/ws?token=test' })),
        getGatewayWsUrl: vi.fn(async () => ({ ok: true, wsUrl: 'ws://127.0.0.1/api/ws?token=test' }))
      }
    })
  })

  afterEach(() => {
    cleanup()
    stopVoicePlayback()
    clearSpokenRepliesForTests()
    $autoSpeakReplies.set(false)
    $hapticsMuted.set(false)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('keeps queued audio alive when the real gateway reducer completes the reply', async () => {
    const $messages = mountSpeech()
    const states = new Map<string, ClientSessionState>()

    const stream = renderMessageStream(SID, {
      states,
      updateSessionState: (id, updater) => {
        const state = updater(states.get(id) ?? createClientSessionState())
        states.set(id, state)
        $messages.set(state.messages)

        return state
      }
    })

    act(() => stream.handleEvent({ type: 'message.start', session_id: SID }))
    act(() => stream.handleEvent({ type: 'message.delta', session_id: SID, payload: { text: '一文目です。' } }))
    await tick(STREAM_DELTA_FLUSH_MS)
    const socket = TestSocket.instances[0]
    expect(socket).toBeDefined()
    act(() => socket.audio())
    expect($voicePlayback.get().status).toBe('speaking')
    act(() =>
      stream.handleEvent({ type: 'message.complete', session_id: SID, payload: { text: '一文目です。二文目です。' } })
    )
    await tick()
    expect(socket.close).not.toHaveBeenCalled()
    expect(TestAudioContext.instances[0].close).not.toHaveBeenCalled()
    expect(socket.sent.map(frame => JSON.parse(frame))).toEqual([
      { text: '一文目です。' },
      { text: '二文目です。' },
      { done: true }
    ])
    act(() => socket.onmessage?.({ data: JSON.stringify({ type: 'end' }) }))
    await tick(1200)
    expect($voicePlayback.get().status).toBe('idle')
    expect(TestSocket.instances).toHaveLength(1)
  })

  it('keeps the active reply when older assistant rows merge during transcript refresh', async () => {
    const $messages = mountSpeech([reply('old-a', '確認します。'), reply('old-b', '結果です。')])
    act(() => $messages.set([...$messages.get(), { id: 'user-2', role: 'user', parts: [assistantTextPart('次は？')] }]))
    act(() => $messages.set([...$messages.get(), reply('assistant-stream-new', '一文目です。二文目です。', true)]))
    await tick()
    const socket = TestSocket.instances.at(-1)!
    act(() => socket.audio())
    const before = TestSocket.instances.length
    act(() =>
      $messages.set([
        reply('old-merged', '確認します。結果です。'),
        { id: 'user-2', role: 'user', parts: [assistantTextPart('次は？')] },
        reply('durable-new', '一文目です。二文目です。')
      ])
    )
    await tick()
    expect(socket.close).not.toHaveBeenCalled()
    expect(TestSocket.instances).toHaveLength(before)
  })

  it.each([false, true])(
    'does not cut or replay a formatted reply when the history window moves (Stop=%s)',
    async stopped => {
      const oldUser: ChatMessage = { id: 'old-user', rowId: 1, role: 'user', parts: [textPart('前の質問')] }
      const $messages = mountSpeech([oldUser, reply('old-answer', '前の回答です。')])
      const user: ChatMessage = { id: 'user-new', role: 'user', parts: [textPart('次の質問')] }
      const raw = '一文目です。  \n二文目です。  \n三文目です。'
      act(() =>
        $messages.set([
          ...$messages.get(),
          user,
          { ...reply('assistant-stream-new', raw, true), parts: [textPart(raw)] }
        ])
      )
      await tick()
      const socket = TestSocket.instances[0]
      act(() => socket.audio())
      act(() => $messages.set($messages.get().map(message => ({ ...message, pending: false }))))

      if (stopped) {
        act(() => stopVoicePlayback())
      }

      // REST hydration invokes the real display formatter, drops an older page,
      // and replaces the optimistic ids with durable rows. None is new input.
      const hydrated = toChatMessages([
        { id: 10, role: 'user', content: '次の質問', timestamp: 100 },
        { id: 11, role: 'assistant', content: raw, timestamp: 101 }
      ])

      act(() => $messages.set(hydrated))
      await tick()

      if (!stopped) {
        expect(socket.close).not.toHaveBeenCalled()
        act(() => socket.onmessage?.({ data: JSON.stringify({ type: 'end' }) }))
      }

      await tick(1200)
      act(() => $messages.set([oldUser, reply('old-answer', '前の回答です。'), ...hydrated]))
      await tick()
      expect(TestSocket.instances).toHaveLength(1)
      expect(socket.sent.map(frame => JSON.parse(frame))).toEqual([{ text: raw }, { done: true }])
      expect($voicePlayback.get().status).toBe('idle')
    }
  )

  it('appends only the new sentence when display whitespace changes before completion', async () => {
    const user: ChatMessage = { id: 'user-1', role: 'user', parts: [textPart('質問')] }
    const $messages = mountSpeech([user])
    const raw = '一文目です。  \n二文目です。'
    act(() => $messages.set([user, { ...reply('assistant-stream-1', raw, true), parts: [textPart(raw)] }]))
    await tick()
    const socket = TestSocket.instances[0]
    act(() => socket.audio())
    act(() => $messages.set([user, reply('assistant-stream-1', '一文目です。\n二文目です。三文目です。')]))
    await tick()
    expect(socket.close).not.toHaveBeenCalled()
    expect(socket.sent.map(frame => JSON.parse(frame))).toEqual([
      { text: raw },
      { text: '三文目です。' },
      { done: true }
    ])
  })

  it('still retries complete text when a finished stream produced no audio', async () => {
    const $messages = mountSpeech()
    act(() => $messages.set([reply('assistant-stream-1', '無音の場合だけやり直します。')]))
    await tick()
    const socket = TestSocket.instances[0]
    act(() => socket.onmessage?.({ data: JSON.stringify({ type: 'fallback' }) }))
    await tick()
    expect(TestSocket.instances).toHaveLength(2)
    expect(TestSocket.instances[1].sent.map(frame => JSON.parse(frame))).toEqual([
      { text: '無音の場合だけやり直します。' },
      { done: true }
    ])
  })

  it.each([false, true])(
    'keeps final audio when hydration merges it with earlier narration (pending=%s)',
    async pending => {
      const user: ChatMessage = { id: 'stored-user', rowId: 10, role: 'user', parts: [textPart('質問')] }
      const $messages = mountSpeech([user, reply('interim', '確認します。')])
      act(() => $messages.set([...$messages.get(), reply('assistant-stream-final', '結果です。', pending)]))
      await tick()
      const socket = TestSocket.instances[0]
      act(() => socket.audio())

      const hydrated = toChatMessages([
        { id: 10, role: 'user', content: '質問', timestamp: 100 },
        {
          id: 11,
          role: 'assistant',
          content: '確認します。',
          timestamp: 101,
          tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'test', arguments: '{}' } }]
        },
        { id: 12, role: 'assistant', content: '結果です。', timestamp: 102 }
      ])

      expect(hydrated.filter(message => message.role === 'assistant')).toHaveLength(1)
      act(() => $messages.set(hydrated))
      await tick()
      expect(socket.close).not.toHaveBeenCalled()
      act(() => socket.onmessage?.({ data: JSON.stringify({ type: 'end' }) }))
      await tick(1200)
      expect(TestSocket.instances).toHaveLength(1)
      expect(socket.sent.map(frame => JSON.parse(frame)).filter(frame => frame.text)).toEqual([{ text: '結果です。' }])
    }
  )

  it('reads a new final suffix once after already-spoken narration is merged into its bubble', async () => {
    const user: ChatMessage = { id: 'stored-user', rowId: 10, role: 'user', parts: [textPart('質問')] }
    const $messages = mountSpeech([user])
    act(() => $messages.set([user, reply('assistant-stream-interim', '確認します。')]))
    await tick()
    const first = TestSocket.instances[0]
    act(() => first.audio())

    const hydrated = toChatMessages([
      { id: 10, role: 'user', content: '質問', timestamp: 100 },
      {
        id: 11,
        role: 'assistant',
        content: '確認します。',
        timestamp: 101,
        tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'test', arguments: '{}' } }]
      },
      { id: 12, role: 'assistant', content: '結果です。', timestamp: 102 }
    ])

    act(() => $messages.set(hydrated))
    act(() => first.onmessage?.({ data: JSON.stringify({ type: 'end' }) }))
    await tick(1200)
    expect(TestSocket.instances).toHaveLength(2)
    const second = TestSocket.instances[1]
    expect(second.sent.map(frame => JSON.parse(frame))).toEqual([{ text: '結果です。' }, { done: true }])
    act(() => second.audio())
    act(() => second.onmessage?.({ data: JSON.stringify({ type: 'end' }) }))
    await tick(1200)
    act(() => $messages.set([...hydrated]))
    await tick()
    expect(TestSocket.instances).toHaveLength(2)
  })

  it('claims a new suffix of the same durable bubble before the narration claim expires', async () => {
    const user: ChatMessage = { id: 'user-1', role: 'user', parts: [textPart('質問')] }
    const $messages = mountSpeech([user])
    act(() => $messages.set([user, { ...reply('durable-11', '確認します。'), rowId: 11 }]))
    await tick()
    const first = TestSocket.instances[0]
    act(() => first.audio(2400))
    act(() => $messages.set([user, { ...reply('durable-11', '確認します。結果です。'), rowId: 11 }]))
    act(() => first.onmessage?.({ data: JSON.stringify({ type: 'end' }) }))
    await tick(400)
    expect(TestSocket.instances).toHaveLength(2)
    expect(TestSocket.instances[1].sent.map(frame => JSON.parse(frame))).toEqual([
      { text: '結果です。' },
      { done: true }
    ])
    const claims = vi.mocked(window.hermesDesktop!.claimAmbientCue).mock.calls.map(([key]) => key)
    expect(claims).toEqual(['speak:durable-11:0', 'speak:durable-11:6'])
    expect(await window.hermesDesktop!.claimAmbientCue(claims[1])).toBe(false)
  })
})
