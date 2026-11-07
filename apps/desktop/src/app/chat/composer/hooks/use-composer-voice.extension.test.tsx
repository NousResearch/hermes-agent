import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { type LiveVoiceEngineIO, VOICE_ENGINE_AREA } from '@/contrib/voice-engines'
import { $voiceLiveStatus } from '@/store/voice-live'

import { useComposerVoice } from './use-composer-voice'

// The composer seam for extension engines: `voice.voice_chat_mode` names a
// `<plugin>:<engine>`, the composer mounts the registered engine, and its
// delegations become `voice-live` turns through the composer's own submit —
// or, when that plugin is not loaded, the conversation runs on chained.
const mocks = vi.hoisted(() => ({
  chainedEnabled: [] as boolean[],
  liveEnabled: [] as boolean[],
  notify: vi.fn(),
  onSubmit: vi.fn(async () => true)
}))

vi.mock('@/lib/tts-lease', () => ({
  CONVERSATION_LEASE: 'conversation',
  READ_ALOUD_LEASE: 'read-aloud',
  syncTtsLease: vi.fn(async () => undefined)
}))

vi.mock('@/lib/wake-indicator', () => ({
  clearWakeIndicator: vi.fn(),
  syncWakeIndicatorWithVoice: vi.fn(() => false)
}))

vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  notify: mocks.notify
}))

vi.mock('@/store/voice-live', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  refreshVoiceLiveStatus: vi.fn(async () => null)
}))

vi.mock('@/store/wake-word', () => ({ resumeWakeAfterVoice: vi.fn(async () => undefined) }))

vi.mock('../scope', async () => {
  const { atom } = await import('nanostores')
  const $messages = atom([])

  return { useComposerScope: () => ({ $messages }), useComposerSurfaceId: () => null }
})

vi.mock('./use-auto-speak-replies', () => ({ useAutoSpeakReplies: vi.fn() }))

const conversation = {
  end: vi.fn(async () => undefined),
  level: 0,
  muted: false,
  start: vi.fn(async () => undefined),
  status: 'idle',
  stopTurn: vi.fn(),
  toggleMute: vi.fn()
}

vi.mock('./use-voice-conversation', () => ({
  useVoiceConversation: ({ enabled }: { enabled: boolean }) => {
    mocks.chainedEnabled.push(enabled)

    return conversation
  }
}))

vi.mock('./use-voice-live-conversation', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useVoiceLiveConversation: ({ enabled }: { enabled: boolean }) => {
    mocks.liveEnabled.push(enabled)

    return conversation
  }
}))

vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: () => ({
    dictate: vi.fn(),
    voiceActivityState: { elapsedSeconds: 0, level: 0, status: 'idle' },
    voiceStatus: 'idle'
  })
}))

const unloads: Array<() => void> = []
const ios: LiveVoiceEngineIO[] = []

function loadEngine() {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext('hermes-xai', dispose => disposers.push(dispose))

  ctx.register({
    area: VOICE_ENGINE_AREA,
    data: {
      label: 'Grok Voice',
      start: async (io: LiveVoiceEngineIO) => {
        ios.push(io)

        return { stop: async () => undefined }
      },
      status: async () => ({ available: true })
    },
    id: 'grok'
  })
  unloads.push(() => disposers.splice(0).forEach(dispose => dispose()))
}

function mountComposer() {
  return renderHook(() =>
    useComposerVoice({
      busy: false,
      clearDraft: vi.fn(),
      disabled: false,
      focusInput: vi.fn(),
      insertText: vi.fn(),
      maxRecordingSeconds: 60,
      onSubmit: mocks.onSubmit,
      onTranscribeAudio: vi.fn(async () => ''),
      sessionId: 'chat-7',
      target: 'main'
    })
  )
}

beforeEach(() => {
  $voiceLiveStatus.set({ available: false, mode: 'hermes-xai:grok', model: '', reason: 'no key', voice: '' })
})

afterEach(() => {
  cleanup()
  unloads.splice(0).forEach(unload => unload())
  ios.length = 0
  mocks.chainedEnabled.length = 0
  mocks.liveEnabled.length = 0
  vi.clearAllMocks()
  $voiceLiveStatus.set(null)
})

describe('useComposerVoice with an extension voice engine', () => {
  it('mounts the selected engine for this chat and submits its delegations as voice-live turns', async () => {
    loadEngine()
    const hook = mountComposer()

    await act(async () => hook.result.current.startConversation())
    await act(async () => undefined)

    expect(ios).toHaveLength(1)
    expect(ios[0].sessionId).toBe('chat-7')
    // Neither built-in engine is enabled while the extension engine runs.
    expect(mocks.chainedEnabled.at(-1)).toBe(false)
    expect(mocks.liveEnabled.at(-1)).toBe(false)

    await act(async () => {
      void ios[0].delegate('turn on the lights', 'User: turn on the lights')
    })

    expect(mocks.onSubmit).toHaveBeenCalledWith('turn on the lights', {
      surface: 'voice-live',
      voiceContext: 'User: turn on the lights'
    })

    await act(async () => hook.result.current.endConversation())
    expect(hook.result.current.voiceConversationActive).toBe(false)
  })

  it('falls back to chained with a notice when the selected engine is not registered', async () => {
    const hook = mountComposer()

    await act(async () => hook.result.current.startConversation())

    expect(ios).toHaveLength(0)
    expect(mocks.chainedEnabled.at(-1)).toBe(true)
    expect(mocks.notify).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'voice-live-unavailable', kind: 'warning' })
    )
  })
})
