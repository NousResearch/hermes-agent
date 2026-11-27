import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestProfile } from '@/api/client'
import { createPluginContext } from '@/contrib/plugin'
import {
  findLiveVoiceEngine,
  type LiveVoiceEngineIO,
  liveVoiceEngineStatus,
  VOICE_ENGINE_AREA
} from '@/contrib/voice-engines'
import type { HermesApiRequest } from '@/global'
import { refreshVoiceSignIn } from '@/store/voice-engine-sign-in'
import { $voiceLiveStatus } from '@/store/voice-live'
import { installRestBridge } from '@/test/rest-bridge'

import { useComposerVoice } from './use-composer-voice'

// Which profile serves an extension voice engine. The engine's desktop half is
// app-wide (it loads in every profile), but its backend answers only in a
// profile that enabled the plugin: the gate refuses every other profile with
// 404 "Plugin not found" (PLG-3). These tests drive the real `ctx.rest` door
// through the preload bridge, so they see the exact (path, profile) a call sends.
const mocks = vi.hoisted(() => ({
  chainedEnabled: [] as boolean[],
  notify: vi.fn(),
  notifyError: vi.fn(),
  onSubmit: vi.fn(async () => true),
  owner: { connectionId: undefined as string | undefined, profile: undefined as string | undefined }
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
  notify: mocks.notify,
  notifyError: mocks.notifyError
}))

vi.mock('@/store/voice-live', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  refreshVoiceLiveStatus: vi.fn(async () => null)
}))

vi.mock('@/store/wake-word', () => ({ resumeWakeAfterVoice: vi.fn(async () => undefined) }))

// The composer of a Bot chat names the Bot as its session owner (ChatRuntimeBoundary
// for a chat opened in place, the tile scope for a tile); the active profile stays.
vi.mock('../scope', async () => {
  const { atom } = await import('nanostores')
  const $messages = atom([])

  return {
    useComposerScope: () => ({ $messages, connectionId: mocks.owner.connectionId, profile: mocks.owner.profile }),
    useComposerSurfaceId: () => null
  }
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
  useVoiceLiveConversation: () => conversation
}))

vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: () => ({
    dictate: vi.fn(),
    voiceActivityState: { elapsedSeconds: 0, level: 0, status: 'idle' },
    voiceStatus: 'idle'
  })
}))

/** One gateway serving several profiles, each with its own plugin consent and xAI sign-in. */
interface FakeProfile {
  enabled: boolean
  account: null | string
}

let profiles: Record<string, FakeProfile> = {}
const pluginRequests: Array<{ path: string; profile: null | string; connectionId: null | string }> = []
const oauthReads: Array<null | string> = []
const secrets: string[] = []
const ios: LiveVoiceEngineIO[] = []
const unloads: Array<() => void> = []

const servedBy = (request: HermesApiRequest) => request.profile ?? 'default'

function installGateway() {
  return installRestBridge(request => {
    const profile = servedBy(request)

    if (request.path.startsWith('/api/plugins/hermes-xai/')) {
      pluginRequests.push({
        connectionId: request.connectionId ?? null,
        path: request.path,
        profile: request.profile ?? null
      })

      if (!profiles[profile]?.enabled) {
        // What the preload bridge rejects with for the backend's PLG-3 refusal.
        throw new Error('404: {"detail":"Plugin not found"}')
      }

      if (request.path.endsWith('/status')) {
        return { account: profiles[profile].account, ready: profiles[profile].account !== null }
      }

      return { secret: `ephemeral-${profile}` }
    }

    if (request.path === '/api/providers/oauth') {
      oauthReads.push(request.profile ?? null)

      return {
        providers: [{ id: 'xai-oauth', name: 'xAI', status: { logged_in: profiles[profile]?.account != null } }]
      }
    }

    throw new Error(`unexpected request ${request.path}`)
  })
}

/** A plugin written like the SDK guide's example: status and session through
 *  `ctx.rest`. `localStatus` answers status() without its backend, so only the
 *  session request meets the plugin gate. */
function loadEngine({ localStatus = false } = {}) {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext('hermes-xai', dispose => disposers.push(dispose))

  ctx.register({
    area: VOICE_ENGINE_AREA,
    data: {
      label: 'Grok Voice',
      signIn: { provider: 'xai-oauth' },
      start: async (io: LiveVoiceEngineIO) => {
        const { secret } = await ctx.rest<{ secret: string }>('/session', { method: 'POST' })
        secrets.push(secret)
        ios.push(io)

        return { stop: async () => undefined }
      },
      status: async () => {
        if (localStatus) {
          return { available: true }
        }

        const answer = await ctx.rest<{ ready: boolean }>('/status')

        return { available: answer.ready, reason: answer.ready ? null : 'not signed in' }
      }
    },
    id: 'grok'
  })
  unloads.push(() => disposers.splice(0).forEach(dispose => dispose()))
}

function mountComposer(sessionId = 'chat-7') {
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
      sessionId,
      target: 'main'
    })
  )
}

async function startCall(hook: ReturnType<typeof mountComposer>) {
  await act(async () => hook.result.current.startConversation())
  await act(async () => undefined)
  await act(async () => undefined)
}

beforeEach(() => {
  $voiceLiveStatus.set({ available: false, mode: 'hermes-xai:grok', model: '', reason: 'no key', voice: '' })
  installGateway()
})

afterEach(() => {
  cleanup()
  unloads.splice(0).forEach(unload => unload())
  ios.length = 0
  secrets.length = 0
  pluginRequests.length = 0
  oauthReads.length = 0
  mocks.chainedEnabled.length = 0
  mocks.owner = { connectionId: undefined, profile: undefined }
  vi.clearAllMocks()
  setApiRequestProfile(null)
  $voiceLiveStatus.set(null)
})

describe('extension voice engines across profiles', () => {
  it('a Bot chat uses the active profile for the voice and the Bot for the Hermes turn', async () => {
    profiles = { chief: { account: null, enabled: false }, default: { account: 'paul', enabled: true } }
    setApiRequestProfile('default')
    mocks.owner = { connectionId: undefined, profile: 'chief' }
    loadEngine()
    const hook = mountComposer('chief-chat')

    await startCall(hook)

    expect(secrets).toEqual(['ephemeral-default'])
    expect(pluginRequests.map(request => request.profile)).toEqual(['default', 'default'])
    expect(ios[0].sessionId).toBe('chief-chat')

    await act(async () => {
      void ios[0].delegate('what is on my calendar', 'User: what is on my calendar')
    })

    // The composer's own submit: the turn runs on the open (Bot) chat.
    expect(mocks.onSubmit).toHaveBeenCalledWith('what is on my calendar', {
      surface: 'voice-live',
      voiceContext: 'User: what is on my calendar'
    })
  })

  it('a Bot on another connection still reaches the voice plugin on the active connection', async () => {
    profiles = { default: { account: 'paul', enabled: true } }
    setApiRequestProfile('default')
    mocks.owner = { connectionId: 'remote-1', profile: 'chief' }
    loadEngine()

    await startCall(mountComposer('remote-chief-chat'))

    expect(secrets).toEqual(['ephemeral-default'])
    expect(pluginRequests.every(request => request.connectionId === null && request.profile === 'default')).toBe(true)
  })

  it('with the plugin enabled in every profile, each active profile mints with its own sign-in', async () => {
    profiles = { default: { account: 'paul', enabled: true }, secondary: { account: 'team', enabled: true } }
    loadEngine()

    setApiRequestProfile('default')
    const first = mountComposer('a')
    await startCall(first)
    await act(async () => first.result.current.endConversation())
    first.unmount()

    setApiRequestProfile('secondary')
    await startCall(mountComposer('b'))

    expect(secrets).toEqual(['ephemeral-default', 'ephemeral-secondary'])
  })

  it('with different accounts, the sign-in the menu shows is the account the call uses', async () => {
    profiles = {
      chief: { account: 'bot-account', enabled: true },
      default: { account: 'paul', enabled: true }
    }
    setApiRequestProfile('default')
    mocks.owner = { connectionId: undefined, profile: 'chief' }
    loadEngine()
    const engine = findLiveVoiceEngine('hermes-xai:grok')!

    // What VoiceEngineRows runs when the voice menu opens (no explicit profile).
    await liveVoiceEngineStatus(engine)
    await refreshVoiceSignIn('xai-oauth')
    await startCall(mountComposer('chief-chat'))

    expect(oauthReads).toEqual(['default'])
    expect(pluginRequests.map(request => request.profile)).toEqual(['default', 'default', 'default'])
    expect(secrets).toEqual(['ephemeral-default'])
  })

  it('a profile that did not enable the plugin gets a clear notice, not a raw 404', async () => {
    profiles = { default: { account: 'paul', enabled: true }, secondary: { account: null, enabled: false } }
    setApiRequestProfile('secondary')
    loadEngine()

    await startCall(mountComposer('b'))

    expect(ios).toHaveLength(0)
    expect(mocks.chainedEnabled.at(-1)).toBe(true)
    const notice = mocks.notify.mock.calls.map(([options]) => options).find(o => o.id === 'voice-live-unavailable')
    expect(notice?.message).toContain('hermes-xai')
    expect(notice?.message).toContain('secondary')
    expect(notice?.message).not.toMatch(/404|Plugin not found/)
  })

  it('a session request refused by the plugin gate gets the same notice and fallback', async () => {
    profiles = { secondary: { account: null, enabled: false } }
    setApiRequestProfile('secondary')
    loadEngine({ localStatus: true })

    await startCall(mountComposer('b'))

    expect(pluginRequests.map(request => request.path)).toEqual(['/api/plugins/hermes-xai/session'])
    expect(ios).toHaveLength(0)
    expect(mocks.chainedEnabled.at(-1)).toBe(true)
    // One warning, no "could not start" error on top of it.
    const warnings = mocks.notify.mock.calls.map(([options]) => options).filter(o => o.kind !== 'info')
    expect(warnings).toHaveLength(1)
    expect(warnings[0]).toMatchObject({ id: 'voice-live-unavailable', kind: 'warning' })
    expect(warnings[0].message).toContain('secondary')
    expect(mocks.notifyError).not.toHaveBeenCalled()
  })

  it('the voice menu row explains the same refusal', async () => {
    profiles = { secondary: { account: null, enabled: false } }
    setApiRequestProfile('secondary')
    loadEngine()

    const status = await liveVoiceEngineStatus(findLiveVoiceEngine('hermes-xai:grok')!)

    expect(status.available).toBe(false)
    expect(status.notEnabled).toBe(true)
  })
})
