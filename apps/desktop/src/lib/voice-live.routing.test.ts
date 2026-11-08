// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useVoiceLiveConversation } from '@/app/chat/composer/hooks/use-voice-live-conversation'
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { voiceLiveBackend, type VoiceLiveHandlers, VoiceLiveSession } from './voice-live'

// The hook reads the OWNER-scoped status to pick the transport; tests swap the
// answer per case. VoiceLiveSession itself stays real (the OpenAI routing test
// below drives it), only the status probe is stubbed.
const statusControl = vi.hoisted(() => ({
  calls: [] as unknown[],
  impl: async (_owner?: unknown): Promise<unknown> => null,
  real: null as null | ((owner?: unknown) => Promise<unknown>)
}))

vi.mock('@/lib/voice-live', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  statusControl.real = actual.fetchVoiceLiveStatus as (owner?: unknown) => Promise<unknown>

  return {
    ...actual,
    fetchVoiceLiveStatus: (owner?: unknown) => {
      statusControl.calls.push(owner)

      return statusControl.impl(owner)
    }
  }
})

// The Azure relay transport is a stand-in here: the hook tests assert WHICH
// transport start() picked, with what owner — the relay's own wire protocol is
// pinned in voice-live-azure.test.ts.
const azureControl = vi.hoisted(() => ({
  constructed: [] as Array<{ handlers: VoiceLiveHandlers; owner: unknown }>,
  sessions: [] as Array<{ completeCommentary: ReturnType<typeof vi.fn>; speak: ReturnType<typeof vi.fn>; think: ReturnType<typeof vi.fn> }>
}))

vi.mock('@/lib/voice-live-azure', () => ({
  VoiceLiveAzureSession: class {
    completeCommentary = vi.fn()
    speak = vi.fn()
    think = vi.fn()
    close = vi.fn()
    instruct = vi.fn()
    setMuted = vi.fn()

    constructor(handlers: VoiceLiveHandlers, owner: unknown) {
      azureControl.constructed.push({ handlers, owner })
      azureControl.sessions.push({ completeCommentary: this.completeCommentary, speak: this.speak, think: this.think })
    }

    async start(): Promise<void> {}
  }
}))

// A GPT-Live session is created on the chat OWNER's (connection, profile) —
// the Bot that owns the chat — never on the window's active scope, mirroring
// the TTS legs (#117014): two Bots with different voices configured must get
// their own voices. The status probe follows the same rule so a Bot tile
// mounts the engine the Bot's own profile selected.

const handlers: VoiceLiveHandlers = {
  onClosed: () => undefined,
  onDelegation: () => undefined,
  onError: () => undefined
}

function installApi() {
  const api = vi.fn(async (_request: unknown) => ({
    ok: true,
    session: { id: 's1' },
    transport: { sdp: 'answer', type: 'answer' }
  }))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  return api
}

/** Enough of RTCPeerConnection/getUserMedia for `start()` to reach the POST. */
function installWebRTC() {
  const channel = { addEventListener: () => undefined, readyState: 'open' as const, send: () => undefined }

  const PeerConnection = class {
    addEventListener = () => undefined
    connectionState = 'new'
    createDataChannel = () => channel
    createOffer = async () => ({})
    iceGatheringState = 'complete'
    localDescription = { sdp: 'v=0\r\n' }
    setLocalDescription = async () => undefined
    setRemoteDescription = async () => undefined
  }

  Object.defineProperty(globalThis, 'RTCPeerConnection', { configurable: true, value: PeerConnection })
  Object.defineProperty(window.navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getAudioTracks: () => [] }) }
  })
}

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(globalThis, 'RTCPeerConnection')
  Reflect.deleteProperty(window.navigator, 'mediaDevices')
  statusControl.calls.length = 0
  statusControl.impl = async () => null
  azureControl.constructed.length = 0
  azureControl.sessions.length = 0
  cleanup()
})

describe('GPT-Live owner routing', () => {
  it('creates the session on the chat owner (connection, profile) ahead of the active scope', async () => {
    installWebRTC()
    const api = installApi()
    setApiRequestConnection('gw-active')
    setApiRequestProfile('research')

    const session = new VoiceLiveSession(handlers, { connectionId: 'gw-bots', profile: 'bot-adam' })
    await session.start([])

    expect(api).toHaveBeenCalledTimes(1)
    const request = api.mock.calls[0][0] as { connectionId?: string; path: string; priority?: string; profile?: string }

    expect(request.path).toBe('/api/audio/voice-live/session')
    expect(request.profile).toBe('bot-adam')
    expect(request.connectionId).toBe('gw-bots')
    expect(request.priority).toBe('foreground')

    // Ownerless chats keep the active scope verbatim (the old bare profileScoped()).
    const ownerless = new VoiceLiveSession(handlers, null)
    await ownerless.start([])
    const plain = api.mock.calls[1][0] as { connectionId?: string; path: string; priority?: string; profile?: string }
    expect(plain.path).toBe('/api/audio/voice-live/session')
    expect(plain.profile).toBe('research')
    expect(plain.connectionId).toBe('gw-active')
    expect(plain.priority).toBeUndefined()
  })
})

describe('voice-live status backend', () => {
  it('propagates backend azure and defaults to openai when the backend predates the field', async () => {
    const api = vi.fn(async (_request: unknown) => ({
      available: true,
      backend: 'azure',
      ok: true,
      mode: 'gpt-live',
      model: 'my-deployment',
      reason: null,
      voice: 'echo'
    }))

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

    const realFetch = statusControl.real as (owner?: unknown) => Promise<null | { backend: string }>
    const azure = await realFetch()

    expect(azure?.backend).toBe('azure')
    expect(voiceLiveBackend(azure as never)).toBe('azure')

    api.mockResolvedValueOnce({
      available: true,
      ok: true,
      mode: 'gpt-live',
      model: 'gpt-live-1',
      reason: null,
      voice: 'alloy'
    } as never)
    const legacy = await realFetch()

    expect(legacy?.backend).toBe('openai')
    expect(voiceLiveBackend(legacy as never)).toBe('openai')
    expect(voiceLiveBackend(null)).toBe('openai')
  })

  it('reads the status on the chat owner (connection, profile), falling back to the active scope', async () => {
    const api = vi.fn(async (_request: unknown) => ({
      available: true,
      backend: 'openai',
      ok: true,
      mode: 'gpt-live',
      model: 'gpt-live-1',
      reason: null,
      voice: 'alloy'
    }))

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    setApiRequestConnection('gw-active')
    setApiRequestProfile('research')

    const realFetch = statusControl.real as (owner?: unknown) => Promise<unknown>

    await realFetch({ connectionId: 'gw-bots', profile: 'bot-adam' })

    const owned = api.mock.calls[0][0] as { connectionId?: string; path: string; priority?: string; profile?: string }

    expect(owned.path).toBe('/api/audio/voice-live/status')
    expect(owned.connectionId).toBe('gw-bots')
    expect(owned.profile).toBe('bot-adam')
    expect(owned.priority).toBe('foreground')

    await realFetch()

    const ambient = api.mock.calls[1][0] as { connectionId?: string; priority?: string; profile?: string }

    expect(ambient.profile).toBe('research')
    expect(ambient.connectionId).toBe('gw-active')
    expect(ambient.priority).toBeUndefined()
  })
})

describe('useVoiceLiveConversation — backend selection at start', () => {
  const azureStatus = {
    available: true,
    backend: 'azure' as const,
    mode: 'gpt-live' as const,
    model: 'my-deployment',
    reason: null,
    voice: 'echo'
  }

  function mountLive(wrapper?: ({ children }: { children: ReactNode }) => ReactNode, overrides: Record<string, unknown> = {}) {
    return renderHook(
      (props: Record<string, unknown>) =>
        useVoiceLiveConversation({
          busy: false,
          consumePendingResponse: vi.fn(),
          enabled: true,
          onSubmit: vi.fn(),
          pendingResponse: () => null,
          seedHistory: () => [],
          ...props,
          ...overrides
        } as Parameters<typeof useVoiceLiveConversation>[0]),
      wrapper ? { wrapper } : undefined
    )
  }

  it('starts the Azure relay session — no RTCPeerConnection, no SDP POST — when the owner-scoped status says azure', async () => {
    const api = vi.fn(async (_request: unknown) => ({ ok: true }))

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    statusControl.impl = async () => azureStatus

    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(
        ComposerScopeProvider,
        { value: { ...MAIN_COMPOSER_SCOPE, connectionId: 'gw-bots', profile: 'bot-adam', target: 'tile:bot' } },
        children
      )

    const hook = mountLive(wrapper)

    await act(async () => {
      await hook.result.current.start()
    })

    // The status probe rode the owner scope, and the owner went to the transport.
    expect(statusControl.calls.at(-1)).toEqual({ connectionId: 'gw-bots', profile: 'bot-adam' })
    expect(azureControl.constructed).toHaveLength(1)
    expect(azureControl.constructed[0].owner).toEqual({ connectionId: 'gw-bots', profile: 'bot-adam' })
    // The OpenAI path would have needed RTCPeerConnection (absent here) and the
    // SDP POST (api never called).
    expect(api).not.toHaveBeenCalled()
  })

  it('keeps the OpenAI WebRTC session when the status is openai or unreadable', async () => {
    installWebRTC()
    const api = installApi()
    statusControl.impl = async () => null // older backend / probe failure

    const hook = mountLive()

    await act(async () => {
      await hook.result.current.start()
    })

    expect(azureControl.constructed).toHaveLength(0)
    expect(api).toHaveBeenCalledTimes(1)
    expect((api.mock.calls[0][0] as { path: string }).path).toBe('/api/audio/voice-live/session')
  })

  it('emits commentary.complete after the full response and on the no-response path', async () => {
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api: vi.fn() } })
    statusControl.impl = async () => azureStatus

    let response: null | { id: string; pending: boolean; text: string } = null

    const hook = renderHook(
      (props: { busy: boolean }) =>
        useVoiceLiveConversation({
          busy: props.busy,
          consumePendingResponse: vi.fn(),
          enabled: true,
          onSubmit: vi.fn(),
          pendingResponse: () => response,
          seedHistory: () => []
        }),
      { initialProps: { busy: false } }
    )

    await act(async () => {
      await hook.result.current.start()
    })

    const { handlers } = azureControl.constructed[0]
    const session = azureControl.sessions[0]

    act(() => {
      handlers.onDelegation('call_1', [{ endMs: 1, speaker: 'user', startMs: 0, text: 'run the tests' }])
    })

    // Full response lands, turn not busy: speak the tail, then seal the commentary.
    response = { id: 'r1', pending: false, text: 'Tests passed.' }
    await act(async () => {
      hook.rerender({ busy: true })
    })
    await act(async () => {
      hook.rerender({ busy: false })
      await new Promise(resolve => setTimeout(resolve, 300))
    })

    expect(session.speak).toHaveBeenCalledWith('call_1', 'Tests passed.')
    expect(session.completeCommentary).toHaveBeenCalledWith('call_1')

    // Second delegation, no speakable reply: the settle path still seals it.
    response = null
    act(() => {
      handlers.onDelegation('call_2', [{ endMs: 2, speaker: 'user', startMs: 1, text: 'list files' }])
    })
    await act(async () => {
      hook.rerender({ busy: true })
    })
    await act(async () => {
      hook.rerender({ busy: false })
      await new Promise(resolve => setTimeout(resolve, 300))
    })

    expect(session.think).toHaveBeenCalledWith('call_2', expect.stringContaining('without a spoken result'))
    expect(session.completeCommentary).toHaveBeenCalledWith('call_2')
  })

  it('seals the Azure function call when the Hermes submission itself fails', async () => {
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api: vi.fn() } })
    statusControl.impl = async () => azureStatus
    const hook = mountLive(undefined, { onSubmit: vi.fn(async () => { throw new Error('gateway offline') }) })

    await act(async () => {
      await hook.result.current.start()
    })

    const { handlers } = azureControl.constructed[0]
    const session = azureControl.sessions[0]
    act(() => {
      handlers.onDelegation('call_failed', [{ endMs: 1, speaker: 'user', startMs: 0, text: 'run the tests' }])
    })
    await act(async () => {
      await Promise.resolve()
    })

    expect(session.speak).toHaveBeenCalledWith('call_failed', expect.stringContaining('could not reach Hermes'))
    expect(session.completeCommentary).toHaveBeenCalledWith('call_failed')
  })
})
