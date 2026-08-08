import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  checkHermesUpdate,
  getActionStatus,
  getElevenLabsVoices,
  getMemoryProviderConfig,
  getStatus,
  mapGatewayProfileParams,
  restartGateway,
  saveMemoryProviderConfig,
  setApiRequestProfile,
  speakText,
  transcribeAudio,
  updateHermes
} from './hermes'

describe('remote gateway profile aliases', () => {
  it('removes the Desktop alias when the remote target is default', () => {
    expect(mapGatewayProfileParams({ profile: 'work', session_id: 's1' }, 'work', 'default')).toEqual({
      session_id: 's1'
    })
  })

  it('injects a named remote profile when a request omits the Desktop alias', () => {
    expect(mapGatewayProfileParams({ key: 'model' }, 'research', 'writer')).toEqual({
      key: 'model',
      profile: 'writer'
    })
  })

  it('preserves unrelated explicit profile scopes', () => {
    expect(mapGatewayProfileParams({ profile: 'reviewer' }, 'research', 'writer')).toEqual({
      profile: 'reviewer'
    })
  })
})

// Contract: every backend-targeted action helper must carry the active gateway
// profile, so a multi-profile / global-remote user's restart, status poll, and
// update hit the backend they're actually on — not the primary/default. The
// System-panel "restart does nothing" bug was these helpers dropping it.
describe('backend action helpers are profile-scoped', () => {
  const api = vi.fn(async (_req: { path: string; profile?: string }) => ({}) as never)

  beforeEach(() => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api }
    api.mockClear()
  })

  afterEach(() => {
    setApiRequestProfile(null)
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  const lastProfile = () => api.mock.calls.at(-1)?.[0].profile

  it('omits profile when none is active (single-profile users unaffected)', () => {
    void getStatus()
    expect(lastProfile()).toBeUndefined()
  })

  it('forwards the active profile to memory provider config calls', () => {
    setApiRequestProfile('coder')

    void getMemoryProviderConfig('honcho')
    void saveMemoryProviderConfig('honcho', { workspace: 'w' })

    for (const call of api.mock.calls) {
      expect(call[0].profile).toBe('coder')
    }
  })

  it('forwards the active profile to every backend action', () => {
    setApiRequestProfile('coder')

    void getStatus()
    void restartGateway()
    void updateHermes()
    void checkHermesUpdate()
    void getActionStatus('gateway-restart')

    for (const call of api.mock.calls) {
      expect(call[0].profile).toBe('coder')
    }
  })

  // Audio endpoints (transcribe / speak / voices) write to the active
  // profile's config in the settings UI but historically called the backend
  // without a profile scope, so playback used the default profile's TTS/voice
  // config instead of the active one (#53441).
  it('forwards the active profile to audio endpoints', () => {
    setApiRequestProfile('jarvis')

    void transcribeAudio('data:audio/webm;base64,AAAA', 'audio/webm')
    void speakText('hello')
    void getElevenLabsVoices()

    expect(api.mock.calls).toHaveLength(3)

    for (const call of api.mock.calls) {
      expect(call[0].profile).toBe('jarvis')
    }
  })
})
