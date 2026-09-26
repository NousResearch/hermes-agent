// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from 'vitest'

import {
  selectedVoiceChatMode,
  setVoiceChatMode,
  HERMES_DESKTOP_GEMINI_LIVE_STORAGE_PREFIX
} from './voice-live'
import { activeGateway } from '@/store/gateway'
import { fetchVoiceLiveStatus, type VoiceLiveStatus } from '@/lib/voice-live'
import { profileScopeKey } from '@/hermes'

vi.mock('@/store/gateway', () => ({
  activeGateway: vi.fn()
}))

vi.mock('@/lib/voice-live', () => ({
  fetchVoiceLiveStatus: vi.fn().mockResolvedValue({
    mode: 'chained',
    available: true,
    reason: null,
    model: '',
    voice: ''
  })
}))

vi.mock('@/hermes', () => ({
  profileScopeKey: vi.fn().mockReturnValue('test-profile'),
  deleteEnvVar: vi.fn(),
  revealEnvVar: vi.fn(),
  setEnvVar: vi.fn()
}))

vi.mock('@/lib/gemini-live', () => ({
  resolveGeminiLiveApiKey: vi.fn().mockResolvedValue('test-key')
}))

function mockStatus(mode: 'chained' | 'gpt-live'): VoiceLiveStatus {
  return {
    mode,
    available: true,
    reason: null,
    model: 'gpt-4o-realtime',
    voice: 'marin'
  }
}

describe('voice-live store', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('selectedVoiceChatMode returns gemini-live when profile-scoped key is set', () => {
    localStorage.setItem(`${HERMES_DESKTOP_GEMINI_LIVE_STORAGE_PREFIX}test-profile`, '1')
    expect(selectedVoiceChatMode(mockStatus('chained'))).toBe('gemini-live')
  })

  it('selectedVoiceChatMode falls back to backend status mode when scoped key is absent', () => {
    expect(selectedVoiceChatMode(mockStatus('gpt-live'))).toBe('gpt-live')
    expect(selectedVoiceChatMode(mockStatus('chained'))).toBe('chained')
    expect(selectedVoiceChatMode(null)).toBe('chained')
  })

  it('setVoiceChatMode("gemini-live") stores scoped preference without calling gateway', async () => {
    const mockGateway = { request: vi.fn() }
    vi.mocked(activeGateway).mockReturnValue(mockGateway as any)

    await setVoiceChatMode('gemini-live')

    expect(localStorage.getItem(`${HERMES_DESKTOP_GEMINI_LIVE_STORAGE_PREFIX}test-profile`)).toBe('1')
    expect(mockGateway.request).not.toHaveBeenCalled()
  })

  it('setVoiceChatMode throws error when gateway is not connected for backend modes', async () => {
    vi.mocked(activeGateway).mockReturnValue(null)

    await expect(setVoiceChatMode('gpt-live')).rejects.toThrow('gateway not connected')
    await expect(setVoiceChatMode('chained')).rejects.toThrow('gateway not connected')
  })

  it('setVoiceChatMode propagates gateway error when config.set fails', async () => {
    const mockGateway = {
      request: vi.fn().mockRejectedValue(new Error('config rejection'))
    }
    vi.mocked(activeGateway).mockReturnValue(mockGateway as any)

    await expect(setVoiceChatMode('gpt-live')).rejects.toThrow('config rejection')
  })

  it('setVoiceChatMode("chained") removes scoped gemini-live key upon success', async () => {
    localStorage.setItem(`${HERMES_DESKTOP_GEMINI_LIVE_STORAGE_PREFIX}test-profile`, '1')

    const mockGateway = { request: vi.fn().mockResolvedValue({ ok: true }) }
    vi.mocked(activeGateway).mockReturnValue(mockGateway as any)

    await setVoiceChatMode('chained')

    expect(mockGateway.request).toHaveBeenCalledWith('config.set', {
      key: 'voice.voice_chat_mode',
      value: 'chained'
    })
    expect(localStorage.getItem(`${HERMES_DESKTOP_GEMINI_LIVE_STORAGE_PREFIX}test-profile`)).toBeNull()
  })
})
