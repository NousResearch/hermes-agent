import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { activate, pin, requestStart } = vi.hoisted(() => ({
  activate: vi.fn(),
  pin: vi.fn(),
  requestStart: vi.fn()
}))

vi.mock('@/store/composer', () => ({ requestVoiceConversationStart: requestStart }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  activateOnCurrentSource: activate,
  normalizeProfileKey: (value: string | null) => value?.trim() || 'default',
  pinNewChatProfile: pin
}))

import { $activeGatewayProfile } from '@/store/profile'

import { startWakeVoiceConversation } from './wake-voice-start'

beforeEach(() => {
  vi.clearAllMocks()
  $activeGatewayProfile.set('default')
})

describe('wake-word voice activation', () => {
  it.each([true, false])('waits for the detected profile before voice (fresh=%s)', async fresh => {
    let complete!: () => void
    activate.mockImplementation(() => new Promise<void>(resolve => (complete = resolve)))
    const startFresh = vi.fn()
    const task = startWakeVoiceConversation({ profile: 'voice', start_new_session: fresh }, startFresh)

    expect(activate).toHaveBeenCalledWith('voice')
    expect(requestStart).not.toHaveBeenCalled()
    expect(startFresh).not.toHaveBeenCalled()
    $activeGatewayProfile.set('voice')
    complete()
    await task

    expect(startFresh).toHaveBeenCalledTimes(fresh ? 1 : 0)
    expect(pin).toHaveBeenCalledTimes(fresh ? 1 : 0)
    expect(requestStart).toHaveBeenCalledOnce()
  })

  it.each(['failure', 'superseded', 'default'])('honors activation ownership: %s', async outcome => {
    const startFresh = vi.fn()

    if (outcome === 'failure') {
      activate.mockRejectedValue(new Error('OAuth unavailable'))
      await expect(startWakeVoiceConversation({ profile: 'voice' }, startFresh)).rejects.toThrow('OAuth unavailable')
      expect(requestStart).not.toHaveBeenCalled()
      expect(startFresh).not.toHaveBeenCalled()

      return
    }

    let complete!: () => void
    activate.mockImplementationOnce(() => new Promise<void>(resolve => (complete = resolve)))
    const first = startWakeVoiceConversation(undefined, startFresh)

    if (outcome === 'superseded') {
      activate.mockResolvedValue(undefined)
      await startWakeVoiceConversation(undefined, startFresh)
    }

    complete()
    await first
    expect(requestStart).toHaveBeenCalledOnce()
    expect(startFresh).toHaveBeenCalledOnce()
    expect(pin).toHaveBeenCalledWith('default')
  })
})
