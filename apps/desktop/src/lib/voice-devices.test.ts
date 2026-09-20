import { afterEach, describe, expect, it, vi } from 'vitest'

import { HERMES_CONFIG_KEY } from '@/app/hooks/use-config-record'
import { queryClient } from '@/lib/query-client'

import {
  applyAudioOutputDevice,
  audioInputConstraints,
  deviceLabel,
  isMissingDeviceError,
  voiceInputDeviceId,
  voiceOutputDeviceId
} from './voice-devices'

function seedConfig(record: unknown) {
  queryClient.setQueryData(HERMES_CONFIG_KEY, record)
}

afterEach(() => {
  queryClient.clear()
  vi.restoreAllMocks()
})

describe('voice device config readers', () => {
  it('reads the configured microphone and speaker ids', () => {
    seedConfig({ voice: { mic_device_id: 'mic-42', speaker_device_id: 'spk-7' } })

    expect(voiceInputDeviceId()).toBe('mic-42')
    expect(voiceOutputDeviceId()).toBe('spk-7')
  })

  it('treats empty, missing and malformed values as "system default"', () => {
    seedConfig({ voice: { mic_device_id: '  ' } })
    expect(voiceInputDeviceId()).toBe('')
    expect(voiceOutputDeviceId()).toBe('')

    seedConfig({ voice: null })
    expect(voiceInputDeviceId()).toBe('')

    seedConfig(undefined)
    expect(voiceInputDeviceId()).toBe('')
    expect(voiceOutputDeviceId()).toBe('')
  })
})

describe('getUserMedia constraints', () => {
  it('pins the configured microphone exactly, keeping the caller constraints', () => {
    seedConfig({ voice: { mic_device_id: 'mic-42' } })

    const constraints = audioInputConstraints({ echoCancellation: true, noiseSuppression: true })

    expect(constraints).toEqual({
      echoCancellation: true,
      noiseSuppression: true,
      deviceId: { exact: 'mic-42' }
    })
  })

  it('requests no device when none is configured', () => {
    seedConfig({ voice: {} })

    expect(audioInputConstraints({ echoCancellation: true })).toEqual({ echoCancellation: true })
  })

  it('recognises the "device is gone" errors only', () => {
    expect(isMissingDeviceError({ name: 'OverconstrainedError' })).toBe(true)
    expect(isMissingDeviceError({ name: 'NotFoundError' })).toBe(true)
    expect(isMissingDeviceError({ name: 'NotAllowedError' })).toBe(false)
    expect(isMissingDeviceError(null)).toBe(false)
  })
})

describe('output device routing', () => {
  function fakeAudio(sink?: (id: string) => Promise<void>) {
    return (sink ? { setSinkId: sink } : {}) as unknown as HTMLAudioElement
  }

  it('routes playback to the configured speaker', async () => {
    seedConfig({ voice: { speaker_device_id: 'spk-7' } })
    const setSinkId = vi.fn().mockResolvedValue(undefined)

    await applyAudioOutputDevice(fakeAudio(setSinkId))

    expect(setSinkId).toHaveBeenCalledWith('spk-7')
  })

  it('leaves the default alone when nothing is configured', async () => {
    seedConfig({ voice: {} })
    const setSinkId = vi.fn()

    await applyAudioOutputDevice(fakeAudio(setSinkId))

    expect(setSinkId).not.toHaveBeenCalled()
  })

  it('does not throw when the element or browser cannot switch sink', async () => {
    seedConfig({ voice: { speaker_device_id: 'spk-7' } })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})

    await applyAudioOutputDevice(fakeAudio())                                    // unsupported
    await applyAudioOutputDevice(fakeAudio(vi.fn().mockRejectedValue(new Error('nope'))))

    expect(warn).toHaveBeenCalledTimes(1)
  })
})

describe('device labels', () => {
  it('falls back to a positional name when permissions hide labels', () => {
    expect(deviceLabel({ label: '' } as MediaDeviceInfo, 1, 'input')).toBe('Microphone 2')
    expect(deviceLabel({ label: 'AirPods' } as MediaDeviceInfo, 0, 'output')).toBe('AirPods')
  })
})
