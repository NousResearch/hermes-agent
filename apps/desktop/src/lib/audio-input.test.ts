import { describe, expect, it } from 'vitest'

import {
  audioCaptureConstraints,
  displayAudioInputLabel,
  isVirtualAudioLabel,
  SYSTEM_DEFAULT_AUDIO_INPUT
} from './audio-input'

describe('isVirtualAudioLabel', () => {
  it('flags mixer and loopback labels', () => {
    expect(isVirtualAudioLabel('RØDE Connect System (Virtual)')).toBe(true)
    expect(isVirtualAudioLabel('Camo Microphone (Virtual)')).toBe(true)
    expect(isVirtualAudioLabel('BlackHole 2ch')).toBe(true)
  })

  it('leaves real hardware alone', () => {
    expect(isVirtualAudioLabel('MacBook Pro Microphone (Built-in)')).toBe(false)
    expect(isVirtualAudioLabel('Alice’s iPhone Microphone')).toBe(false)
  })
})

describe('audioCaptureConstraints', () => {
  it('omits deviceId for the system default', () => {
    expect(audioCaptureConstraints(null).deviceId).toBeUndefined()
    expect(audioCaptureConstraints(SYSTEM_DEFAULT_AUDIO_INPUT).deviceId).toBeUndefined()
  })

  it('pins a chosen device exactly', () => {
    expect(audioCaptureConstraints('abc')).toEqual({
      deviceId: { exact: 'abc' },
      echoCancellation: true,
      noiseSuppression: true
    })
  })

  it('merges caller extras (wake capture) without unpinning the device', () => {
    expect(audioCaptureConstraints('abc', { autoGainControl: true, channelCount: 1 })).toEqual({
      autoGainControl: true,
      channelCount: 1,
      deviceId: { exact: 'abc' },
      echoCancellation: true,
      noiseSuppression: true
    })
    // An extras deviceId must not override the explicit pin.
    expect(audioCaptureConstraints('abc', { deviceId: 'other' }).deviceId).toEqual({ exact: 'abc' })
  })
})

describe('displayAudioInputLabel', () => {
  const devices = [
    { deviceId: 'default', groupId: 'g0', label: 'Default - RØDE Connect System (Virtual)' },
    { deviceId: 'built-in', groupId: 'g1', label: 'MacBook Pro Microphone (Built-in)' }
  ]

  it('uses the OS default row when no pin is stored', () => {
    expect(displayAudioInputLabel(null, devices, 'System default')).toBe(
      'Default - RØDE Connect System (Virtual)'
    )
  })

  it('uses the pinned device label', () => {
    expect(displayAudioInputLabel('built-in', devices, 'System default')).toBe(
      'MacBook Pro Microphone (Built-in)'
    )
  })
})
