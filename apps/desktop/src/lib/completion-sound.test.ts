// Check completion cue ownership races without opening a real audio device.
import { afterEach, expect, it, vi } from 'vitest'

import { getAudioContext } from '@/lib/audio-context'
import {
  $completionSoundVariantId,
  DEFAULT_COMPLETION_SOUND_VARIANT_ID,
  NO_COMPLETION_SOUND_VARIANT_ID
} from '@/store/completion-sound'
import { $hapticsMuted, setHapticsMuted } from '@/store/haptics'

import { playCompletionSound, previewCompletionSound } from './completion-sound'

vi.mock('@/lib/audio-context', () => ({ getAudioContext: vi.fn(() => null) }))

const original = window.hermesDesktop

afterEach(() => {
  window.hermesDesktop = original
  setHapticsMuted(false)
  $completionSoundVariantId.set(DEFAULT_COMPLETION_SOUND_VARIANT_ID)
  vi.clearAllMocks()
})

it('rechecks mute after ownership resolves, including a failed claim and a later retry', async () => {
  for (const fail of [false, true]) {
    let resolve!: (owns: boolean) => void
    let reject!: (error: Error) => void

    const pending = new Promise<boolean>((yes, no) => {
      resolve = yes
      reject = no
    })

    const claim = vi.fn(() => pending)
    window.hermesDesktop = { ...original, claimAmbientCue: claim }
    setHapticsMuted(false)
    playCompletionSound('session')
    expect(claim).toHaveBeenCalledWith('sound:session')
    setHapticsMuted(true)

    if (fail) {
      reject(new Error('Ownership bridge unavailable'))
    } else {
      resolve(true)
    }

    // Drain the real ownership helper and its completion callback.
    await pending.catch(() => undefined)
    await Promise.resolve()
    expect(getAudioContext).not.toHaveBeenCalled()
    playCompletionSound('session')
    expect(claim).toHaveBeenCalledTimes(1)
  }

  setHapticsMuted(false)
  window.hermesDesktop = { ...original, claimAmbientCue: vi.fn(async () => true) }
  playCompletionSound('session')
  await Promise.resolve()
  await Promise.resolve()
  expect(getAudioContext).toHaveBeenCalledTimes(1)
  expect($hapticsMuted.get()).toBe(false)
})

it('stays silent on None and never claims the cue from an audible peer', () => {
  const claim = vi.fn(async () => true)
  window.hermesDesktop = { ...original, claimAmbientCue: claim }
  $completionSoundVariantId.set(NO_COMPLETION_SOUND_VARIANT_ID)

  playCompletionSound('session')
  previewCompletionSound()

  expect(claim).not.toHaveBeenCalled()
  expect(getAudioContext).not.toHaveBeenCalled()
})
