// Check saved mute on a fresh module load, before any settings page mounts.
import { afterEach, expect, it, vi } from 'vitest'

import { getAudioContext } from '@/lib/audio-context'

vi.mock('@/lib/audio-context', () => ({ getAudioContext: vi.fn(() => null) }))

const key = 'hermes.desktop.hapticsMuted'

afterEach(() => {
  localStorage.removeItem(key)
})

it('loads saved mute before automatic playback and persists later choices', async () => {
  localStorage.setItem(key, 'true')
  const { $hapticsMuted, setHapticsMuted } = await import('./haptics')
  const { playCompletionSound } = await import('@/lib/completion-sound')
  expect($hapticsMuted.get()).toBe(true)
  playCompletionSound()
  expect(getAudioContext).not.toHaveBeenCalled()

  setHapticsMuted(false)
  expect(localStorage.getItem(key)).toBe('false')
  playCompletionSound()
  expect(getAudioContext).toHaveBeenCalledTimes(1)
  setHapticsMuted(true)
  expect(localStorage.getItem(key)).toBe('true')
})
