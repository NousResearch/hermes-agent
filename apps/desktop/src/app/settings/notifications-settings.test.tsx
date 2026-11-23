// Check that the visible sound control silences real cue callers without audio output.
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { getAudioContext } from '@/lib/audio-context'
import { playCompletionSound, previewCompletionSound } from '@/lib/completion-sound'
import { startThinkingSound, stopThinkingSound } from '@/lib/thinking-sound'
import { playWakeSound } from '@/lib/wake-sound'
import {
  $completionSoundVolume,
  DEFAULT_COMPLETION_SOUND_VOLUME,
  setCompletionSoundVariantId,
  setCompletionSoundVolume
} from '@/store/completion-sound'
import { $hapticsMuted, setHapticsMuted } from '@/store/haptics'

import { NotificationsSettings } from './notifications-settings'

vi.mock('@/lib/audio-context', () => ({ getAudioContext: vi.fn(() => null) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))
// Keep native alerts disabled while exercising the separate real sound owner.
vi.mock('@/store/native-notifications', async () => {
  const { atom } = await import('nanostores')

  return {
    $nativeNotifyPrefs: atom({ enabled: false, kinds: {} }),
    NATIVE_NOTIFICATION_KINDS: [],
    sendTestNativeNotification: vi.fn(),
    setNativeNotifyEnabled: vi.fn(),
    setNativeNotifyKind: vi.fn()
  }
})

beforeEach(() => {
  vi.useFakeTimers()
  setHapticsMuted(false)
  setCompletionSoundVariantId(1)
  setCompletionSoundVolume(DEFAULT_COMPLETION_SOUND_VOLUME)
  vi.clearAllMocks()
})

afterEach(() => {
  cleanup()
  stopThinkingSound()
  setHapticsMuted(false)
  vi.useRealTimers()
})

it('offers a persisted mute that silences automatic completion, wake and thinking cues with native alerts off', () => {
  render(
    <MemoryRouter>
      <NotificationsSettings subpage="sounds" />
    </MemoryRouter>
  )
  const toggle = screen.getByRole('switch', { name: 'Mute interface sounds' })

  // Native alert preferences alone do not own the foreground completion cue.
  playCompletionSound()
  expect(getAudioContext).toHaveBeenCalledTimes(1)
  vi.mocked(getAudioContext).mockClear()
  fireEvent.click(toggle)
  expect($hapticsMuted.get()).toBe(true)
  expect(localStorage.getItem('hermes.desktop.hapticsMuted')).toBe('true')

  playCompletionSound()
  playWakeSound()
  startThinkingSound()
  act(() => vi.advanceTimersByTime(2_000))
  expect(getAudioContext).not.toHaveBeenCalled()

  // An explicit audition remains available without enabling automatic cues.
  previewCompletionSound()
  expect(getAudioContext).toHaveBeenCalledTimes(1)
  vi.mocked(getAudioContext).mockClear()

  cleanup()
  render(
    <MemoryRouter>
      <NotificationsSettings subpage="sounds" />
    </MemoryRouter>
  )
  const restoredToggle = screen.getByRole('switch', { name: 'Mute interface sounds' })
  expect(restoredToggle.getAttribute('aria-checked')).toBe('true')
  fireEvent.click(restoredToggle)
  expect(localStorage.getItem('hermes.desktop.hapticsMuted')).toBe('false')
  previewCompletionSound()
  expect(getAudioContext).toHaveBeenCalledTimes(1)
})

it('silences cues in this window when a peer window saves mute', () => {
  localStorage.setItem('hermes.desktop.hapticsMuted', 'true')
  window.dispatchEvent(new StorageEvent('storage', { key: 'hermes.desktop.hapticsMuted' }))
  playCompletionSound()
  playWakeSound()
  startThinkingSound()
  act(() => vi.advanceTimersByTime(2_000))
  expect($hapticsMuted.get()).toBe(true)
  expect(getAudioContext).not.toHaveBeenCalled()

  window.dispatchEvent(new StorageEvent('storage', { key: 'unrelated' }))
  expect($hapticsMuted.get()).toBe(true)
  localStorage.setItem('hermes.desktop.hapticsMuted', 'false')
  window.dispatchEvent(new StorageEvent('storage', { key: 'hermes.desktop.hapticsMuted' }))
  expect($hapticsMuted.get()).toBe(false)
  playCompletionSound()
  expect(getAudioContext).toHaveBeenCalledTimes(1)

  setHapticsMuted(true)
  localStorage.removeItem('hermes.desktop.hapticsMuted')
  window.dispatchEvent(new StorageEvent('storage', { key: 'hermes.desktop.hapticsMuted' }))
  expect($hapticsMuted.get()).toBe(false)
})

it('offers a completion volume slider that persists, quantizes and silences at its floor', () => {
  render(
    <MemoryRouter>
      <NotificationsSettings subpage="sounds" />
    </MemoryRouter>
  )
  const slider = screen.getByRole('slider', { name: 'Completion Sound — Volume' }) as HTMLInputElement

  expect(slider.value).toBe('0.48')
  expect(screen.getByText('0.48×')).toBeTruthy()

  fireEvent.change(slider, { target: { value: '1.35' } })
  expect($completionSoundVolume.get()).toBe(1.35)
  expect(localStorage.getItem('hermes.desktop.completionSoundVolume')).toBe('1.35')
  expect(screen.getByText('1.35×')).toBeTruthy()

  // Quantized where it is applied, so the readout is the gain the cue plays at.
  fireEvent.change(slider, { target: { value: '2.4376' } })
  expect($completionSoundVolume.get()).toBe(2.44)

  // At the floor the cue never reaches the audio context — the same silence
  // contract the mute toggle has.
  fireEvent.change(slider, { target: { value: '0' } })
  expect(screen.getByText('0.00×')).toBeTruthy()
  vi.mocked(getAudioContext).mockClear()
  playCompletionSound()
  expect(getAudioContext).not.toHaveBeenCalled()

  // "None" schedules nothing, so the slider has nothing to act on.
  act(() => setCompletionSoundVariantId(0))
  expect(slider.disabled).toBe(true)
})
