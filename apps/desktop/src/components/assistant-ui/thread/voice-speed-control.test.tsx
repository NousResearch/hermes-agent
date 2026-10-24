// The speed control next to read-aloud: opens the preset menu, persists a
// choice, reflects the active rate on the trigger, and resets state between
// tests through the same store seam the rest of the suite uses.

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { setVoicePlaybackSpeed } from '@/store/voice-playback-speed'

import { VoiceSpeedControl } from './voice-speed-control'

const STORAGE_KEY = 'hermes.desktop.voicePlaybackSpeed'

// Radix menus use pointer capture and scrollIntoView; jsdom has neither.
beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.scrollIntoView ??= () => undefined
})

afterEach(() => {
  cleanup()
  window.localStorage.removeItem(STORAGE_KEY)
})

const openMenu = async () => {
  // Radix's dropdown trigger opens on pointerdown; fire the full mouse
  // sequence a real click produces (project-menu.test.tsx, #67500).
  const trigger = screen.getByTestId('voice-speed-control')

  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)

  await screen.findByTestId('voice-speed-1.5')
}

describe('VoiceSpeedControl', () => {
  it('renders 1x with no badge while the default rate is active', () => {
    render(<VoiceSpeedControl />)

    const trigger = screen.getByTestId('voice-speed-control')

    expect(trigger.textContent).not.toContain('×')
  })

  it('persists a preset choice and shows the rate on the trigger', async () => {
    render(<VoiceSpeedControl />)
    await openMenu()

    fireEvent.click(screen.getByTestId('voice-speed-1.5'))

    await waitFor(() => expect(window.localStorage.getItem(STORAGE_KEY)).toBe('1.5'))
    expect(screen.getByTestId('voice-speed-control').textContent).toContain('1.5×')
  })

  it('marks the default row and returns the badge to hidden at 1x', async () => {
    render(<VoiceSpeedControl />)
    await openMenu()

    const normalRow = screen.getByTestId('voice-speed-1')

    expect(normalRow.textContent).toContain('normal')

    // Back at the default via the store seam: the badge hides and the record
    // is removed. (Menu re-open is guarded by Radix's interaction-mode timer;
    // the row-selection path is covered by the test above.)
    fireEvent.click(screen.getByTestId('voice-speed-2'))
    await waitFor(() => expect(window.localStorage.getItem(STORAGE_KEY)).toBe('2'))

    act(() => {
      setVoicePlaybackSpeed(1)
    })
    await waitFor(() => expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull())
    expect(screen.getByTestId('voice-speed-control').textContent).not.toContain('×')
  })
})
