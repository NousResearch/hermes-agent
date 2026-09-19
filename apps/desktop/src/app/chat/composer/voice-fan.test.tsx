/**
 * The fan's wake-word disc must surface the same failure/arming reason the
 * old WakeWordButton tooltip showed (missing STT/TTS, no mic permission,
 * still installing, etc.) — wake-word.ts populates notice on exactly those
 * paths. Without this, clicking the ear in the default (unfolded) layout
 * gives no visible explanation when the backend can't start.
 */
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { $wakeWord } from '@/store/wake-word'

import type { ChatBarState, VoiceStatus } from './types'
import { VoiceFan } from './voice-fan'

vi.mock('./control-classes', () => ({ ACTIVE_ICON_BTN: '', GHOST_ICON_BTN: '' }))

const STATE: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { active: false, enabled: true }
}

function renderFan(voiceStatus: VoiceStatus = 'idle') {
  render(
    <I18nProvider configClient={null}>
      <VoiceFan
        autoSpeak={false}
        disabled={false}
        onDictate={vi.fn()}
        onToggleAutoSpeak={vi.fn()}
        state={STATE}
        voiceStatus={voiceStatus}
      />
    </I18nProvider>
  )

  // Discs render in a portal only once the fan opens (hover/focus on the
  // hub) — the wake disc doesn't exist in the DOM at all until then.
  const hub = screen.getByRole('button', { name: 'Voice dictation' })
  fireEvent.pointerEnter(hub)
}

afterEach(() => {
  cleanup()
  $wakeWord.set({ available: true, enabled: true, listening: false, notice: '', pending: false, phrase: '' })
})

describe('VoiceFan wake-word disc', () => {
  it('has a plain label when there is no notice', () => {
    $wakeWord.set({ available: true, enabled: true, listening: false, notice: '', pending: false, phrase: '' })
    renderFan()

    const disc = screen.getByRole('button', { name: /wake word/i })

    expect(disc.getAttribute('aria-label')).not.toMatch(/—/)
  })

  it('surfaces the failure/arming notice in the disc label', () => {
    $wakeWord.set({
      available: true,
      enabled: true,
      listening: false,
      notice: 'arming — first use may take a minute while the engine installs',
      pending: false,
      phrase: ''
    })
    renderFan()

    const disc = screen.getByRole('button', { name: /wake word.*arming/i })

    expect(disc.getAttribute('aria-label')).toContain('arming')
  })

  // The initial-notice test above alone can pass even if `wake.notice` were
  // dropped from the items useMemo's dependency array, since the label is
  // still correct on first render regardless. The real regression path is
  // the notice changing AFTER mount (a click fails, the store updates,
  // wake-word.ts:69/224/239/300/324) — this proves the disc's already-
  // mounted label actually reacts to that, not just the initial paint.
  it('updates the disc label when notice changes after mount', () => {
    $wakeWord.set({ available: true, enabled: true, listening: false, notice: '', pending: false, phrase: '' })
    renderFan()

    expect(screen.getByRole('button', { name: /wake word/i }).getAttribute('aria-label')).not.toMatch(/—/)

    act(() => {
      $wakeWord.set({
        ...$wakeWord.get(),
        notice: 'Failed to open the client microphone for wake word'
      })
    })

    const disc = screen.getByRole('button', { name: /wake word.*failed to open/i })

    expect(disc.getAttribute('aria-label')).toContain('Failed to open the client microphone')
  })
})
