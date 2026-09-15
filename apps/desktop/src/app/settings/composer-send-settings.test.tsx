// @vitest-environment jsdom
import {
  COMPOSER_SEND_DEFAULT_MODE,
  DOUBLE_ENTER_DEFAULT_MS,
  HOLD_DEFAULT_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_SCOPE,
  TYPING_IDLE_DEFAULT_MS
} from '@hermes/shared'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerSendPrefs } from '@/store/composer-send'

import { ComposerSendSettings } from './composer-send-settings'

/**
 * The graded half of the send-mode feature: which rows exist for which state,
 * and whether a control persists what it says it does.
 *
 * The store is REAL here — only the IPC bridge and the i18n table are stubbed —
 * so a change that stopped persisting would fail these, not pass them.
 */

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  set: vi.fn(),
  haptic: vi.fn()
}))

const WORDS = {
  title: 'Send with',
  description: 'Which keypress commits a message.',
  modeEnter: 'Enter',
  modeDoubleEnter: 'Double tap',
  modePause: 'Pause',
  modeHold: 'Hold',
  modeModEnter: 'Enter + modifier',
  doubleTapTitle: 'Double-tap window',
  doubleTapDescription: 'How fast the two Enter presses have to land.',
  doubleTapUnit: 'ms',
  holdMsTitle: 'Hold time',
  holdMsDescription: 'How long Enter has to stay down.',
  holdMsUnit: 'ms',
  typingIdleTitle: 'Typing pause',
  typingIdleDescription: 'How long you have to stop typing.',
  typingIdleUnit: 'ms',
  graceTitle: 'Hold a guessed send',
  graceDescription: 'Guessed sends wait a moment so Esc can take them back.',
  graceOff: 'Never',
  graceInferred: 'Guessed only',
  graceAll: 'Every send',
  graceMsTitle: 'Hold duration',
  graceMsDescription: 'How long a held send waits before it goes.',
  graceMsUnit: 'ms',
  fileHint: (path: string) => `Saved in ${path} — edit that file for exact values.`
}

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      composer: {
        placeholderSendChord: (chord: string) => `${chord} sends`,
        placeholderSendDoubleTap: 'tap it twice to send',
        placeholderSendEnterSends: 'Enter sends',
        placeholderSendNewline: 'send · Shift+Enter for newline',
        placeholderSendPause: 'sends once you stop typing'
      },
      keybinds: { composerSend: WORDS }
    }
  })
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: (...args: unknown[]) => mocks.haptic(...args)
}))

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => mocks.notify(...args),
  notifyError: (...args: unknown[]) => mocks.notifyError(...args)
}))

// The row shell is presentation; this test is about which rows exist and what
// their controls do.
vi.mock('./primitives', () => ({
  ListRow: ({ action, description, hint, title }: Record<string, unknown>) => (
    <div>
      <span>{title as string}</span>
      <span>{description as string}</span>
      {hint ? <span>{hint as string}</span> : null}
      {action as never}
    </div>
  )
}))

const DEFAULTS = {
  mode: COMPOSER_SEND_DEFAULT_MODE,
  doubleEnterMs: DOUBLE_ENTER_DEFAULT_MS,
  holdMs: HOLD_DEFAULT_MS,
  typingIdleMs: TYPING_IDLE_DEFAULT_MS,
  sendGrace: SEND_GRACE_DEFAULT_SCOPE,
  sendGraceMs: SEND_GRACE_DEFAULT_MS
}

const setPrefs = (next: Partial<typeof DEFAULTS>) => {
  act(() => {
    $composerSendPrefs.set({ ...DEFAULTS, ...next })
  })
}

const later = () => Promise.resolve()

beforeEach(() => {
  mocks.set.mockReset()
  mocks.notify.mockReset()
  mocks.haptic.mockReset()
  mocks.set.mockImplementation(async (prefs: unknown) => ({ ...(prefs as object), path: '/tmp/composer-send.json' }))

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { composerSend: { get: async () => DEFAULTS, set: mocks.set } },
    writable: true
  })

  $composerSendPrefs.set(DEFAULTS)
})

afterEach(() => {
  cleanup()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ComposerSendSettings', () => {
  it('offers every mode, including the pause that the schema already carried', () => {
    const { getByText } = render(<ComposerSendSettings />)

    for (const label of ['Enter', 'Double tap', 'Pause', 'Hold', 'Enter + modifier']) {
      expect(getByText(label)).toBeTruthy()
    }
  })

  it('persists the chosen mode and says what the switch did', async () => {
    const { getByText } = render(<ComposerSendSettings />)

    fireEvent.click(getByText('Pause'))
    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ mode: 'pause' }))
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('shows both timing rows for pause — it measures the double-tap too', () => {
    setPrefs({ mode: 'pause' })

    const { getByText } = render(<ComposerSendSettings />)

    expect(getByText('Typing pause')).toBeTruthy()
    expect(getByText('Double-tap window')).toBeTruthy()
  })

  it('offers the hold time only where a long press is the send', () => {
    setPrefs({ mode: 'hold' })

    const { getByText } = render(<ComposerSendSettings />)

    expect(getByText('Hold time')).toBeTruthy()
  })

  it('hides both timing rows for a mode that measures neither', () => {
    setPrefs({ mode: 'enter' })

    const { queryByText } = render(<ComposerSendSettings />)

    expect(queryByText('Typing pause')).toBeNull()
    expect(queryByText('Double-tap window')).toBeNull()
  })

  it('offers the grace window on every mode — default Enter is the one that needs undo-send', () => {
    setPrefs({ mode: 'enter' })

    const { getByText } = render(<ComposerSendSettings />)

    expect(getByText('Hold a guessed send')).toBeTruthy()
    expect(getByText('Hold duration')).toBeTruthy()
  })

  it('hides the hold duration exactly when nothing can be held', () => {
    setPrefs({ sendGrace: 'off' })

    const { queryByText } = render(<ComposerSendSettings />)

    expect(queryByText('Hold duration')).toBeNull()
  })

  it('persists the grace scope', async () => {
    const { getByText } = render(<ComposerSendSettings />)

    fireEvent.click(getByText('Every send'))
    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ sendGrace: 'all' }))
  })

  it('names the file it persists to, so the hand-edit path is discoverable', () => {
    setPrefs({ mode: 'pause' })

    const { getAllByText } = render(<ComposerSendSettings />)

    expect(getAllByText(/composer-send\.json/).length).toBeGreaterThan(0)
  })
})
