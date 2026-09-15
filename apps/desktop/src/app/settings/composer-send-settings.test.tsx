// @vitest-environment jsdom
import {
  COMPOSER_SEND_DEFAULT_MODE,
  DOUBLE_ENTER_DEFAULT_MS,
  HOLD_DEFAULT_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
  TYPING_IDLE_DEFAULT_MS
} from '@hermes/shared'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerSendPrefs } from '@/store/composer-send'
import { stubResizeObserver } from '@/test/jsdom'

import { ComposerSendSettings } from './composer-send-settings'

/**
 * The graded half of the send-mode feature: which rows exist for which state,
 * and whether a control persists what it says it does.
 *
 * The store is REAL here — only the IPC bridge and the i18n table are stubbed —
 * so a change that stopped persisting would fail these, not pass them.
 */

// Radix's Popover measures its content with ResizeObserver, which jsdom lacks.
stubResizeObserver()

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
  modeModEnter: 'Enter + modifier',
  doubleTapTitle: 'Double-tap window',
  doubleTapDescription: 'How fast the two Enter presses have to land.',
  doubleTapUnit: 'ms',
  holdToggleTitle: 'Press and hold Enter',
  holdToggleDescription: 'Keep the key down to send.',
  holdMsTitle: 'Hold time',
  holdMsDescription: 'How long Enter has to stay down.',
  holdMsUnit: 'ms',
  typingIdleTitle: 'Typing pause',
  typingIdleDescription: 'How long you have to stop typing.',
  typingIdleUnit: 'ms',
  graceTitle: 'Wait before sending',
  graceDescription: 'Sends you pick here wait a moment before they go.',
  gracePopoverHint: 'Which sends should wait?',
  graceReasonEnter: 'A bare Enter',
  graceReasonDoubleTap: 'A double tap',
  graceReasonPause: 'Enter after a pause',
  graceReasonHold: 'Press and hold',
  graceNone: 'None',
  graceAll: 'All',
  graceSome: (count: string, total: string) => `${count} of ${total}`,
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
  sendOnHold: false,
  typingIdleMs: TYPING_IDLE_DEFAULT_MS,
  sendGraceFor: SEND_GRACE_DEFAULT_REASONS,
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

    for (const label of ['Enter', 'Double tap', 'Pause', 'Enter + modifier']) {
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

  it('offers press-and-hold everywhere a bare Enter does not already send', () => {
    setPrefs({ mode: 'pause' })

    const { getByText } = render(<ComposerSendSettings />)

    expect(getByText('Press and hold Enter')).toBeTruthy()
  })

  it('withholds press-and-hold in `enter`, where the press has already sent', () => {
    setPrefs({ mode: 'enter' })

    const { queryByText } = render(<ComposerSendSettings />)

    expect(queryByText('Press and hold Enter')).toBeNull()
  })

  it('keeps the hold time off until the gesture is switched on', () => {
    setPrefs({ mode: 'pause', sendOnHold: false })

    const { queryByText } = render(<ComposerSendSettings />)

    expect(queryByText('Hold time')).toBeNull()
  })

  it('shows the hold time once the gesture is on, and persists the switch', async () => {
    setPrefs({ mode: 'pause', sendOnHold: true })

    const { getByText, getByRole } = render(<ComposerSendSettings />)

    expect(getByText('Hold time')).toBeTruthy()

    fireEvent.click(getByRole('switch', { name: 'Press and hold Enter' }))
    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ sendOnHold: false }))
  })

  it('hides both timing rows for a mode that measures neither', () => {
    setPrefs({ mode: 'enter' })

    const { queryByText } = render(<ComposerSendSettings />)

    expect(queryByText('Typing pause')).toBeNull()
    expect(queryByText('Double-tap window')).toBeNull()
  })

  it('offers the delay on every mode — default Enter is the one that needs undo-send', () => {
    setPrefs({ mode: 'enter' })

    const { getByText } = render(<ComposerSendSettings />)

    expect(getByText('Wait before sending')).toBeTruthy()
    expect(getByText('Hold duration')).toBeTruthy()
  })

  it('summarises the delay as a count, not a fixed scope', () => {
    setPrefs({ sendGraceFor: ['pause', 'hold'] })

    const { getByText } = render(<ComposerSendSettings />)

    expect(getByText('2 of 4')).toBeTruthy()
  })

  it('hides the duration exactly when nothing is set to wait', () => {
    setPrefs({ sendGraceFor: [] })

    const { queryByText } = render(<ComposerSendSettings />)

    expect(queryByText('Hold duration')).toBeNull()
  })

  it('persists one situation at a time', async () => {
    setPrefs({ sendGraceFor: ['pause'] })

    const { getByText, getByRole } = render(<ComposerSendSettings />)

    fireEvent.click(getByRole('button', { name: '1 of 4' }))
    fireEvent.click(getByText('Press and hold'))

    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ sendGraceFor: ['pause', 'hold'] }))
  })

  it('can turn a single situation back off without touching the others', async () => {
    setPrefs({ sendGraceFor: ['pause', 'hold'] })

    const { getByText, getByRole } = render(<ComposerSendSettings />)

    fireEvent.click(getByRole('button', { name: '2 of 4' }))
    fireEvent.click(getByText('Enter after a pause'))

    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ sendGraceFor: ['hold'] }))
  })

  it('names the file it persists to, so the hand-edit path is discoverable', () => {
    setPrefs({ mode: 'pause' })

    const { getAllByText } = render(<ComposerSendSettings />)

    expect(getAllByText(/composer-send\.json/).length).toBeGreaterThan(0)
  })
})
