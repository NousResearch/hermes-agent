// @vitest-environment jsdom
import {
  DOUBLE_ENTER_DEFAULT_MS,
  HOLD_DEFAULT_MS,
  IDLE_SEND_DEFAULT_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
  TYPING_IDLE_DEFAULT_MS
} from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerSendPrefs } from '@/store/composer-prefs'
import { stubResizeObserver } from '@/test/jsdom'

import { ComposerSendSettings } from './composer-send-settings'

/**
 * The graded half of the send settings: which rows exist for which state, and
 * whether a control persists what it says it does.
 *
 * The store is REAL here — only the config write and the i18n table are stubbed
 * — so a control that stopped persisting would fail these, not pass them. The
 * write is the gateway config (`desktop.composer.*`), which is where the values
 * live now; the panel has no storage of its own.
 */

const mocks = vi.hoisted(() => ({
  haptic: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn(),
  save: vi.fn()
}))

const WORDS = {
  title: 'Enter and sending',
  description: 'Keep a stray Enter from sending.',
  gateLabel: 'Keep a bare Enter from sending',
  gateDescription: 'A lone press never commits.',
  newlineLabel: 'A bare Enter starts a new line',
  newlineDescription: 'Off means the press does nothing at all.',
  gesturesTitle: 'Other ways to send',
  gesturesDisabled: 'Available once a bare Enter stops sending.',
  gestureDoubleTap: 'Double tap',
  gesturePause: 'Enter after a pause',
  gestureHold: 'Press and hold',
  gestureIdle: 'Send when I stop typing',
  gestureDoubleTapDesc: 'Enter twice in quick succession sends.',
  gesturePauseDesc: 'A single Enter sends once you have stopped typing.',
  gestureHoldDesc: 'Keep the key down and the press becomes a send.',
  gestureIdleDesc: 'Sends on its own if you stop typing.',
  doubleTapTitle: 'Double-tap window',
  doubleTapDescription: 'How fast the two Enter presses have to land.',
  doubleTapUnit: 'ms',
  holdMsTitle: 'Hold time',
  holdMsDescription: 'How long Enter has to stay down.',
  holdMsUnit: 'ms',
  idleMsTitle: 'Idle time',
  idleMsDescription: 'How long the composer waits before sending on its own.',
  idleMsUnit: 'ms',
  typingIdleTitle: 'Typing pause',
  typingIdleDescription: 'How long you have to stop typing.',
  typingIdleUnit: 'ms',
  graceTitle: 'Wait before sending',
  graceDescription: 'Sends you pick here wait a moment before they go.',
  gracePopoverHint: 'Which sends should wait?',
  gracePressHint: 'While a send is waiting',
  gracePressCommit: 'Pressing Enter sends it now',
  graceReasonEnter: 'A bare Enter',
  graceReasonDoubleTap: 'A double tap',
  graceReasonPause: 'The pause send',
  graceReasonLongPress: 'The long press',
  graceReasonHold: 'The long press',
  graceNone: 'None',
  graceAll: 'All',
  graceSome: (count: string, total: string) => `${count} of ${total}`,
  graceMsTitle: 'Hold duration',
  graceMsDescription: 'How long a held send waits before it goes.',
  graceMsUnit: 'ms',
  fileHint: (path: string) => `Saved in ${path} — edit that file for exact values.`
}

// Radix's Popover measures its content with ResizeObserver, which jsdom lacks.
stubResizeObserver()

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      composer: {
        placeholderSendChord: (chord: string) => `${chord} sends`,
        placeholderSendDoubleTap: 'tap Enter twice to send',
        placeholderSendEnterSends: 'Enter sends',
        placeholderSendHold: 'hold Enter to send',
        placeholderSendIdle: 'and it sends if you stop typing',
        placeholderSendNewline: 'Enter starts a new line',
        placeholderSendPause: 'Enter after a pause sends'
      },
      keybinds: { composerSend: WORDS },
      settings: { config: { autosaveFailed: 'Could not save settings' } }
    }
  })
}))

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    saveHermesConfig: (...args: unknown[]) => mocks.save(...args)
  }
})

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
  commitOnPress: true,
  doubleEnterMs: DOUBLE_ENTER_DEFAULT_MS,
  enterNewline: true,
  enterSends: true,
  holdMs: HOLD_DEFAULT_MS,
  idleSendMs: IDLE_SEND_DEFAULT_MS,
  sendOnDoubleTap: false,
  sendOnHold: false,
  sendOnIdle: false,
  sendOnPause: false,
  sendGraceFor: SEND_GRACE_DEFAULT_REASONS,
  sendGraceMs: SEND_GRACE_DEFAULT_MS,
  typingIdleMs: TYPING_IDLE_DEFAULT_MS
}

const setPrefs = (next: Partial<typeof DEFAULTS>) => {
  act(() => {
    $composerSendPrefs.set({ ...DEFAULTS, ...next })
  })
}

/** A configuration where the gestures are reachable at all. */
const newlineMode = (next: Partial<typeof DEFAULTS> = {}) => setPrefs({ enterSends: false, ...next })

const later = () => Promise.resolve()

/** The config record the panel asked the gateway to save. */
const savedComposerRecord = () =>
  (mocks.save.mock.calls.at(-1)?.[0] as { desktop?: { composer?: Record<string, unknown> } } | undefined)?.desktop
    ?.composer

const open = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <ComposerSendSettings />
    </QueryClientProvider>
  )

beforeEach(() => {
  mocks.save.mockReset()
  mocks.save.mockResolvedValue({ ok: true })
  mocks.notify.mockReset()
  mocks.notifyError.mockReset()
  mocks.haptic.mockReset()

  $composerSendPrefs.set(DEFAULTS)
})

afterEach(cleanup)

describe('ComposerSendSettings', () => {
  it('leads with the gate, and hides the newline choice until it can matter', () => {
    const { getByRole, getByText, queryByText } = open()

    expect(getByText('Keep a bare Enter from sending')).toBeTruthy()
    expect(getByRole('switch', { name: 'Keep a bare Enter from sending' })).toBeTruthy()
    // Nothing can break the line while the press still sends.
    expect(queryByText('A bare Enter starts a new line')).toBeNull()
  })

  it('persists the gate, which is stored inverted from how it reads', async () => {
    const { getByRole } = open()

    fireEvent.click(getByRole('switch', { name: 'Keep a bare Enter from sending' }))
    await later()

    expect(savedComposerRecord()).toMatchObject({ enter_sends: false })
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('offers the line break as its own choice once the gate is closed', async () => {
    newlineMode()

    const { getByRole, getByText } = open()

    expect(getByText('A bare Enter starts a new line')).toBeTruthy()

    fireEvent.click(getByRole('switch', { name: 'A bare Enter starts a new line' }))
    await later()

    expect(savedComposerRecord()).toMatchObject({ enter_newline: false })
  })

  it('hides the gestures entirely while Enter sends on the press', () => {
    const { queryByText } = open()

    // They cannot fire, and a row of dead switches reads as a broken panel. The
    // note is what says where they went.
    for (const label of ['Double tap', 'Press and hold', 'Send when I stop typing']) {
      expect(queryByText(label)).toBeNull()
    }

    expect(queryByText('Available once a bare Enter stops sending.')).toBeTruthy()
  })

  it('brings the gestures back once Enter only breaks the line', () => {
    newlineMode()

    const { getByRole, getByText, queryByText } = open()

    expect(getByText('Press and hold')).toBeTruthy()
    expect(getByRole('switch', { name: 'Press and hold' })).toBeTruthy()
    expect(queryByText('Available once a bare Enter stops sending.')).toBeNull()
  })

  it('persists a gesture switch', async () => {
    newlineMode()

    const { getByRole } = open()

    fireEvent.click(getByRole('switch', { name: 'Press and hold' }))
    await later()

    expect(savedComposerRecord()).toMatchObject({ send_on_hold: true })
  })

  it('shows each window only for the gesture that is switched on', () => {
    newlineMode({ sendOnHold: true })

    const { getByText, queryByText } = open()

    expect(getByText('Hold time')).toBeTruthy()
    expect(queryByText('Idle time')).toBeNull()
    expect(queryByText('Typing pause')).toBeNull()
  })

  it('shows the idle window, which is the only gesture that acts with no key', () => {
    newlineMode({ sendOnIdle: true })

    const { getByText } = open()

    expect(getByText('Idle time')).toBeTruthy()
  })

  it('offers the delay in every configuration — default Enter is where undo-send matters most', () => {
    const { getByText } = open()

    expect(getByText('Wait before sending')).toBeTruthy()
    expect(getByText('Hold duration')).toBeTruthy()
  })

  it('summarises the delay as a count, not a fixed scope', () => {
    setPrefs({ sendGraceFor: ['pause', 'hold'] })

    const { getByText } = open()

    expect(getByText('2 of 4')).toBeTruthy()
  })

  it('hides the duration exactly when nothing is set to wait', () => {
    setPrefs({ sendGraceFor: [] })

    const { queryByText } = open()

    expect(queryByText('Hold duration')).toBeNull()
  })

  it('persists one situation at a time', async () => {
    setPrefs({ sendGraceFor: ['pause'] })

    const { getByRole, getByText } = open()

    fireEvent.click(getByRole('button', { name: '1 of 4' }))
    fireEvent.click(getByText('The long press'))

    await later()

    expect(savedComposerRecord()).toMatchObject({ send_grace_for: ['pause', 'hold'] })
  })

  it('can turn a single situation back off without touching the others', async () => {
    setPrefs({ sendGraceFor: ['pause', 'hold'] })

    const { getByRole, getByText } = open()

    fireEvent.click(getByRole('button', { name: '2 of 4' }))
    fireEvent.click(getByText('The pause send'))

    await later()

    expect(savedComposerRecord()).toMatchObject({ send_grace_for: ['hold'] })
  })

  it('persists the press-during-a-wait switch, which is not one of the situations', async () => {
    setPrefs({ sendGraceFor: ['pause'] })

    const { getByRole, getByText } = open()

    fireEvent.click(getByRole('button', { name: '1 of 4' }))
    fireEvent.click(getByText('Pressing Enter sends it now'))

    await later()

    // The situation set is untouched: this is a different question.
    expect(savedComposerRecord()).toMatchObject({ commit_on_press: false, send_grace_for: ['pause'] })

    fireEvent.click(getByText('Pressing Enter sends it now'))

    await later()

    expect(savedComposerRecord()).toMatchObject({ commit_on_press: true })
  })

  it('names where it persists to, so the hand-edit path is discoverable', () => {
    newlineMode({ sendOnHold: true })

    const { getAllByText } = open()

    expect(getAllByText(/desktop\.composer/).length).toBeGreaterThan(0)
  })

  it('keeps the last known-good values when the write fails', async () => {
    mocks.save.mockRejectedValue(new Error('gateway down'))

    const { getByRole } = open()

    fireEvent.click(getByRole('switch', { name: 'Keep a bare Enter from sending' }))

    // The atom moved optimistically, then went back: the panel and the composer
    // read this atom on every keystroke, so a failed write must not leave either
    // of them claiming a setting the gateway never took.
    await waitFor(() => expect($composerSendPrefs.get()).toEqual(DEFAULTS))
    expect(mocks.notifyError).toHaveBeenCalled()
  })
})
