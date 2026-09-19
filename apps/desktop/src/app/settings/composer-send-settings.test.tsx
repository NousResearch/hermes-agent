// @vitest-environment jsdom
import {
  DOUBLE_ENTER_DEFAULT_MS,
  HOLD_DEFAULT_MS,
  IDLE_SEND_DEFAULT_MS,
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
 * The graded half of the send settings: which rows exist for which state, and
 * whether a control persists what it says it does.
 *
 * The store is REAL here — only the IPC bridge and the i18n table are stubbed —
 * so a control that stopped persisting would fail these, not pass them.
 */

const mocks = vi.hoisted(() => ({
  haptic: vi.fn(),
  notify: vi.fn(),
  notifyError: vi.fn(),
  set: vi.fn()
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
  graceReasonEnter: 'A bare Enter',
  graceReasonDoubleTap: 'A double tap',
  graceReasonPause: 'The pause send',
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
  it('leads with the gate, and hides the newline choice until it can matter', () => {
    const { getByRole, getByText, queryByText } = render(<ComposerSendSettings />)

    expect(getByText('Keep a bare Enter from sending')).toBeTruthy()
    expect(getByRole('switch', { name: 'Keep a bare Enter from sending' })).toBeTruthy()
    // Nothing can break the line while the press still sends.
    expect(queryByText('A bare Enter starts a new line')).toBeNull()
  })

  it('persists the gate, which is stored inverted from how it reads', async () => {
    const { getByRole } = render(<ComposerSendSettings />)

    fireEvent.click(getByRole('switch', { name: 'Keep a bare Enter from sending' }))
    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ enterSends: false }))
    expect(mocks.notify).toHaveBeenCalled()
  })

  it('offers the line break as its own choice once the gate is closed', async () => {
    newlineMode()

    const { getByRole, getByText } = render(<ComposerSendSettings />)

    expect(getByText('A bare Enter starts a new line')).toBeTruthy()

    fireEvent.click(getByRole('switch', { name: 'A bare Enter starts a new line' }))
    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ enterNewline: false }))
  })

  it('hides the gestures entirely while Enter sends on the press', () => {
    const { queryByText } = render(<ComposerSendSettings />)

    // They cannot fire, and a row of dead switches reads as a broken panel. The
    // note is what says where they went.
    for (const label of ['Double tap', 'Press and hold', 'Send when I stop typing']) {
      expect(queryByText(label)).toBeNull()
    }

    expect(queryByText('Available once a bare Enter stops sending.')).toBeTruthy()
  })

  it('brings the gestures back once Enter only breaks the line', () => {
    newlineMode()

    const { getByRole, getByText, queryByText } = render(<ComposerSendSettings />)

    expect(getByText('Press and hold')).toBeTruthy()
    expect(getByRole('switch', { name: 'Press and hold' })).toBeTruthy()
    expect(queryByText('Available once a bare Enter stops sending.')).toBeNull()
  })

  it('persists a gesture switch', async () => {
    newlineMode()

    const { getByRole } = render(<ComposerSendSettings />)

    fireEvent.click(getByRole('switch', { name: 'Press and hold' }))
    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ sendOnHold: true }))
  })

  it('shows each window only for the gesture that is switched on', () => {
    newlineMode({ sendOnHold: true })

    const { getByText, queryByText } = render(<ComposerSendSettings />)

    expect(getByText('Hold time')).toBeTruthy()
    expect(queryByText('Idle time')).toBeNull()
    expect(queryByText('Typing pause')).toBeNull()
  })

  it('shows the idle window, which is the only gesture that acts with no key', () => {
    newlineMode({ sendOnIdle: true })

    const { getByText } = render(<ComposerSendSettings />)

    expect(getByText('Idle time')).toBeTruthy()
  })

  it('offers the delay in every configuration — default Enter is where undo-send matters most', () => {
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

    const { getByRole, getByText } = render(<ComposerSendSettings />)

    fireEvent.click(getByRole('button', { name: '1 of 4' }))
    fireEvent.click(getByText('The long press'))

    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ sendGraceFor: ['pause', 'hold'] }))
  })

  it('can turn a single situation back off without touching the others', async () => {
    setPrefs({ sendGraceFor: ['pause', 'hold'] })

    const { getByRole, getByText } = render(<ComposerSendSettings />)

    fireEvent.click(getByRole('button', { name: '2 of 4' }))
    fireEvent.click(getByText('The pause send'))

    await later()

    expect(mocks.set).toHaveBeenCalledWith(expect.objectContaining({ sendGraceFor: ['hold'] }))
  })

  it('names the file it persists to, so the hand-edit path is discoverable', () => {
    newlineMode({ sendOnHold: true })

    const { getAllByText } = render(<ComposerSendSettings />)

    expect(getAllByText(/composer-send\.json/).length).toBeGreaterThan(0)
  })
})
