// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import type { PetActionCenterLiveStatus, PetActionCenterState } from '@/store/pet-action-center'
import type { PetActionCenterControl } from '@/store/pet-overlay'

import { PetActionCenter } from './pet-action-center'

// ── Test helpers ──────────────────────────────────────────────────────────

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const ac = en.pet.actionCenter
type TestItem = NonNullable<PetActionCenterState['items'][number]>
type TestLiveTurnItem = Extract<TestItem, { kind: 'live-turn' }>

let controlMock: ReturnType<typeof vi.fn>
let setFocusableMock: ReturnType<typeof vi.fn>
let setIgnoreMouseMock: ReturnType<typeof vi.fn>

function installDesktopMock() {
  controlMock = vi.fn()
  setFocusableMock = vi.fn()
  setIgnoreMouseMock = vi.fn()

  desktopWindow.hermesDesktop = {
    petOverlay: {
      close: vi.fn().mockResolvedValue({ ok: true }),
      control: controlMock,
      onControl: vi.fn(() => () => {}),
      onState: vi.fn(() => () => {}),
      open: vi.fn().mockResolvedValue({ ok: true }),
      pushState: vi.fn(),
      setBounds: vi.fn(),
      setFocusable: setFocusableMock,
      setIgnoreMouse: setIgnoreMouseMock
    }
  } as unknown as Window['hermesDesktop']
}

function uninstallDesktopMock() {
  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
}

function makeApprovalItem(
  overrides: Partial<NonNullable<PetActionCenterState['items'][number]>> = {}
): NonNullable<PetActionCenterState['items'][number]> {
  return {
    actionable: true,
    allowPermanent: true,
    allowedActions: ['approve-once', 'approve-session', 'approve-always', 'deny'],
    blocking: true,
    choices: null,
    command: 'npm test',
    description: 'Run the test suite',
    detail: 'npm test',
    id: 'approval-1',
    kind: 'approval',
    profile: 'default',
    profileLabel: 'default',
    receivedAt: 1000,
    sessionId: 'session-1',
    sessionTitle: 'My Session',
    smartDenied: false,
    storedSessionId: 'stored-1',
    summary: 'Run the test suite',
    ...overrides
  } as NonNullable<PetActionCenterState['items'][number]>
}

function makeClarifyItem(
  overrides: Partial<NonNullable<PetActionCenterState['items'][number]>> = {}
): NonNullable<PetActionCenterState['items'][number]> {
  return {
    actionable: true,
    allowedActions: ['clarify-respond', 'clarify-skip'],
    blocking: true,
    choices: ['Red', 'Blue'],
    detail: 'Red\nBlue',
    id: 'clarify-1',
    kind: 'clarify',
    profile: 'default',
    profileLabel: 'default',
    question: 'Which color?',
    receivedAt: 2000,
    sessionId: 'session-2',
    sessionTitle: null,
    storedSessionId: null,
    summary: 'Which color?',
    ...overrides
  } as NonNullable<PetActionCenterState['items'][number]>
}

function makeLiveTurnItem(
  overrides: Partial<TestLiveTurnItem> = {}
): TestLiveTurnItem {
  return {
    actionable: true,
    activityKind: null,
    activityName: null,
    allowedActions: ['steer', 'queue', 'stop', 'open-in-app'],
    blocking: false,
    connectionState: 'open',
    detail: null,
    id: 'live-1',
    kind: 'live-turn',
    profile: 'work-profile-id',
    profileLabel: 'Work',
    queuedCount: 0,
    receivedAt: 3000,
    sessionId: 'runtime-secret-id',
    sessionTitle: 'Release prep',
    status: 'working',
    storedSessionId: 'stored-secret-id',
    summary: null,
    turnStartedAt: null,
    ...overrides
  }
}

function makeLiveStatus(
  overrides: Partial<PetActionCenterLiveStatus> = {}
): PetActionCenterLiveStatus {
  return {
    activityKind: null,
    activityName: null,
    connectionState: 'open',
    queuedCount: 0,
    status: 'waiting',
    turnStartedAt: null,
    ...overrides
  }
}

function makeState(
  items: NonNullable<PetActionCenterState['items'][number]>[] = [],
  overrides: Partial<PetActionCenterState> = {}
): PetActionCenterState {
  return {
    action: null,
    actionableCount: items.filter(i => i.actionable).length,
    attentionCount: items.length,
    blockingCount: items.filter(i => i.blocking).length,
    items,
    secureInputCount: 0,
    selectedItemId: items[0]?.id ?? null,
    ...overrides
  }
}

// Collect emitted controls for assertion.
function emittedControls(): PetActionCenterControl[] {
  return controlMock.mock.calls.map(call => call[0] as PetActionCenterControl)
}

// ── Setup / teardown ──────────────────────────────────────────────────────

beforeEach(() => {
  installDesktopMock()
})

afterEach(() => {
  cleanup()
  uninstallDesktopMock()
})

// ── Tests ─────────────────────────────────────────────────────────────────

describe('PetActionCenter — contract 1: no auto-focus/expand', () => {
  it('does not render the panel or call setFocusable when no user click has happened', () => {
    const state = makeState([makeApprovalItem()])

    render(<PetActionCenter state={state} />)

    // The trigger button is present but the panel is NOT open.
    expect(screen.queryByRole('dialog')).toBeNull()
    // No focusable calls happened (no auto-focus steal).
    expect(setFocusableMock).not.toHaveBeenCalled()
  })

  it('opens the panel after explicit user click on the trigger without calling setFocusable', () => {
    const state = makeState([makeApprovalItem()])

    render(<PetActionCenter state={state} />)

    const trigger = screen.getByRole('button', { name: ac.open })
    fireEvent.click(trigger)

    // Panel is now open (getByRole throws if absent, so this suffices).
    const dialog = screen.getByRole('dialog')
    expect(dialog).not.toBeNull()
    expect(dialog.getAttribute('aria-modal')).toBe('true')
    // setFocusable is NOT called by the action center itself — focus is
    // managed by the parent overlay via onOpenChange. Without the callback,
    // the component must never call setFocusable directly.
    expect(setFocusableMock).not.toHaveBeenCalled()
  })

  it('contains focus inside the dialog after an explicit open', async () => {
    render(<PetActionCenter state={makeState([makeApprovalItem()])} />)

    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    await waitFor(() => expect(document.activeElement).toBe(dialog))
    expect(dialog.tabIndex).toBe(-1)
  })
})

describe('PetActionCenter — contract 2: count + navigation', () => {
  it('shows count and allows deterministic next navigation with an opaque item id', () => {
    const items = [
      makeApprovalItem({ id: 'a-1', command: 'cmd-1', description: 'desc-1', receivedAt: 1000 }),
      makeApprovalItem({ id: 'a-2', command: 'cmd-2', description: 'desc-2', receivedAt: 2000 })
    ]

    const state = makeState(items, { selectedItemId: 'a-1' })

    render(<PetActionCenter state={state} />)

    // Open the panel.
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')

    // Count is shown.
    expect(within(dialog).getByText(ac.itemCount(2, 2))).not.toBeNull()

    // Next button navigates to a-2.
    const nextBtn = within(dialog).getByRole('button', { name: /next item/i })
    fireEvent.click(nextBtn)

    const controls = emittedControls()
    const selectControl = controls.find(c => c.type === 'action-center-select')
    expect(selectControl).toEqual({ type: 'action-center-select', itemId: 'a-2' })
  })

  it('allows deterministic previous navigation with an opaque item id', () => {
    const items = [
      makeApprovalItem({ id: 'opaque-first', command: 'cmd-1', receivedAt: 1000 }),
      makeApprovalItem({ id: 'opaque-second', command: 'cmd-2', receivedAt: 2000 })
    ]

    render(<PetActionCenter state={makeState(items, { selectedItemId: 'opaque-first' })} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: ac.previous }))

    expect(emittedControls()).toEqual([{ type: 'action-center-select', itemId: 'opaque-second' }])
  })
})

describe('PetActionCenter — contract 4: render only present capabilities', () => {
  it('renders only approve-once and deny when those are the only allowed actions', () => {
    const item = makeApprovalItem({
      allowedActions: ['approve-once', 'deny'],
      allowPermanent: false
    })

    const state = makeState([item])

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')

    expect(within(dialog).getByRole('button', { name: /run once/i })).not.toBeNull()
    expect(within(dialog).getByRole('button', { name: /reject/i })).not.toBeNull()
    expect(within(dialog).queryByRole('button', { name: /allow this session/i })).toBeNull()
    expect(within(dialog).queryByRole('button', { name: /always allow/i })).toBeNull()
  })
})

describe('PetActionCenter — contract 3: approval shows description + command preview', () => {
  it('renders description and command preview inside the panel', () => {
    const item = makeApprovalItem({
      description: 'Run the full test suite',
      command: 'npm run test -- --verbose'
    })

    const state = makeState([item])

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText('Run the full test suite')).not.toBeNull()
    expect(within(dialog).getByText('npm run test -- --verbose')).not.toBeNull()
  })
})

describe('PetActionCenter — contract 5: approve-always opens inline confirmation', () => {
  it('opens an inline two-step confirmation and focuses its safe visible Cancel action', async () => {
    const item = makeApprovalItem({
      allowedActions: ['approve-once', 'approve-always', 'deny'],
      allowPermanent: true
    })

    const state = makeState([item])

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')

    // Click "Always allow…" — this opens a confirm, not a direct action.
    const alwaysBtn = within(dialog).getByRole('button', { name: /always allow/i })
    fireEvent.click(alwaysBtn)

    // Confirmation text appears.
    expect(within(dialog).getByText(/always allow this command/i)).not.toBeNull()

    // The destructive confirm button exists but is NOT the element with focus.
    const confirmBtn = within(dialog).getByRole('button', { name: /^always allow$/i })
    expect(confirmBtn).not.toBeNull()
    const cancelBtn = within(dialog).getByRole('button', { name: ac.cancel })
    await waitFor(() => expect(document.activeElement).toBe(cancelBtn))
    expect(confirmBtn).not.toBe(document.activeElement)

    // No approval control was emitted yet.
    expect(emittedControls().filter(c => c.type === 'action-center-approval')).toHaveLength(0)
  })

  it.each(['Cancel', 'Escape'] as const)(
    '%s cancels only the confirmation, restores its exact source button, and emits no action',
    async dismissWith => {
      const item = makeApprovalItem({
        allowedActions: ['approve-once', 'approve-always', 'deny'],
        allowPermanent: true
      })

      const state = makeState([item])

      render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))
      const dialog = screen.getByRole('dialog')
      const sourceButton = within(dialog).getByRole('button', { name: /always allow/i })

      fireEvent.click(sourceButton)
      const cancelButton = within(dialog).getByRole('button', { name: ac.cancel })
      await waitFor(() => expect(document.activeElement).toBe(cancelButton))
      expect(sourceButton.isConnected).toBe(true)

      if (dismissWith === 'Cancel') {
        fireEvent.click(cancelButton)
      } else {
        fireEvent.keyDown(cancelButton, { key: 'Escape' })
      }

      expect(within(dialog).queryByText(/always allow this command/i)).toBeNull()
      expect(screen.getByRole('dialog')).toBe(dialog)
      await waitFor(() => expect(document.activeElement).toBe(sourceButton))
      expect(emittedControls().filter(c => c.type === 'action-center-approval')).toHaveLength(0)
    }
  )

  it.each([
    ['always confirmation', ac.approveAlways],
    ['deny reason', ac.deny]
  ] as const)('hides item navigation during %s and restores it on cancellation', async (_label, sourceName) => {
    const items = [makeApprovalItem({ id: 'a-1' }), makeApprovalItem({ id: 'a-2' })]

    render(<PetActionCenter state={makeState(items)} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByRole('button', { name: ac.next })).not.toBeNull()

    fireEvent.click(within(dialog).getByRole('button', { name: sourceName }))

    expect(within(dialog).queryByRole('button', { name: ac.previous })).toBeNull()
    expect(within(dialog).queryByRole('button', { name: ac.next })).toBeNull()

    const cancelButton = within(dialog).getByRole('button', { name: ac.cancel })
    await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true))
    fireEvent.click(cancelButton)

    expect(within(dialog).getByRole('button', { name: ac.previous })).not.toBeNull()
    expect(within(dialog).getByRole('button', { name: ac.next })).not.toBeNull()
  })
})

describe('PetActionCenter — contract 6: deny opens optional reason', () => {
  it('deny opens a reason input; reason is sent only with deny; Escape cancels reason only', () => {
    const item = makeApprovalItem({
      allowedActions: ['approve-once', 'deny']
    })

    const state = makeState([item])

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')

    // Click Reject — opens reason input.
    const denyBtn = within(dialog).getByRole('button', { name: /reject/i })
    fireEvent.click(denyBtn)

    // Reason input appears.
    const reasonInput = within(dialog).getByRole('textbox', { name: ac.denyReason })
    expect(reasonInput).not.toBeNull()

    // Type a reason and submit.
    fireEvent.change(reasonInput, { target: { value: 'unsafe' } })
    const denySubmit = within(dialog).getByRole('button', { name: /send rejection/i })
    fireEvent.click(denySubmit)

    const controls = emittedControls()
    const approvalControl = controls.find(c => c.type === 'action-center-approval')
    expect(approvalControl).toEqual({
      type: 'action-center-approval',
      itemId: item.id,
      choice: 'deny',
      reason: 'unsafe'
    })
  })

  it.each(['Cancel', 'Escape'] as const)(
    '%s cancels only the deny reason, restores its exact source button, and emits no action',
    async dismissWith => {
      const state = makeState([makeApprovalItem({ allowedActions: ['approve-once', 'deny'] })])

      render(<PetActionCenter state={state} />)
      fireEvent.click(screen.getByRole('button', { name: ac.open }))

      const dialog = screen.getByRole('dialog')
      const sourceButton = within(dialog).getByRole('button', { name: ac.deny })
      fireEvent.click(sourceButton)

      const reasonInput = within(dialog).getByRole('textbox', { name: ac.denyReason })
      await waitFor(() => expect(document.activeElement).toBe(reasonInput))
      expect(sourceButton.isConnected).toBe(true)

      if (dismissWith === 'Cancel') {
        fireEvent.click(within(dialog).getByRole('button', { name: ac.cancel }))
      } else {
        fireEvent.keyDown(reasonInput, { key: 'Escape' })
      }

      expect(within(dialog).queryByRole('textbox', { name: ac.denyReason })).toBeNull()
      expect(screen.getByRole('dialog')).toBe(dialog)
      await waitFor(() => expect(document.activeElement).toBe(sourceButton))
      expect(emittedControls().filter(c => c.type === 'action-center-approval')).toHaveLength(0)
    }
  )
})

describe('PetActionCenter — contract 7: clarify choices + Other + skip + IME-safe Enter', () => {
  it('renders backend choices, supports Other exclusivity, skip, and IME-safe Enter', () => {
    const item = makeClarifyItem({
      choices: ['Red', 'Blue'],
      allowedActions: ['clarify-respond', 'clarify-skip']
    })

    const state = makeState([item])

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')

    // Choices are rendered.
    expect(within(dialog).getByText('Red')).not.toBeNull()
    expect(within(dialog).getByText('Blue')).not.toBeNull()

    // Other textarea is present.
    const otherInput = within(dialog).getByRole('textbox', { name: ac.other })
    expect(otherInput).not.toBeNull()

    // Selecting a choice, then typing in Other clears the choice (exclusivity).
    const redBtn = within(dialog).getByRole('button', { name: /red/i })
    fireEvent.click(redBtn)
    expect(within(dialog).getByRole<HTMLButtonElement>('button', { name: ac.clarifySubmit }).disabled).toBe(false)

    // Now type in Other — should clear the selected choice.
    fireEvent.change(otherInput, { target: { value: 'Green' } })

    // Submit with Enter (non-composing).
    fireEvent.keyDown(otherInput, { key: 'Enter' })

    const controls = emittedControls()
    const clarifyControl = controls.find(c => c.type === 'action-center-clarify')
    expect(clarifyControl).toEqual({
      type: 'action-center-clarify',
      itemId: item.id,
      answer: 'Green'
    })
  })

  it('does not submit on Enter during IME composition', () => {
    const item = makeClarifyItem({
      choices: null,
      allowedActions: ['clarify-respond', 'clarify-skip']
    })

    const state = makeState([item])

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    const textInput = within(dialog).getByRole('textbox', { name: ac.other })

    fireEvent.change(textInput, { target: { value: 'テスト' } })
    fireEvent(
      textInput,
      new KeyboardEvent('keydown', {
        bubbles: true,
        cancelable: true,
        isComposing: true,
        key: 'Enter'
      })
    )

    // No clarify control emitted.
    expect(emittedControls().filter(c => c.type === 'action-center-clarify')).toHaveLength(0)
  })

  it('skip sends an empty answer', () => {
    const item = makeClarifyItem({
      choices: ['Red'],
      allowedActions: ['clarify-respond', 'clarify-skip']
    })

    const state = makeState([item])

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    const skipBtn = within(dialog).getByRole('button', { name: /skip/i })
    fireEvent.click(skipBtn)

    const controls = emittedControls()
    const clarifyControl = controls.find(c => c.type === 'action-center-clarify')
    expect(clarifyControl).toEqual({
      type: 'action-center-clarify',
      itemId: item.id,
      answer: ''
    })
  })

  it('never submits an empty open-ended answer except through explicit Skip', () => {
    const item = makeClarifyItem({ choices: null })
    const state = makeState([item])

    const { rerender } = render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    const textInput = within(dialog).getByRole('textbox', { name: ac.other })
    const continueButton = within(dialog).getByRole<HTMLButtonElement>('button', { name: ac.clarifySubmit })

    expect(continueButton.disabled).toBe(true)
    fireEvent.change(textInput, { target: { value: '   ' } })
    expect(continueButton.disabled).toBe(true)
    fireEvent.keyDown(textInput, { key: 'Enter' })
    expect(emittedControls().filter(c => c.type === 'action-center-clarify')).toHaveLength(0)

    fireEvent.change(textInput, { target: { value: 'A real answer' } })
    expect(continueButton.disabled).toBe(false)

    rerender(<PetActionCenter state={makeState([item], { action: { status: 'submitting', itemId: item.id } })} />)
    expect(continueButton.disabled).toBe(true)
    fireEvent.keyDown(textInput, { key: 'Enter' })
    expect(emittedControls().filter(c => c.type === 'action-center-clarify')).toHaveLength(0)

    rerender(<PetActionCenter state={state} />)
    fireEvent.change(textInput, { target: { value: '' } })
    expect(continueButton.disabled).toBe(true)
    fireEvent.keyDown(textInput, { key: 'Enter' })
    expect(emittedControls().filter(c => c.type === 'action-center-clarify')).toHaveLength(0)

    fireEvent.click(within(dialog).getByRole('button', { name: ac.skip }))
    expect(emittedControls()).toEqual([{ type: 'action-center-clarify', itemId: item.id, answer: '' }])
  })
})

describe('PetActionCenter — contract 8: submitting disables duplicate actions', () => {
  it('disables action buttons while submitting and re-enables on error', () => {
    const item = makeApprovalItem({
      allowedActions: ['approve-once', 'deny']
    })

    const state = makeState([item], {
      action: { status: 'submitting', itemId: item.id }
    })

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')

    const runBtn = within(dialog).getByRole<HTMLButtonElement>('button', { name: /run once/i })
    expect(runBtn.disabled).toBe(true)
  })

  it('disables every selected item action while a different item is submitting', () => {
    const first = makeApprovalItem({ id: 'approval-a' })
    const second = makeApprovalItem({ id: 'approval-b', allowedActions: ['approve-once', 'deny'] })

    const state = makeState([first, second], {
      action: { status: 'submitting', itemId: first.id },
      selectedItemId: second.id
    })

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByRole<HTMLButtonElement>('button', { name: ac.approveOnce }).disabled).toBe(true)
    expect(within(dialog).getByRole<HTMLButtonElement>('button', { name: ac.deny }).disabled).toBe(true)
  })

  it('renders recoverable error state with localized-neutral message', () => {
    const item = makeApprovalItem({
      allowedActions: ['approve-once', 'deny']
    })

    const state = makeState([item], {
      action: { status: 'error', itemId: item.id, errorCode: 'rpc-failed' }
    })

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText(/something went wrong/i)).not.toBeNull()

    // Buttons are re-enabled (error is recoverable).
    const runBtn = within(dialog).getByRole<HTMLButtonElement>('button', { name: /run once/i })
    expect(runBtn.disabled).toBe(false)
  })

  it('announces success/stale status in a live region', () => {
    const item = makeApprovalItem({
      allowedActions: ['approve-once', 'deny']
    })

    // Open while the item is still pending.
    const { rerender } = render(
      <PetActionCenter state={makeState([item], { action: { status: 'success', itemId: item.id } })} />
    )

    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    // The resolved item may disappear before the terminal status reaches the
    // overlay. The live announcement must survive that removal.
    rerender(<PetActionCenter state={makeState([], { action: { status: 'success', itemId: item.id } })} />)
    const liveRegion = screen.getByRole('status')
    expect(liveRegion.textContent?.match(/done/i) !== null).toBe(true)

    // Stale is terminal too and follows the same item-independent contract.
    rerender(<PetActionCenter state={makeState([], { action: { status: 'stale', itemId: item.id } })} />)
    expect(screen.getByRole('status').textContent?.match(/no longer pending/i) !== null).toBe(true)
  })
})

describe('PetActionCenter — contract 9: keyboard + focus restoration', () => {
  it('wraps Tab from the last control to the first control', () => {
    render(<PetActionCenter state={makeState([makeApprovalItem()])} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    const first = within(dialog).getByRole('button', { name: ac.close })
    const last = within(dialog).getByRole('button', { name: ac.deny })
    last.focus()

    fireEvent.keyDown(last, { key: 'Tab' })

    expect(document.activeElement).toBe(first)
  })

  it('wraps Shift+Tab from the first control to the last control', () => {
    render(<PetActionCenter state={makeState([makeApprovalItem()])} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    const first = within(dialog).getByRole('button', { name: ac.close })
    const last = within(dialog).getByRole('button', { name: ac.deny })
    first.focus()

    fireEvent.keyDown(first, { key: 'Tab', shiftKey: true })

    expect(document.activeElement).toBe(last)
  })

  it('keeps focus inside the dialog when tabbing through a nested layer', async () => {
    render(<PetActionCenter state={makeState([makeApprovalItem({ allowedActions: ['approve-once', 'deny'] })])} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    fireEvent.click(within(dialog).getByRole('button', { name: ac.deny }))

    const last = within(dialog).getByRole('button', { name: ac.denySubmit })
    await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true))
    last.focus()
    fireEvent.keyDown(last, { key: 'Tab' })

    expect(dialog.contains(document.activeElement)).toBe(true)
    expect(document.activeElement).toBe(within(dialog).getByRole('button', { name: ac.close }))
  })

  it('Escape on the panel closes it and restores focus to the trigger', async () => {
    const state = makeState([makeApprovalItem()])

    render(<PetActionCenter state={state} />)

    const trigger = screen.getByRole('button', { name: ac.open })
    fireEvent.click(trigger)

    const dialog = screen.getByRole('dialog')
    expect(dialog).not.toBeNull()

    // Escape closes the panel.
    fireEvent.keyDown(dialog, { key: 'Escape' })

    expect(screen.queryByRole('dialog')).toBeNull()
    // Focus is restored to the trigger button.
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('button', { name: ac.open }))
    )
  })
})

describe('PetActionCenter — contract 10: flat layout, one surface, tokens', () => {
  it('uses a single elevated surface (one dialog) and no raw color literals', () => {
    const state = makeState([makeApprovalItem()])

    const { container } = render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    // One dialog.
    expect(screen.getAllByRole('dialog')).toHaveLength(1)

    // No raw hex colors in inline styles.
    const allElements = container.querySelectorAll('*')

    for (const el of allElements) {
      if (el instanceof HTMLElement && el.style.length > 0) {
        for (let i = 0; i < el.style.length; i++) {
          const prop = el.style[i]
          const val = el.style.getPropertyValue(prop)
          expect(val).not.toMatch(/#[0-9a-f]{3,8}/i)
        }
      }
    }
  })
})

describe('PetActionCenter — contract 11: gateway-less control channel', () => {
  it('all controls go through sendPetOverlayControl — no gateway imports', () => {
    const item = makeApprovalItem({
      allowedActions: ['approve-once']
    })

    const state = makeState([item])

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    fireEvent.click(within(dialog).getByRole('button', { name: /run once/i }))

    const controls = emittedControls()
    const approvalControl = controls.find(c => c.type === 'action-center-approval')
    expect(approvalControl).toEqual({
      type: 'action-center-approval',
      itemId: item.id,
      choice: 'approve-once'
    })
  })
})

describe('PetActionCenter — contract 12: i18n strings', () => {
  it('renders user-facing strings from the i18n catalog, not literals', () => {
    const state = makeState([makeApprovalItem()])

    render(<PetActionCenter state={state} />)

    // Trigger button uses i18n string.
    expect(screen.getByRole('button', { name: ac.open })).not.toBeNull()
  })
})

describe('PetActionCenter — stale item handling', () => {
  it('does not crash when the selected item disappears from the list', () => {
    const items = [makeApprovalItem({ id: 'a-1' })]
    const state = makeState(items, { selectedItemId: 'gone-id' })

    // Should not throw.
    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    // Falls back to first item.
    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText('npm test')).not.toBeNull()
  })
})

describe('PetActionCenter — contract 13: secure input count hint', () => {
  it('shows a secure-input count hint when secureInputCount > 0', () => {
    const state = makeState([makeApprovalItem()], { secureInputCount: 2 })

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText(/2 secure inputs need attention/i)).not.toBeNull()
  })
})

describe('PetActionCenter — Pet-6C2 live visibility and status', () => {
  it('renders a collapsed trigger for a working zero-attention item without opening or focusing until click', () => {
    const item = makeLiveTurnItem()

    render(<PetActionCenter state={makeState([item], { attentionCount: 0 })} />)

    const trigger = screen.getByRole('button')
    expect(trigger.textContent).not.toMatch(/pending/i)
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(document.activeElement).not.toBe(trigger)
    expect(setFocusableMock).not.toHaveBeenCalled()

    fireEvent.click(trigger)

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).queryByText(/0 pending actions/i)).toBeNull()
    expect(within(dialog).getByText('Work')).not.toBeNull()
    expect(within(dialog).getByText('Release prep')).not.toBeNull()
  })

  it('shows safe live metadata, queue/connection state, advancing elapsed time, and a stable status announcement', () => {
    vi.useFakeTimers()
    vi.setSystemTime(120_000)
    const clearIntervalSpy = vi.spyOn(globalThis, 'clearInterval')
    const setIntervalSpy = vi.spyOn(globalThis, 'setInterval')

    const item = makeLiveTurnItem({
      activityKind: 'reasoning',
      activityName: 'Reviewing patch',
      connectionState: 'connecting',
      queuedCount: 2,
      turnStartedAt: 62_000
    })

    const { rerender, unmount } = render(<PetActionCenter state={makeState([item], { attentionCount: 0 })} />)
    fireEvent.click(screen.getByRole('button'))

    const strip = screen.getByTestId('live-status-strip')
    expect(within(strip).getByText('Work')).not.toBeNull()
    expect(within(strip).getByText('Release prep')).not.toBeNull()
    expect(within(strip).getByText('Working')).not.toBeNull()
    expect(within(strip).getByText('Reasoning')).not.toBeNull()
    expect(within(strip).getByText('Reviewing patch')).not.toBeNull()
    expect(within(strip).getByText('2 queued')).not.toBeNull()
    expect(within(strip).getByText('Connecting')).not.toBeNull()
    expect(within(strip).getByText('00:58')).not.toBeNull()

    const announcement = within(strip).getByTestId('live-status-announcement')
    const announcedStatus = announcement.textContent

    act(() => vi.advanceTimersByTime(2_000))

    expect(within(strip).getByText('01:00')).not.toBeNull()
    expect(announcement.textContent).toBe(announcedStatus)

    rerender(
      <PetActionCenter
        state={makeState([makeLiveTurnItem({ ...item, turnStartedAt: 125_000 })], { attentionCount: 0 })}
      />
    )
    expect(within(strip).getByText('00:00')).not.toBeNull()

    rerender(
      <PetActionCenter
        state={makeState([makeLiveTurnItem({ ...item, turnStartedAt: Number.NaN })], { attentionCount: 0 })}
      />
    )
    expect(within(strip).getByText('00:00')).not.toBeNull()

    const timerId = setIntervalSpy.mock.results.at(-1)?.value
    unmount()
    expect(clearIntervalSpy).toHaveBeenCalledWith(timerId)

    clearIntervalSpy.mockRestore()
    setIntervalSpy.mockRestore()
    vi.useRealTimers()
  })

  it('renders an attached prompt live status exactly once above the prompt detail', () => {
    const item = makeApprovalItem({
      liveStatus: makeLiveStatus({ activityKind: 'tool', activityName: 'Terminal' })
    })

    render(<PetActionCenter state={makeState([item])} />)
    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getAllByTestId('live-status-strip')).toHaveLength(1)
    const children = Array.from(dialog.children)
    expect(children.indexOf(within(dialog).getByTestId('live-status-strip'))).toBeLessThan(
      children.findIndex(child => child.textContent?.includes('Approval needed'))
    )
  })

  it('uses localized safe fallbacks and never renders runtime or stored session ids', () => {
    const item = makeLiveTurnItem({
      sessionId: 'runtime-do-not-render',
      sessionTitle: null,
      storedSessionId: 'stored-do-not-render'
    })

    const { container, rerender } = render(<PetActionCenter state={makeState([item], { attentionCount: 0 })} />)
    fireEvent.click(screen.getByRole('button'))

    expect(container.textContent).toContain('Untitled session')
    expect(container.textContent).not.toContain('runtime-do-not-render')
    expect(container.textContent).not.toContain('stored-do-not-render')

    rerender(
      <PetActionCenter
        state={makeState([makeLiveTurnItem({ id: 'new', sessionTitle: null, storedSessionId: null })], {
          attentionCount: 0,
          selectedItemId: 'new'
        })}
      />
    )
    expect(container.textContent).toContain('New session')
  })
})

describe('PetActionCenter — Pet-6C2 live actions', () => {
  it('sends an idle draft only when non-empty, keeps Enter IME/Shift-safe, and clears on send success', async () => {
    const item = makeLiveTurnItem({ allowedActions: ['send'], status: 'idle' })

    const base = makeState([item], { attentionCount: 0 })
    const { rerender } = render(<PetActionCenter state={base} />)
    fireEvent.click(screen.getByRole('button'))

    const dialog = screen.getByRole('dialog')
    const input = within(dialog).getByRole<HTMLTextAreaElement>('textbox', { name: /message session/i })
    const sendButton = within(dialog).getByRole<HTMLButtonElement>('button', { name: /^send$/i })

    expect(sendButton.disabled).toBe(true)
    fireEvent.change(input, { target: { value: '   ' } })
    expect(sendButton.disabled).toBe(true)

    fireEvent.change(input, { target: { value: '  ship it  ' } })
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
    fireEvent(
      input,
      new KeyboardEvent('keydown', { bubbles: true, cancelable: true, isComposing: true, key: 'Enter' })
    )
    expect(emittedControls()).toHaveLength(0)

    fireEvent.keyDown(input, { key: 'Enter' })
    expect(emittedControls()).toEqual([
      { type: 'action-center-submit', itemId: item.id, text: 'ship it' }
    ])

    controlMock.mockClear()
    fireEvent.click(sendButton)
    expect(emittedControls()).toEqual([
      { type: 'action-center-submit', itemId: item.id, text: 'ship it' }
    ])

    rerender(<PetActionCenter state={{ ...base, action: { status: 'success', itemId: item.id } }} />)
    await waitFor(() => expect(input.value).toBe(''))
  })

  it('offers explicit online Steer, Queue, and Stop controls and emits exact opaque controls', () => {
    const item = makeLiveTurnItem()

    const base = makeState([item], { attentionCount: 0 })
    const { rerender } = render(<PetActionCenter state={base} />)
    fireEvent.click(screen.getByRole('button'))
    const dialog = screen.getByRole('dialog')
    const input = within(dialog).getByRole<HTMLTextAreaElement>('textbox', { name: /message session/i })
    fireEvent.change(input, { target: { value: '  redirect  ' } })

    fireEvent.keyDown(input, { key: 'Enter' })
    fireEvent.click(within(dialog).getByRole('button', { name: /^queue$/i }))
    fireEvent.click(within(dialog).getByRole('button', { name: /^stop$/i }))

    expect(emittedControls()).toEqual([
      { type: 'action-center-steer', itemId: item.id, text: 'redirect' },
      { type: 'action-center-queue', itemId: item.id, text: 'redirect' },
      { type: 'action-center-stop', itemId: item.id }
    ])
    expect(JSON.stringify(emittedControls())).not.toContain(item.profile)
    expect(JSON.stringify(emittedControls())).not.toContain(item.sessionId)

    rerender(<PetActionCenter state={{ ...base, action: { status: 'stopped', itemId: item.id } }} />)
    expect(input.value).toBe('  redirect  ')
  })

  it('makes offline queue-only behavior explicit and Enter queues without exposing Steer or Stop', () => {
    const item = makeLiveTurnItem({
      allowedActions: ['queue'],
      connectionState: 'closed'
    })

    render(<PetActionCenter state={makeState([item], { attentionCount: 0 })} />)
    fireEvent.click(screen.getByRole('button'))
    const dialog = screen.getByRole('dialog')

    expect(within(dialog).queryByRole('button', { name: /^steer$/i })).toBeNull()
    expect(within(dialog).queryByRole('button', { name: /^stop$/i })).toBeNull()
    expect(within(dialog).getByRole('button', { name: /^queue$/i })).not.toBeNull()

    const input = within(dialog).getByRole('textbox', { name: /queue message/i })
    fireEvent.change(input, { target: { value: '  later  ' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(emittedControls()).toEqual([
      { type: 'action-center-queue', itemId: item.id, text: 'later' }
    ])
  })

  it('retains a rejected steer draft and queues only after the explicit Queue click', async () => {
    const item = makeLiveTurnItem()
    const base = makeState([item], { attentionCount: 0 })
    const { rerender } = render(<PetActionCenter state={base} />)
    fireEvent.click(screen.getByRole('button'))
    const input = screen.getByRole<HTMLInputElement>('textbox', { name: /message session/i })
    fireEvent.change(input, { target: { value: 'keep this draft' } })

    rerender(
      <PetActionCenter
        state={{ ...base, action: { status: 'steer-rejected', itemId: item.id } }}
      />
    )

    expect(input.value).toBe('keep this draft')
    expect(screen.getByText(/could not be steered/i)).not.toBeNull()
    expect(emittedControls()).toHaveLength(0)

    fireEvent.click(screen.getByRole('button', { name: /queue this message/i }))
    expect(emittedControls()).toEqual([
      { type: 'action-center-queue', itemId: item.id, text: 'keep this draft' }
    ])

    rerender(<PetActionCenter state={{ ...base, action: { status: 'queued', itemId: item.id } }} />)
    await waitFor(() => expect(input.value).toBe(''))
  })

  it('clears only a matching successful draft and ignores a stale terminal action from another item', async () => {
    const first = makeLiveTurnItem({ id: 'live-first' })
    const second = makeLiveTurnItem({ id: 'live-second', sessionTitle: 'Second session' })

    const { rerender } = render(
      <PetActionCenter state={makeState([first, second], { attentionCount: 0, selectedItemId: first.id })} />
    )

    fireEvent.click(screen.getByRole('button'))
    let input = screen.getByRole<HTMLInputElement>('textbox', { name: /message session/i })
    fireEvent.change(input, { target: { value: 'first draft' } })
    fireEvent.click(screen.getByRole('button', { name: /^steer$/i }))

    rerender(
      <PetActionCenter
        state={makeState([first, second], {
          action: { status: 'steered', itemId: first.id },
          attentionCount: 0,
          selectedItemId: first.id
        })}
      />
    )
    await waitFor(() => expect(input.value).toBe(''))

    fireEvent.change(input, { target: { value: 'keep after opening' } })
    fireEvent.click(screen.getByRole('button', { name: /open in hermes/i }))
    rerender(
      <PetActionCenter
        state={makeState([first, second], {
          action: { status: 'success', itemId: first.id },
          attentionCount: 0,
          selectedItemId: first.id
        })}
      />
    )
    expect(input.value).toBe('keep after opening')

    rerender(
      <PetActionCenter
        state={makeState([first, second], {
          action: { status: 'steered', itemId: first.id },
          attentionCount: 0,
          selectedItemId: second.id
        })}
      />
    )
    input = screen.getByRole<HTMLInputElement>('textbox', { name: /message session/i })
    fireEvent.change(input, { target: { value: 'second draft' } })
    rerender(
      <PetActionCenter
        state={makeState([first, second], {
          action: { status: 'queued', itemId: first.id },
          attentionCount: 0,
          selectedItemId: second.id
        })}
      />
    )

    expect(input.value).toBe('second draft')
  })

  it('renders no live text or stop controls while waiting and acknowledges done/failed exactly', () => {
    const waiting = makeLiveTurnItem({
      allowedActions: ['steer', 'queue', 'stop'],
      status: 'waiting'
    })

    const { rerender } = render(<PetActionCenter state={makeState([waiting])} />)
    fireEvent.click(screen.getByRole('button'))

    expect(screen.queryByRole('textbox', { name: /message session/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /^steer$/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /^queue$/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /^stop$/i })).toBeNull()

    const done = makeLiveTurnItem({ id: 'done', allowedActions: ['acknowledge'], status: 'done' })
    rerender(<PetActionCenter state={makeState([done], { selectedItemId: done.id })} />)
    fireEvent.click(screen.getByRole('button', { name: /acknowledge|dismiss/i }))
    expect(emittedControls().at(-1)).toEqual({ type: 'action-center-acknowledge', itemId: done.id })

    const failed = makeLiveTurnItem({ id: 'failed', allowedActions: ['acknowledge'], status: 'failed' })
    rerender(<PetActionCenter state={makeState([failed], { selectedItemId: failed.id })} />)
    fireEvent.click(screen.getByRole('button', { name: /acknowledge|dismiss/i }))
    expect(emittedControls().at(-1)).toEqual({ type: 'action-center-acknowledge', itemId: failed.id })
  })

  it('renders Open in Hermes for every capable item kind and emits only the opaque item id', () => {
    const items = [
      makeApprovalItem({ id: 'approval-open', allowedActions: ['approve-once', 'open-in-app'] }),
      makeClarifyItem({ id: 'clarify-open', allowedActions: ['clarify-respond', 'open-in-app'] }),
      makeLiveTurnItem({ id: 'live-open', allowedActions: ['steer', 'open-in-app'] })
    ]

    const { rerender } = render(
      <PetActionCenter state={makeState([items[0]!], { selectedItemId: items[0]!.id })} />
    )

    fireEvent.click(screen.getByRole('button', { name: ac.open }))

    for (const item of items) {
      rerender(<PetActionCenter state={makeState([item], { selectedItemId: item.id })} />)
      fireEvent.click(screen.getByRole('button', { name: /open in hermes/i }))
    }

    expect(emittedControls()).toEqual(
      items.map(item => ({ type: 'action-center-open-session', itemId: item.id }))
    )
  })

  it('renders only allowed live actions and the global submitting mutex disables input and buttons', () => {
    const item = makeLiveTurnItem({ allowedActions: ['queue', 'open-in-app'] })

    const state = makeState([item], {
      action: { status: 'submitting', itemId: 'some-other-item' },
      attentionCount: 0
    })

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button'))
    const dialog = screen.getByRole('dialog')

    expect(within(dialog).queryByRole('button', { name: /^steer$/i })).toBeNull()
    expect(within(dialog).queryByRole('button', { name: /^stop$/i })).toBeNull()
    expect(within(dialog).getByRole<HTMLButtonElement>('button', { name: /^queue$/i }).disabled).toBe(true)
    expect(within(dialog).getByRole<HTMLButtonElement>('button', { name: /open in hermes/i }).disabled).toBe(true)
    expect(within(dialog).getByRole<HTMLTextAreaElement>('textbox', { name: /queue message/i }).disabled).toBe(true)
  })
})

describe('PetActionCenter — Pet-6C2 live feedback', () => {
  it.each([
    ['success', 'Done'],
    ['stale', 'No longer pending'],
    ['steered', 'Instruction sent'],
    ['queued', 'Message queued'],
    ['stopped', 'Session stopped'],
    ['acknowledged', 'Dismissed']
  ] as const)('announces %s even after the acted-on item is removed', (status, message) => {
    const item = makeLiveTurnItem()
    const { rerender } = render(<PetActionCenter state={makeState([item], { attentionCount: 0 })} />)
    fireEvent.click(screen.getByRole('button'))

    rerender(
      <PetActionCenter
        state={makeState([], {
          action: { status, itemId: item.id },
          attentionCount: 0
        })}
      />
    )

    expect(screen.getByRole('status').textContent).toContain(message)
  })

  it('keeps steer rejection and generic errors attached to the selected item', () => {
    const first = makeLiveTurnItem({ id: 'first' })
    const second = makeLiveTurnItem({ id: 'second' })

    const { rerender } = render(
      <PetActionCenter
        state={makeState([first, second], {
          action: { status: 'steer-rejected', itemId: first.id },
          attentionCount: 0,
          selectedItemId: second.id
        })}
      />
    )

    fireEvent.click(screen.getByRole('button'))

    expect(screen.queryByText(/could not be steered/i)).toBeNull()

    rerender(
      <PetActionCenter
        state={makeState([first, second], {
          action: { status: 'steer-rejected', itemId: second.id },
          attentionCount: 0,
          selectedItemId: second.id
        })}
      />
    )
    expect(screen.getByRole('status').textContent).toMatch(/could not be steered/i)

    rerender(
      <PetActionCenter
        state={makeState([first, second], {
          action: { status: 'error', itemId: first.id, errorCode: 'rpc-failed' },
          attentionCount: 0,
          selectedItemId: second.id
        })}
      />
    )
    expect(screen.queryByText(/something went wrong/i)).toBeNull()
  })
})
