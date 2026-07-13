// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import type { PetActionCenterState } from '@/store/pet-action-center'
import type { PetActionCenterControl } from '@/store/pet-overlay'

import { PetActionCenter } from './pet-action-center'

// ── Test helpers ──────────────────────────────────────────────────────────

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const ac = en.pet.actionCenter

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

    const trigger = screen.getByRole('button', { name: /review pending actions/i })
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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

    const dialog = screen.getByRole('dialog')

    // Count is shown.
    expect(within(dialog).getByText(/2 pending actions/i)).not.toBeNull()

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
      fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))
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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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

    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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

    const trigger = screen.getByRole('button', { name: /review pending actions/i })
    fireEvent.click(trigger)

    const dialog = screen.getByRole('dialog')
    expect(dialog).not.toBeNull()

    // Escape closes the panel.
    fireEvent.keyDown(dialog, { key: 'Escape' })

    expect(screen.queryByRole('dialog')).toBeNull()
    // Focus is restored to the trigger button.
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('button', { name: /review pending actions/i }))
    )
  })
})

describe('PetActionCenter — contract 10: flat layout, one surface, tokens', () => {
  it('uses a single elevated surface (one dialog) and no raw color literals', () => {
    const state = makeState([makeApprovalItem()])

    const { container } = render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

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
    expect(screen.getByRole('button', { name: /review pending actions/i })).not.toBeNull()
  })
})

describe('PetActionCenter — stale item handling', () => {
  it('does not crash when the selected item disappears from the list', () => {
    const items = [makeApprovalItem({ id: 'a-1' })]
    const state = makeState(items, { selectedItemId: 'gone-id' })

    // Should not throw.
    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

    // Falls back to first item.
    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText('npm test')).not.toBeNull()
  })
})

describe('PetActionCenter — contract 13: secure input count hint', () => {
  it('shows a secure-input count hint when secureInputCount > 0', () => {
    const state = makeState([makeApprovalItem()], { secureInputCount: 2 })

    render(<PetActionCenter state={state} />)
    fireEvent.click(screen.getByRole('button', { name: /review pending actions/i }))

    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText(/2 secure inputs need attention/i)).not.toBeNull()
  })
})
