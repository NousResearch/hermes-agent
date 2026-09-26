// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'

import { type GuardedModelSwitchResult, surfaceModelSwitchConfirm } from './guarded-model-switch'

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

afterEach(cleanup)

/** Shaped like the gateway's: blank lines separate the guard's paragraphs. */
const GUARD_MESSAGE =
  'This session holds ~206,172 tokens of context.\nSwitching re-reads it.\n\nContext window shrinks (1,000,000 → 256,000).'

type SwitchResult = GuardedModelSwitchResult & { value: string }

function ask() {
  const requestConfirmed = vi.fn(async (): Promise<SwitchResult> => ({ value: 'perplexity/sonar-deep-research' }))

  const answer = surfaceModelSwitchConfirm({
    confirmMessage: GUARD_MESSAGE,
    failureMessage: 'Model switch failed',
    model: 'perplexity/sonar-deep-research',
    requestConfirmed
  })

  return { answer, requestConfirmed }
}

// The applier through the real `confirm()` atom, the real ConfirmHost mount
// and the real ConfirmDialog, so the choices a user actually sees are
// asserted (#112458: the toast it replaced had one "Confirm" and an ✕).
describe('surfaceModelSwitchConfirm through the shell dialog', () => {
  it('offers a decline that keeps the current model and keeps the guard paragraphs', async () => {
    render(<ConfirmHost />)

    const { answer, requestConfirmed } = ask()

    expect(await screen.findByRole('dialog')).toBeTruthy()

    // The gateway's line breaks survive: the description is not one run-on line.
    const description = screen.getByText(/Context window shrinks/)

    expect(description.textContent).toBe(GUARD_MESSAGE)

    fireEvent.click(screen.getByRole('button', { name: 'Keep current model' }))

    await expect(answer).resolves.toBe(false)
    expect(requestConfirmed).not.toHaveBeenCalled()
  })

  it('resends with confirmation only after "Switch anyway"', async () => {
    render(<ConfirmHost />)

    const { answer, requestConfirmed } = ask()

    fireEvent.click(await screen.findByRole('button', { name: 'Switch anyway' }))

    await expect(answer).resolves.toBe(true)
    expect(requestConfirmed).toHaveBeenCalledTimes(1)
  })
})
