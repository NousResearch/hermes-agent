import { fireEvent, render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { SendHoldChip } from './send-hold-chip'

/**
 * The visible half of the send grace window. A hold with no indicator is
 * indistinguishable from the app hanging, so these pin the three things that
 * make it honest: it says what is happening, it can be taken back with a
 * mouse, and it can never re-submit the draft it is holding.
 */
describe('SendHoldChip', () => {
  it('names what is about to happen and how to stop it', () => {
    const { getByText } = render(<SendHoldChip label="Sending… Esc to cancel" onCancel={vi.fn()} />)

    expect(getByText('Sending… Esc to cancel')).toBeTruthy()
  })

  it('announces itself, because nothing else signals a withheld send', () => {
    const { getByRole } = render(<SendHoldChip label="Sending… Esc to cancel" onCancel={vi.fn()} />)

    expect(getByRole('status')).toBeTruthy()
  })

  it('is the mouse path to the same cancel Esc performs', () => {
    const onCancel = vi.fn()
    const { getByRole } = render(<SendHoldChip label="Sending… Esc to cancel" onCancel={onCancel} />)

    fireEvent.click(getByRole('button'))

    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('is a plain button, not a submit — clicking it must not re-submit the draft', () => {
    const { getByRole } = render(<SendHoldChip label="Sending… Esc to cancel" onCancel={vi.fn()} />)

    // A button inside a form defaults to type="submit", which would send the
    // very draft the user is trying to take back.
    expect(getByRole('button').getAttribute('type')).toBe('button')
  })
})
