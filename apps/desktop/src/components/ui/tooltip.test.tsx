import { describe, expect, it, vi } from 'vitest'

import { suppressNonKeyboardFocusOpen } from './tooltip'

// Radix opens tooltips on ANY trigger focus; menus/dialogs restore focus to
// their trigger on close, which left tips stuck open after a mouse pick (e.g.
// the composer model pill). The trigger's focus handler must preventDefault —
// which Radix's composed handler honors — for non-keyboard focus only.

const focusEvent = (matchesImpl: (selector: string) => boolean) => {
  const preventDefault = vi.fn()

  return {
    event: {
      currentTarget: { matches: matchesImpl } as unknown as HTMLElement,
      preventDefault
    } as unknown as React.FocusEvent<HTMLElement>,
    preventDefault
  }
}

describe('suppressNonKeyboardFocusOpen', () => {
  it('suppresses the focus-open when focus is not keyboard-visible (menu close restore)', () => {
    const { event, preventDefault } = focusEvent(selector => selector !== ':focus-visible')

    suppressNonKeyboardFocusOpen(event)

    expect(preventDefault).toHaveBeenCalledOnce()
  })

  it('keeps the focus-open for keyboard (Tab) focus — a11y path', () => {
    const { event, preventDefault } = focusEvent(selector => selector === ':focus-visible')

    suppressNonKeyboardFocusOpen(event)

    expect(preventDefault).not.toHaveBeenCalled()
  })

  it('fails open when :focus-visible is unsupported', () => {
    const { event, preventDefault } = focusEvent(() => {
      throw new Error('unsupported selector')
    })

    suppressNonKeyboardFocusOpen(event)

    expect(preventDefault).not.toHaveBeenCalled()
  })
})
