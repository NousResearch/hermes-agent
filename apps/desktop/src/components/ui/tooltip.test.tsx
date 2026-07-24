import { describe, expect, it, vi } from 'vitest'

// Radix opens tooltips on ANY trigger focus; menus/dialogs restore focus to
// their trigger on close, which left tips stuck open after a mouse pick (e.g.
// the composer model pill). The trigger's focus handler must mark the tooltip
// to suppress its next open for non-keyboard focus only.
//
// IMPORTANT: We cannot use event.preventDefault() on the focus event because
// in Electron/Chromium, preventDefault() on a focus event can prevent the
// subsequent click event from firing on the same element (#66854). Instead,
// we use a context-based approach where the trigger sets a ref flag and the
// Tooltip Root's onOpenChange handler checks it before opening.

const focusEvent = (matchesImpl: (selector: string) => boolean) => {
  return {
    event: {
      currentTarget: { matches: matchesImpl } as unknown as HTMLElement
    } as unknown as React.FocusEvent<HTMLElement>
  }
}

describe('Tooltip mouse-focus suppression', () => {
  it('marks suppressNextOpen when focus is not keyboard-visible (menu close restore)', () => {
    const suppressNextOpen = { current: false }
    const ctx = { suppressNextOpen }
    const { event } = focusEvent(selector => selector !== ':focus-visible')

    // Simulate the TooltipTrigger onFocus handler logic
    try {
      if (!event.currentTarget.matches(':focus-visible')) {
        ctx.suppressNextOpen.current = true
      }
    } catch {
      // Selector unsupported — don't suppress
    }

    expect(suppressNextOpen.current).toBe(true)
  })

  it('does not mark suppressNextOpen for keyboard (Tab) focus — a11y path', () => {
    const suppressNextOpen = { current: false }
    const ctx = { suppressNextOpen }
    const { event } = focusEvent(selector => selector === ':focus-visible')

    // Simulate the TooltipTrigger onFocus handler logic
    try {
      if (!event.currentTarget.matches(':focus-visible')) {
        ctx.suppressNextOpen.current = true
      }
    } catch {
      // Selector unsupported — don't suppress
    }

    expect(suppressNextOpen.current).toBe(false)
  })

  it('fails open when :focus-visible is unsupported', () => {
    const suppressNextOpen = { current: false }
    const ctx = { suppressNextOpen }
    const { event } = focusEvent(() => {
      throw new Error('unsupported selector')
    })

    // Simulate the TooltipTrigger onFocus handler logic
    try {
      if (!event.currentTarget.matches(':focus-visible')) {
        ctx.suppressNextOpen.current = true
      }
    } catch {
      // Selector unsupported — don't suppress
    }

    expect(suppressNextOpen.current).toBe(false)
  })

  it('onOpenChange suppresses open when suppressNextOpen is true', () => {
    const suppressNextOpen = { current: true }
    const onOpenChange = vi.fn()

    // Simulate the Tooltip Root onOpenChange handler logic
    const handleOpenChange = (open: boolean) => {
      if (suppressNextOpen.current && open) {
        suppressNextOpen.current = false
        return // Suppress the open
      }
      suppressNextOpen.current = false
      onOpenChange?.(open)
    }

    handleOpenChange(true)

    expect(suppressNextOpen.current).toBe(false)
    expect(onOpenChange).not.toHaveBeenCalled()
  })

  it('onOpenChange allows open when suppressNextOpen is false', () => {
    const suppressNextOpen = { current: false }
    const onOpenChange = vi.fn()

    // Simulate the Tooltip Root onOpenChange handler logic
    const handleOpenChange = (open: boolean) => {
      if (suppressNextOpen.current && open) {
        suppressNextOpen.current = false
        return // Suppress the open
      }
      suppressNextOpen.current = false
      onOpenChange?.(open)
    }

    handleOpenChange(true)

    expect(suppressNextOpen.current).toBe(false)
    expect(onOpenChange).toHaveBeenCalledWith(true)
  })
})
