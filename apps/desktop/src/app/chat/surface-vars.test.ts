import { describe, expect, it } from 'vitest'

import { chatSurfaceRoot, clearSurfaceVar, setSurfaceVar, STATUS_STACK_VAR } from './surface-vars'

/**
 * Measured-height vars must be scoped per chat surface. Session tiles render a
 * full ChatView — thread, composer, and status stack — beside the workspace
 * pane, so a single global slot let a background tab's tall status stack
 * inflate the foreground thread's bottom clearance and shove its
 * jump-to-bottom button into mid-screen.
 */
function surface(): { root: HTMLElement; stack: HTMLElement } {
  const root = document.createElement('div')
  root.setAttribute('data-chat-surface', '')

  const stack = document.createElement('div')
  root.append(stack)
  document.body.append(root)

  return { root, stack }
}

describe('per-surface measured-height vars', () => {
  it('publishes onto the owning chat surface, not the document root', () => {
    const { root, stack } = surface()

    setSurfaceVar(stack, STATUS_STACK_VAR, '96px')

    expect(root.style.getPropertyValue(STATUS_STACK_VAR)).toBe('96px')
    expect(document.documentElement.style.getPropertyValue(STATUS_STACK_VAR)).toBe('')
  })

  it('keeps two visible surfaces independent', () => {
    const background = surface()
    const foreground = surface()

    setSurfaceVar(background.stack, STATUS_STACK_VAR, '160px')
    setSurfaceVar(foreground.stack, STATUS_STACK_VAR, '0px')

    expect(background.root.style.getPropertyValue(STATUS_STACK_VAR)).toBe('160px')
    expect(foreground.root.style.getPropertyValue(STATUS_STACK_VAR)).toBe('0px')
  })

  it('clearing one surface leaves the other intact', () => {
    const a = surface()
    const b = surface()

    setSurfaceVar(a.stack, STATUS_STACK_VAR, '48px')
    setSurfaceVar(b.stack, STATUS_STACK_VAR, '80px')
    clearSurfaceVar(a.stack, STATUS_STACK_VAR)

    expect(a.root.style.getPropertyValue(STATUS_STACK_VAR)).toBe('')
    expect(b.root.style.getPropertyValue(STATUS_STACK_VAR)).toBe('80px')
  })

  it('falls back to the document root outside a chat surface (popped-out composer)', () => {
    const orphan = document.createElement('div')
    document.body.append(orphan)

    expect(chatSurfaceRoot(orphan)).toBe(document.documentElement)

    setSurfaceVar(orphan, STATUS_STACK_VAR, '24px')
    expect(document.documentElement.style.getPropertyValue(STATUS_STACK_VAR)).toBe('24px')

    clearSurfaceVar(orphan, STATUS_STACK_VAR)
    expect(document.documentElement.style.getPropertyValue(STATUS_STACK_VAR)).toBe('')
  })
})
