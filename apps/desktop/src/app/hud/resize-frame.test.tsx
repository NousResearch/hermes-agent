// @vitest-environment jsdom
import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('../contrib/wiring', () => ({ WiredPane: () => null }))

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
Object.assign(globalThis, { ResizeObserver: ResizeObserverStub })

import { HudShell } from './hud-shell'

/** #108793: the resize frame's hit-testability rides the shell's engagement
 * gates (styles.css keys `pointer-events` off them). The contract under test:
 * an idle HUD carries NONE of the gates, so the invisible frame falls out of
 * the hit test entirely; each arm — caret in the composer, a held band, a
 * solid-input host — re-arms it on its own. */
describe('HudShell resize-frame engagement gates', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  const shellOf = (container: HTMLElement) => {
    const shell = container.querySelector('[data-hud-shell]')

    expect(shell).not.toBeNull()

    return shell as HTMLElement
  }

  it('an idle HUD carries none of the engagement gates the frame keys on', () => {
    const { container } = render(
      <MemoryRouter>
        <HudShell />
      </MemoryRouter>
    )
    const shell = shellOf(container)

    expect(shell.hasAttribute('data-hud-typing')).toBe(false)
    expect(shell.hasAttribute('data-hud-held')).toBe(false)
    expect(shell.getAttribute('data-hud-input')).toBe('click-through')
    // The frame itself is mounted (the resize grammar is not conditional) but
    // no handle is mid-drag.
    const handles = shell.querySelectorAll('[data-hud-resize]')

    expect(handles.length).toBeGreaterThan(0)
    handles.forEach(handle => {
      expect(handle.hasAttribute('data-hud-grabbing')).toBe(false)
    })
  })

  it('focus in the composer arms the typing gate the frame re-arms on', () => {
    const { container } = render(
      <MemoryRouter>
        <HudShell />
      </MemoryRouter>
    )
    const shell = shellOf(container)

    // useHudGlass stamps data-hud-typing live from the :focus query — the
    // same gate the frost and the scrim run on.
    const composer = document.createElement('input')

    composer.setAttribute('data-slot', 'composer-rich-input')
    act(() => {
      shell.appendChild(composer)
      composer.focus()
      composer.dispatchEvent(new FocusEvent('focusin', { bubbles: true }))
    })

    expect(shell.hasAttribute('data-hud-typing')).toBe(true)
  })

  it('a solid-input host keeps the frame live unconditionally', () => {
    Object.assign(window, { hermesDesktop: { hud: { windowing: { solid: true } } } })

    const { container } = render(
      <MemoryRouter>
        <HudShell />
      </MemoryRouter>
    )

    expect(shellOf(container).getAttribute('data-hud-input')).toBe('solid')
  })
})
