import { render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $sessions } from '@/store/session'
import { $sessionColorOverrides, $sessionTileColors } from '@/store/session-color'

import { makeCwdSession } from '../../test/session-info'

import { SessionTabTint } from './session-tab-tint'

afterEach(() => {
  $sessions.set([])
  $sessionColorOverrides.set({})
  $sessionTileColors.set({})
})

/** The wrapper the painter owns — `display: contents`, so it paints nothing
 *  itself and only carries the custom properties the tab inherits. */
function wrapperOf(container: HTMLElement): HTMLElement | null {
  return container.querySelector<HTMLElement>('span.contents')
}

describe('SessionTabTint', () => {
  it('paints the tab through the three surface vars PaneTab reads', () => {
    const a = makeCwdSession('/x')

    $sessionTileColors.set({ [a.id]: '#4a9eff' })

    const { container } = render(
      <SessionTabTint session={a} storedSessionId={a.id}>
        <div data-slot="pane-tab" />
      </SessionTabTint>
    )

    const wrapper = wrapperOf(container)

    expect(wrapper).not.toBeNull()
    expect(wrapper?.style.getPropertyValue('--pane-tab-active-accent')).toBe('#4a9eff')
    expect(wrapper?.style.getPropertyValue('--pane-tab-strip-bg')).toContain('color-mix(in srgb, #4a9eff 12%')
    expect(wrapper?.style.getPropertyValue('--pane-tab-active-bg')).toContain('color-mix(in srgb, #4a9eff 20%')
    // The tab itself is untouched and still rendered.
    expect(container.querySelector('[data-slot="pane-tab"]')).not.toBeNull()
  })

  it('adds no wrapper at all when the session has no colour', () => {
    const a = makeCwdSession('/x')

    const { container } = render(
      <SessionTabTint session={a} storedSessionId={a.id}>
        <div data-slot="pane-tab" />
      </SessionTabTint>
    )

    expect(wrapperOf(container)).toBeNull()
    expect(container.querySelector('[data-slot="pane-tab"]')).not.toBeNull()
  })

  it('resolves the row from the stored id when the caller holds no row', () => {
    const a = makeCwdSession('/x')

    $sessions.set([a])
    $sessionTileColors.set({ [a.id]: '#7bc86c' })

    const { container } = render(
      <SessionTabTint storedSessionId={a.id}>
        <div data-slot="pane-tab" />
      </SessionTabTint>
    )

    expect(wrapperOf(container)?.style.getPropertyValue('--pane-tab-active-accent')).toBe('#7bc86c')
  })

  it('paints NOTHING for a colour it cannot put in color-mix()', () => {
    // A malformed custom property is invalid at computed-value time, which would
    // take the tab's surface and seam with it — so the guard refuses the paint
    // rather than trusting whatever reached the store.
    const a = makeCwdSession('/x')

    $sessionTileColors.set({ [a.id]: 'not-a-colour' })

    const { container } = render(
      <SessionTabTint session={a} storedSessionId={a.id}>
        <div data-slot="pane-tab" />
      </SessionTabTint>
    )

    expect(wrapperOf(container)).toBeNull()
  })
})
