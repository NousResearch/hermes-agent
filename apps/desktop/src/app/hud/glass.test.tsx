import { act, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useHudGlass } from './glass'

const setFrost = vi.fn()

/** The HUD's real shape as far as the frost is concerned: the shell, the
 *  composer input that owns the caret, and — when it is open — the completion
 *  drawer that takes the surface over. */
function Harness({ backing, drawer }: { backing: boolean; drawer?: boolean }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useHudGlass(ref, backing)

  return (
    <div data-hud-shell ref={ref}>
      <input data-slot="composer-rich-input" />
      {drawer && <div data-slot="composer-completion-drawer" />}
    </div>
  )
}

const frostState = () => setFrost.mock.calls.at(-1)?.[0]

const nextFrame = () => act(() => new Promise(resolve => requestAnimationFrame(() => resolve(undefined))))

afterEach(() => {
  setFrost.mockClear()
})

Object.assign(window, { hermesDesktop: { hud: { setFrost } } })

describe('useHudGlass', () => {
  // The bug this replaced: the caller widened the gate to "recent or held", so
  // a turn merely running raised the window's material while the scrim that is
  // supposed to be painted over it — focus-gated in styles.css — stayed down.
  // On a light theme that is a white slab under the band's white ink.
  it('leaves the window bare while a turn runs with the composer unfocused', () => {
    render(<Harness backing />)

    expect(frostState()).toBe(false)
  })

  it('frosts once the caret is in the composer, and lets go when it leaves', () => {
    const { container } = render(<Harness backing />)
    const input = container.querySelector('input')!

    act(() => input.focus())
    expect(frostState()).toBe(true)

    act(() => input.blur())
    expect(frostState()).toBe(false)
  })

  it('stays off while the band is not covering the window, however engaged', () => {
    const { container } = render(<Harness backing={false} />)

    act(() => container.querySelector('input')!.focus())

    expect(frostState()).toBe(false)
  })

  it('drops the frost when a completion drawer takes the surface over', async () => {
    const { container, rerender } = render(<Harness backing />)

    act(() => container.querySelector('input')!.focus())
    expect(frostState()).toBe(true)

    rerender(<Harness backing drawer />)
    await nextFrame()

    expect(frostState()).toBe(false)
  })
})
