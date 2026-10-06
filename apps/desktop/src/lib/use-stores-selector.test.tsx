/**
 * `useStoresSelector` — the multi-store form of `useStoreSelector`.
 *
 * It exists because subscribing to one store while the selector reads another
 * looks correct for exactly as long as the subscribed store happens to churn
 * on its own. The composer's bot-chat flag was written that way and rode
 * `$sessionStates`' per-token republish to stay roughly right.
 */

import { render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { useStoresSelector } from './use-session-slice'

describe('a selector recomputes for every store it was given', () => {
  it('follows a change in any of them', () => {
    const left = atom(0)
    const right = atom(0)

    function Probe() {
      const total = useStoresSelector([left, right], () => left.get() + right.get())

      return <span data-testid="total">{total}</span>
    }

    render(<Probe />)

    expect(screen.getByTestId('total').textContent).toBe('0')

    act(() => left.set(2))

    expect(screen.getByTestId('total').textContent).toBe('2')

    // The store a single-subscription version would have missed.
    act(() => right.set(5))

    expect(screen.getByTestId('total').textContent).toBe('7')
  })

  it('re-renders only when the derived scalar actually changes', () => {
    const source = atom(1)
    const renders = vi.fn()

    function Probe() {
      const even = useStoresSelector([source], () => source.get() % 2 === 0)
      renders()

      return <span data-testid="even">{String(even)}</span>
    }

    render(<Probe />)
    const baseline = renders.mock.calls.length

    // 1 -> 3: the store moved, the scalar did not.
    act(() => source.set(3))

    expect(renders.mock.calls.length).toBe(baseline)

    act(() => source.set(4))

    expect(renders.mock.calls.length).toBeGreaterThan(baseline)
    expect(screen.getByTestId('even').textContent).toBe('true')
  })

  it('keeps its subscriptions across renders that pass a fresh array literal', () => {
    const source = atom(0)
    const listen = vi.spyOn(source, 'listen')

    function Probe({ label }: { label: string }) {
      const value = useStoresSelector([source], () => source.get())

      return (
        <span data-testid="value">
          {label}
          {value}
        </span>
      )
    }

    const view = render(<Probe label="a" />)
    const initial = listen.mock.calls.length

    view.rerender(<Probe label="b" />)
    view.rerender(<Probe label="c" />)

    expect(listen.mock.calls.length).toBe(initial)
  })

  it('drops every subscription on unmount', () => {
    const left = atom(0)
    const right = atom(0)

    function Probe() {
      return <span>{useStoresSelector([left, right], () => left.get() + right.get())}</span>
    }

    const view = render(<Probe />)

    expect(left.lc).toBeGreaterThan(0)
    expect(right.lc).toBeGreaterThan(0)

    view.unmount()

    expect(left.lc).toBe(0)
    expect(right.lc).toBe(0)
  })
})
