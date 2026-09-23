import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { FanMenu } from './fan-menu'

vi.mock('./tooltip', () => ({
  Tip: ({ children, side }: { children: ReactNode; side: string }) => (
    <span data-testid="tip" data-tip-side={side}>
      {children}
    </span>
  )
}))

afterEach(cleanup)

describe('FanMenu hub tooltip', () => {
  it.each([
    ['vertical', 'left'],
    ['horizontal', 'top'],
    ['arc', 'bottom']
  ] as const)('keeps the %s hub tooltip outside its open fan', (direction, side) => {
    render(
      <FanMenu
        direction={direction}
        hub={{ id: 'hub', icon: 'H', label: 'Hub', onSelect: vi.fn() }}
        items={[{ id: 'option', icon: 'O', label: 'Option', onSelect: vi.fn() }]}
        label="Fan"
      />
    )

    const hub = screen.getByRole('button', { name: 'Hub' })

    expect(hub.closest('[data-tip-side]')?.getAttribute('data-tip-side')).toBe('top')
    fireEvent.pointerEnter(hub)
    expect(hub.closest('[data-tip-side]')?.getAttribute('data-tip-side')).toBe(side)
  })

  it('keeps an explicit side override when the fan opens', () => {
    render(
      <FanMenu
        direction="arc"
        hub={{ id: 'hub', icon: 'H', label: 'Hub', onSelect: vi.fn() }}
        items={[]}
        label="Fan"
        tipAnchor="right"
      />
    )

    const hub = screen.getByRole('button', { name: 'Hub' })

    fireEvent.pointerEnter(hub)
    expect(hub.closest('[data-tip-side]')?.getAttribute('data-tip-side')).toBe('right')
  })
})
