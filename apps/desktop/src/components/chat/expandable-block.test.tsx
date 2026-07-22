import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ExpandableBlock } from './expandable-block'

class TestResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {}

  observe(target: Element) {
    Object.defineProperty(target, 'scrollHeight', { configurable: true, value: 200 })
    this.callback([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
  }

  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

describe('ExpandableBlock', () => {
  afterEach(cleanup)

  it('keeps the bottom fade click-through while limiting the toggle hit area', async () => {
    render(
      <ExpandableBlock>
        <code>{['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'].join('\n')}</code>
      </ExpandableBlock>
    )

    const toggle = await screen.findByRole('button', { name: 'Expand' })
    const fade = toggle.parentElement

    expect(fade?.className).toContain('pointer-events-none')
    expect(fade?.className).toContain('inset-x-0')
    expect(toggle.className).toContain('pointer-events-auto')
    expect(toggle.className).toContain('w-9')
    expect(toggle.className).not.toContain('inset-x-0')

    fireEvent.click(toggle)

    expect(screen.getByRole('button', { name: 'Collapse' }).getAttribute('aria-expanded')).toBe('true')
  })
})
