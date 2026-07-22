import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ExpandableBlock } from './expandable-block'

// jsdom has no ResizeObserver and reports scrollHeight === 0, so the block
// never flips to `overflowing` on its own. Mirror the project's test pattern
// (see approval-mode-menu.test.tsx): stub RO to fire immediately, then force a
// tall scrollHeight on the observed node so the collapse button mounts.
class TestResizeObserver {
  constructor(private cb: ResizeObserverCallback) {}
  observe(target: Element) {
    Object.defineProperty(target, 'scrollHeight', { configurable: true, value: 400 })
    this.cb([] as unknown as ResizeObserverEntry[], this as unknown as ResizeObserver)
  }
  unobserve() {}
  disconnect() {}
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('ExpandableBlock', () => {
  it('gives code cards (collapseAlign="end") a right-pinned button clear of the horizontal scrollbar', () => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)

    const { container } = render(
      <ExpandableBlock collapseAlign="end">
        <pre data-testid="content">{"const x = 1\n".repeat(20)}</pre>
      </ExpandableBlock>
    )

    const inner = container.querySelector('[data-testid="content"]')!.parentElement!
    const button = screen.getByRole('button', { name: /expand|collapse/i })

    // Inner container must allow horizontal scroll and pad its right edge so
    // the scrollbar stays draggable across the full width except the button.
    expect(inner.className).toContain('overflow-x-auto')
    expect(inner.className).toContain('pr-2')

    // The button must NOT span the full bottom edge (the old bug) — it is
    // pinned to the right so it cannot overlay the scrollbar.
    expect(button.className).toContain('right-0')
    expect(button.className).not.toContain('inset-x-0')
  })

  it('keeps the full-width centered button for the plain-text fallback (collapseAlign default)', () => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)

    const { container } = render(
      <ExpandableBlock>
        <pre data-testid="content">{"plain text\n".repeat(20)}</pre>
      </ExpandableBlock>
    )

    const inner = container.querySelector('[data-testid="content"]')!.parentElement!
    const button = screen.getByRole('button', { name: /expand|collapse/i })

    // Fallback has no horizontal scroll, so it keeps the original full-width,
    // centered button — and does not gain the end-aligned padding.
    expect(button.className).toContain('inset-x-0')
    expect(button.className).not.toContain('right-0')
    expect(inner.className).not.toContain('pr-2')
  })

  it('toggles expanded state on button click without breaking the control', () => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)

    render(
      <ExpandableBlock collapseAlign="end">
        <pre data-testid="content">{"line\n".repeat(20)}</pre>
      </ExpandableBlock>
    )

    const button = screen.getByRole('button', { name: /expand|collapse/i })

    expect(button.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(button)
    expect(button.getAttribute('aria-expanded')).toBe('true')
  })
})
