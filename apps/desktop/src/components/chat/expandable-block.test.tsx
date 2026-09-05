import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setCodeBlockCollapse } from '@/store/code-block-collapse'

import { ExpandableBlock } from './expandable-block'

// jsdom has no ResizeObserver and reports scrollHeight === 0, so the block
// never flips to `overflowing` on its own. Stub RO to fire immediately and
// force a tall scrollHeight on the observed node so the toggle mounts.
let stubScrollHeight = 400

class TestResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {}

  observe(target: Element) {
    Object.defineProperty(target, 'scrollHeight', { configurable: true, value: stubScrollHeight })
    this.callback([{ target } as ResizeObserverEntry], this as unknown as ResizeObserver)
  }

  unobserve() {}
  disconnect() {}
}

beforeEach(() => {
  stubScrollHeight = 400
  setCodeBlockCollapse('compact')
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('ExpandableBlock', () => {
  it('lets horizontal scroll through and keeps the last line selectable', () => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)

    const { container } = render(
      <ExpandableBlock>
        <pre data-testid="content">{'const x = 1\n'.repeat(20)}</pre>
      </ExpandableBlock>
    )

    const inner = container.querySelector('[data-testid="content"]')!.parentElement!
    const toggle = screen.getByRole('button', { name: /expand|collapse/i })
    const fade = toggle.parentElement!

    // Inner container allows horizontal scroll so wide code gets a scrollbar:
    // platform overlay (`scrollbar-overlay`), not the always-on classic gutter.
    expect(inner.className).toContain('overflow-x-auto')
    expect(inner.className).toContain('scrollbar-overlay')

    // The full-width fade is a pure cue: it spans the bottom edge but must not
    // intercept pointer events, so the scrollbar drag and text selection on the
    // last line pass through to the content underneath.
    expect(fade.className).toContain('pointer-events-none')
    expect(fade.className).toContain('inset-x-0')

    // Only the compact toggle is clickable, and it is pinned to the right edge
    // rather than spanning the full width (the old bug).
    expect(toggle.className).toContain('pointer-events-auto')
    expect(toggle.className).toContain('w-9')
    expect(toggle.className).not.toContain('inset-x-0')
  })

  it('still toggles expanded state when the compact control is clicked', () => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)

    render(
      <ExpandableBlock>
        <pre data-testid="content">{'line\n'.repeat(20)}</pre>
      </ExpandableBlock>
    )

    const toggle = screen.getByRole('button', { name: 'Expand' })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')

    fireEvent.click(toggle)

    expect(screen.getByRole('button', { name: 'Collapse' }).getAttribute('aria-expanded')).toBe('true')
  })

  it('never folds when the preference is off: no cap, no toggle', () => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)
    setCodeBlockCollapse('off')

    const { container } = render(
      <ExpandableBlock>
        <pre data-testid="content">{'line\n'.repeat(20)}</pre>
      </ExpandableBlock>
    )

    const inner = container.querySelector('[data-testid="content"]')!.parentElement!

    expect(inner.className).not.toMatch(/max-h-/)
    expect(screen.queryByRole('button', { name: /expand|collapse/i })).toBeNull()
  })

  it('folds later under tall than under compact', () => {
    vi.stubGlobal('ResizeObserver', TestResizeObserver)
    // Taller than the compact threshold (121px), shorter than the tall one (321px).
    stubScrollHeight = 200

    const compact = render(
      <ExpandableBlock>
        <pre data-testid="content">{'line\n'.repeat(12)}</pre>
      </ExpandableBlock>
    )

    const compactInner = compact.container.querySelector('[data-testid="content"]')!.parentElement!

    expect(compactInner.className).toContain('max-h-[7.5rem]')
    expect(screen.getByRole('button', { name: 'Expand' })).toBeTruthy()
    compact.unmount()

    setCodeBlockCollapse('tall')

    const tall = render(
      <ExpandableBlock>
        <pre data-testid="content">{'line\n'.repeat(12)}</pre>
      </ExpandableBlock>
    )

    const tallInner = tall.container.querySelector('[data-testid="content"]')!.parentElement!

    expect(tallInner.className).toContain('max-h-[20rem]')
    expect(screen.queryByRole('button', { name: /expand|collapse/i })).toBeNull()
  })
})
