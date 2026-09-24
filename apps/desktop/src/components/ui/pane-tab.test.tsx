import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneTab, PaneTabLabel, PaneTabStrip } from './pane-tab'

afterEach(cleanup)

describe('PaneTab close gestures', () => {
  it('middle-click closes — pointer events only, no auxclick', () => {
    const onClose = vi.fn()
    render(
      <PaneTab onClose={onClose}>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    const tab = screen.getByText('tab')
    fireEvent.pointerDown(tab, { button: 1 })
    fireEvent.pointerUp(tab, { button: 1 })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('⌘-click (metaKey + button 0) closes — the Mac middle-click equivalent', () => {
    const onClose = vi.fn()
    render(
      <PaneTab onClose={onClose}>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    fireEvent.pointerDown(screen.getByText('tab'), { button: 0, metaKey: true })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('⌘-click preempts the shell drag/activate pointerdown handler', () => {
    const onClose = vi.fn()
    const onPointerDown = vi.fn()
    render(
      <PaneTab onClose={onClose} onPointerDown={onPointerDown}>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    fireEvent.pointerDown(screen.getByText('tab'), { button: 0, metaKey: true })
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onPointerDown).not.toHaveBeenCalled()
  })

  it('⌘-click swallows the follow-up activation click (capture phase)', () => {
    const onClose = vi.fn()
    const onActivate = vi.fn()
    render(
      <PaneTab onClose={onClose}>
        <PaneTabLabel as="button" onClick={onActivate}>
          tab
        </PaneTabLabel>
      </PaneTab>
    )

    fireEvent.click(screen.getByText('tab'), { button: 0, metaKey: true })
    expect(onActivate).not.toHaveBeenCalled()
  })

  it('plain left-click neither closes nor blocks activation', () => {
    const onClose = vi.fn()
    const onActivate = vi.fn()
    const onPointerDown = vi.fn()
    render(
      <PaneTab onClose={onClose} onPointerDown={onPointerDown}>
        <PaneTabLabel as="button" onClick={onActivate}>
          tab
        </PaneTabLabel>
      </PaneTab>
    )

    fireEvent.pointerDown(screen.getByText('tab'), { button: 0 })
    fireEvent.click(screen.getByText('tab'), { button: 0 })
    expect(onClose).not.toHaveBeenCalled()
    expect(onPointerDown).toHaveBeenCalledTimes(1)
    expect(onActivate).toHaveBeenCalledTimes(1)
  })

  it('does nothing without an onClose (uncloseable workspace tab)', () => {
    const onPointerDown = vi.fn()
    render(
      <PaneTab onPointerDown={onPointerDown}>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    fireEvent.pointerDown(screen.getByText('tab'), { button: 0, metaKey: true })
    expect(onPointerDown).toHaveBeenCalledTimes(1)
  })
})

describe('PaneTab hover close button', () => {
  it('clicking the ✕ closes without activating or dragging the tab', () => {
    const onClose = vi.fn()
    const onActivate = vi.fn()
    const onPointerDown = vi.fn()
    render(
      <PaneTab onClose={onClose} onPointerDown={onPointerDown}>
        <PaneTabLabel as="button" onClick={onActivate}>
          tab
        </PaneTabLabel>
      </PaneTab>
    )

    const close = screen.getByRole('button', { name: 'Close' })
    fireEvent.pointerDown(close, { button: 0 })
    fireEvent.click(close, { button: 0 })
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onActivate).not.toHaveBeenCalled()
    expect(onPointerDown).not.toHaveBeenCalled()
  })

  it('renders no ✕ without an onClose', () => {
    render(
      <PaneTab>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
  })

  it('renders no ✕ on a vertical rail tab (middle/⌘-click only there)', () => {
    const onClose = vi.fn()
    render(
      <PaneTab onClose={onClose} vertical>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
  })
})

describe('PaneTabStrip overflow', () => {
  // The strip has exactly two overflow behaviours and they are mutually
  // exclusive: scroll one row (default) or wrap onto several. Asserted as a
  // contract between the prop and the two properties that actually decide it —
  // the list's wrapping and the bar's height being a floor vs a fixed value —
  // rather than by freezing the full class string, which churns on every
  // restyle.

  const listOf = (): HTMLElement => screen.getByRole('tablist')
  const barOf = (): HTMLElement => listOf().parentElement!
  const classes = (el: HTMLElement): string[] => el.className.split(/\s+/).filter(Boolean)

  it('scrolls a single row by default — the strip stays exactly one tab tall', () => {
    render(
      <PaneTabStrip>
        <PaneTab>
          <PaneTabLabel>one</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
    )

    expect(classes(listOf())).toContain('overflow-x-auto')
    expect(classes(listOf())).not.toContain('flex-wrap')
    // Fixed height, not a floor: a scrolling strip can never grow a second row.
    expect(classes(barOf())).toContain('h-7')
    expect(classes(barOf())).not.toContain('min-h-7')
  })

  it('wraps onto more rows when asked, and stops scrolling', () => {
    render(
      <PaneTabStrip wrap>
        <PaneTab>
          <PaneTabLabel>one</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
    )

    expect(classes(listOf())).toContain('flex-wrap')
    // Both halves matter: a wrapping list inside a fixed-height bar clips the
    // rows it just created instead of showing them.
    expect(classes(listOf())).not.toContain('overflow-x-auto')
    expect(classes(barOf())).toContain('min-h-7')
    expect(classes(barOf())).not.toContain('h-7')
  })

  it('pins a wrapped tab to ONE row — height:100% would resolve against the whole bar', () => {
    const { rerender } = render(
      <PaneTabStrip wrap>
        <PaneTab>
          <PaneTabLabel>one</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
    )

    // The var the tab's height reads. Set here, a tab is one row tall; unset,
    // it falls back to 100% and stretches to every row in the bar.
    expect(classes(barOf())).toContain('[--pane-tab-h:1.75rem]')

    rerender(
      <PaneTabStrip>
        <PaneTab>
          <PaneTabLabel>one</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
    )

    // Unwrapped strips must NOT declare it: the fallback is what keeps a
    // single-row tab filling a bar whose height it does not know.
    expect(classes(barOf())).not.toContain('[--pane-tab-h:1.75rem]')
  })

  it('wraps INSIDE the titlebar band too — titlebar and wrap are independent', () => {
    // Regression: `titlebar` short-circuited the class ladder before `wrap`,
    // so a titlebar-hosted strip silently kept a one-row h-full bar. The
    // feature shipped, the prop threaded through, and nothing wrapped.
    render(
      <PaneTabStrip titlebar wrap>
        <PaneTab>
          <PaneTabLabel>one</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
    )

    expect(classes(listOf())).toContain('flex-wrap')
    // The band must be a FLOOR, never a fixed height, or the rows the list
    // creates are clipped by the bar that contains them.
    expect(classes(barOf())).toContain('min-h-full')
    expect(classes(barOf())).not.toContain('h-full')
    // Still the titlebar: drag region and flex-1 survive wrapping.
    expect(classes(barOf())).toContain('flex-1')
    expect(classes(barOf())).toContain('[-webkit-app-region:drag]')
  })

  it('keeps trailing chrome one row tall so the chevron stays on the first row', () => {
    render(
      <PaneTabStrip trailing={<button type="button">chevron</button>} wrap>
        <PaneTab>
          <PaneTabLabel>one</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
    )

    const trailing = screen.getByText('chevron').parentElement!

    expect(classes(trailing)).toContain('h-7')
  })
})
