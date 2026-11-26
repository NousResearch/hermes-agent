import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ComposerTriggerPopover } from './trigger-popover'

function slashItem(command: string) {
  return {
    id: command,
    type: 'slash',
    label: command.slice(1),
    metadata: { command, display: command, group: 'Skills', meta: '', rawText: command }
  }
}

function rect(top: number, bottom: number): DOMRect {
  return {
    bottom,
    height: bottom - top,
    left: 0,
    right: 320,
    toJSON: () => ({}),
    top,
    width: 320,
    x: 0,
    y: top
  }
}

function mockDrawerViewport(drawer: HTMLElement) {
  Object.defineProperty(drawer, 'clientHeight', { configurable: true, value: 200 })
  Object.defineProperty(drawer, 'clientTop', { configurable: true, value: 1 })
  vi.spyOn(drawer, 'getBoundingClientRect').mockReturnValue(rect(100, 302))
}

function mockRowPosition(row: HTMLElement, top: number, bottom: number) {
  return vi.spyOn(row, 'getBoundingClientRect').mockReturnValue(rect(top, bottom))
}

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(document.body, 'scrollTop')
  Reflect.deleteProperty(document.documentElement, 'clientWidth')
  Reflect.deleteProperty(document.documentElement, 'clientHeight')
})

describe('ComposerTriggerPopover keyboard scrolling', () => {
  const items = [slashItem('/first'), slashItem('/second'), slashItem('/third')]

  function popover(activeIndex: number, onHover = vi.fn(), nextItems = items) {
    return (
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerTriggerPopover
          activeIndex={activeIndex}
          items={nextItems}
          kind="/"
          loading={false}
          onHover={onHover}
          onPick={vi.fn()}
        />
      </I18nProvider>
    )
  }

  it('keeps keyboard navigation visible and restores the group header on wrap', () => {
    const { rerender } = render(popover(0))
    const drawer = document.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement
    const secondRow = screen.getAllByRole('button')[1]

    // The drawer is portalled out of the clipped pane; the page behind it must
    // still never be scrolled by keyboard navigation.
    Object.defineProperty(document.body, 'scrollTop', { configurable: true, value: 48, writable: true })
    mockDrawerViewport(drawer)
    mockRowPosition(secondRow, 290, 330)
    drawer.scrollTop = 96
    rerender(popover(1))

    const activeRow = document.querySelector('[data-highlighted]') as HTMLElement

    expect(activeRow.textContent).toContain('/second')
    expect(drawer.scrollTop).toBe(125)
    expect(document.body.scrollTop).toBe(48)

    drawer.scrollTop = 96
    rerender(popover(0))

    expect(drawer.scrollTop).toBe(0)
  })

  it('uses the nearest drawer edge for upward, visible, and oversized rows', () => {
    const { rerender } = render(popover(0))
    const drawer = document.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement
    const rows = screen.getAllByRole('button')

    mockDrawerViewport(drawer)
    mockRowPosition(rows[1], 80, 120)
    const thirdRowRect = mockRowPosition(rows[2], 150, 180)

    drawer.scrollTop = 50
    rerender(popover(1))
    expect(drawer.scrollTop).toBe(29)

    rerender(popover(2))
    expect(drawer.scrollTop).toBe(29)

    thirdRowRect.mockReturnValue(rect(80, 340))
    drawer.scrollTop = 50
    rerender(popover(2, vi.fn(), [...items]))
    expect(drawer.scrollTop).toBe(50)

    thirdRowRect.mockReturnValue(rect(150, 400))
    rerender(popover(2, vi.fn(), [...items, slashItem('/fourth')]))
    expect(drawer.scrollTop).toBe(99)

    thirdRowRect.mockReturnValue(rect(0, 250))
    drawer.scrollTop = 100
    rerender(popover(2, vi.fn(), [...items, slashItem('/fifth')]))
    expect(drawer.scrollTop).toBe(49)
  })

  it('does not scroll for a hover echo and consumes the hover marker', () => {
    const onHover = vi.fn()
    const { rerender } = render(popover(0, onHover))
    const rows = screen.getAllByRole('button')
    const drawer = document.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement

    mockDrawerViewport(drawer)
    mockRowPosition(rows[2], 311, 331)
    drawer.scrollTop = 40
    fireEvent.mouseEnter(rows[2])
    expect(onHover).toHaveBeenCalledWith(2)

    rerender(popover(2, onHover))
    expect(drawer.scrollTop).toBe(40)

    rerender(popover(0, onHover))
    rerender(popover(2, onHover))

    expect(drawer.scrollTop).toBe(30)
    expect((document.querySelector('[data-highlighted]') as HTMLElement).textContent).toContain('/third')
  })

  it('does not leave a stale hover marker when the active row is hovered', () => {
    const onHover = vi.fn()
    const { rerender } = render(popover(1, onHover))
    const drawer = document.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement
    const activeRow = screen.getAllByRole('button')[1]

    mockDrawerViewport(drawer)
    mockRowPosition(activeRow, 311, 331)
    fireEvent.mouseEnter(activeRow)
    expect(onHover).toHaveBeenCalledWith(1)

    rerender(popover(1, onHover, [...items, slashItem('/fourth')]))

    expect(drawer.scrollTop).toBe(30)
  })

  it('portals the drawer out of a clipping ancestor and flips below when space above is tight', () => {
    const view = render(<div data-testid="clip">{popover(0)}</div>)
    const clip = view.getByTestId('clip')
    const drawer = document.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement
    const anchor = clip.querySelector('span[aria-hidden]') as HTMLElement

    // The drawer leaves the clipped composer subtree but stays in the shell root,
    // so the window-top zone's `overflow: clip` can never crop it.
    expect(clip.contains(drawer)).toBe(false)
    expect(view.container.contains(drawer)).toBe(true)

    Object.defineProperty(anchor, 'offsetParent', { configurable: true, value: clip })
    vi.spyOn(clip, 'getBoundingClientRect').mockReturnValue({
      ...rect(100, 300),
      left: 40,
      right: 360,
      width: 320,
      x: 40
    })
    Object.defineProperty(drawer, 'scrollHeight', { configurable: true, value: 400 })
    Object.defineProperty(document.documentElement, 'clientWidth', { configurable: true, value: 1024 })
    Object.defineProperty(document.documentElement, 'clientHeight', { configurable: true, value: 800 })

    view.rerender(<div data-testid="clip">{popover(0)}</div>)

    expect(drawer.className).toContain('fixed')
    // 4px gap below a 300px anchor; the 352px cap is the drawer's 22rem maximum.
    expect(drawer.style.top).toBe('304px')
    expect(drawer.style.maxHeight).toBe('352px')
    expect(drawer.style.left).toBe('48px')
  })

  it('keeps the drawer inside the HUD shell instead of the main window root', () => {
    const view = render(
      <div id="root">
        <div data-hud-shell="">{popover(0)}</div>
      </div>
    )
    const hud = view.container.querySelector('[data-hud-shell]') as HTMLElement
    const root = view.container.querySelector('#root') as HTMLElement
    const drawer = document.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement

    expect(drawer.parentElement).toBe(hud)
    expect(drawer.parentElement).not.toBe(root)
  })
})
