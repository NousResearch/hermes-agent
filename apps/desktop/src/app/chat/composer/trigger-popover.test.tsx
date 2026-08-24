import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ComposerTriggerPopover } from './trigger-popover'

function renderPopover(kind: '@' | '/', loading = false) {
  const onHover = vi.fn()
  const onPick = vi.fn()

  const rendered = render(
    <I18nProvider configClient={null} initialLocale="zh">
      <ComposerTriggerPopover
        activeIndex={0}
        items={[]}
        kind={kind}
        loading={loading}
        onHover={onHover}
        onPick={onPick}
      />
    </I18nProvider>
  )

  return { ...rendered, onHover, onPick }
}

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
})

describe('ComposerTriggerPopover i18n', () => {
  it('renders localized empty lookup copy for @ references', () => {
    const { container } = renderPopover('@')

    expect(screen.getByText('没有匹配项。')).toBeTruthy()
    expect(container.textContent).toContain('试试')
    expect(container.textContent).toContain('@file:')
    expect(container.textContent).toContain('或')
    expect(container.textContent).toContain('@folder:')
  })

  it('renders localized loading copy for slash commands', () => {
    renderPopover('/', true)

    // While loading the popover shows only the spinner + loading copy — the
    // `/help` empty-state hint is reserved for the resolved (not-loading) state.
    expect(screen.getByText('查找中…')).toBeTruthy()
  })

  it('renders the slash empty-state hint when not loading', () => {
    const { container } = renderPopover('/')

    expect(screen.getByText('没有匹配项。')).toBeTruthy()
    expect(container.textContent).toContain('/help')
  })
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
    const { container, rerender } = render(popover(0))
    const drawer = container.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement
    const ancestor = drawer.parentElement as HTMLElement
    const secondRow = screen.getAllByRole('button')[1]

    mockDrawerViewport(drawer)
    mockRowPosition(secondRow, 290, 330)
    ancestor.scrollTop = 48
    drawer.scrollTop = 96
    rerender(popover(1))

    const activeRow = container.querySelector('[data-highlighted]') as HTMLElement

    expect(activeRow.textContent).toContain('/second')
    expect(drawer.scrollTop).toBe(125)
    expect(ancestor.scrollTop).toBe(48)

    drawer.scrollTop = 96
    rerender(popover(0))

    expect(drawer.scrollTop).toBe(0)
  })

  it('uses the nearest drawer edge for upward, visible, and oversized rows', () => {
    const { container, rerender } = render(popover(0))
    const drawer = container.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement
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
    const { container, rerender } = render(popover(0, onHover))
    const rows = screen.getAllByRole('button')
    const drawer = container.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement

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
    expect((container.querySelector('[data-highlighted]') as HTMLElement).textContent).toContain('/third')
  })

  it('does not leave a stale hover marker when the active row is hovered', () => {
    const onHover = vi.fn()
    const { container, rerender } = render(popover(1, onHover))
    const drawer = container.querySelector('[data-slot="composer-completion-drawer"]') as HTMLElement
    const activeRow = screen.getAllByRole('button')[1]

    mockDrawerViewport(drawer)
    mockRowPosition(activeRow, 311, 331)
    fireEvent.mouseEnter(activeRow)
    expect(onHover).toHaveBeenCalledWith(1)

    rerender(popover(1, onHover, [...items, slashItem('/fourth')]))

    expect(drawer.scrollTop).toBe(30)
  })
})
