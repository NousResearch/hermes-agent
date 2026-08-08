import type { Unstable_TriggerItem } from '@assistant-ui/core'
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

describe('ComposerTriggerPopover i18n', () => {
  afterEach(() => {
    cleanup()
  })

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

describe('ComposerTriggerPopover keyboard-driven scroll', () => {
  const items: readonly Unstable_TriggerItem[] = Array.from({ length: 8 }, (_, i) => ({
    id: `cmd${i}|${i}`,
    label: `cmd${i}`,
    type: 'slash'
  }))

  function renderListPopover(activeIndex: number) {
    const onHover = vi.fn()
    const onPick = vi.fn()

    const rendered = render(
      <I18nProvider configClient={null} initialLocale="zh">
        <ComposerTriggerPopover
          activeIndex={activeIndex}
          items={items}
          kind="/"
          loading={false}
          onHover={onHover}
          onPick={onPick}
        />
      </I18nProvider>
    )

    const rerenderAt = (index: number) =>
      rendered.rerender(
        <I18nProvider configClient={null} initialLocale="zh">
          <ComposerTriggerPopover
            activeIndex={index}
            items={items}
            kind="/"
            loading={false}
            onHover={onHover}
            onPick={onPick}
          />
        </I18nProvider>
      )

    return { ...rendered, onHover, onPick, rerenderAt }
  }

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('scrolls the highlighted row into view as the active index advances', () => {
    // jsdom has no layout, so scrollIntoView is the observable contract here.
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView

    const { container, rerenderAt } = renderListPopover(0)

    rerenderAt(3)

    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' })
    const highlighted = container.querySelector('[data-highlighted]')
    expect(highlighted?.textContent).toContain('/cmd3')
  })

  it('pins the drawer to the top when the highlight wraps back to the first row', () => {
    Element.prototype.scrollIntoView = vi.fn()

    const { rerenderAt } = renderListPopover(6)
    const drawer = screen.getByRole('listbox')

    rerenderAt(7)
    drawer.scrollTop = 120

    rerenderAt(0)

    expect(drawer.scrollTop).toBe(0)
  })

  it('does not scroll for hover-driven highlight changes', () => {
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView

    const { container, onHover, rerenderAt } = renderListPopover(0)
    scrollIntoView.mockClear()

    const rows = container.querySelectorAll('button')
    fireEvent.mouseEnter(rows[4]!)

    expect(onHover).toHaveBeenCalledWith(4)

    // The parent echoes the hover back as the new active index — that update
    // must not move the drawer out from under the pointer.
    rerenderAt(4)

    expect(scrollIntoView).not.toHaveBeenCalled()

    // The next keyboard step scrolls again.
    rerenderAt(5)

    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' })
  })

  it('scrolls when the keyboard wraps back to the last hovered row', () => {
    // Hover the final row, ArrowDown wraps to 0 (pinning the drawer to the
    // top), ArrowUp wraps back to the final row. The hover marker must have
    // been consumed by then — a stale marker would suppress this scroll and
    // leave the highlighted row off-screen.
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView

    const { container, rerenderAt } = renderListPopover(0)

    const rows = container.querySelectorAll('button')
    fireEvent.mouseEnter(rows[7]!)
    rerenderAt(7)
    scrollIntoView.mockClear()

    rerenderAt(0)
    rerenderAt(7)

    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' })
  })
})
