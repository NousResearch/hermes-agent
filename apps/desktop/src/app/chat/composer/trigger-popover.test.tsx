import { cleanup, render, screen } from '@testing-library/react'
import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ComposerTriggerPopover } from './trigger-popover'

function makeItem(label: string, id?: string): Unstable_TriggerItem {
  return { id: id ?? label, label, type: 'simple' }
}

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

function renderPopoverWithItems(kind: '@' | '/', activeIndex = 0, items?: readonly Unstable_TriggerItem[]) {
  const onHover = vi.fn()
  const onPick = vi.fn()

  const rendered = render(
    <I18nProvider configClient={null} initialLocale="zh">
      <ComposerTriggerPopover
        activeIndex={activeIndex}
        items={items ?? [makeItem('cmd1'), makeItem('cmd2'), makeItem('cmd3')]}
        kind={kind}
        loading={false}
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

describe('ComposerTriggerPopover keyboard scroll', () => {
  beforeEach(() => {
    // jsdom does not implement scrollIntoView — stub it so the effect doesn't throw.
    Element.prototype.scrollIntoView = vi.fn()
  })

  afterEach(() => {
    cleanup()
  })

  it('scrolls the active item into view when activeIndex changes', () => {
    const { rerender } = renderPopoverWithItems('/', 0)

    // The initial render with activeIndex=0 fires the effect once, scrolling the
    // first item into view.
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled()

    const scrollIntoView = Element.prototype.scrollIntoView as ReturnType<typeof vi.fn>

    // Change activeIndex to 2 (third item).
    rerender(
      <I18nProvider configClient={null} initialLocale="zh">
        <ComposerTriggerPopover
          activeIndex={2}
          items={[makeItem('cmd1'), makeItem('cmd2'), makeItem('cmd3')]}
          kind="/"
          loading={false}
          onHover={vi.fn()}
          onPick={vi.fn()}
        />
      </I18nProvider>
    )

    // The effect triggers again with the new activeIndex — scrollIntoView is
    // called once on mount (activeIndex=0) and once after re-render
    // (activeIndex=2), each time with { block: 'nearest' } to scroll the
    // active row into the visible area without moving the container.
    expect(scrollIntoView).toHaveBeenCalledTimes(2)
    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' })
  })
})
