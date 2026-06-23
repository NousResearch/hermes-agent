import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ComposerTriggerPopover } from './trigger-popover'

function renderPopover(kind: '@' | '/' | '$', loading = false) {
  const onHover = vi.fn()
  const onPick = vi.fn()

  const rendered = render(
    <I18nProvider configClient={null} initialLocale="zh">
      <ComposerTriggerPopover activeIndex={0} items={[]} kind={kind} loading={loading} onHover={onHover} onPick={onPick} />
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

  it('renders the skill marker empty-state hint when not loading', () => {
    const { container } = renderPopover('$')

    expect(screen.getByText('没有匹配项。')).toBeTruthy()
    expect(container.textContent).toContain('$skill')
  })
})
