import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'
import { resetThreadScroll, setThreadAtBottom } from '@/store/thread-scroll'

import { ComposerStatusStack } from './index'

class TestResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

describe('ComposerStatusStack scroll treatment', () => {
  beforeEach(() => {
    setThreadAtBottom(false, 'sess-a')
  })

  afterEach(() => {
    cleanup()
    $subagentsBySession.set({})
    resetThreadScroll('sess-a')
  })

  it('dims only the status content while keeping the dock card opaque', () => {
    const view = render(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerStatusStack queue={<div>Queued task</div>} sessionId="sess-a" />
          <ComposerStatusStack queue={<div>Sibling task</div>} sessionId="sess-b" />
        </I18nProvider>
      </MemoryRouter>
    )

    const card = view.container.querySelector<HTMLElement>('[class*="bg-(--composer-fill)"]')
    const dimmedContent = screen.getByText('Queued task').closest<HTMLElement>('.opacity-30')

    expect(card).not.toBeNull()
    expect(card?.classList.contains('opacity-30')).toBe(false)
    expect(dimmedContent).not.toBeNull()
    expect(dimmedContent).not.toBe(card)
    expect(card?.contains(dimmedContent)).toBe(true)
    expect(screen.getByText('Sibling task').closest('.opacity-30')).toBeNull()
  })

  it('surfaces a finished subagent row and lets the user dismiss it', () => {
    upsertSubagent('sess-a', {
      goal: 'Finished task',
      status: 'completed',
      subagent_id: 'finished',
      summary: 'Done'
    })

    render(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerStatusStack queue={null} sessionId="sess-a" />
        </I18nProvider>
      </MemoryRouter>
    )

    const header = screen.getByRole('button', { name: /1 Subagent/ })
    fireEvent.click(header)
    expect(screen.getByText('Finished task')).toBeTruthy()
    expect(screen.getByText('Done')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))

    expect(screen.queryByText('Finished task')).toBeNull()
  })
})
