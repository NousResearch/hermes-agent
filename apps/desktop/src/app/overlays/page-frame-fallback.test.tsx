import { cleanup, render, screen } from '@testing-library/react'
import { Suspense } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { lazyView } from '@/lib/lazy-view'
import { $shellChoice } from '@/store/shell-variant'

import { PageFrameFallback, SettingsFrameFallback } from './page-frame-fallback'
import { WorkspacePageFallback } from './page-header-skeleton'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

// A page whose chunk never arrives: the Suspense fallback is all there is to see.
const PendingPage = lazyView<object>(() => new Promise(() => undefined))

function renderPending(fallback: React.ReactNode, route = '/') {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <Suspense fallback={fallback}>
        <PendingPage />
      </Suspense>
    </MemoryRouter>
  )
}

describe('a routed page while its code loads', () => {
  it('shows the page frame: its panel and a busy header, not an empty window', () => {
    renderPending(<PageFrameFallback frame="schedules" onClose={vi.fn()} />)

    expect(screen.getByRole('complementary', { name: en.cron.title })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 2, name: en.cron.title })).toBeTruthy()
    const header = screen.getByLabelText(en.common.loading)
    expect(header.getAttribute('aria-busy')).toBe('true')
    expect(header.querySelector('h1')).toBeTruthy()
  })

  it('frames the Settings route as an agent page when it names an agent', () => {
    renderPending(<SettingsFrameFallback onClose={vi.fn()} />, '/settings?agentProfile=work')

    expect(screen.getByRole('complementary', { name: en.simplifiedAgents.agents })).toBeTruthy()
    expect(screen.queryByRole('complementary', { name: en.commandCenter.settings })).toBeNull()
  })

  it('gives a workspace page its known title in the page column', () => {
    renderPending(<WorkspacePageFallback frame="library" />)

    expect(screen.getByRole('heading', { level: 1, name: en.simplifiedAgents.library })).toBeTruthy()
  })

  it('leaves classic with its blank wait', () => {
    $shellChoice.set(false)
    const { container } = renderPending(<PageFrameFallback frame="schedules" onClose={vi.fn()} />)

    expect(container.textContent).toBe('')
  })
})
