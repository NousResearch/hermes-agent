import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $goalsBySession, type SessionGoal } from '@/store/goals'

import { ComposerStatusStack } from './index'

// The stack measures itself into a surface var — jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const SID = 'sess-goal-1'

const goal = (status: SessionGoal['status'], title = 'ship the feature', detail?: string): SessionGoal => ({
  detail,
  status,
  title,
  updatedAt: Date.now()
})

function renderStack(sessionId: null | string = SID) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={sessionId} />
      </I18nProvider>
    </MemoryRouter>
  )
}

describe('ComposerStatusStack goal indicator', () => {
  beforeEach(() => {
    $goalsBySession.set({})
  })

  afterEach(() => {
    cleanup()
    $goalsBySession.set({})
  })

  it('renders nothing when the session has no goal', () => {
    const view = renderStack()

    expect(view.container.firstChild).toBeNull()
  })

  it('shows an active goal with its title', () => {
    $goalsBySession.set({ [SID]: goal('active') })

    renderStack()

    expect(screen.getByText('Goal active')).toBeTruthy()
    expect(screen.getByText('ship the feature')).toBeTruthy()
  })

  it('labels a paused goal as paused', () => {
    $goalsBySession.set({ [SID]: goal('paused') })

    renderStack()

    expect(screen.getByText('Goal paused')).toBeTruthy()
    expect(screen.getByText('ship the feature')).toBeTruthy()
  })

  it('shows the continuation detail line for an active goal', () => {
    $goalsBySession.set({ [SID]: goal('active', 'ship it', 'Continuing toward goal (3/20)') })

    renderStack()

    expect(screen.getByText('Continuing toward goal (3/20)')).toBeTruthy()
  })

  it('scopes the indicator to the goal-owning session', () => {
    $goalsBySession.set({ 'other-session': goal('active') })

    const view = renderStack()

    expect(view.container.firstChild).toBeNull()
  })
})
