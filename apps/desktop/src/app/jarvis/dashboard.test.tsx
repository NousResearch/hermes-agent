import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { JarvisDashboard } from './dashboard'
import { initialJarvisUiState } from './projector'
import type { JarvisUiState } from './types'

function fixtureState(overrides: {
  result?: string
  taskPhase?: JarvisUiState['task']['phase']
} = {}): JarvisUiState {
  return {
    ...initialJarvisUiState(),
    result: overrides.result,
    sessionId: 's1',
    task: { id: 't1', phase: overrides.taskPhase ?? 'idle' },
    activity: [
      { at: 1, label: 'Terminal', sessionId: 's1', taskId: 't1', toolCallId: 'tool-1', type: 'tool.started' },
      { at: 2, detail: 'done', label: 'Terminal', sessionId: 's1', taskId: 't1', toolCallId: 'tool-1', type: 'tool.completed' }
    ]
  }
}

function renderDashboard(ui: React.ReactElement) {
  return render(
    <I18nProvider configClient={null} initialLocale="pl">
      {ui}
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
})

describe('JarvisDashboard', () => {
  it('prioritizes the verified result and real status without fake intelligence metrics', () => {
    renderDashboard(
      <JarvisDashboard connected state={fixtureState({ result: 'Notatka została utworzona.', taskPhase: 'verified' })}>
        <div data-testid="real-chat">Real transcript and composer</div>
      </JarvisDashboard>
    )

    expect(screen.getByRole('heading', { name: 'Notatka została utworzona.' })).toBeTruthy()
    expect(screen.getByText('Zakończono')).toBeTruthy()
    expect(screen.getByTestId('real-chat')).toBeTruthy()
    expect(screen.queryByText(/42%|poziom inteligencji/i)).toBeNull()
  })

  it('renders the desktop composition with conversation and activity landmarks but no product navigation', () => {
    renderDashboard(
      <JarvisDashboard connected state={fixtureState({ taskPhase: 'running' })}>
        <div data-testid="real-chat">Real transcript and composer</div>
      </JarvisDashboard>
    )

    expect(screen.getByTestId('jarvis-dashboard').getAttribute('data-layout')).toBe('desktop')
    expect(screen.queryByRole('navigation')).toBeNull()
    expect(screen.getByRole('main', { name: 'Rozmowa z Jarvisem' })).toBeTruthy()
    expect(screen.getByRole('complementary', { name: 'Co robi Jarvis' })).toBeTruthy()
    expect(screen.getByTestId('real-chat')).toBeTruthy()
  })

  it('renders the tablet composition with an activity drawer that does not remount the conversation', () => {
    renderDashboard(
      <JarvisDashboard connected layout="tablet" state={fixtureState({ taskPhase: 'running' })}>
        <div data-testid="real-chat">Real transcript and composer</div>
      </JarvisDashboard>
    )

    expect(screen.getByTestId('jarvis-dashboard').getAttribute('data-layout')).toBe('tablet')
    const activityButton = screen.getByRole('button', { name: 'Pokaż aktywność' })
    expect(activityButton.className).toContain('min-h-11')
    expect(activityButton.getAttribute('aria-expanded')).toBe('false')
    expect(activityButton.getAttribute('aria-controls')).toBeTruthy()
    expect(screen.queryByRole('dialog', { name: 'Co robi Jarvis' })).toBeNull()

    fireEvent.click(activityButton)

    const drawer = screen.getByRole('dialog', { name: 'Co robi Jarvis' })
    expect(activityButton.getAttribute('aria-expanded')).toBe('true')
    expect(drawer.getAttribute('data-activity-surface')).toBe('drawer')
    expect(screen.getByTestId('real-chat')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Zamknij aktywność' }))

    expect(screen.queryByRole('dialog', { name: 'Co robi Jarvis' })).toBeNull()
    expect(activityButton.getAttribute('aria-expanded')).toBe('false')
  })

  it('renders the mobile composition with a local activity sheet control', () => {
    renderDashboard(
      <JarvisDashboard connected layout="mobile" state={fixtureState({ taskPhase: 'approval' })}>
        <div data-testid="real-chat">Real transcript and composer</div>
      </JarvisDashboard>
    )

    expect(screen.getByTestId('jarvis-dashboard').getAttribute('data-layout')).toBe('mobile')
    expect(screen.getByTestId('jarvis-core').getAttribute('data-compact')).toBe('true')
    expect(screen.queryByRole('navigation')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Pokaż aktywność' }))

    expect(screen.getByRole('dialog', { name: 'Co robi Jarvis' }).getAttribute('data-activity-surface')).toBe(
      'bottom-sheet'
    )

    fireEvent.keyDown(globalThis.document, { key: 'Escape' })

    expect(screen.queryByRole('dialog', { name: 'Co robi Jarvis' })).toBeNull()
  })

  it('uses a profile display name for the empty greeting and never hardcodes Chris', () => {
    renderDashboard(
      <JarvisDashboard connected profileDisplayName="Ada" state={fixtureState()}>
        <div />
      </JarvisDashboard>
    )

    expect(screen.getByText('Ada, od czego zaczynamy?')).toBeTruthy()
    expect(screen.queryByText(/Chris/i)).toBeNull()
  })

  it('renders visible connection status from the dashboard prop', () => {
    const { rerender } = renderDashboard(
      <JarvisDashboard connected state={fixtureState()}>
        <div />
      </JarvisDashboard>
    )

    expect(screen.getByText('Połączono')).toBeTruthy()

    rerender(
      <I18nProvider configClient={null} initialLocale="pl">
        <JarvisDashboard connected={false} state={fixtureState()}>
          <div />
        </JarvisDashboard>
      </I18nProvider>
    )

    expect(screen.getByText('Brak połączenia')).toBeTruthy()
  })
})
