import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { JarvisDashboard } from './dashboard'
import type { JarvisNewsItem } from './news'
import { initialJarvisUiState } from './projector'
import type { JarvisUiState } from './types'

function fixtureState(
  overrides: {
    result?: string
    taskPhase?: JarvisUiState['task']['phase']
  } = {}
): JarvisUiState {
  return {
    ...initialJarvisUiState(),
    result: overrides.result,
    sessionId: 's1',
    task: { id: 't1', phase: overrides.taskPhase ?? 'idle' },
    activity: [
      { at: 1, label: 'Terminal', sessionId: 's1', taskId: 't1', toolCallId: 'tool-1', type: 'tool.started' },
      {
        at: 2,
        detail: 'done',
        label: 'Terminal',
        sessionId: 's1',
        taskId: 't1',
        toolCallId: 'tool-1',
        type: 'tool.completed'
      }
    ]
  }
}

function renderDashboard(ui: React.ReactElement) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="pl">
        {ui}
      </I18nProvider>
    </MemoryRouter>
  )
}

afterEach(() => {
  cleanup()
})

describe('Agent CzesiekDashboard', () => {
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

  it('offers the playbook from the dashboard without opening it over an empty setup', () => {
    renderDashboard(
      <JarvisDashboard connected state={fixtureState()}>
        <div data-testid="real-chat">Real transcript and composer</div>
      </JarvisDashboard>
    )

    expect(screen.getByRole('button', { name: 'Podpowiedzi' })).toBeTruthy()
    expect(screen.queryByTestId('jarvis-tips')).toBeNull()
  })

  it('renders the desktop composition with conversation and activity landmarks but no product navigation', () => {
    renderDashboard(
      <JarvisDashboard connected state={fixtureState({ taskPhase: 'running' })}>
        <div data-testid="real-chat">Real transcript and composer</div>
      </JarvisDashboard>
    )

    expect(screen.getByTestId('jarvis-dashboard').getAttribute('data-layout')).toBe('desktop')
    expect(screen.queryByRole('navigation')).toBeNull()
    expect(screen.getByRole('main', { name: 'Rozmowa z Agentem Czeskiem' })).toBeTruthy()
    expect(screen.getByRole('complementary', { name: 'Co robi Agent Czesiek' })).toBeTruthy()
    expect(screen.getByTestId('real-chat')).toBeTruthy()
  })

  it('collapses and restores the right rail without losing the conversation', () => {
    renderDashboard(
      <JarvisDashboard connected rail={<div data-testid="rail-card">Card</div>} state={fixtureState()}>
        <div data-testid="real-chat">Real transcript and composer</div>
      </JarvisDashboard>
    )

    expect(screen.getByTestId('rail-card')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Zwiń prawy panel' }))
    expect(screen.queryByTestId('jarvis-rail')).toBeNull()
    expect(screen.getByTestId('real-chat')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Pokaż prawy panel' }))
    expect(screen.getByTestId('rail-card')).toBeTruthy()
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
    expect(screen.queryByRole('dialog', { name: 'Co robi Agent Czesiek' })).toBeNull()

    fireEvent.click(activityButton)

    const drawer = screen.getByRole('dialog', { name: 'Co robi Agent Czesiek' })
    expect(activityButton.getAttribute('aria-expanded')).toBe('true')
    expect(drawer.getAttribute('data-activity-surface')).toBe('drawer')
    expect(screen.getByTestId('real-chat')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Zamknij aktywność' }))

    expect(screen.queryByRole('dialog', { name: 'Co robi Agent Czesiek' })).toBeNull()
    expect(activityButton.getAttribute('aria-expanded')).toBe('false')
  })

  it('renders the mobile composition with a local activity sheet control', () => {
    renderDashboard(
      <JarvisDashboard connected layout="mobile" state={fixtureState({ taskPhase: 'approval' })}>
        <div data-testid="real-chat">Real transcript and composer</div>
      </JarvisDashboard>
    )

    expect(screen.getByTestId('jarvis-dashboard').getAttribute('data-layout')).toBe('mobile')
    expect(screen.getByTestId('jarvis-core').getAttribute('data-compact')).toBe('false')
    expect(screen.queryByRole('navigation')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Pokaż aktywność' }))

    expect(screen.getByRole('dialog', { name: 'Co robi Agent Czesiek' }).getAttribute('data-activity-surface')).toBe(
      'bottom-sheet'
    )

    fireEvent.keyDown(globalThis.document, { key: 'Escape' })

    expect(screen.queryByRole('dialog', { name: 'Co robi Agent Czesiek' })).toBeNull()
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

  it('only raises a connection chip when the connection is lost', () => {
    // Connected is the status bar's job; a chip would say the same thing twice.
    const { rerender } = renderDashboard(
      <JarvisDashboard connected state={fixtureState()}>
        <div />
      </JarvisDashboard>
    )

    expect(screen.queryByText('Połączono')).toBeNull()

    rerender(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="pl">
          <JarvisDashboard connected={false} state={fixtureState()}>
            <div />
          </JarvisDashboard>
        </I18nProvider>
      </MemoryRouter>
    )

    expect(screen.getByText('Brak połączenia')).toBeTruthy()
  })

  it('charts the session from the same events the activity log shows', () => {
    renderDashboard(
      <JarvisDashboard connected state={fixtureState({ taskPhase: 'verified' })}>
        <div />
      </JarvisDashboard>
    )

    const panel = screen.getByRole('complementary', { name: 'Co robi Agent Czesiek' })

    fireEvent.click(within(panel).getByRole('button', { name: 'Statystyki' }))

    // One matched start→complete pair in the fixture, nothing invented.
    expect(within(panel).getByRole('heading', { name: 'Liczby tej sesji' })).toBeTruthy()
    expect(within(panel).getByText('Uruchomienia narzędzi')).toBeTruthy()
    expect(within(panel).getByRole('img', { name: '2 zdarzenia' })).toBeTruthy()
    expect(within(panel).getByText('Terminal')).toBeTruthy()
    // The fixture's pair spans 1ms: a measured median, not a placeholder.
    expect(within(panel).getByText('1 ms')).toBeTruthy()
  })

  it('shows no statistics or news for a session that has produced nothing', () => {
    renderDashboard(
      <JarvisDashboard connected state={{ ...initialJarvisUiState(), sessionId: 's1' }}>
        <div />
      </JarvisDashboard>
    )

    const panel = screen.getByRole('complementary', { name: 'Co robi Agent Czesiek' })

    fireEvent.click(within(panel).getByRole('button', { name: 'Statystyki' }))
    expect(
      within(panel).getByText('Statystyki pojawią się, gdy Agent Czesiek zacznie pracować w tej rozmowie.')
    ).toBeTruthy()
    expect(within(panel).queryByRole('img')).toBeNull()

    fireEvent.click(within(panel).getByRole('button', { name: 'Powiadomienia' }))
    expect(
      within(panel).getByText(
        'Nic nowego. Ta lista wypełnia się, gdy Agent Czesiek pracuje i gdy pojawiają się aktualizacje.'
      )
    ).toBeTruthy()
  })

  it('surfaces the digest and routes its update action to the shell', () => {
    const onOpenUpdate = vi.fn()

    const news: JarvisNewsItem[] = [
      {
        action: 'update-client',
        detail: 'Add Polish TTS',
        id: 'release:abc',
        kind: 'release',
        title: 'Nowa wersja Agenta Cześka',
        tone: 'accent'
      }
    ]

    renderDashboard(
      <JarvisDashboard connected news={news} onOpenUpdate={onOpenUpdate} state={fixtureState()}>
        <div />
      </JarvisDashboard>
    )

    const panel = screen.getByRole('complementary', { name: 'Co robi Agent Czesiek' })

    fireEvent.click(within(panel).getByRole('button', { name: 'Powiadomienia' }))
    expect(within(panel).getByText('Add Polish TTS')).toBeTruthy()

    fireEvent.click(within(panel).getByRole('button', { name: 'Otwórz aktualizację' }))
    expect(onOpenUpdate).toHaveBeenCalledWith('client')
  })

  it('flags items needing attention on the collapsed mobile control', () => {
    const news: JarvisNewsItem[] = [
      { id: 'a1', kind: 'approval', title: 'Czeka na Twoją zgodę', tone: 'warn' },
      { id: 'f1', kind: 'failure', title: 'Zadanie nie powiodło się', tone: 'warn' }
    ]

    renderDashboard(
      <JarvisDashboard connected layout="mobile" news={news} state={fixtureState({ taskPhase: 'approval' })}>
        <div />
      </JarvisDashboard>
    )

    expect(screen.getByRole('button', { name: 'Pokaż aktywność' }).textContent).toContain('2')
  })
})
