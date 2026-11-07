import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetElapsedTimerRegistryForTests } from '@/components/chat/activity-timer'
import { I18nProvider } from '@/i18n'
import { $providerWaitSessions, setSessionProviderWait } from '@/store/provider-wait'
import { $activeSessionId, $turnStartedAt } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { $todosBySession } from '@/store/todos'

import { SPINNER_VERBS } from './spinner-label'
import { ResponseLoadingIndicator } from './status'

function renderIndicator(reasoningLabel?: string) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ResponseLoadingIndicator reasoningLabel={reasoningLabel} />
    </I18nProvider>
  )
}

// The row reads `usage.output`; once a slice exists the primary view also takes
// the turn clock from it, so the fixture carries both.
const setSessionOutputTokens = (sid: string, output: number, turnStartedAt: number) =>
  $sessionStates.set({
    ...$sessionStates.get(),
    [sid]: { turnStartedAt, usage: { calls: 1, input: 0, output, total: output } } as never
  })

describe('ResponseLoadingIndicator timer', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
    // useViewedInterval gates ticking on document focus + visibility; jsdom's
    // hasFocus() is unreliable across runners, so pin it (same as the
    // background-sync backstop tests).
    vi.spyOn(globalThis.document, 'hasFocus').mockReturnValue(true)
    __resetElapsedTimerRegistryForTests()
  })

  afterEach(() => {
    cleanup()
    $activeSessionId.set(null)
    $turnStartedAt.set(null)
    $providerWaitSessions.set({})
    $sessionStates.set({})
    $todosBySession.set({})
    __resetElapsedTimerRegistryForTests()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('preserves each running session timer while switching between sessions', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    const sessionA = renderIndicator()

    act(() => vi.advanceTimersByTime(5_000))
    expect(screen.getAllByText((_, node) => node?.textContent === '5s').length).toBeGreaterThan(0)
    sessionA.unmount()

    $activeSessionId.set('session-b')
    $turnStartedAt.set(Date.now())
    const sessionB = renderIndicator()

    act(() => vi.advanceTimersByTime(3_000))
    expect(screen.getAllByText((_, node) => node?.textContent === '3s').length).toBeGreaterThan(0)
    sessionB.unmount()

    $activeSessionId.set('session-a')
    $turnStartedAt.set(new Date('2026-01-01T00:00:00.000Z').getTime())
    renderIndicator()

    expect(screen.getAllByText((_, node) => node?.textContent === '8s').length).toBeGreaterThan(0)
  })

  it('names a prolonged provider wait in the existing response status row', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    setSessionProviderWait('session-a', '⏳ waiting on local-model — 30s with no output yet')

    renderIndicator()

    expect(screen.getByText('⏳ waiting on local-model — 30s with no output yet')).toBeTruthy()
  })

  it('never leaves the row wordless: a per-turn verb when nothing else names the work', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    renderIndicator()

    const label = screen.getByRole('status').textContent ?? ''

    expect(SPINNER_VERBS.some(verb => label.startsWith(`${verb}…`))).toBe(true)
  })

  it('names the in-progress todo, Claude Code activeForm style', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    $todosBySession.set({
      'session-a': [
        { content: 'Plan the row', id: '1', status: 'completed' },
        { content: 'Ship the row', id: '2', status: 'in_progress' }
      ]
    })
    renderIndicator()

    expect(screen.getByText('Shipping the row…')).toBeTruthy()
  })

  it('prefers the newest reasoning step over the todo', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    $todosBySession.set({ 'session-a': [{ content: 'Ship the row', id: '2', status: 'in_progress' }] })
    renderIndicator('Reading the adapter')

    expect(screen.getByText('Reading the adapter…')).toBeTruthy()
    expect(screen.queryByText('Shipping the row…')).toBeNull()
  })

  it('lets a named wait outrank the spinner words', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    setSessionProviderWait('session-a', 'waiting on provider')
    renderIndicator('Reading the adapter')

    expect(screen.getByText('waiting on provider')).toBeTruthy()
    expect(screen.queryByText('Reading the adapter…')).toBeNull()
  })

  it('counts only the tokens this turn spent, hiding the count until there are any', () => {
    // Its own session id: the turn baseline is keyed by session + turn start,
    // and every test here shares one frozen clock.
    const startedAt = Date.now()

    $activeSessionId.set('session-tokens')
    setSessionOutputTokens('session-tokens', 1_000, startedAt)
    renderIndicator()

    expect(screen.queryByText(/tokens/)).toBeNull()

    act(() => setSessionOutputTokens('session-tokens', 9_500, startedAt))

    expect(screen.getByText('· 8.5k tokens')).toBeTruthy()
  })

  it('keeps the ticking timer out of the live region accessibility tree', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    renderIndicator()

    act(() => vi.advanceTimersByTime(2_000))

    const status = screen.getByRole('status')
    const timer = [...status.querySelectorAll('[aria-hidden="true"]')].find(el => el.textContent === '2s')

    expect(status.getAttribute('aria-live')).toBe('polite')
    expect(timer).toBeDefined()
  })
})
