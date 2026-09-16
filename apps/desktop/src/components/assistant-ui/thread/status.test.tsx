import { act, cleanup, render, screen, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { __resetElapsedTimerRegistryForTests } from '@/components/chat/activity-timer'
import { I18nProvider } from '@/i18n'
import { $providerWaitSessions, setSessionProviderWait } from '@/store/provider-wait'
import { $activeSessionId, $busy, $turnStartedAt } from '@/store/session'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'

import { BackgroundResumeNotice, ResponseLoadingIndicator } from './status'

function renderIndicator() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ResponseLoadingIndicator />
    </I18nProvider>
  )
}

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
})

// The status line sits between tool rows and thinking headers, which the
// transcript rests at a fade. Without the mark it reads a shade brighter than
// both — the one line in the column claiming emphasis it hasn't earned.
describe('status line', () => {
  afterEach(cleanup)

  it('is marked as transcript scaffolding', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    const { container } = renderIndicator()

    expect(container.querySelector('[role="status"]')?.hasAttribute('data-conversation-scaffold')).toBe(true)
  })
})

const runningSubagent = (id: string, activity: string): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: 'Background work',
  id,
  parentId: null,
  startedAt: 0,
  status: 'running',
  stream: [{ at: 0, kind: 'progress', text: activity }],
  taskCount: 1,
  taskIndex: 0,
  updatedAt: 0
})

const mountedView = (runtimeId: string, busy = false): SessionView => ({
  ...({} as SessionView),
  $busy: atom(busy),
  $runtimeId: atom<null | string>(runtimeId),
  kind: 'tile'
})

describe('BackgroundResumeNotice session scope', () => {
  beforeEach(() => {
    $activeSessionId.set('session-b')
    $busy.set(true)
    $subagentsBySession.set({
      'session-a': [runningSubagent('owner-child', 'Owner activity')]
    })
  })

  afterEach(() => {
    cleanup()
    $activeSessionId.set(null)
    $busy.set(false)
    $subagentsBySession.set({})
  })

  it('paints background wait only in the simultaneously mounted owner surface', () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <section data-testid="owner-surface">
          <SessionViewProvider value={mountedView('session-a')}>
            <BackgroundResumeNotice />
          </SessionViewProvider>
        </section>
        <section data-testid="other-surface">
          <SessionViewProvider value={mountedView('session-b')}>
            <BackgroundResumeNotice />
          </SessionViewProvider>
        </section>
      </I18nProvider>
    )

    const owner = within(screen.getByTestId('owner-surface'))
    const other = within(screen.getByTestId('other-surface'))

    expect(owner.getByRole('status').textContent).not.toContain('Owner activity')
    expect(other.queryByRole('status')).toBeNull()

    act(() => $subagentsBySession.set({ 'session-a': [] }))

    expect(owner.queryByRole('status')).toBeNull()
    expect(other.queryByRole('status')).toBeNull()
  })
})
