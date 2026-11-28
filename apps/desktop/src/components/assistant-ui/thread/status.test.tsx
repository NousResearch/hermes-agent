import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { __resetElapsedTimerRegistryForTests } from '@/components/chat/activity-timer'
import { I18nProvider } from '@/i18n'
import { $providerWaitSessions, setSessionProviderWait } from '@/store/provider-wait'
import { $activeSessionId, $turnStartedAt } from '@/store/session'

import { ResponseLoadingIndicator } from './status'

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

  it('shows prefill counts and percent instead of a bare percent', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    setSessionProviderWait('session-a', '⚙ processing prompt — 46,500 / 80,000 tokens (58%)')

    renderIndicator()

    expect(screen.getByText('Processing prompt — 46,500 / 80,000 tokens')).toBeTruthy()
    expect(screen.getByText('58%')).toBeTruthy()
  })

  it('marks an estimated prefill total with ~', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    setSessionProviderWait('session-a', '⚙ processing prompt — 12,288 / ~39,551 tokens (31%)')

    renderIndicator()

    expect(screen.getByText('Processing prompt — 12,288 / ~39,551 tokens')).toBeTruthy()
    expect(screen.getByText('31%')).toBeTruthy()
  })

  it('shows the processed count with no bar when no honest total exists', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    setSessionProviderWait('session-a', '⚙ processing prompt — 12,288 tokens read')

    renderIndicator()

    expect(screen.getByText('Processing prompt — 12,288 tokens read')).toBeTruthy()
    expect(screen.queryByText(/%$/)).toBeNull()
  })

  it('names queue waiting distinctly from active input processing', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    setSessionProviderWait('session-a', '⏳ waiting for a free slot — the local server is busy with another request')

    renderIndicator()

    expect(screen.getByText('Waiting for a free slot — another request is using the local server')).toBeTruthy()
  })

  it('names a generic first-chunk wait honestly, without its frozen seconds', () => {
    $activeSessionId.set('session-a')
    $turnStartedAt.set(Date.now())
    setSessionProviderWait('session-a', '⏳ waiting on qwen — 60s waiting for the first stream chunk')

    renderIndicator()

    expect(screen.getByText('Waiting on qwen for the first response — no progress info yet')).toBeTruthy()
    expect(screen.queryByText(/60s waiting for the first stream chunk/)).toBeNull()
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
