import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ErrorBoundary } from './error-boundary'

// The real assistant-ui stale-index throw the root boundary must survive
// (open chat / session switch render race), reproduced verbatim so the
// recoverable-pattern match is exercised against the actual error text.
const TAP_ERROR = 'tapClientLookup: Index 23 out of bounds (length: 18)'

// Throws purely from `box.error` so a render replay (React dev) throws
// identically; the test mutates the box only from timers, never during render —
// modelling a transient race that clears once the boundary remounts against
// fresh state.
function makeBomb(box: { error: Error | null }) {
  return function Bomb() {
    if (box.error) {
      throw box.error
    }

    return <div>recovered</div>
  }
}

const RELOAD_WINDOW = { name: 'Reload window', role: 'button' } as const

const countRecoverWarnings = (calls: unknown[][]) =>
  calls.filter(call => call.some(value => String(value).includes('auto-recovering from transient render error'))).length

describe('ErrorBoundary root auto-recovery', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('recovers the root boundary from a transient stale-index render race', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const box: { error: Error | null } = { error: new Error(TAP_ERROR) }
    const Bomb = makeBomb(box)

    // Disarm before the scheduled next-tick reset re-renders the subtree, so the
    // race genuinely resolves on recovery instead of throwing forever.
    queueMicrotask(() => {
      box.error = null
    })

    render(
      <ErrorBoundary label="root">
        <Bomb />
      </ErrorBoundary>
    )

    await waitFor(() => expect(screen.getByText('recovered')).toBeTruthy())
    expect(screen.queryByRole(RELOAD_WINDOW.role, { name: RELOAD_WINDOW.name })).toBeNull()
    expect(countRecoverWarnings(warnSpy.mock.calls)).toBeGreaterThanOrEqual(1)
  })

  it('stops auto-recovering a persistent error after the cap and leaves the fallback up', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    // Never disarmed: the boundary must not spin a reset -> throw -> reset loop
    // forever — it caps recovery and surfaces the fallback to the user.
    const box: { error: Error | null } = { error: new Error(TAP_ERROR) }
    const Bomb = makeBomb(box)

    render(
      <ErrorBoundary label="root">
        <Bomb />
      </ErrorBoundary>
    )

    // The fallback showing up at all IS the cap working: with unbounded recovery
    // the boundary would reset -> throw -> reset forever and 'Reload window'
    // would never render (this waitFor would hang). The recovery attempts are
    // bounded by MAX_RECOVERIES (3), never an unbounded storm.
    await waitFor(() => expect(screen.getByRole(RELOAD_WINDOW.role, { name: RELOAD_WINDOW.name })).toBeTruthy())
    const warnings = countRecoverWarnings(warnSpy.mock.calls)
    expect(warnings).toBeGreaterThanOrEqual(1)
    expect(warnings).toBeLessThanOrEqual(3)
  })

  it('does not auto-recover a non-root boundary', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const box: { error: Error | null } = { error: new Error(TAP_ERROR) }
    const Bomb = makeBomb(box)

    render(
      <ErrorBoundary fallback={() => <div>scoped-fallback</div>} label="thread">
        <Bomb />
      </ErrorBoundary>
    )

    await waitFor(() => expect(screen.getByText('scoped-fallback')).toBeTruthy())
    expect(countRecoverWarnings(warnSpy.mock.calls)).toBe(0)
  })

  it('does not auto-recover an unrecognized error even at the root', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const box: { error: Error | null } = { error: new Error('some unrelated application error') }
    const Bomb = makeBomb(box)

    render(
      <ErrorBoundary label="root">
        <Bomb />
      </ErrorBoundary>
    )

    await waitFor(() => expect(screen.getByRole(RELOAD_WINDOW.role, { name: RELOAD_WINDOW.name })).toBeTruthy())
    expect(countRecoverWarnings(warnSpy.mock.calls)).toBe(0)
  })
})
