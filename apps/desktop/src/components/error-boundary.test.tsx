import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ErrorBoundary } from './error-boundary'

const TAP_LOOKUP_ERROR = new Error('tapClientLookup: Index 6 out of bounds (length: 2)')
const RELOAD_WINDOW = { name: 'Reload window', role: 'button' } as const

function makeBomb(box: { error: Error | null }) {
  return function Bomb() {
    if (box.error) {
      throw box.error
    }

    return <div>recovered</div>
  }
}

const recoveryWarningCount = (calls: unknown[][]) =>
  calls.filter(call => call.some(value => String(value).includes('auto-recovering from tapClientLookup'))).length

describe('ErrorBoundary tapClientLookup recovery', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('recovers the root boundary after a transient lookup race clears', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const box: { error: Error | null } = { error: TAP_LOOKUP_ERROR }
    const Bomb = makeBomb(box)

    render(
      <ErrorBoundary label="root">
        <Bomb />
      </ErrorBoundary>
    )

    box.error = null
    act(() => vi.runOnlyPendingTimers())

    expect(screen.getByText('recovered')).toBeTruthy()
    expect(screen.queryByRole(RELOAD_WINDOW.role, { name: RELOAD_WINDOW.name })).toBeNull()
    expect(recoveryWarningCount(warnSpy.mock.calls)).toBe(1)
  })

  it('stops retrying a persistent lookup error after the recovery budget is exhausted', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const box: { error: Error | null } = { error: TAP_LOOKUP_ERROR }
    const Bomb = makeBomb(box)

    render(
      <ErrorBoundary label="root">
        <Bomb />
      </ErrorBoundary>
    )

    for (let attempt = 0; attempt < 4; attempt += 1) {
      act(() => vi.runOnlyPendingTimers())
    }

    expect(screen.getByRole(RELOAD_WINDOW.role, { name: RELOAD_WINDOW.name })).toBeTruthy()
    expect(recoveryWarningCount(warnSpy.mock.calls)).toBe(3)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not auto-recover the same error in a scoped boundary', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const Bomb = makeBomb({ error: TAP_LOOKUP_ERROR })

    render(
      <ErrorBoundary fallback={() => <div>scoped fallback</div>} label="thread">
        <Bomb />
      </ErrorBoundary>
    )

    act(() => vi.runAllTimers())

    expect(screen.getByText('scoped fallback')).toBeTruthy()
    expect(recoveryWarningCount(warnSpy.mock.calls)).toBe(0)
  })

  it.each([
    ['a renamed lookup error', new Error('useClientLookup: Index 6 out of bounds (length: 2)')],
    ['an unrelated render error', new Error('some unrelated application error')]
  ])('does not auto-recover %s at root', (_label, error) => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const box: { error: Error | null } = { error }
    const Bomb = makeBomb(box)

    render(
      <ErrorBoundary label="root">
        <Bomb />
      </ErrorBoundary>
    )

    act(() => vi.runAllTimers())

    expect(screen.getByRole(RELOAD_WINDOW.role, { name: RELOAD_WINDOW.name })).toBeTruthy()
    expect(recoveryWarningCount(warnSpy.mock.calls)).toBe(0)
  })
})
