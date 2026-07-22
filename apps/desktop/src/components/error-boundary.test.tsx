import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ErrorBoundary } from './error-boundary'

const USE_LOOKUP_ERROR = new Error('useClientLookup: Index 0 out of bounds (length: 0)')
const TAP_LOOKUP_ERROR = new Error('tapClientLookup: Index 0 out of bounds (length: 0)')

function makeBomb(box: { error: Error | null }) {
  return function Bomb() {
    if (box.error) {
      throw box.error
    }

    return <div>recovered</div>
  }
}

function Fallback({ reset }: { reset: () => void }) {
  return <button onClick={reset}>manual reset</button>
}

function countLogCalls(spy: ReturnType<typeof vi.spyOn>, text: string) {
  return spy.mock.calls.filter((call: unknown[]) => call.some((value: unknown) => String(value).includes(text))).length
}

describe('ErrorBoundary client lookup recovery', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it.each([USE_LOOKUP_ERROR, TAP_LOOKUP_ERROR])('recovers exact root lookup errors after the first delay', error => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const recoveredSpy = vi.spyOn(console, 'info').mockImplementation(() => undefined)
    const box: { error: Error | null } = { error }
    const Bomb = makeBomb(box)

    render(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )

    expect(countLogCalls(warnSpy, 'client lookup recovery attempt 1/3')).toBe(1)
    act(() => vi.advanceTimersByTime(249))
    expect(screen.getByRole('button', { name: 'manual reset' })).toBeTruthy()

    box.error = null
    act(() => vi.advanceTimersByTime(1))

    expect(screen.getByText('recovered')).toBeTruthy()
    expect(countLogCalls(recoveredSpy, 'client lookup recovery recovered after attempt 1')).toBe(1)
  })

  it('waits for a later retry when the client store recovers slowly', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const box: { error: Error | null } = { error: USE_LOOKUP_ERROR }
    const Bomb = makeBomb(box)

    render(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )

    act(() => vi.advanceTimersByTime(250))
    expect(countLogCalls(warnSpy, 'client lookup recovery attempt')).toBe(2)

    box.error = null
    act(() => vi.advanceTimersByTime(999))
    expect(screen.getByRole('button', { name: 'manual reset' })).toBeTruthy()

    act(() => vi.advanceTimersByTime(1))
    expect(screen.getByText('recovered')).toBeTruthy()
  })

  it('stops after three persistent errors and reports exhaustion once', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const Bomb = makeBomb({ error: USE_LOOKUP_ERROR })

    render(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )

    act(() => vi.advanceTimersByTime(250))
    act(() => vi.advanceTimersByTime(1_000))
    act(() => vi.advanceTimersByTime(3_000))

    expect(countLogCalls(warnSpy, 'client lookup recovery attempt')).toBe(3)
    expect(countLogCalls(warnSpy, 'client lookup recovery exhausted')).toBe(1)
    expect(screen.getByRole('button', { name: 'manual reset' })).toBeTruthy()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not restore the retry budget until a recovery remains stable for 30 seconds', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const box: { error: Error | null } = { error: USE_LOOKUP_ERROR }
    const Bomb = makeBomb(box)

    const view = render(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )

    box.error = null
    act(() => vi.advanceTimersByTime(250))
    expect(screen.getByText('recovered')).toBeTruthy()

    act(() => vi.advanceTimersByTime(29_999))
    box.error = USE_LOOKUP_ERROR
    view.rerender(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )
    expect(countLogCalls(warnSpy, 'client lookup recovery attempt 2/3')).toBe(1)

    box.error = null
    act(() => vi.advanceTimersByTime(1_000))
    act(() => vi.advanceTimersByTime(30_000))

    box.error = USE_LOOKUP_ERROR
    view.rerender(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )
    expect(countLogCalls(warnSpy, 'client lookup recovery attempt 1/3')).toBe(2)
  })

  it('clears the retry budget when the user manually resets the boundary', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const box: { error: Error | null } = { error: USE_LOOKUP_ERROR }
    const Bomb = makeBomb(box)

    const view = render(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )

    act(() => vi.advanceTimersByTime(250))
    act(() => vi.advanceTimersByTime(1_000))
    act(() => vi.advanceTimersByTime(3_000))
    box.error = null
    fireEvent.click(screen.getByRole('button', { name: 'manual reset' }))
    expect(screen.getByText('recovered')).toBeTruthy()

    box.error = USE_LOOKUP_ERROR
    view.rerender(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )
    expect(countLogCalls(warnSpy, 'client lookup recovery attempt 1/3')).toBe(2)
  })

  it('clears pending timers when unmounted', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const Bomb = makeBomb({ error: USE_LOOKUP_ERROR })

    const { unmount } = render(
      <ErrorBoundary fallback={Fallback} label="root">
        <Bomb />
      </ErrorBoundary>
    )

    unmount()
    act(() => vi.runAllTimers())

    expect(vi.getTimerCount()).toBe(0)
    expect(countLogCalls(warnSpy, 'client lookup recovery attempt')).toBe(1)
  })

  it('does not double-schedule recovery in StrictMode', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const Bomb = makeBomb({ error: USE_LOOKUP_ERROR })

    render(
      <StrictMode>
        <ErrorBoundary fallback={Fallback} label="root">
          <Bomb />
        </ErrorBoundary>
      </StrictMode>
    )

    expect(countLogCalls(warnSpy, 'client lookup recovery attempt 1/3')).toBe(1)
    expect(vi.getTimerCount()).toBe(1)
  })

  it('does not auto-recover unrelated root errors', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const UnrelatedBomb = makeBomb({ error: new Error('tapClientResource: Index 0 out of bounds (length: 0)') })

    render(
      <ErrorBoundary fallback={Fallback} label="root">
        <UnrelatedBomb />
      </ErrorBoundary>
    )

    expect(vi.getTimerCount()).toBe(0)
    expect(warnSpy).not.toHaveBeenCalled()
  })

  it('does not auto-recover lookup errors outside the root boundary', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const ScopedBomb = makeBomb({ error: USE_LOOKUP_ERROR })

    render(
      <ErrorBoundary fallback={Fallback} label="thread">
        <ScopedBomb />
      </ErrorBoundary>
    )

    expect(vi.getTimerCount()).toBe(0)
    expect(warnSpy).not.toHaveBeenCalled()
  })

  it('auto-recovers lookup errors in an explicitly enabled scoped boundary', () => {
    const box: { error: Error | null } = { error: USE_LOOKUP_ERROR }
    const ScopedBomb = makeBomb(box)

    render(
      <ErrorBoundary fallback={Fallback} label="chat-routes" recoverClientLookup>
        <ScopedBomb />
      </ErrorBoundary>
    )

    box.error = null
    act(() => vi.advanceTimersByTime(250))

    expect(screen.getByText('recovered')).toBeTruthy()
  })

  it('resets a scoped failure when its connection identity changes', () => {
    const box: { error: Error | null } = { error: new Error('render failed') }
    const ScopedBomb = makeBomb(box)

    const view = render(
      <ErrorBoundary fallback={Fallback} label="chat-routes" resetKeys={['connecting', 'session-a']}>
        <ScopedBomb />
      </ErrorBoundary>
    )

    box.error = null
    view.rerender(
      <ErrorBoundary fallback={Fallback} label="chat-routes" resetKeys={['open', 'session-a']}>
        <ScopedBomb />
      </ErrorBoundary>
    )

    expect(screen.getByText('recovered')).toBeTruthy()
  })
})
