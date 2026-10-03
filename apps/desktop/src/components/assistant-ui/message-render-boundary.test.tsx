import { act, cleanup, render, screen } from '@testing-library/react'
import { Component, type ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MessageRenderBoundary } from './message-render-boundary'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

function Boom({ error }: { error: Error | null }): null {
  if (error) {
    throw error
  }

  return null
}

const lookupError = new Error('useClientLookup: Index 2 out of bounds (length: 2)')

const outerCaught: Error[] = []

// Records what propagates past MessageRenderBoundary, so the tests can tell
// a re-thrown error apart from a swallowed one.
class RecordingBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error) {
    outerCaught.push(error)
  }

  render() {
    return this.state.error ? null : this.props.children
  }
}

const lookupErrors = [
  ['useClientLookup', lookupError],
  ['tapClientLookup', new Error('tapClientLookup: Index 2 out of bounds (length: 2)')],
  ['tapClientResource', new Error('tapClientResource: Index 2 out of bounds (length: 2)')]
] as const

describe('MessageRenderBoundary', () => {
  it.each(lookupErrors)('swallows the transient %s out-of-bounds store race', (_label, error) => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    const { container } = render(
      <MessageRenderBoundary resetKey="a">
        <Boom error={error} />
      </MessageRenderBoundary>
    )

    expect(container.innerHTML).toBe('')
    spy.mockRestore()
  })

  it('recovers on the next consistent snapshot when resetKey changes', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    const { rerender } = render(
      <MessageRenderBoundary resetKey="a">
        <Boom error={lookupError} />
      </MessageRenderBoundary>
    )

    rerender(
      <MessageRenderBoundary resetKey="b">
        <Boom error={null} />
      </MessageRenderBoundary>
    )

    rerender(
      <MessageRenderBoundary resetKey="b">
        <div>recovered</div>
      </MessageRenderBoundary>
    )

    expect(screen.getByText('recovered')).toBeTruthy()
    spy.mockRestore()
  })

  it('re-throws unrelated errors so real bugs still surface', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    expect(() =>
      render(
        <MessageRenderBoundary resetKey="a">
          <Boom error={new Error('genuine render bug')} />
        </MessageRenderBoundary>
      )
    ).toThrow('genuine render bug')

    spy.mockRestore()
  })

  it('recovers on the retry timer without a resetKey change', () => {
    // The mid-turn race: the message list shrinks and regrows while
    // ids/roles/count stay stable, so resetKey never changes. The boundary
    // must self-retry on a timer instead of rendering null for the rest of
    // the turn.
    vi.useFakeTimers()
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    let failing = true

    function MaybeBoom() {
      if (failing) {
        throw new Error('useClientLookup: index 3 out of bounds')
      }

      return <div>turn content</div>
    }

    render(
      <MessageRenderBoundary resetKey="0:m1:user">
        <MaybeBoom />
      </MessageRenderBoundary>
    )

    expect(screen.queryByText('turn content')).toBeNull()

    failing = false

    act(() => {
      vi.advanceTimersByTime(0)
    })

    // Recovered through the retry timer alone; resetKey never changed.
    expect(screen.getByText('turn content')).toBeTruthy()
    spy.mockRestore()
  })

  it('recovers when the store settles after the zero-ms burst', () => {
    // Symptom A (#122167): the consistent store snapshot can arrive a few
    // frames after the lookup race fires. Retries that all burn in one
    // zero-ms burst exhaust the budget while the store is still
    // inconsistent, and the turn row then stays null until an unrelated
    // structural change. Retries must span the settle window.
    vi.useFakeTimers()
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    let failing = true

    function MaybeBoom() {
      if (failing) {
        throw new Error('useClientLookup: index 3 out of bounds')
      }

      return <div>late-settling content</div>
    }

    render(
      <MessageRenderBoundary resetKey="0:m1:user">
        <MaybeBoom />
      </MessageRenderBoundary>
    )

    expect(screen.queryByText('late-settling content')).toBeNull()

    // Burn the immediate burst while the store is still inconsistent: each
    // zero-ms retry arms the next one, so one advance only spends one retry.
    for (let burst = 0; burst < 6; burst += 1) {
      act(() => {
        vi.advanceTimersByTime(0)
      })
    }

    expect(screen.queryByText('late-settling content')).toBeNull()

    // The store settles a few frames later; the row must come back on its
    // own, with no resetKey change.
    failing = false

    act(() => {
      vi.advanceTimersByTime(1000)
    })

    expect(screen.getByText('late-settling content')).toBeTruthy()
    spy.mockRestore()
  })

  it('stops retrying after the transient retry cap', () => {
    // If the lookup stays out of bounds the boundary must give up instead of
    // looping a setState/render cycle forever: initial render plus 5 retries
    // (spread over the backoff schedule), then it stays null and arms no
    // further timer.
    vi.useFakeTimers()
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    let attempts = 0

    function AlwaysBoom(): null {
      attempts += 1
      throw lookupError
    }

    const { container } = render(
      <MessageRenderBoundary resetKey="a">
        <AlwaysBoom />
      </MessageRenderBoundary>
    )

    // React dev mode replays a failed render once per attempt, and an error
    // during the initial mount gets an extra sync retry from the root, so
    // measure the per-attempt cost from the first retry instead of guessing.
    const mountAttempts = attempts

    act(() => {
      vi.advanceTimersByTime(0)
    })

    const perRetry = attempts - mountAttempts

    // The remaining retries are spread over the backoff schedule. Timer
    // callbacks batch their setState inside act(), so each advance spends
    // exactly one retry; step past the whole schedule.
    for (let step = 0; step < 6; step += 1) {
      act(() => {
        vi.advanceTimersByTime(1000)
      })
    }

    // The boundary gave up: it stays null and arms no further timer.
    expect(vi.getTimerCount()).toBe(0)
    expect(container.innerHTML).toBe('')

    const settledAttempts = attempts

    act(() => {
      vi.advanceTimersByTime(10000)
    })

    expect(attempts).toBe(settledAttempts)
    spy.mockRestore()
  })

  it('resets the retry budget after a successful recovery', () => {
    // The cap bounds a single streak of consecutive transient catches. A
    // recovered boundary must get a fresh budget, otherwise enough separate
    // races over a long session would permanently blank the turn.
    vi.useFakeTimers()
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    let failing = true

    function MaybeBoom() {
      if (failing) {
        throw lookupError
      }

      return <div>turn content</div>
    }

    const { rerender } = render(
      <MessageRenderBoundary resetKey="a">
        <MaybeBoom />
      </MessageRenderBoundary>
    )

    // The mount is the first streak; five more follow. With a lifetime
    // budget the sixth streak would find the cap exhausted and stay blank.
    // Each rerender needs a fresh element: React bails out on an identical
    // element reference and the child would never re-render (or re-throw).
    for (let streak = 0; streak < 6; streak += 1) {
      expect(screen.queryByText('turn content')).toBeNull()

      failing = false

      act(() => {
        vi.advanceTimersByTime(0)
      })

      expect(screen.getByText('turn content')).toBeTruthy()

      failing = true

      rerender(
        <MessageRenderBoundary resetKey="a">
          <MaybeBoom />
        </MessageRenderBoundary>
      )
    }

    spy.mockRestore()
  })

  it('does not schedule a retry for non-transient errors', () => {
    vi.useFakeTimers()
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    outerCaught.length = 0

    render(
      <RecordingBoundary>
        <MessageRenderBoundary resetKey="a">
          <Boom error={new Error('boom')} />
        </MessageRenderBoundary>
      </RecordingBoundary>
    )

    // MessageRenderBoundary re-threw, the outer boundary caught it, and no
    // retry timer was armed for a failure that cannot heal itself.
    expect(outerCaught.map(error => error.message)).toContain('boom')
    expect(vi.getTimerCount()).toBe(0)
    spy.mockRestore()
  })
})
