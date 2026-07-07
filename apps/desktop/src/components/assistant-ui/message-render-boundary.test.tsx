import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { MessageRenderBoundary } from './message-render-boundary'

afterEach(cleanup)

function Boom({ error }: { error: Error | null }): null {
  if (error) {
    throw error
  }

  return null
}

const lookupError = new Error('tapClientLookup: Index 2 out of bounds (length: 2)')

describe('MessageRenderBoundary', () => {
  it('renders children when nothing throws', () => {
    render(
      <MessageRenderBoundary resetKey="a">
        <div>content</div>
      </MessageRenderBoundary>
    )

    expect(screen.getByText('content')).toBeTruthy()
  })

  it('swallows the transient tapClientLookup out-of-bounds store race', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    const { container } = render(
      <MessageRenderBoundary resetKey="a">
        <Boom error={lookupError} />
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

  it('contains unrelated errors to an inline fallback instead of unwinding to the root', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    render(
      <MessageRenderBoundary resetKey="a">
        <Boom error={new Error('genuine render bug')} />
      </MessageRenderBoundary>
    )

    expect(screen.getByRole('alert').textContent).toBe('This message failed to render.')
    expect(spy).toHaveBeenCalledWith(
      '[message-render-boundary]',
      expect.objectContaining({ message: 'genuine render bug' })
    )
    spy.mockRestore()
  })

  it('does not tag the transient lookup race in the console', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    render(
      <MessageRenderBoundary resetKey="a">
        <Boom error={lookupError} />
      </MessageRenderBoundary>
    )

    const tagged = spy.mock.calls.filter(call => call[0] === '[message-render-boundary]')
    expect(tagged).toEqual([])
    spy.mockRestore()
  })

  it('recovers from a contained error when resetKey changes', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    const { rerender } = render(
      <MessageRenderBoundary resetKey="a">
        <Boom error={new Error('genuine render bug')} />
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
})
