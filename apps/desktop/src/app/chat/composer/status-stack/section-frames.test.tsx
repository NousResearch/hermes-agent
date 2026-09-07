import { cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { $todosBySession } from '@/store/todos'

import { ComposerStatusStack } from './index'

/**
 * Sections in the stack are framed siblings: each is wrapped in its own
 * `[data-status-section]` element, which is the hook the seam rule in
 * styles.css hangs a hairline on. Without the wrapper, two adjacent sections
 * paint as one block and the lower one reads as a footnote of the one above.
 */
describe('ComposerStatusStack section frames', () => {
  beforeAll(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        disconnect() {}
        observe() {}
      }
    )
  })

  afterEach(() => {
    cleanup()
    $todosBySession.set({})
  })

  const renderStack = (queue: React.ReactNode = null) =>
    render(
      <MemoryRouter>
        <ComposerStatusStack queue={queue} sessionId="session-1" />
      </MemoryRouter>
    )

  it('frames every rendered section separately', () => {
    $todosBySession.set({
      'session-1': [{ content: 'Wire the status stack', id: '1', status: 'in_progress' }]
    })

    const view = renderStack(<div data-testid="queue-section">Queued</div>)

    expect(view.container.querySelectorAll('[data-status-section]').length).toBe(2)
  })

  it('keeps each section inside its own frame, so a seam always has one owner per side', () => {
    $todosBySession.set({
      'session-1': [{ content: 'Wire the status stack', id: '1', status: 'in_progress' }]
    })

    const view = renderStack(<div data-testid="queue-section">Queued</div>)
    const todo = view.container.querySelector('[data-testid="queue-section"]')
    const todoFrame = todo?.closest('[data-status-section]')
    const otherFrame = [...view.container.querySelectorAll('[data-status-section]')].find(f => f !== todoFrame)

    expect(todoFrame).toBeTruthy()
    expect(otherFrame).toBeTruthy()
    expect(otherFrame?.contains(todo as Node)).toBe(false)
  })

  it('frames a lone section too, so one section and many share the same shape', () => {
    const view = renderStack(<div data-testid="queue-section">Queued</div>)

    expect(view.container.querySelectorAll('[data-status-section]').length).toBe(1)
  })
})
