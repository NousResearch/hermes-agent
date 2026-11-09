import { act, cleanup, render } from '@testing-library/react'
import { useRef, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TIMELINE_REVEAL_EVENT } from './timeline-data'
import { TranscriptWindowProvider } from './transcript-window'
import { useTimelineReveal } from './use-timeline-reveal'

vi.mock('@/app/chat/session-view', () => ({ useSessionView: () => view }))
const view = { $messages: { get: () => [] } }

beforeEach(() => vi.stubGlobal('CSS', { escape: (id: string) => id }))
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function Harness({ fetchPage, signal }: { fetchPage: () => Promise<string | null>; signal: AbortSignal }) {
  const [id, setId] = useState('latest')

  return (
    <TranscriptWindowProvider
      value={{
        olderAvailable: true,
        expandWindow: vi.fn(),
        revealRow: async () => {
          const next = await fetchPage()

          if (next && !signal.aborted) {
            setId(next)
          }

          return next
        }
      }}
    >
      <List id={id} />
    </TranscriptWindowProvider>
  )
}

function List({ id }: { id: string }) {
  const viewport = useRef<HTMLDivElement>(null)
  useTimelineReveal({
    viewport,
    groups: [{ id, weight: 1 }],
    hiddenCount: 0,
    renderBudget: 600,
    olderAvailable: true,
    revealBudget: vi.fn(),
    expandWindow: vi.fn(),
    prepare: vi.fn(),
    sessionKey: 'session'
  })

  return (
    <div data-testid="viewport" ref={viewport}>
      <div data-message-id={id}>{id}</div>
    </div>
  )
}

describe('timeline direct reveal', () => {
  it('waits for a single bounded around-page commit and resolves the mounted id', async () => {
    const controller = new AbortController()
    const fetchPage = vi.fn(async () => 'old-target')
    const complete = vi.fn()
    const ui = render(<Harness fetchPage={fetchPage} signal={controller.signal} />)
    await act(async () => {
      ui.getByTestId('viewport').dispatchEvent(
        new CustomEvent(TIMELINE_REVEAL_EVENT, {
          detail: { id: 'history:42', rowId: 42, signal: controller.signal, complete }
        })
      )
    })
    expect(fetchPage).toHaveBeenCalledOnce()
    expect(complete).toHaveBeenCalledWith('old-target')
  })

  it('leaves existing content when a direct read fails', async () => {
    const controller = new AbortController()
    const complete = vi.fn()
    const ui = render(<Harness fetchPage={async () => null} signal={controller.signal} />)
    await act(async () => {
      ui.getByTestId('viewport').dispatchEvent(
        new CustomEvent(TIMELINE_REVEAL_EVENT, { detail: { id: 'history:42', signal: controller.signal, complete } })
      )
    })
    expect(complete).toHaveBeenCalledWith(false)
    expect(ui.getByText('latest')).toBeTruthy()
  })
})
