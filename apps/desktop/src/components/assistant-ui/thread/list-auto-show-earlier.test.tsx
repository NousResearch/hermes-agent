import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const SCROLL_H = 20000
const CLIENT_H = 600

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
  configurable: true,
  get: () => SCROLL_H
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get: () => CLIENT_H
})

beforeEach(() => window.localStorage.clear())

async function settle(ticks = 20) {
  await act(async () => {
    for (let tick = 0; tick < ticks; tick += 1) {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }
  })
}

// Heavy enough that the transcript outgrows RENDER_BUDGET (600 units) and the
// list hides older turns behind "Show earlier" — the shape of a branched
// session that inherits a long parent history.
const heavyText = 'x'.repeat(5000)
const createdAt = new Date('2026-08-01T00:00:00.000Z')

function heavyTranscript(turns: number): ThreadMessage[] {
  return Array.from({ length: turns }, (_, index) => [
    {
      id: `u-${index}`,
      role: 'user',
      content: [{ type: 'text', text: heavyText }],
      attachments: [],
      createdAt,
      metadata: { custom: {} }
    } as ThreadMessage,
    {
      id: `a-${index}`,
      role: 'assistant',
      content: [{ type: 'text', text: heavyText }],
      status: { type: 'complete', reason: 'stop' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as ThreadMessage
  ]).flat()
}

function Harness({ messages }: { messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({ isRunning: false, messages, onNew: async () => {} })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread sessionKey="long" />
    </AssistantRuntimeProvider>
  )
}

const mountedGroups = (container: HTMLElement) => container.querySelectorAll('[data-slot="aui_message-group"]').length

async function settledGroupCount(container: HTMLElement): Promise<number> {
  let count = mountedGroups(container)

  for (let round = 0; round < 20; round += 1) {
    await settle(10)
    const next = mountedGroups(container)

    if (next === count) {
      return count
    }

    count = next
  }

  return count
}

describe('top-edge auto Show earlier', () => {
  it('pages older turns in when the reader wheels up at the clamped top, not mid-transcript', async () => {
    const turns = 60
    const { container } = render(<Harness messages={heavyTranscript(turns)} />)
    const viewport = container.querySelector('[data-slot="aui_thread-viewport"]') as HTMLElement

    // Let the stepped first-paint → full-page backfill finish; from here only
    // Show earlier grows the DOM.
    const windowed = await settledGroupCount(container)

    expect(windowed).toBeGreaterThan(0)
    expect(windowed).toBeLessThan(turns)

    // Leave the bottom lock the way a reader does: a scroll-up event.
    viewport.scrollTop = SCROLL_H - CLIENT_H
    act(() => viewport.dispatchEvent(new Event('scroll')))
    viewport.scrollTop = 2000
    act(() => viewport.dispatchEvent(new Event('scroll')))
    await settle(5)

    act(() => viewport.dispatchEvent(new WheelEvent('wheel', { deltaY: -120 })))
    await settle(5)

    expect(mountedGroups(container)).toBe(windowed)

    // At the clamped top the browser emits no further scroll events — only
    // the wheel says "I want to read earlier".
    viewport.scrollTop = 0
    act(() => viewport.dispatchEvent(new Event('scroll')))
    await settle(5)
    act(() => viewport.dispatchEvent(new WheelEvent('wheel', { deltaY: -120 })))
    await settle()

    expect(mountedGroups(container)).toBeGreaterThan(windowed)
  })
})
