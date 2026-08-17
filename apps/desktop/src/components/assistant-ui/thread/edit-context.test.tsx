// Thread deliberately keeps cwd/gateway/sessionId OUT of the messageComponents
// memo deps: those values change on every session switch, and reminting the
// component types mid-switch remounts the whole outgoing transcript. The
// mounted edit composer still has to see a same-session change (e.g. a cwd
// remap). It used to read the values from a render-time ref, but a mounted
// composer never re-reads the ref when the change leaves every
// ThreadMessageList prop referentially equal (Thread and ThreadMessageList
// are both memo'd, so the wrapper never re-renders). The values now travel
// through ThreadEditContext, whose propagation reaches the mounted consumer
// through the memo bail-out. These tests pin both directions: the composer
// sees the change, and the transcript still does not remount.
import { ExportedMessageRepository } from '@assistant-ui/react'
import { AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { Thread } from '.'

interface MockComposerProps {
  cwd: string | null
  gateway: unknown
  sessionId: string | null
}

const composerRenders = vi.hoisted(() => [] as MockComposerProps[])

vi.mock('./user-edit-composer', () => ({
  UserEditComposer: (props: MockComposerProps) => {
    composerRenders.push(props)

    return <div data-testid="edit-composer">{props.cwd}</div>
  }
}))

const createdAt = new Date('2026-05-01T00:00:00.000Z')

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (str: string) => str })

Element.prototype.scrollTo = function scrollTo() {}

afterEach(() => {
  cleanup()
})

beforeEach(() => {
  composerRenders.length = 0
})

// jsdom returns 0 for offset*; the virtualizer reads those to size its
// viewport. Fall through to client* or a sane default so virtualized
// items render (same stub as user-message-edit.test.tsx).
function stubOffsetDimension(
  prop: 'offsetHeight' | 'offsetWidth',
  clientProp: 'clientHeight' | 'clientWidth',
  fallback: number
) {
  const previous = Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)

  Object.defineProperty(HTMLElement.prototype, prop, {
    configurable: true,
    get() {
      return previous?.get?.call(this) || (this as HTMLElement)[clientProp] || fallback
    }
  })
}

stubOffsetDimension('offsetWidth', 'clientWidth', 800)
stubOffsetDimension('offsetHeight', 'clientHeight', 600)

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'edit me please' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

const noopAsync = async () => {}

// The repository must stay referentially stable across rerenders: a new
// object would make the incremental runtime resync the transcript and
// unmount the open composer, defeating the test.
function Harness({ cwd, sessionKey }: { cwd: string; sessionKey: string }) {
  const [repository] = useState(() => ExportedMessageRepository.fromArray([userMessage(), assistantMessage()]))

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    setMessages: () => {},
    onNew: noopAsync,
    onEdit: noopAsync,
    onCancel: noopAsync,
    onReload: noopAsync
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread cwd={cwd} sessionKey={sessionKey} />
    </AssistantRuntimeProvider>
  )
}

describe('thread edit context', () => {
  it('passes a same-session cwd change to the mounted edit composer', async () => {
    const { rerender } = render(<Harness cwd="/old" sessionKey="k1" />)

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    await screen.findByTestId('edit-composer')

    expect(composerRenders.at(-1)?.cwd).toBe('/old')

    // Same session, same messages: every ThreadMessageList prop stays
    // referentially equal, so only context propagation can reach the
    // mounted composer.
    await act(async () => {
      rerender(<Harness cwd="/new" sessionKey="k1" />)
    })

    expect(composerRenders.at(-1)?.cwd).toBe('/new')
    expect(screen.getByTestId('edit-composer').textContent).toBe('/new')
  })

  it('still passes the new cwd after a session switch', async () => {
    const { rerender } = render(<Harness cwd="/old" sessionKey="k1" />)

    await act(async () => {
      rerender(<Harness cwd="/new" sessionKey="k2" />)
    })

    fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
    await screen.findByTestId('edit-composer')

    expect(composerRenders.at(-1)?.cwd).toBe('/new')
  })

  it('does not remount the transcript when cwd changes', async () => {
    // The perf invariant behind keeping cwd out of the memo deps: a cwd
    // change with identical messages must not remint the component types,
    // so the mounted message DOM nodes survive the rerender.
    const { rerender } = render(<Harness cwd="/old" sessionKey="k1" />)

    await waitFor(() => {
      expect(screen.getByText('done')).toBeTruthy()
      expect(screen.getByText('edit me please')).toBeTruthy()
    })

    const assistantBefore = screen.getByText('done')
    const userBefore = screen.getByText('edit me please')

    await act(async () => {
      rerender(<Harness cwd="/new" sessionKey="k1" />)
    })

    expect(screen.getByText('done')).toBe(assistantBefore)
    expect(screen.getByText('edit me please')).toBe(userBefore)
  })
})
