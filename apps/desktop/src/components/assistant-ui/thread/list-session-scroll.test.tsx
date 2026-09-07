import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { setActiveProfile } from '@/store/profile'
import { saveThreadScrollPosition } from '@/store/thread-scroll'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const SCROLL_H = 5000
const CLIENT_H = 600
let scrollHeightValue = SCROLL_H

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
  configurable: true,
  get() {
    return scrollHeightValue
  }
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get() {
    return CLIENT_H
  }
})

beforeEach(() => {
  scrollHeightValue = SCROLL_H
  window.localStorage.clear()
  setActiveProfile('default')
})

const VIEWPORT_SLOT = 'aui_thread-viewport'

function viewportEl(container: HTMLElement): HTMLElement {
  const el = container.querySelector(`[data-slot="${VIEWPORT_SLOT}"]`) as HTMLElement | null
  expect(el).toBeTruthy()

  return el!
}

async function settleScroll(ticks = 3) {
  await act(async () => {
    for (let tick = 0; tick < ticks; tick += 1) {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }
  })
}

const createdAt = new Date('2026-08-01T00:00:00.000Z')

function sessionMessages(key: string, turns = 1): ThreadMessage[] {
  return Array.from({ length: turns }, (_, index) => [
    {
      id: `u-${key}-${index}`,
      role: 'user',
      content: [{ type: 'text', text: `message ${index} in ${key}` }],
      attachments: [],
      createdAt,
      metadata: { custom: {} }
    } as ThreadMessage,
    {
      id: `a-${key}-${index}`,
      role: 'assistant',
      content: [{ type: 'text', text: `response ${index} in ${key}` }],
      status: { type: 'complete', reason: 'stop' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as ThreadMessage
  ]).flat()
}

interface ScrollHarnessProps {
  messages: ThreadMessage[]
  sessionKey: string | null
}

function ScrollHarness({ messages, sessionKey }: ScrollHarnessProps) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread sessionKey={sessionKey} />
    </AssistantRuntimeProvider>
  )
}

describe('list session-scroll restore', () => {
  it('restores a reading offset on return after switching away', async () => {
    saveThreadScrollPosition('a', { fromBottom: 800, kind: 'offset' })

    const { container, rerender } = render(<ScrollHarness messages={sessionMessages('a')} sessionKey="a" />)
    const vp = viewportEl(container)

    await settleScroll()

    expect(vp.scrollTop).toBe(SCROLL_H - 800 - CLIENT_H)

    rerender(<ScrollHarness messages={sessionMessages('b')} sessionKey="b" />)
    const vpB = viewportEl(container)

    await settleScroll()

    expect(vpB.scrollTop).toBe(SCROLL_H - CLIENT_H)

    rerender(<ScrollHarness messages={sessionMessages('a')} sessionKey="a" />)
    const vpA = viewportEl(container)

    await settleScroll()

    expect(vpA.scrollTop).toBe(SCROLL_H - 800 - CLIENT_H)
  })

  it('keeps a clamped cold offset parked until the transcript is tall enough', async () => {
    saveThreadScrollPosition('b', { fromBottom: 800, kind: 'offset' })
    scrollHeightValue = CLIENT_H

    const { container, rerender } = render(<ScrollHarness messages={[]} sessionKey="b" />)
    const vp = viewportEl(container)

    await settleScroll()

    scrollHeightValue = 1000
    rerender(<ScrollHarness messages={sessionMessages('b')} sessionKey="b" />)
    await settleScroll(20)

    expect(vp.scrollTop).toBe(0)

    rerender(<ScrollHarness messages={sessionMessages('b', 2)} sessionKey="b" />)
    await settleScroll()

    expect(vp.scrollTop).toBe(0)

    scrollHeightValue = 2000
    rerender(<ScrollHarness messages={sessionMessages('b', 3)} sessionKey="b" />)
    await settleScroll()

    expect(vp.scrollTop).toBe(2000 - CLIENT_H - 800)
  })

})
