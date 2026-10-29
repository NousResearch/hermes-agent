import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import { setActiveProfile } from '@/store/profile'
import { getThreadScrollPosition, saveThreadScrollPosition } from '@/store/thread-scroll'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

/**
 * #131165's self-perpetuating loop: a remembered reading position beyond the
 * mounted window's scroll range clamps the restore to the TOP of the bounded
 * window, input cancels the loop there, and the cleanup then persists the
 * clamp's metrics as the session's reading position — so every later open
 * strands at the same place. The reporter measured {fromBottom: 7416.47}
 * reopening the transcript 7416px above the live tail, on every open.
 */
stubThreadEnvironment()
stubThreadViewportSize()

const SCROLL_H = 5000
const CLIENT_H = 600
const CLEARANCE_H = 120
const VIEWPORT_SLOT = 'aui_thread-viewport'
const CLEARANCE_SLOT = 'aui_composer-clearance'

let viewportScrollHeight = SCROLL_H
let viewportClientHeight = CLIENT_H
let clearanceHeight = CLEARANCE_H

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
  configurable: true,
  get() {
    return viewportScrollHeight
  }
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get() {
    if (this.getAttribute?.('data-slot') === CLEARANCE_SLOT) {
      return clearanceHeight
    }

    return viewportClientHeight
  }
})

beforeEach(() => {
  viewportScrollHeight = SCROLL_H
  viewportClientHeight = CLIENT_H
  clearanceHeight = CLEARANCE_H
  window.localStorage.clear()
  setActiveProfile('default')
  rescopeConnectionScopedStores(null)
})

const createdAt = new Date('2026-08-01T00:00:00.000Z')

function messages(key: string, turns = 1): ThreadMessage[] {
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

function Harness({ sessionKey }: { sessionKey: string }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages: messages(sessionKey),
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread clampToComposer sessionKey={sessionKey} />
    </AssistantRuntimeProvider>
  )
}

function viewportEl(container: HTMLElement): HTMLElement {
  const el = container.querySelector(`[data-slot="${VIEWPORT_SLOT}"]`) as HTMLElement | null
  expect(el).toBeTruthy()

  return el!
}

describe('a stranded landing is never persisted as the reading position', () => {
  it('keeps the remembered position when input cancels a restore that is still clamped', () => {
    // 9000px from the bottom of a 4400px scroll range: unreachable in the
    // mounted window, so the restore clamps at its top.
    saveThreadScrollPosition('stranded', { fromBottom: 9000, kind: 'offset' })

    const { container, unmount } = render(<Harness sessionKey="stranded" />)
    const vp = viewportEl(container)

    // A wheel notch is input the loop must yield to — at the clamped top no
    // scroll event fires, so the reader cannot cancel it any other way.
    act(() => {
      vp.dispatchEvent(new WheelEvent('wheel', { deltaY: -1 }))
    })

    unmount()

    expect(getThreadScrollPosition('stranded')).toEqual({ fromBottom: 9000, kind: 'offset' })
  })

  it('still records a reader who genuinely moves during the unfinished load', () => {
    saveThreadScrollPosition('stranded-reader', { fromBottom: 9000, kind: 'offset' })

    const { container, unmount } = render(<Harness sessionKey="stranded-reader" />)
    const vp = viewportEl(container)

    act(() => {
      vp.scrollTop = 3000
      vp.dispatchEvent(new Event('scroll'))
    })

    unmount()

    // Their real distance from the bottom (5000 - 3000 - 600), not the
    // remembered target and not the clamp's metrics.
    expect(getThreadScrollPosition('stranded-reader')).toEqual({ fromBottom: 1400, kind: 'offset' })
  })
})
