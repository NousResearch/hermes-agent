import { act, cleanup, render, screen } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'

import { setStickyUserMessage } from '@/store/sticky-user-message'

import {
  assistantMessage,
  stubThreadEnvironment,
  stubThreadViewportSize,
  ThreadRuntime,
  userMessage
} from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const PROMPT = 'keep me pinned'

// The pin's carrier is Tailwind utilities on the container (sticky z-40, in
// StickyHumanMessageContainer), but jsdom does not load the app's stylesheet
// (the vitest `css.include` filter carries only status-stack.css), so computed
// style would read static for pinned and unpinned bubbles alike. These two
// rules are the compiled form of exactly those utilities — nothing else — and
// they exist so the assertions below can read the BEHAVIOR (computed position
// and stacking) instead of class strings: a carrier change that renders the
// same pin still passes, and a regression that keeps the strings but drops the
// pin fails. If the gate's carrier classes change, update the rules with them.
beforeAll(() => {
  const style = window.document.createElement('style')
  style.textContent = '.sticky { position: sticky; }\n.z-40 { z-index: 40; }'
  window.document.head.append(style)
})

afterAll(() => {
  window.document.head.lastElementChild?.remove()
})

const pin = (root: HTMLElement) => {
  const computed = getComputedStyle(root)

  return { position: computed.position, zIndex: computed.zIndex }
}

const PINNED = { position: 'sticky', zIndex: '40' }
const UNPINNED = { position: 'static', zIndex: 'auto' }

const userRoot = (container: HTMLElement) => {
  const root = container.querySelector<HTMLElement>('[data-slot="aui_user-message-root"]')

  expect(root).toBeTruthy()

  return root!
}

function renderThread() {
  return render(
    <ThreadRuntime messages={[userMessage('user-1', PROMPT), assistantMessage()]}>
      <Thread />
    </ThreadRuntime>
  )
}

afterEach(() => {
  cleanup()
  act(() => setStickyUserMessage(true))
})

describe('sticky user message toggle', () => {
  // The shipped behavior: every user bubble is a sticky layer so the prompt
  // that started the turn stays on screen while its reply streams.
  it('pins the user bubble by default', async () => {
    const { container } = renderThread()
    await screen.findByText(PROMPT)

    expect(pin(userRoot(container))).toEqual(PINNED)
  })

  // Opt-out (#38372/#80146): the gate is the container itself, so the bubble
  // scrolls like any other message and the switch is the one lever every
  // pinnable surface reads. The clamp (#39721) is independent and stays.
  it('drops the pin while the switch is off and restores it on', async () => {
    act(() => setStickyUserMessage(false))
    const { container } = renderThread()
    await screen.findByText(PROMPT)

    expect(pin(userRoot(container))).toEqual(UNPINNED)

    act(() => setStickyUserMessage(true))

    expect(pin(userRoot(container))).toEqual(PINNED)
  })

  it('unpins live without a remount', async () => {
    const { container } = renderThread()
    await screen.findByText(PROMPT)

    const pinned = userRoot(container)

    expect(pin(pinned)).toEqual(PINNED)

    act(() => setStickyUserMessage(false))

    expect(userRoot(container)).toBe(pinned)
    expect(pin(pinned)).toEqual(UNPINNED)

    act(() => setStickyUserMessage(true))

    expect(pin(pinned)).toEqual(PINNED)
  })
})
